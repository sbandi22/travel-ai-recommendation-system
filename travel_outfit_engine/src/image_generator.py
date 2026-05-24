"""
Image generator for the travel outfit engine.

Backend: Hugging Face Inference API (FLUX.1-schnell via router.huggingface.co).
Requires HF_API_KEY environment variable.

Prompt structure:
  Structured 5-part template:
    1. Anatomy anchor  — locks gender, symmetry, framing
    2. Pose lock       — explicit stance prevents unnatural contortion
    3. Outfit block    — top + bottom + outerwear + footwear + accessories
    4. Scene block     — setting + weather + time-of-day lighting
    5. Quality tags    — photography style, lens, focus keywords

Best-candidate selection: generates CANDIDATES images, picks largest file size
as proxy for sharpness and detail.
"""

import os
import random
import uuid
import hashlib
import time
from pathlib import Path

import httpx

from src.prompt_builder import (
    StyleVariant,
    get_variants,
    pick_color_palette,
    resolve_accessories,
)

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path("generated_images")
IMAGE_WIDTH  = 512
IMAGE_HEIGHT = 768
HF_MODEL     = "black-forest-labs/FLUX.1-schnell"
HF_API_URL   = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# Minimum acceptable image size in bytes.
# Anything below this is likely a blank, corrupted, or collapsed output.
MIN_IMAGE_BYTES = 20_000

# Number of candidates to generate and pick the best from.
# Best = largest file size, which correlates strongly with detail and realism.
CANDIDATES = 1

# ─── Prompt building ──────────────────────────────────────────────────────────

_ACTIVITY_SETTING = {
    "Hiking":   "mountain trail backdrop, natural wilderness",
    "Museum":   "modern art gallery interior, white walls, polished floor",
    "Dinner":   "upscale restaurant, ambient candlelight, elegant interior",
    "Business": "contemporary office lobby, glass and steel architecture",
    "Beach":    "sandy beach, ocean waves, warm coastal light",
    "Walking":  "urban city sidewalk, blurred storefronts in background",
}

_WEATHER_LIGHTING = {
    "Sunny":  "bright golden sunlight, sharp natural shadows",
    "Cloudy": "soft overcast diffused light, even neutral tones",
    "Rain":   "moody rainy atmosphere, wet reflective surfaces",
    "Snowy":  "crisp cold winter light, blue-white tones",
}

_TIME_LIGHTING = {
    "morning":   "warm early morning golden hour light",
    "afternoon": "bright clear midday natural light",
    "evening":   "warm amber evening light, soft shadows",
}

# Negative prompt — suppresses the most common failure modes:
# distorted anatomy, bad hands, blurry output, wrong gender clothing
_NEGATIVE_BASE = (
    # Style exclusions
    "cartoon, anime, illustration, painting, sketch, 3D render, CGI, "
    "watermark, text overlay, logo, signature, "
    # Quality exclusions
    "blurry, out of focus, low resolution, grainy, jpeg artefacts, "
    "overexposed, underexposed, harsh flash, dark shadows on face, "
    # Anatomy — most important section for reducing distortion
    "bad anatomy, deformed body, extra limbs, missing limbs, floating limbs, "
    "extra fingers, fused fingers, mutated hands, poorly drawn hands, "
    "disfigured face, asymmetric face, uneven eyes, cross-eyed, lazy eye, "
    "tilted head, crooked nose, misaligned features, "
    "long neck, short torso, disproportionate legs, "
    # Framing exclusions
    "cropped feet, cut-off figure, partial body, head cut off, "
    "multiple people, duplicate person, crowd, "
    # Texture/skin exclusions
    "plastic skin, waxy face, airbrushed skin, unrealistic clothing texture"
)

_NEGATIVE_CATEGORY = {
    "Athletic":   "suit, formal dress, heels, office wear",
    "Formal":     "sportswear, gym wear, ripped clothing, sneakers",
    "Rain Ready": "summer shorts, sandals, dry sunny beach",
    "Casual":     "ball gown, tuxedo, military uniform",
}

_NEGATIVE_GENDER = {
    "male":   "dress, skirt, feminine clothing",
    "female": "suit and tie, masculine silhouette",
}


def _build_prompt(
    event_features: dict,
    variant: StyleVariant,
    gender: str,
    outfit_items: list[str] | None,
    color_palette: str | None = None,
    accessory_override: str | None = None,
) -> str:
    """
    Build a structured prompt in 5 explicit sections.

    Section order matters for FLUX — it weights earlier tokens more strongly,
    so anatomy and framing constraints come FIRST, before clothing detail.

    1. Anatomy anchor  — symmetry + framing locks (prevents distortion)
    2. Pose lock       — explicit stance prevents unnatural contortion
    3. Outfit block    — clothing, footwear, accessories
    4. Scene + light   — setting, weather, time of day
    5. Quality tags    — photography style reinforcement
    """
    is_female   = gender.lower() in ("female", "woman", "women", "f")
    gender_word = "woman" if is_female else "man"

    # 1. Anatomy anchor — placed FIRST so FLUX weights it highest
    # "symmetrical face" and "two hands" are the most effective distortion guards
    anatomy = (
        f"full body fashion photograph of one {gender_word}, "
        "symmetrical face, both eyes level, realistic body proportions, "
        "complete figure from head to feet, two hands, five fingers each"
    )

    # 2. Pose lock — upright standing reduces limb contortion artifacts
    pose = "standing upright, natural relaxed pose, facing camera"

    # 3. Outfit block
    # Strip banned item types for women — shirts/tees/cargos have no place in
    # women's outfit prompts; the variant's women_* fields handle alternatives.
    _WOMEN_BANNED = {"shirt", "tee", "t-shirt", "tshirt", "cargo", "jeans", "trousers", "shorts", "pants"}
    if outfit_items and is_female:
        outfit_items = [
            item for item in outfit_items
            if not any(b in item.lower() for b in _WOMEN_BANNED)
        ] or None  # fall back to variant fields if everything was filtered

    if outfit_items:
        clothing = ", ".join(outfit_items)
    elif is_female and variant.women_top.lower() in ("dress", "none", ""):
        clothing = variant.women_bottom
    elif is_female and variant.women_top:
        clothing = f"{variant.women_top}, {variant.women_bottom}"
        if variant.women_outerwear and variant.women_outerwear.lower() != "none":
            clothing += f", {variant.women_outerwear}"
    else:
        clothing = f"{variant.top}, {variant.bottom}"
        if variant.outerwear and variant.outerwear.lower() != "none":
            clothing += f", {variant.outerwear}"

    footwear    = variant.women_footwear    if is_female and variant.women_footwear    else variant.footwear
    accessories = variant.women_accessories if is_female and variant.women_accessories else variant.accessories
    active_accessories = accessory_override or accessories
    active_colors      = color_palette or variant.colors
    outfit_block = f"wearing {clothing}, {footwear}, {active_accessories}, {active_colors}"

    # 4. Scene + lighting
    setting      = _ACTIVITY_SETTING.get(event_features.get("activity", ""), "neutral clean studio")
    weather_note = _WEATHER_LIGHTING.get(event_features.get("weather", "Sunny"), "natural daylight")
    time_note    = _TIME_LIGHTING.get(event_features.get("time", "afternoon"), "natural light")
    scene_block  = f"{setting}, {weather_note}, {time_note}"

    # 5. Quality tags
    quality = (
        "editorial fashion photography, Vogue magazine, "
        "85mm f/1.8 lens, sharp focus on clothing, "
        "photorealistic, professional colour grading, 8K resolution"
    )

    return f"{anatomy}, {pose}, {outfit_block}, {scene_block}, {quality}"


def _build_negative(category: str, gender: str) -> str:
    parts = [
        _NEGATIVE_BASE,
        _NEGATIVE_CATEGORY.get(category, ""),
        _NEGATIVE_GENDER.get(gender.lower(), ""),
    ]
    return ", ".join(p for p in parts if p)


# ─── Prompt cache ─────────────────────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# ─── Filename ─────────────────────────────────────────────────────────────────

def _filename(category: str, label: str, prompt_key: str = "") -> str:
    # Use prompt hash as uid so the filename is deterministic — survives server restarts.
    uid      = prompt_key[:8] if prompt_key else uuid.uuid4().hex[:8]
    safe_cat = category.lower().replace(" ", "_")
    safe_lbl = label.lower().replace(" ", "_")
    return f"{safe_cat}_{safe_lbl}_{uid}.jpg"


# ─── Core generation ──────────────────────────────────────────────────────────

def _call_api(prompt: str, negative: str, seed: int) -> bytes:
    """Call HF Inference API and return raw image bytes."""
    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        raise EnvironmentError("HF_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative,
            "num_inference_steps": 4,
            "guidance_scale": 3.5,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "seed": seed,
        },
    }
    response = httpx.post(HF_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.content


def _generate_one(
    prompt: str,
    negative: str,
    category: str,
    variant: StyleVariant,
    gender: str,
) -> str:
    """
    Generate CANDIDATES images with different seeds, save the best one.

    Best = largest file size. This is a reliable proxy for image quality:
    - Blurry / collapsed outputs are small (few bytes of uniform colour)
    - Detailed, sharp images compress less and produce larger JPEG files
    If all candidates are below MIN_IMAGE_BYTES, the generation is rejected.
    """
    key = _cache_key(prompt)

    # Deterministic filepath — same prompt always maps to the same filename.
    # This lets us skip generation even after a server restart if the file exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / _filename(category, f"{gender}_{variant.name}", key)

    if filepath.exists() and filepath.stat().st_size > MIN_IMAGE_BYTES:
        print(f"    [disk-cache] reusing {filepath}")
        _prompt_cache[key] = str(filepath)
        return str(filepath)

    if key in _prompt_cache:
        cached = Path(_prompt_cache[key])
        if cached.exists() and cached.stat().st_size > MIN_IMAGE_BYTES:
            print(f"    [mem-cache] reusing {cached}")
            return str(cached)

    base_seed = abs(hash(prompt)) % 99999
    candidates: list[bytes] = []

    for i in range(CANDIDATES):
        seed = base_seed + i
        try:
            data = _call_api(prompt, negative, seed)
            candidates.append(data)
            print(f"    [candidate {i+1}/{CANDIDATES}] {len(data):,} bytes")
        except Exception as e:
            print(f"    [candidate {i+1}/{CANDIDATES}] failed: {e}")
            time.sleep(2)

    if not candidates:
        raise RuntimeError("All generation candidates failed.")

    # Pick the largest — best proxy for sharpness and detail
    best = max(candidates, key=len)

    if len(best) < MIN_IMAGE_BYTES:
        raise RuntimeError(
            f"Best candidate only {len(best):,} bytes — likely blank or corrupt. "
            "Try re-running or check the API."
        )

    filepath.write_bytes(best)
    _prompt_cache[key] = str(filepath)
    return str(filepath)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_outfit_images(
    category: str,
    event_features: dict,
    outfit_items: list[str] | None = None,
    n: int = 1,
    gender: str = "male",
) -> list[str]:
    """
    Generate `n` outfit image variations for the given category and context.

    Args:
        category:       Predicted outfit category ("Casual", "Formal",
                        "Athletic", "Rain Ready").
        event_features: Dict with keys: temp, rain, activity, weather, time.
        outfit_items:   Optional specific clothing items.
        n:              Number of variations (default 1).
        gender:         "male" or "female" (default "male").

    Returns:
        List of local file paths to saved images.
    """
    variants = get_variants(category)
    n        = min(n, len(variants))
    negative = _build_negative(category, gender)
    activity = event_features.get("activity", "")
    saved_paths = []

    # Seed variant selection from event context so the same event always
    # produces the same variant, but different events/genders get variety.
    variant_seed = abs(hash(
        f"{category}{gender}{activity}"
        f"{event_features.get('temp', 0)}"
        f"{event_features.get('day', '')}"
        f"{event_features.get('slot', '')}"
        f"{event_features.get('city', '')}"
    ))
    rng = random.Random(variant_seed)
    # Shuffle a copy so we cycle through all variants without repeating
    shuffled = list(variants)
    rng.shuffle(shuffled)

    for i in range(n):
        variant       = shuffled[i % len(shuffled)]
        palette       = pick_color_palette(seed=variant_seed + i)
        accessories   = resolve_accessories(activity, variant.women_accessories if gender == "female" and variant.women_accessories else variant.accessories)
        prompt        = _build_prompt(event_features, variant, gender, outfit_items, palette, accessories)

        print(f"  [{i+1}/{n}] [{gender}] {category} - {variant.name}")
        print(f"        Palette : {palette}")
        print(f"        Prompt  : {prompt[:140]}...")

        path = _generate_one(prompt, negative, category, variant, gender)
        saved_paths.append(path)
        print(f"        Saved   -> {path}")

    return saved_paths


def generate_outfit_image(
    category: str,
    event_features: dict,
    outfit_items: list[str] | None = None,
    gender: str = "male",
) -> str:
    """Convenience wrapper — generate a single image and return its path."""
    return generate_outfit_images(category, event_features, outfit_items, n=1, gender=gender)[0]
