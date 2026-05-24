"""
Prompt builder for outfit image generation.

Each StyleVariant stores outfit components as separate labeled fields
(top, bottom, outerwear, footwear, accessories) rather than a prose blob.
build_prompt() assembles these into a two-section format:
  1. Narrative description  — sets scene, model, clothing story
  2. Keyword reinforcement  — comma-separated tags that diffusion models
                              weight strongly to reinforce critical attributes

This two-section layout is recommended for FLUX.1 and SDXL-family models:
the narrative anchors composition while the tags reinforce fine details.

Public API
----------
build_prompt(event_features, category, style_variant, gender, outfit_items) -> str
build_negative_prompt(category, gender) -> str
get_variants(category) -> list[StyleVariant]
STYLE_VARIANTS: dict[str, list[StyleVariant]]
"""

import random
from dataclasses import dataclass, field


# ─── Style variant definition ─────────────────────────────────────────────────

@dataclass(frozen=True)
class StyleVariant:
    """
    Holds all the outfit attributes for one named style.

    Outfit fields are split into explicit components (top, bottom, outerwear)
    rather than a single prose 'layering' string so that build_prompt() can
    emit clearly labeled sections. Women's fields fall back to men's when empty.
    """
    name: str           # e.g. "Streetwear", "Minimalist"
    style_label: str    # short style tag used in the keyword block, e.g. "urban streetwear"

    # Shared visual attributes
    colors: str         # full palette description
    textures: str       # fabric / material descriptions
    pose: str           # model pose and framing direction

    # ── Men's outfit components ──────────────────────────────────────────────
    top: str            # shirt / tee / sweater
    bottom: str         # trousers / jeans / shorts / skirt
    outerwear: str      # jacket / coat / layer worn over top ("none" if absent)
    footwear: str
    accessories: str

    # ── Women's outfit components (fall back to men's fields when empty) ─────
    women_top: str        = field(default="")
    women_bottom: str     = field(default="")
    women_outerwear: str  = field(default="")
    women_footwear: str   = field(default="")
    women_accessories: str = field(default="")


# ─── Per-category style variants ─────────────────────────────────────────────
# Five named variants per category — each differs in silhouette, palette,
# layering structure, footwear, and accessories.

STYLE_VARIANTS: dict[str, list[StyleVariant]] = {

    # ── CASUAL ────────────────────────────────────────────────────────────────
    "Casual": [
        StyleVariant(
            name="Streetwear",
            style_label="urban streetwear casual",
            colors="olive green, washed white, indigo denim, gum sole tan",
            textures="heavyweight washed cotton, vintage denim, rubberised sole",
            pose="relaxed three-quarter stance, hands in pockets, slight forward lean",
            top="oversized relaxed-fit graphic tee with dropped shoulder, slightly cropped",
            bottom="wide-leg olive cargo trousers with side cargo pockets, cuffs rolled once",
            outerwear="unbuttoned light-wash denim jacket, sleeves pushed up",
            footwear="chunky-soled retro low-top sneakers in white with gum rubber sole",
            accessories="minimal silver chain necklace, clean washed cotton cap",
            women_top="dress",
            women_bottom="flowy tiered midi sundress in warm ivory linen with thin spaghetti straps and subtle terracotta floral print",
            women_outerwear="none",
            women_footwear="tan leather flat sandals with ankle strap",
            women_accessories="woven raffia tote bag, layered gold necklace, small hoop earrings",
        ),
        StyleVariant(
            name="Minimalist",
            style_label="clean minimalist",
            colors="crisp white, warm stone grey, natural ivory",
            textures="smooth cotton piqué polo, stonewashed linen, clean leather",
            pose="straight upright pose, arms relaxed at sides, direct camera gaze",
            top="fitted short-sleeve polo shirt in crisp white, collar open, clean finish",
            bottom="tailored wide-leg linen trousers in warm stone, pressed crease",
            outerwear="none",
            footwear="white leather tennis sneakers with thin flat sole",
            accessories="slim leather strap watch, no bag",
            women_top="dress",
            women_bottom="sleeveless fitted sheath dress in crisp white cotton with a clean round neckline, knee-length",
            women_outerwear="none",
            women_footwear="white leather pointed-toe mules with low block heel",
            women_accessories="minimal gold chain necklace, small structured white leather bag",
        ),
        StyleVariant(
            name="Smart Casual",
            style_label="smart casual",
            colors="navy blue, warm beige, deep burgundy",
            textures="ponte fabric trousers, fine-gauge merino wool knit, suede",
            pose="confident contrapposto, weight on one leg, slight profile angle",
            top="fine-knit merino sweater over a collared shirt with subtle collar peek",
            bottom="tapered beige chinos",
            outerwear="none",
            footwear="tan suede chelsea boots",
            accessories="leather belt, slim leather wallet",
            women_top="fitted fine-knit merino sweater in burgundy, tucked into waistband",
            women_bottom="high-waist A-line midi skirt in navy with subtle front pleat",
            women_outerwear="none",
            women_footwear="pointed-toe tan suede ankle boots",
            women_accessories="leather belt bag at hip, delicate layered gold necklace",
        ),
        StyleVariant(
            name="Earth Tones",
            style_label="relaxed earth-tone casual",
            colors="terracotta, warm sand, rust brown",
            textures="crinkled linen, raw-edge cotton, woven canvas",
            pose="candid mid-walk pose, soft natural movement, looking slightly off-camera",
            top="loose linen button-down shirt in terracotta, half-tucked",
            bottom="relaxed wide-leg linen cargo pants in sand, cuffs rolled",
            outerwear="none",
            footwear="tan leather slip-on loafers",
            accessories="woven straw tote bag, minimal gold-tone bracelet",
            women_top="dress",
            women_bottom="romantic floral midi frock in terracotta and sand tones, with puffed short sleeves, smocked bodice, and full skirt",
            women_outerwear="none",
            women_footwear="tan leather strappy block-heel sandals",
            women_accessories="woven rattan tote, layered gold bangles, small hoop earrings",
        ),
        StyleVariant(
            name="Monochrome",
            style_label="all-black monochrome",
            colors="all-black with matte and satin finish contrast",
            textures="heavyweight cotton, matte nylon, patent leather",
            pose="leaning against wall, arms crossed loosely, editorial upward gaze",
            top="fitted black ribbed turtleneck",
            bottom="slim black nylon jogger pants",
            outerwear="unzipped black leather bomber jacket",
            footwear="sleek black leather high-top sneakers",
            accessories="black nylon backpack, silver-tone chain bracelet",
            women_top="dress",
            women_bottom="sleek black wrap mini dress in matte jersey with long sleeves and a V-neckline",
            women_outerwear="unzipped black moto jacket with silver zip hardware",
            women_footwear="black leather ankle boots with small block heel",
            women_accessories="black structured mini bag, silver hoop earrings",
        ),
        StyleVariant(
            name="Resort Linen",
            style_label="resort linen relaxed",
            colors="natural ecru, sage green, warm ivory",
            textures="stonewashed linen, woven cotton, natural leather",
            pose="relaxed candid standing, fabric catching a light breeze, warm outdoor light",
            top="relaxed linen camp-collar short-sleeve shirt in soft sage green, half-tucked",
            bottom="wide-leg linen trousers in natural ecru, cuffs rolled once",
            outerwear="none",
            footwear="tan woven leather slide sandals",
            accessories="simple leather bracelet, no bag",
            women_top="none",
            women_bottom="wide-leg linen jumpsuit with a square neckline, tie-waist belt, and wide flowing legs in natural ecru",
            women_outerwear="none",
            women_footwear="leather flat thong sandals with thin ankle wrap",
            women_accessories="simple gold chain necklace, small woven clutch",
        ),
    ],

    # ── FORMAL ────────────────────────────────────────────────────────────────
    "Formal": [
        StyleVariant(
            name="Classic Business",
            style_label="classic business formal",
            colors="charcoal grey, crisp white, navy silk tie",
            textures="super-120 wool suiting, broadcloth cotton shirt, woven silk tie",
            pose="upright power stance, jacket buttoned, direct confident gaze",
            top="French-cuffed white broadcloth dress shirt with navy tie",
            bottom="charcoal grey suit trousers with knife-edge crease",
            outerwear="single-breasted two-button charcoal wool suit jacket with white pocket square",
            footwear="polished black cap-toe oxford shoes",
            accessories="silver cufflinks, slim leather briefcase, dress wristwatch",
            women_top="fitted ivory silk shell camisole with pearl-button placket",
            women_bottom="tailored high-waist midi pencil skirt in charcoal with a subtle back kick pleat",
            women_outerwear="tailored single-breasted blazer in charcoal",
            women_footwear="pointed-toe black leather court pumps with 7cm heel",
            women_accessories="pearl stud earrings, structured leather tote bag",
        ),
        StyleVariant(
            name="Elegant Evening",
            style_label="black tie elegant evening",
            colors="midnight navy, champagne, ivory",
            textures="velvet peak lapels, satin-finish trousers, fine lightweight wool",
            pose="side profile with chin slightly raised, sophisticated composed posture",
            top="pleated ivory dress shirt with black bow tie",
            bottom="slim-fit satin-stripe dinner suit trousers",
            outerwear="midnight navy dinner suit jacket with velvet peak lapels",
            footwear="patent leather opera pumps in black",
            accessories="mother-of-pearl cufflinks, dress watch with crocodile strap",
            women_top="draped at one shoulder, column gown bodice",
            women_bottom="floor-length column skirt with subtle side slit",
            women_outerwear="none",
            women_footwear="strappy heeled satin sandals in champagne",
            women_accessories="small crystal-embellished evening clutch, long diamond drop earrings",
        ),
        StyleVariant(
            name="Modern Professional",
            style_label="modern business professional",
            colors="medium grey, pale blue shirt, warm white",
            textures="four-way stretch performance wool, no-iron cotton poplin, full-grain leather",
            pose="mid-stride walking pose, jacket open, relaxed confident energy",
            top="pale blue open-collar performance dress shirt",
            bottom="medium grey slim tailored flat-front trousers",
            outerwear="slim notch-lapel grey performance wool blazer",
            footwear="tan leather derby shoes with rubber commuter sole",
            accessories="slim leather portfolio folder, minimalist silver watch",
            women_top="fitted sleeveless silk shell top in pale blue, tucked in",
            women_bottom="tailored midi A-line skirt in medium grey with a structured waistband",
            women_outerwear="structured blazer in matching grey",
            women_footwear="tan leather block-heel ankle boots",
            women_accessories="leather portfolio tote, gold minimalist watch",
        ),
        StyleVariant(
            name="Power Suit",
            style_label="power suit authoritative",
            colors="deep charcoal chalk stripe, stark white",
            textures="heavy herringbone wool, fine poplin, visible silk lining at lapel",
            pose="frontal commanding pose, jacket buttoned, hands clasped in front",
            top="stark white poplin dress shirt, bold silk necktie with dimple",
            bottom="double-breasted suit high-rise trousers with side adjusters",
            outerwear="double-breasted six-button chalk-stripe suit jacket",
            footwear="hand-burnished cognac cap-toe brogues",
            accessories="gold dimple tie bar, folded silk pocket square, leather portfolio bag",
            women_top="fitted satin camisole with wide sculptural neckline, tucked in",
            women_bottom="high-waist fitted midi power skirt in chalk stripe with a front kick pleat",
            women_outerwear="structured double-breasted chalk-stripe blazer",
            women_footwear="pointed stiletto heels in black patent leather",
            women_accessories="bold sculptural collar necklace, sleek black leather briefcase",
        ),
        StyleVariant(
            name="Creative Professional",
            style_label="creative business smart",
            colors="forest green blazer, off-white turtleneck, dark charcoal trousers",
            textures="nubby tweed blazer, fine-rib merino turtleneck, smooth gabardine",
            pose="three-quarter angle, one hand on lapel, thoughtful composed expression",
            top="slim merino turtleneck in off-white",
            bottom="cropped tapered charcoal gabardine trousers",
            outerwear="unstructured forest green tweed sport coat",
            footwear="dark brown crepe-sole chelsea boots",
            accessories="round wire-frame glasses, slim leather messenger bag",
            women_top="fitted fine-rib merino turtleneck in off-white",
            women_bottom="structured high-waist midi wrap skirt in charcoal with an asymmetric draped hem",
            women_outerwear="relaxed forest green tweed blazer",
            women_footwear="low-block-heel suede chelsea boots in dark brown",
            women_accessories="oversized tortoiseshell glasses, leather hobo bag, architectural earrings",
        ),
    ],

    # ── ATHLETIC ──────────────────────────────────────────────────────────────
    "Athletic": [
        StyleVariant(
            name="Performance Run",
            style_label="performance running sportswear",
            colors="electric blue, fluorescent yellow, clean white",
            textures="moisture-wicking Dri-FIT mesh, lightweight ripstop, compression knit",
            pose="dynamic mid-run stride, torso forward lean, arms bent at 90 degrees pumping",
            top="fitted long-sleeve compression running top",
            bottom="technical running shorts with inner liner",
            outerwear="ultralight packable running jacket, half-unzipped",
            footwear="neon-accented road running shoes with thick stack foam sole",
            accessories="GPS running watch, reflective hi-vis armband, wireless earbuds",
            women_top="fitted long-sleeve compression crop top with racerback cutout",
            women_bottom="high-waist 7-inch running shorts or compression tights",
            women_outerwear="lightweight running jacket, half-unzipped",
            women_footwear="women's neon-accented road running shoes",
            women_accessories="GPS running watch, hair in high ponytail, wireless earbuds",
        ),
        StyleVariant(
            name="Gym Training",
            style_label="gym training activewear",
            colors="matte black, dark charcoal grey, deep crimson red",
            textures="four-way stretch spandex, heavyweight ribbed cotton, rubber grip texture",
            pose="confident upright gym stance, weight on both feet, natural athletic posture",
            top="fitted sleeveless muscle tank top",
            bottom="tapered jogger pants with elasticated ankle",
            outerwear="none",
            footwear="cross-training shoes with wide base and lateral support",
            accessories="workout gloves tucked into waistband, stainless steel water bottle in hand",
            women_top="sports crop bra with cut-out back detail",
            women_bottom="high-waist 7/8 compression leggings with side pocket",
            women_outerwear="none",
            women_footwear="women's cross-training shoes",
            women_accessories="small resistance band around wrist, water bottle, hair in sleek high bun",
        ),
        StyleVariant(
            name="Outdoor Active",
            style_label="outdoor adventure activewear",
            colors="burnt orange, slate grey, forest green",
            textures="ripstop nylon shell, merino wool base layer, Vibram rubber",
            pose="mid-stride on slight uphill incline, arms swinging naturally",
            top="merino wool short-sleeve base layer tee",
            bottom="technical convertible hiking trousers",
            outerwear="lightweight packable wind jacket tied around waist",
            footwear="trail running shoes with lugged rubber outsole",
            accessories="small running hydration pack, wide-brim sun hat, polarised wraparound sunglasses",
            women_top="fitted merino wool short-sleeve athletic crop top",
            women_bottom="women's convertible hiking trousers or technical leggings",
            women_outerwear="lightweight wind jacket tied around waist",
            women_footwear="women's trail running shoes",
            women_accessories="hydration vest with soft flask pockets, sun cap, polarised sunglasses",
        ),
        StyleVariant(
            name="Athleisure",
            style_label="sporty athleisure",
            colors="heather grey, cream white, pastel sage green",
            textures="cloud-soft brushed fleece, heavyweight jersey, rubberised sole",
            pose="relaxed casual stance, one hand on hip, slight side angle",
            top="fitted heavyweight jersey tank",
            bottom="slim tapered jogger in heather grey",
            outerwear="zip-up fleece hoodie, half-unzipped",
            footwear="retro chunky sneakers in white with pastel accents",
            accessories="small zip-around belt bag, simple silicone wristband",
            women_top="fitted sports crop top in cream",
            women_bottom="high-waist flare leggings in heather grey",
            women_outerwear="cropped zip-up fleece hoodie in sage",
            women_footwear="chunky retro platform sneakers in white and sage",
            women_accessories="small belt bag, beaded scrunchie on wrist, tiny gold hoop earrings",
        ),
        StyleVariant(
            name="Court Sport",
            style_label="court sports performance",
            colors="clean white, royal blue, red accent stripe",
            textures="technical pique knit polo, stretch woven court shorts, perforated leather",
            pose="athletic ready-position stance, slight crouch, weight evenly distributed forward",
            top="classic technical polo shirt with tonal stripe",
            bottom="mid-thigh stretch court shorts",
            outerwear="none",
            footwear="low-cut court sneakers with herringbone rubber outsole",
            accessories="terry cloth wristband, sports bag strap visible over shoulder",
            women_top="fitted sleeveless athletic tank top with princess seam",
            women_bottom="pleated tennis skirt with built-in compression shorts",
            women_outerwear="none",
            women_footwear="women's low-cut court sneakers",
            women_accessories="terry wristband, racket bag visible, hair in sleek ponytail",
        ),
    ],

    # ── RAIN READY ────────────────────────────────────────────────────────────
    "Rain Ready": [
        StyleVariant(
            name="Sleek Urban Rain",
            style_label="sleek urban rain fashion",
            colors="jet black, gunmetal grey, matte silver hardware",
            textures="bonded Gore-Tex 3L shell, water-resistant coated denim, natural rubber",
            pose="walking confidently through rain, umbrella held at angle, slight forward lean",
            top="fitted black merino turtleneck",
            bottom="slim water-resistant coated denim jeans tucked into boot",
            outerwear="streamlined hooded rain mac fully buttoned with storm placket",
            footwear="tall black rubber Chelsea rain boots with slip-resistant sole",
            accessories="compact auto-open umbrella, waterproof zip-around crossbody bag",
            women_top="fitted black merino turtleneck",
            women_bottom="sleek water-resistant midi skirt in jet black with opaque black thermal tights",
            women_outerwear="fitted belted hooded rain mac in jet black",
            women_footwear="tall glossy black rubber rain boots",
            women_accessories="compact umbrella, sleek waterproof crossbody bag",
        ),
        StyleVariant(
            name="Colourful Wet Weather",
            style_label="bold colourful wet weather look",
            colors="canary yellow, cobalt blue, clean white",
            textures="high-gloss PVC-coated nylon, ribbed jersey cotton lining, thick vulcanised rubber",
            pose="playful mid-jump over puddle, coat billowing, joyful expression",
            top="fitted white ribbed turtleneck",
            bottom="straight dark indigo jeans",
            outerwear="oversized bright yellow PVC rain slicker with oversized hood",
            footwear="glossy canary yellow tall rubber rain boots",
            accessories="clear transparent PVC structured tote, matching yellow umbrella",
            women_top="fitted white ribbed turtleneck",
            women_bottom="bright cobalt blue midi skirt with dark opaque thermal tights",
            women_outerwear="oversized yellow rain slicker belted with matching sash",
            women_footwear="glossy yellow tall rubber rain boots",
            women_accessories="transparent PVC bucket bag, yellow umbrella, small gold studs",
        ),
        StyleVariant(
            name="Trench Classic",
            style_label="classic belted trench",
            colors="camel tan, khaki, ivory lining",
            textures="tightly woven gabardine cotton, jersey lining, smooth vegetable-tanned leather",
            pose="belting trench coat at waist, three-quarter body turn, windswept collar detail",
            top="fine-knit ivory rollneck sweater",
            bottom="slim tapered dark trousers",
            outerwear="double-breasted D-ring belted trench coat in camel with storm flap",
            footwear="tan zip-side leather ankle boots with low stacked heel",
            accessories="silk pocket square tucked in breast pocket, structured leather tote",
            women_top="fine-knit ivory rollneck",
            women_bottom="wool A-line midi skirt in warm camel, hem peeking below trench coat",
            women_outerwear="double-breasted belted trench coat in camel",
            women_footwear="heeled camel leather ankle boots",
            women_accessories="silk scarf tied loosely at neck, structured leather handbag",
        ),
        StyleVariant(
            name="Layered Storm Proof",
            style_label="layered storm-proof outdoors",
            colors="olive drab, dark navy, burnt sienna",
            textures="waxed cotton outer shell, Polartec fleece mid-layer, stretch merino wool",
            pose="bracing against wind, one hand holding hood, dynamic outdoor stride",
            top="merino wool long-sleeve base layer",
            bottom="waterproof softshell cargo trousers",
            outerwear="waxed cotton hooded jacket over fleece gilet, all zipped up",
            footwear="waterproof lug-sole leather hiking boots, laced high",
            accessories="ribbed merino beanie, neck gaiter, waterproof roll-top rucksack",
            women_top="merino wool thermal fitted long-sleeve crop top",
            women_bottom="waterproof softshell trousers with ankle drawcord",
            women_outerwear="waxed cotton hooded anorak over fleece mid-layer",
            women_footwear="women's lug-sole waterproof ankle boots",
            women_accessories="fitted merino beanie, neck gaiter, waterproof roll-top backpack",
        ),
        StyleVariant(
            name="Minimalist Rain",
            style_label="clean minimalist rain look",
            colors="stone grey, warm off-white, pale navy",
            textures="bonded technical softshell, fine-rib merino knit, matte rubber",
            pose="calm upright stance under awning, arms at sides, composed expression",
            top="fine-rib merino turtleneck in off-white",
            bottom="slim tapered softshell trousers in stone grey",
            outerwear="fitted hooded technical softshell jacket in grey, all zipped",
            footwear="minimalist waterproof low-top sneaker boot in light grey",
            accessories="slim waterproof daypack, no umbrella",
            women_top="ribbed merino turtleneck in off-white",
            women_bottom="water-resistant midi skirt in stone grey with opaque thermal tights",
            women_outerwear="fitted hooded softshell jacket in grey",
            women_footwear="minimalist waterproof leather ankle boot with 3cm heel",
            women_accessories="sleek waterproof backpack, small gold stud earrings",
        ),
    ],
}


# ─── Context lookup tables ────────────────────────────────────────────────────

def _temp_descriptor(temp_f: int) -> str:
    if temp_f >= 90:
        return "extreme heat, ultra-lightweight breathable fabrics, minimal layers"
    if temp_f >= 80:
        return "hot weather, lightweight breathable fabrics, short sleeves preferred"
    if temp_f >= 70:
        return "warm weather, light comfortable layering"
    if temp_f >= 58:
        return "mild weather, transitional layering, optional light jacket"
    if temp_f >= 45:
        return "cool weather, medium-weight layers, jacket required"
    return "cold weather, heavy insulating layers, warmth essential"


_WEATHER_LIGHTING: dict[str, str] = {
    "Sunny":  "bright direct sunlight, crisp sharp shadows, warm golden tones",
    "Cloudy": "soft even overcast diffused light, no harsh shadows, cool neutral tones",
    "Rain":   "moody rain atmosphere, rain streaks visible, glistening wet surfaces, dark dramatic sky",
    "Snowy":  "crisp cold winter light, blue-white palette, snow-dusted surfaces",
    "Windy":  "dynamic outdoor light, clothes and hair in natural movement, slightly overcast",
}

_TIME_LIGHTING: dict[str, str] = {
    "morning":   "early morning golden hour light, long soft warm shadows",
    "afternoon": "bright even midday natural light, clear outdoor visibility",
    "evening":   "warm amber low evening light, soft deep shadows, interior or city-lit ambience",
}

_ACTIVITY_SETTING: dict[str, str] = {
    "Hiking":   "mountain trail, rocky terrain and pine trees in background",
    "Museum":   "clean contemporary art gallery interior, white walls, polished terrazzo floor",
    "Dinner":   "upscale restaurant interior, soft candlelight, blurred elegant table settings",
    "Business": "modern glass-and-steel office lobby, city skyline visible through floor-to-ceiling windows",
    "Beach":    "sandy beach shoreline, gentle surf, warm coastal light",
    "Walking":  "urban city sidewalk, soft-focus storefronts and pedestrians in background",
}

# Photography keyword tags that reinforce quality and realism in the keyword block
_PHOTO_TAGS = (
    "editorial fashion photography, Vogue magazine aesthetic, "
    "shot on Sony A7R V, 85mm f/1.4 prime lens, shallow depth of field, "
    "sharp focus on clothing and fabric detail, photorealistic, "
    "professional colour grading, 8K resolution"
)


# ─── Public API ───────────────────────────────────────────────────────────────

def build_prompt(
    event_features: dict,
    category: str,
    style_variant: StyleVariant,
    gender: str = "male",
    outfit_items: list[str] | None = None,
    color_palette: str | None = None,
    activity_accessories: str | None = None,
) -> str:
    """
    Build a two-section prompt: narrative + keyword tag reinforcement.

    Section 1 (narrative): full sentence description of model, outfit, setting,
    lighting — anchors composition and clothing story.

    Section 2 (keywords): comma-separated attribute tags — reinforce fine
    clothing detail, photography style, and quality. FLUX.1 and SDXL-family
    models weight the keyword section strongly for detail fidelity.

    Args:
        event_features: Dict with keys: temp, rain, activity, weather, time.
        category:       Predicted outfit category.
        style_variant:  StyleVariant instance with structured outfit fields.
        gender:         "male" or "female".
        outfit_items:   Optional specific item names (overrides variant fields).

    Returns:
        Full prompt string ready for FLUX.1-schnell / SDXL / DALL-E 3.
    """
    temp        = event_features.get("temp", 72)
    weather     = event_features.get("weather", "Sunny")
    time_of_day = event_features.get("time", "afternoon")
    activity    = event_features.get("activity", "")
    rain        = event_features.get("rain", 0)

    is_female   = gender.lower() in ("female", "woman", "women", "f")
    gender_word = "female" if is_female else "male"

    # ── Section 1: Narrative ──────────────────────────────────────────────────

    # Subject — explicit anatomy spec reduces distortion
    subject = (
        f"Full body fashion editorial photograph of a real {gender_word} model, "
        "realistic body proportions, complete figure visible from head to feet"
    )

    # Resolve outfit fields by gender
    if outfit_items:
        outfit_desc = f"wearing {', '.join(outfit_items)}"
    else:
        top       = (style_variant.women_top       if is_female and style_variant.women_top       else style_variant.top)
        bottom    = (style_variant.women_bottom     if is_female and style_variant.women_bottom    else style_variant.bottom)
        outerwear = (style_variant.women_outerwear  if is_female and style_variant.women_outerwear else style_variant.outerwear)

        # One-piece garments (dress, frock, jumpsuit) — women_top signals this
        # by being empty or "dress", and the full description lives in women_bottom
        if is_female and (not top or top.lower() in ("dress", "none")):
            outfit_desc = f"wearing {bottom}"
        else:
            outfit_desc = f"wearing {top}, {bottom}"

        if outerwear and outerwear.lower() != "none":
            outfit_desc += f", {outerwear}"

    footwear    = (style_variant.women_footwear    if is_female and style_variant.women_footwear    else style_variant.footwear)
    accessories = (style_variant.women_accessories if is_female and style_variant.women_accessories else style_variant.accessories)

    # Context signals
    temp_note    = _temp_descriptor(temp)
    weather_note = _WEATHER_LIGHTING.get(weather, "natural daylight")
    time_note    = _TIME_LIGHTING.get(time_of_day, "natural daylight")
    setting      = _ACTIVITY_SETTING.get(activity, "neutral clean fashion studio backdrop")

    rain_note = ""
    if rain >= 60:
        rain_note = "heavy rain in background, wet reflective pavement, puddles visible"
    elif rain >= 30:
        rain_note = "light drizzle in background, slight moisture on surfaces"

    atmosphere_parts = [p for p in [weather_note, rain_note, time_note, temp_note] if p]

    active_colors = color_palette or style_variant.colors
    active_accessories = activity_accessories or accessories

    narrative = (
        f"{subject}. "
        f"{outfit_desc}. "
        f"Colors: {active_colors}. "
        f"Fabrics: {style_variant.textures}. "
        f"Footwear: {footwear}. "
        f"Accessories: {active_accessories}. "
        f"Style: {style_variant.style_label}. "
        f"Pose: {style_variant.pose}. "
        f"Setting: {setting}. "
        f"Lighting: {', '.join(atmosphere_parts)}."
    )

    # ── Section 2: Keyword tag reinforcement ─────────────────────────────────
    keywords = (
        f"full body portrait, {style_variant.style_label}, {category.lower()} fashion, "
        f"{gender_word} fashion model, realistic clothing, "
        f"{active_colors}, {style_variant.textures}, "
        f"{footwear}, {_PHOTO_TAGS}, "
        "no text, no watermark, no illustration"
    )

    return f"{narrative}\n\n{keywords}"


def build_negative_prompt(category: str = "", gender: str = "male") -> str:
    """
    Negative prompt for Stable Diffusion / FLUX backends.
    DALL-E 3 ignores negative prompts — use only with SD-family pipelines.
    """
    base = (
        "cartoon, anime, illustration, painting, drawing, sketch, 3D render, CGI, "
        "watermark, text overlay, logo, signature, "
        "blurry, out of focus, low resolution, jpeg compression artefacts, film grain, "
        "bad anatomy, deformed body, extra limbs, missing limbs, fused fingers, "
        "mutated hands, poorly drawn hands, disfigured face, "
        "cropped feet, cut-off figure, partial body, floating limbs, "
        "unrealistic clothing proportions, plastic-looking skin, "
        "overexposed highlights, underexposed shadows, harsh direct flash"
    )

    category_extras: dict[str, str] = {
        "Athletic": "formal suit, evening dress, office attire, high heels",
        "Formal":   "sportswear, gym wear, ripped clothing, casual hoodies, trainers",
        "Rain Ready": "dry sunny scene, summer shorts, flip-flops, no rain gear",
        "Casual":   "ball gown, tuxedo, military uniform, swimwear",
    }

    is_female   = gender.lower() in ("female", "woman", "women", "f")
    gender_neg  = "masculine clothing, male silhouette, beard" if is_female else "feminine clothing, female silhouette, dress, skirt"

    parts = [base, category_extras.get(category, ""), gender_neg]
    return ", ".join(p for p in parts if p)


# ─── Color palette pool ───────────────────────────────────────────────────────
# Independent of StyleVariant so the same variant can be rendered in different
# palettes across multiple generations, preventing color repetition.

COLOR_PALETTES: list[str] = [
    "soft blush pink, powder lavender, pale mint green, ivory white",
    "cobalt electric blue, warm fire orange, clean white accent",
    "deep forest green, ivory cream, warm tan leather accent",
    "terracotta orange, sandy dune beige, burnt sienna, sage green",
    "all-black: jet black, matte charcoal, satin black, silver hardware",
    "crisp white, warm off-white, pearl light grey",
    "deep navy blue, bright white, gold and tan accent",
    "mustard golden yellow, coral red, warm teal, cream",
    "dusty rose, warm copper, warm sand, soft ivory",
    "sky blue, seafoam green, white, warm sandy beige",
    "deep sapphire, rich ruby red, champagne gold",
    "camel, warm taupe, off-white, chocolate brown",
    "olive drab green, khaki, natural ivory, rust orange",
    "ash grey, slate grey, charcoal, crisp warm white",
    "vibrant emerald green, deep magenta, gold, black",
]

# ─── Activity-aware accessory guidance ───────────────────────────────────────
# Prevents bags appearing in every image and keeps accessories context-relevant.

_ACTIVITY_ACCESSORIES: dict[str, str] = {
    "Museum":   "small structured mini bag or simple tote, delicate minimal jewelry only",
    "Dinner":   "small elegant clutch or structured evening bag, fine jewelry, elegant watch",
    "Beach":    "woven straw tote or canvas beach bag, sunglasses, minimal light jewelry",
    "Hiking":   "backpack or hydration vest, wide-brim sun hat, polarised sunglasses",
    "Business": "structured briefcase or professional tote bag, minimalist dress watch",
    "Walking":  "small crossbody bag at most, sunglasses if sunny, keep accessories minimal",
}


def pick_color_palette(seed: int | None = None) -> str:
    """
    Return a color palette string from the pool.

    Seeded so the same event always gets the same palette (deterministic),
    but different seeds produce varied palettes across generations.
    """
    rng = random.Random(seed)
    return rng.choice(COLOR_PALETTES)


def resolve_accessories(activity: str, base_accessories: str) -> str:
    """
    Return context-appropriate accessories for the given activity.

    Falls back to base_accessories when the activity has no specific guidance.
    This prevents bags/accessories from appearing in every image regardless
    of whether they make sense for the scene.
    """
    return _ACTIVITY_ACCESSORIES.get(activity, base_accessories)


# ─── Convenience ─────────────────────────────────────────────────────────────

def get_variants(category: str) -> list[StyleVariant]:
    """Return the list of StyleVariant objects for a given category."""
    return STYLE_VARIANTS.get(category, STYLE_VARIANTS["Casual"])
