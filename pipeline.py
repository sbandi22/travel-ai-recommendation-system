"""
TravelIQ — Main Pipeline
User enters city, state, month, year, travel style, and detailed preferences.
Attractions are fetched comprehensively from Google Places (multi-type, paginated, text search).
Features are extracted from Review, Crowd, and Weather modules.
Attractions are ranked by LLM personalised to travel style.
Itinerary is planned by OR-Tools + OpenAI with fully dynamic scheduling.
"""

import json
import math
import time
import threading
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
from openai        import OpenAI
from modules.review    import ReviewModel
from modules.crowd     import CrowdModel
from modules.weather   import WeatherModel
from modules.ranker    import LLMRanker
from modules.itinerary import ItineraryPlanner
from dotenv import load_dotenv, dotenv_values
load_dotenv()

config         = dotenv_values(".env")
GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]

# ── Attraction categories ─────────────────────────────────────────────
CATEGORIES = [
    "outdoor", "indoor", "cultural", "hiking",
    "theme_park", "beach", "nightlife", "food", "wellness",
]

VALID_STYLES = [
    "adventure", "cultural", "relaxation", "foodie",
    "family",    "luxury",   "budget",     "nightlife",
    "wellness",  "sports",   "romantic",   "eco",
    "photography","history", "art",        "popular",
]

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Place types to search across ─────────────────────────────────────
NEARBY_TYPES = [
    "tourist_attraction",
    "museum",
    "park",
    "art_gallery",
    "amusement_park",
    "aquarium",
    "zoo",
    "natural_feature",
    "restaurant",
    "cafe",
    "bar",
    "spa",
    "stadium",
    "shopping_mall",
    "campground",
]

# Types that are only fetched when the traveler's style explicitly calls for them.
# Skipping irrelevant types stops spas/campgrounds from consuming the 60-result cap.
TYPE_REQUIRES_STYLE = {
    "spa":           {"wellness", "relaxation"},
    "campground":    {"adventure", "eco"},
    "bar":           {"nightlife", "foodie"},
    "shopping_mall": {"budget", "luxury"},
}

# Maximum straight-line distance from city centre to accept a result.
# Eliminates parks/campgrounds that are hours away but returned by Text Search.
MAX_RESULT_KM = 60

# ── Style-specific text searches ──────────────────────────────────────
# These run IN ADDITION to the generic NEARBY_TYPES searches so each
# travel style surfaces the venues that matter most to it.
STYLE_SEARCHES = {
    "adventure"  : [
        "rock climbing kayaking white water rafting {city} {state}",
        "guided hiking tours outdoor adventure sports {city} {state}",
    ],
    "cultural"   : [
        "historic landmarks heritage sites cultural centers {city} {state}",
        "science centers ethnic neighborhoods cultural festivals {city} {state}",
    ],
    "relaxation" : [
        "day spa wellness retreat botanical gardens {city} {state}",
        "scenic viewpoints peaceful nature walks quiet parks {city} {state}",
    ],
    "foodie"     : [
        "food tours cooking classes farmers market food hall {city} {state}",
        "best local restaurants cult favorite eats culinary experiences {city} {state}",
    ],
    "family"     : [
        "family activities children museum interactive exhibits {city} {state}",
        "zoo aquarium theme park kid friendly fun {city} {state}",
    ],
    "luxury"     : [
        "luxury experiences private tours fine dining upscale {city} {state}",
        "rooftop lounge exclusive venues premium sightseeing {city} {state}",
    ],
    "budget"     : [
        "free attractions public parks free events cheap activities {city} {state}",
        "budget friendly tours low cost outdoor experiences {city} {state}",
    ],
    "nightlife"  : [
        "rooftop bars live music venues jazz clubs nightlife {city} {state}",
        "entertainment districts cocktail bars night life {city} {state}",
    ],
    "wellness"   : [
        "yoga studio meditation center nature retreat {city} {state}",
        "holistic spa sound bath forest bathing wellness {city} {state}",
    ],
    "sports"     : [
        "stadiums arenas sports venues recreational parks {city} {state}",
        "sports museum athletic outdoor activities {city} {state}",
    ],
    "romantic"   : [
        "romantic restaurants sunset viewpoints couples experiences {city} {state}",
        "scenic gardens waterfront intimate venues date night {city} {state}",
    ],
    "eco"        : [
        "nature reserves wildlife sanctuary conservation area {city} {state}",
        "botanical garden eco tours sustainable experiences {city} {state}",
    ],
    "photography": [
        "scenic vistas iconic viewpoints photography spots {city} {state}",
        "colorful markets unique architecture murals street art {city} {state}",
    ],
    "history"    : [
        "historic sites battlefields monuments heritage districts {city} {state}",
        "archaeological sites old town walking history tour {city} {state}",
    ],
    "art"        : [
        "art galleries street art sculpture gardens {city} {state}",
        "theaters live performance venues art studios creative districts {city} {state}",
    ],
    "popular"    : [
        "iconic landmarks must see tourist attractions {city} {state}",
        "most visited top rated famous sights {city} {state}",
        "famous monuments historic sites world famous {city} {state}",
    ],
}


# ── City Geocoder ─────────────────────────────────────────────────────
def geocode_city(city: str, state: str) -> tuple:
    """Returns (lat, lon) for a city. Raises ValueError if not found."""
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": f"{city}, {state}", "key": GOOGLE_API_KEY},
    ).json()
    if not resp.get("results"):
        raise ValueError(f"Could not geocode: {city}, {state}")
    loc = resp["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


# ── Hotel Finder ──────────────────────────────────────────────────────
def find_hotel(city: str, travel_style: str, group: str,
               group_ages: str, budget_per_meal: str, days: int,
               lat_c: float, lon_c: float) -> dict:
    """
    Finds the best hotel for the trip using:
      1. Google Places Nearby Search (type=lodging) near city centre
      2. Place Details for top candidates
      3. OpenAI selects the best fit based on traveler profile

    Returns a dict with: name, formatted_address, lat, lon,
                         rating, price_level, reason
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    price_sym = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

    # Map meal budget → expected hotel price level
    budget_num = int("".join(filter(str.isdigit, budget_per_meal)) or "20")
    if budget_num < 20:
        preferred_levels = {1, 2}
    elif budget_num < 40:
        preferred_levels = {2, 3}
    else:
        preferred_levels = {3, 4}

    print(f"\n  [Hotel] Searching lodging near {city} (budget tier: "
          f"{'/'.join(price_sym.get(p,'?') for p in sorted(preferred_levels))})...")

    nearby_resp = requests.get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
        params={
            "location" : f"{lat_c},{lon_c}",
            "radius"   : 12000,       # 12 km from city centre
            "type"     : "lodging",
            "key"      : GOOGLE_API_KEY,
        },
    ).json()

    raw_places = nearby_resp.get("results", [])

    # Fetch details for up to 12 candidates (filter by price level)
    candidates = []
    for place in raw_places[:20]:
        det = requests.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={
                "place_id" : place["place_id"],
                "fields"   : ("name,rating,user_ratings_total,price_level,"
                              "formatted_address,geometry,editorial_summary"),
                "key"      : GOOGLE_API_KEY,
            },
        ).json().get("result", {})

        price = det.get("price_level") or place.get("price_level")
        # Accept exact match OR one level off so we don't return an empty list
        if price not in preferred_levels and price not in {min(preferred_levels)-1,
                                                            max(preferred_levels)+1}:
            continue

        geo = (det.get("geometry") or {}).get("location", {})
        lat = geo.get("lat") or place.get("geometry", {}).get("location", {}).get("lat")
        lon = geo.get("lng") or place.get("geometry", {}).get("location", {}).get("lng")
        if not lat or not lon:
            continue

        candidates.append({
            "name"             : det.get("name") or place.get("name", ""),
            "formatted_address": det.get("formatted_address", ""),
            "rating"           : det.get("rating") or place.get("rating", 0),
            "user_ratings_total": det.get("user_ratings_total", 0),
            "price_level"      : price or 2,
            "editorial_summary": (det.get("editorial_summary") or {}).get("overview", ""),
            "lat"              : lat,
            "lon"              : lon,
        })

        if len(candidates) >= 12:
            break

    if not candidates:
        print("  Warning: No hotels found in budget range — relaxing filter")
        # Fallback: take top-rated from raw results regardless of price
        for place in raw_places[:5]:
            geo = place.get("geometry", {}).get("location", {})
            candidates.append({
                "name"             : place.get("name", "Unknown Hotel"),
                "formatted_address": place.get("vicinity", ""),
                "rating"           : place.get("rating", 3.5),
                "user_ratings_total": place.get("user_ratings_total", 0),
                "price_level"      : place.get("price_level", 2),
                "editorial_summary": "",
                "lat"              : geo.get("lat"),
                "lon"              : geo.get("lng"),
            })

    if not candidates:
        print("  Warning: Hotel search returned no results — skipping hotel")
        return None

    # ── OpenAI selection ─────────────────────────────────────────────
    hotel_lines = "\n".join(
        f"{i+1}. {h['name']} | "
        f"{price_sym.get(h['price_level'], '?')} | "
        f"Rating: {h['rating']} ({h['user_ratings_total']:,} reviews) | "
        f"Addr: {h['formatted_address'][:70]} | "
        f"Summary: {(h['editorial_summary'] or 'N/A')[:80]}"
        for i, h in enumerate(candidates)
    )

    prompt = f"""You are a world-class travel concierge at a luxury travel agency. \
A real traveler is paying hundreds of dollars for this trip and EXPECTS the best possible \
hotel recommendation. Your reputation depends on this choice — pick wrong and you've ruined \
their vacation.

Traveler profile:
  Travel style : {travel_style}
  Group        : {group}{f"  |  Ages: {group_ages}" if group_ages else ""}
  Duration     : {days} nights
  Budget tier  : {"/".join(price_sym.get(p, "?") for p in sorted(preferred_levels))}
  City         : {city}

Hotels available:
{hotel_lines}

Selection criteria (in order of priority):
1. LOCATION IS KING — pick the hotel closest to the city centre / major tourist attractions.
   A traveler visiting {city} wants to walk to famous sights, not waste hours commuting.
2. Price level must match the budget tier — do not upsell beyond {price_sym.get(max(preferred_levels), "$$$$")}
3. Highest genuine rating — a 4.7★ with 2,000+ reviews beats a 4.9★ with 8 reviews every time
4. Travel style fit — {travel_style} traveler specifically wants {
    "a boutique, character-filled property near outdoor trailheads or nature" if travel_style in ("adventure","eco") else
    "a serene, spa-quality hotel with peaceful surroundings" if travel_style == "wellness" else
    "something stylish, Instagram-worthy, and centrally located" if travel_style in ("photography","romantic","luxury") else
    "central location within walking distance of great restaurants and nightlife" if travel_style in ("foodie","nightlife") else
    "family-friendly with spacious rooms, amenities, and safe neighbourhood" if travel_style == "family" else
    "a well-located, comfortable base that is centrally situated for sightseeing"
}

CRITICAL: Think like a seasoned travel agent who has visited {city} personally. \
Prioritise the hotel that the traveler will thank you for, not just the one that \
technically matches the criteria on paper.

Respond ONLY with valid JSON (no markdown):
{{"name": "<exact hotel name from list>", "reason": "<one concrete sentence: why this hotel will delight this specific traveler>"}}"""

    print("  [Hotel] Asking OpenAI to select the best match...")
    try:
        resp2 = client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
        )
        pick        = json.loads(resp2.choices[0].message.content.strip())
        chosen_name = pick.get("name", "")
        reason      = pick.get("reason", "Best available option")

        # Find the exact candidate (exact match first, then partial)
        hotel = (next((h for h in candidates if h["name"] == chosen_name), None) or
                 next((h for h in candidates
                       if chosen_name.lower() in h["name"].lower() or
                       h["name"].lower() in chosen_name.lower()), None) or
                 sorted(candidates, key=lambda h: h.get("rating", 0), reverse=True)[0])

        hotel = dict(hotel)
        hotel["reason"] = reason
        print(f"  ✓ Hotel selected: {hotel['name']} "
              f"({price_sym.get(hotel['price_level'], '?')} | ★{hotel['rating']})")
        return hotel

    except Exception as e:
        print(f"  Warning: Hotel OpenAI selection failed ({e}) — using highest-rated")
        best = sorted(candidates, key=lambda h: h.get("rating", 0), reverse=True)[0]
        best = dict(best)
        best["reason"] = "Highest-rated option in budget range"
        return best


# ── OpenAI Batch Categorizer ──────────────────────────────────────────
def categorize_attractions(attractions: list) -> dict:
    """Categorize all attractions in one OpenAI call."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    attraction_list = "\n".join(
        f"{i+1}. Name: {a['name']} | Google Types: {', '.join(a['raw_types'])}"
        for i, a in enumerate(attractions)
    )
    prompt = f"""You are a travel categorization assistant.

Given this list of attractions, assign each exactly ONE category from:
{CATEGORIES}

Guidelines:
- museums, galleries, historic sites, science centers, cultural institutions → cultural
- concert halls, symphony halls, opera houses, performing arts centers, theaters → cultural
- trails, mountains, canyons, trailheads → hiking
- parks, gardens, open spaces, campgrounds, nature reserves, botanical gardens → outdoor
- amusement parks, theme parks, rides → theme_park
- beaches, waterfront, lakes → beach
- bars, cocktail bars, pubs, nightclubs, dive bars, rooftop bars, jazz clubs → nightlife
  (NOTE: theaters and concert halls are NOT nightlife — they are cultural)
- restaurants, cafes, food markets, food halls, culinary experiences → food
- shopping malls, covered centers → indoor
- spas, wellness centers, yoga studios, meditation → wellness
- stadiums, arenas, sports fields → outdoor
- when in doubt → outdoor

Attractions:
{attraction_list}

Respond ONLY with valid JSON: {{"attraction name": "category", ...}}
No markdown, no explanation."""

    print("\nCategorizing attractions with OpenAI...")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        cats = json.loads(raw)
        print(f"  ✓ Categorized {len(cats)} attractions")
        return cats
    except json.JSONDecodeError:
        print("  Warning: Could not parse categories — defaulting to 'outdoor'")
        return {a["name"]: "outdoor" for a in attractions}


# ── OpenAI Knowledge-Based Sentiment ─────────────────────────────────
def get_openai_attraction_sentiment(attractions: list, city: str, month: int) -> dict:
    """
    One OpenAI call that returns a knowledge-based sentiment/reputation score (0–1)
    for each attraction, drawn from the model's world knowledge.

    Useful when Google reviews are few or the DistilBERT review model has
    insufficient text to form a reliable signal.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

    lines = "\n".join(
        f"{i+1}. {a['name']} | type: {a.get('attraction_type','?')} | "
        f"Google rating: {a.get('google_rating',0)} ({a.get('user_ratings_total',0):,} reviews)"
        for i, a in enumerate(attractions)
    )

    prompt = f"""You are an expert travel curator who has personally visited every major \
city in the world. A tourist spending hundreds of dollars on a trip to {city} is trusting \
your judgement to score these attractions honestly.

City   : {city}
Month  : {month_names.get(month, month)}

For each attraction/restaurant/venue below, rate its overall reputation and visitor
sentiment score from 0.0 (avoid at all costs) to 1.0 (once-in-a-lifetime, unmissable).

Score based on:
- FAME AND ICONIC STATUS: world-famous landmarks (Times Square, Statue of Liberty, Golden Gate, \
  Eiffel Tower equivalents for {city}) must score 0.95–1.0 — tourists ALWAYS want to see these
- GENUINE VISITOR LOVE: attractions that consistently wow visitors vs ones that disappoint
- UNIQUENESS TO {city.upper()}: something the tourist can ONLY experience in {city} scores higher
- QUALITY vs TYPE: a legendary restaurant outscores an average one of the same type
- SEASONAL FIT: rate lower if the attraction is notably poor in {month_names.get(month, month)}
- TOURIST TRAP PENALTY: lower score for places that are famous but disappoint most visitors

Score 0.9–1.0 = iconic, unmissable, every visitor should go
Score 0.7–0.9 = highly recommended, worth the time
Score 0.5–0.7 = decent, situational — depends on traveler interests
Score 0.3–0.5 = below average, only for very specific interests
Score 0.0–0.3 = avoid, tourist trap, or genuinely poor quality

Attractions:
{lines}

Respond ONLY with valid JSON: {{"attraction name": score_float, ...}}
Scores must be floats between 0.0 and 1.0.
No markdown, no explanation."""

    print("\n  [Sentiment] Fetching OpenAI knowledge-based sentiment scores...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        scores = json.loads(raw)
        # Ensure all values are valid floats in [0,1]
        return {k: max(0.0, min(1.0, float(v))) for k, v in scores.items()}
    except Exception as e:
        print(f"  Warning: OpenAI sentiment failed ({e}) — using 0.7 default")
        return {a["name"]: 0.7 for a in attractions}


# ── Comprehensive Google Places Fetcher ───────────────────────────────
def fetch_attractions(city: str, state: str, max_results: int = 60,
                      lat_c: float = None, lon_c: float = None,
                      travel_style: str = "") -> list:
    """
    Comprehensive multi-strategy attraction fetcher — fully parallelised.

    Strategies (all run with ThreadPoolExecutor):
      1. Text Search per attraction type in parallel (no 50 km cap, covers ~100 km region)
      2. Style-specific + generic text searches in parallel
      3. Place Details for all results in parallel (biggest speedup — 60 concurrent calls)
      4. Deduplication by place_id
      5. Batch categorize with OpenAI

    lat_c / lon_c  : pre-geocoded city centre (avoids a redundant API call).
    travel_style   : one of VALID_STYLES — drives additional targeted searches.
    """
    print(f"\nFetching comprehensive attractions for {city}, {state}...")

    # ── Geocode (skip if caller already has coordinates) ─────────────
    if lat_c is None or lon_c is None:
        lat_c, lon_c = geocode_city(city, state)
    print(f"  Coordinates: ({lat_c:.4f}, {lon_c:.4f})")

    lock       = threading.Lock()
    seen_ids   = set()
    all_places = []

    # ── Text search by attraction type (no radius cap, covers ~100 km) ──
    TYPE_LABELS = {
        "tourist_attraction": "top tourist attractions",
        "museum":             "museums",
        "park":               "parks and nature reserves",
        "art_gallery":        "art galleries",
        "amusement_park":     "amusement parks and theme parks",
        "aquarium":           "aquariums",
        "zoo":                "zoos and wildlife sanctuaries",
        "natural_feature":    "scenic natural landmarks",
        "restaurant":         "popular restaurants",
        "cafe":               "cafes and coffee shops",
        "bar":                "bars and nightlife",
        "spa":                "spas and wellness centres",
        "stadium":            "stadiums and sports venues",
        "shopping_mall":      "shopping malls and markets",
        "campground":         "campgrounds and outdoor recreation",
    }

    styles_set = set(travel_style if isinstance(travel_style, list) else [travel_style])

    def _type_text_search(place_type: str):
        required = TYPE_REQUIRES_STYLE.get(place_type)
        if required and not (required & styles_set):
            return   # skip type irrelevant to this traveler's style
        label = TYPE_LABELS.get(place_type, place_type.replace("_", " "))
        _text_search(f"{label} near {city} {state}")

    # ── Text search with pagination (up to 3 pages = 60 results) ─────
    def _text_search(query: str, pages: int = 3):
        params = {"query": query, "key": GOOGLE_API_KEY}
        local  = []
        for _ in range(pages):
            resp  = requests.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params=params
            ).json()
            local.extend(resp.get("results", []))
            token = resp.get("next_page_token")
            if not token:
                break
            time.sleep(2)   # Google requires a short delay before next_page_token is valid
            params = {"pagetoken": token, "key": GOOGLE_API_KEY}
        with lock:
            for p in local:
                if p["place_id"] not in seen_ids:
                    seen_ids.add(p["place_id"])
                    all_places.append(p)

    # ── PHASE 1: Text search per attraction type in parallel ─────────
    print(f"  [parallel] Firing {len(NEARBY_TYPES)} type text-searches simultaneously...")
    with ThreadPoolExecutor(max_workers=len(NEARBY_TYPES)) as pool:
        list(pool.map(_type_text_search, NEARBY_TYPES))

    # ── PHASE 2: All text searches in parallel ───────────────────────
    styles_list   = list(styles_set)
    style_queries = [q.format(city=city, state=state)
                     for style in styles_list
                     for q in STYLE_SEARCHES.get(style, [])]
    generic_queries = [
        f"iconic landmarks must see tourist attractions {city} {state}",
        f"most famous sights monuments top visited places {city} {state}",
        f"best attractions {city} {state} local hidden gems unique experiences",
        f"best local restaurants food experiences {city} {state} must try",
        f"scenic viewpoints parks nature {city} {state} underrated",
    ]
    all_text = style_queries + generic_queries
    print(f"  [parallel] Firing {len(all_text)} text searches simultaneously...")
    with ThreadPoolExecutor(max_workers=max(len(all_text), 1)) as pool:
        list(pool.map(_text_search, all_text))

    # ── Post-filter: remove results too far from city centre ──────────
    before_filter = len(all_places)
    all_places = [
        p for p in all_places
        if _haversine_km(
            lat_c, lon_c,
            p.get("geometry", {}).get("location", {}).get("lat", lat_c),
            p.get("geometry", {}).get("location", {}).get("lng", lon_c),
        ) <= MAX_RESULT_KM
    ]
    removed = before_filter - len(all_places)
    if removed:
        print(f"  Removed {removed} out-of-range results (>{MAX_RESULT_KM} km from city centre)")

    # Sort by review count — iconic landmarks (100k+ reviews) float to the top,
    # obscure spas and campgrounds (few reviews) sink to the bottom before the cap.
    all_places.sort(key=lambda p: p.get("user_ratings_total", 0), reverse=True)

    places = all_places[:max_results]
    print(f"  Found {len(places)} unique places (capped at {max_results})")

    # ── PHASE 3: Fetch Place Details for all places in parallel ──────
    def _fetch_details(place):
        try:
            resp   = requests.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id" : place["place_id"],
                    "fields"   : "name,rating,user_ratings_total,reviews,types,"
                                 "opening_hours,geometry,price_level,formatted_address,"
                                 "editorial_summary",
                    "key"      : GOOGLE_API_KEY,
                }
            ).json()
            result  = resp.get("result", {})
            reviews = [r["text"] for r in result.get("reviews", []) if r.get("text")]
            oh      = result.get("opening_hours", {})
            geo     = result.get("geometry", {}).get("location", {})
            lat     = geo.get("lat") or place.get("geometry", {}).get("location", {}).get("lat")
            lon     = geo.get("lng") or place.get("geometry", {}).get("location", {}).get("lng")
            return {
                "name"                  : place.get("name", result.get("name", "")),
                "place_id"              : place["place_id"],
                "raw_types"             : place.get("types", result.get("types", [])),
                "google_rating"         : place.get("rating", result.get("rating", 0)),
                "user_ratings_total"    : result.get("user_ratings_total", 0),
                "reviews"               : reviews,
                "open_now"              : oh.get("open_now", True),
                "opening_hours_periods" : oh.get("periods", []),
                "lat"                   : lat,
                "lon"                   : lon,
                "price_level"           : result.get("price_level"),
                "formatted_address"     : result.get("formatted_address", ""),
                "editorial_summary"     : result.get("editorial_summary", {}).get("overview", ""),
            }
        except Exception as e:
            print(f"  Warning: details fetch failed for {place.get('name','?')}: {e}")
            return None

    print(f"  [parallel] Fetching details for {len(places)} places with 20 workers...")
    with ThreadPoolExecutor(max_workers=20) as pool:
        raw_details = list(pool.map(_fetch_details, places))
    attractions = [a for a in raw_details if a is not None]

    # ── Batch categorize with OpenAI ─────────────────────────────────
    categories = categorize_attractions(attractions)
    for a in attractions:
        a["attraction_type"] = categories.get(a["name"], "outdoor")
        print(f"  ✓ {a['name'][:45]:<45} → {a['attraction_type']}")

    return attractions


# ── Main Pipeline ─────────────────────────────────────────────────────
class TravelIQPipeline:
    def __init__(self,
                 review_model_path    : str = "models/best_model.pt",
                 review_tokenizer_path: str = "models/traveliq_tokenizer",
                 crowd_model_path     : str = "models/sarimax_models.pkl"):

        print("Loading TravelIQ modules...")
        self.review  = ReviewModel(review_model_path, review_tokenizer_path)
        self.crowd   = CrowdModel(crowd_model_path)
        self.weather = WeatherModel(openai_api_key=OPENAI_API_KEY)
        self.ranker  = LLMRanker(api_key=OPENAI_API_KEY)
        self.planner = ItineraryPlanner(openai_api_key=OPENAI_API_KEY,
                                        google_api_key=GOOGLE_API_KEY)
        print("All modules loaded.\n")

    def process_attractions(self, attractions: list,
                            city: str, state: str,
                            year: int, month: int,
                            travel_style: str = "adventure",
                            group: str = "solo",
                            interests: str = "") -> list:
        """
        Extract features for all attractions — fully batched.

        Steps (all parallelised where possible):
          1. OpenAI knowledge-based sentiment  )  fired in parallel
          2. Dynamic weather weights via OpenAI )
          3. DistilBERT batch inference — all reviews in one pass (~20x faster than per-review)
          4. Crowd + weather score lookup per attraction
        """
        t0 = time.time()
        print(f"\nExtracting features for {len(attractions)} attractions...")

        # ── Steps 1 + 2 in PARALLEL ──────────────────────────────────
        def _get_sentiment():
            return get_openai_attraction_sentiment(attractions, city, month)
        def _get_weather_weights():
            return self.weather.get_dynamic_weights(travel_style, month, group, interests)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_sent = pool.submit(_get_sentiment)
            fut_wx   = pool.submit(_get_weather_weights)
            ai_sentiment = fut_sent.result()
            try:
                weather_weights = fut_wx.result()
            except Exception as e:
                print(f"  Warning: Dynamic weights failed ({e}) — using defaults")
                weather_weights = None
        print(f"  ✓ OpenAI batch calls done ({time.time()-t0:.1f}s)")

        # ── Step 3: Batch DistilBERT over ALL reviews at once ────────
        # Collect (attraction_index, review_text) pairs
        t1 = time.time()
        review_pairs = [
            (i, r)
            for i, a in enumerate(attractions)
            for r in a.get("reviews", [])
        ]
        if review_pairs:
            all_texts  = [r for _, r in review_pairs]
            preds      = self.review.predict_batch(all_texts)   # one batched forward pass
            # Map predictions back: model_sent[i] = fraction positive for attraction i
            pos_count  = {}
            tot_count  = {}
            for (i, _), pred in zip(review_pairs, preds):
                pos_count[i] = pos_count.get(i, 0) + (1 if pred["sentiment"] == "Positive" else 0)
                tot_count[i] = tot_count.get(i, 0) + 1
            model_sent = {i: pos_count[i] / tot_count[i] for i in tot_count}
        else:
            model_sent = {}
        print(f"  ✓ DistilBERT batch inference done — {len(review_pairs)} reviews ({time.time()-t1:.1f}s)")

        # ── Step 4: Scalar signals per attraction ────────────────────
        results = []
        for i, attraction in enumerate(attractions):
            name          = attraction["name"]
            google_rating = attraction.get("google_rating", 0)
            n_reviews     = attraction.get("user_ratings_total", len(attraction.get("reviews", [])))
            a_type        = attraction.get("attraction_type", "outdoor")

            # Blended sentiment
            rating_signal = (google_rating - 1) / 4 if google_rating > 0 else 0.5
            ai_score      = (ai_sentiment or {}).get(name, 0.7)
            ms            = model_sent.get(i)
            if ms is not None:
                review_weight   = min(n_reviews / 5.0, 0.6)
                remaining       = 1.0 - review_weight
                sentiment_score = (review_weight * ms +
                                   (remaining * 0.5) * rating_signal +
                                   (remaining * 0.5) * ai_score)
            else:
                sentiment_score = 0.5 * rating_signal + 0.5 * ai_score

            # Crowd index
            try:
                crowd_index = self.crowd.get_crowd_index(city, state, year, month)
            except Exception:
                crowd_index = min(0.3 + (google_rating / 5) * 0.5, 1.0) if google_rating else 0.5

            # Weather suitability
            try:
                weather_score = self.weather.get_suitability(city, month, a_type, weights=weather_weights)
            except Exception:
                weather_score = 0.5

            results.append({
                "name"              : name,
                "attraction_type"   : a_type,
                "google_rating"     : google_rating,
                "user_ratings_total": n_reviews,
                "crowd_index"       : float(crowd_index),
                "weather_score"     : float(weather_score),
                "sentiment_score"   : float(sentiment_score),
                "editorial_summary" : attraction.get("editorial_summary", ""),
            })

        print(f"  ✓ Feature extraction complete ({time.time()-t0:.1f}s total)")
        return results

    def run(self, city: str, state: str, month: int, year: int,
            travel_style, interests: str, group: str,
            days: int, start_day: int,
            transport_mode: str  = "driving",
            wake_hour: int       = 8,
            pace: str            = "moderate",
            dietary: str         = "none",
            budget_per_meal: str = "$20",
            fitness: str         = "moderate",
            group_ages: str      = "",
            must_visit: list     = ()) -> dict:
        """
        Full pipeline: fetch → extract features → rank → build itinerary.
        Returns {"ranked": [...], "itinerary": {...}}
        travel_style can be a string (single) or list of up to 3 styles.
        """
        # Normalise travel_style to a list; derive a primary style for single-style modules
        styles_list    = travel_style if isinstance(travel_style, list) else [travel_style]
        styles_list    = [s for s in styles_list if s in VALID_STYLES] or ["popular"]
        primary_style  = styles_list[0]
        styles_label   = " + ".join(styles_list)
        print(f"\n[Pipeline] Travel styles: {styles_label}")

        # 1a. Geocode city once — reused by both fetch and hotel search
        lat_c, lon_c = geocode_city(city, state)

        T = {}   # timing dict — printed at the end for profiling

        # 1b+1c+1d. Fetch attractions, hotel search, AND weather in PARALLEL.
        # Weather only needs city + start_date which we already know here —
        # no reason to wait until plan() starts 60-80s later.
        start_date = datetime(year, month, start_day)
        print("\n[Pipeline] Launching attraction fetch + hotel search + weather in parallel...")
        t = time.time()
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_attractions = pool.submit(
                fetch_attractions,
                city=city, state=state, max_results=60,
                lat_c=lat_c, lon_c=lon_c, travel_style=styles_list
            )
            fut_hotel = pool.submit(
                find_hotel,
                city=city, travel_style=primary_style, group=group,
                group_ages=group_ages, budget_per_meal=budget_per_meal,
                days=days, lat_c=lat_c, lon_c=lon_c
            )
            fut_weather = pool.submit(
                self.planner.get_weather_per_day,
                city, start_date, days
            )
            attractions   = fut_attractions.result()
            hotel         = fut_hotel.result()
            weather_ready = fut_weather.result()
        T["1_fetch+hotel+weather"] = time.time() - t

        # 2. Extract features (batched DistilBERT + parallel OpenAI)
        t = time.time()
        results = self.process_attractions(
            attractions, city, state, year, month,
            travel_style=primary_style, group=group, interests=interests,
        )
        T["2_features"] = time.time() - t

        # 3. Rank with LLM
        t = time.time()
        ranked = self.ranker.rank(
            results,
            city         = city,
            month        = month,
            travel_style = styles_label,
            year         = year,
            interests    = interests,
            group        = group,
            days         = days,
        )
        T["3_rank"] = time.time() - t

        seen, deduped = set(), []
        for r in ranked:
            if r["name"] not in seen:
                seen.add(r["name"])
                deduped.append(r)

        # Keep 7 per day (max 28). Smaller input = OR-Tools solves faster + schedules more %.
        cutoff = min(max(days * 7, 18), 28)
        deduped = deduped[:cutoff]
        print(f"\n[Pipeline] Using top {len(deduped)} ranked attractions (cutoff: {cutoff})")

        # 3b. Pin must-visit places at the top of the list.
        # If a place was already fetched and ranked, move it to the front.
        # If it wasn't fetched (e.g. niche spot), synthesise a minimal entry.
        if must_visit:
            existing = {r["name"].lower(): r for r in deduped}
            pinned   = []
            for mv in must_visit:
                mv_name  = mv.get("name", "")
                mv_lower = mv_name.lower()
                if mv_lower in existing:
                    # Promote to front; remove from current position
                    deduped  = [r for r in deduped if r["name"].lower() != mv_lower]
                    pinned.append(existing[mv_lower])
                else:
                    # Synthesise minimal entry from validated coordinates
                    pinned.append({
                        "name"                : mv_name,
                        "lat"                 : mv.get("lat"),
                        "lon"                 : mv.get("lon"),
                        "attraction_type"     : (mv.get("types") or ["tourist_attraction"])[0],
                        "google_rating"       : mv.get("rating", 4.0),
                        "user_ratings_total"  : 0,
                        "formatted_address"   : mv.get("address", ""),
                        "opening_hours_periods": [],
                        "open_now"            : True,
                        "editorial_summary"   : "",
                        "price_level"         : None,
                        "utility_score"       : 999.0,
                        "ranking_reason"      : "User-specified must-visit",
                        "sentiment_score"     : 0.8,
                        "crowd_index"         : 0.5,
                        "weather_score"       : 0.5,
                    })
            # Prepend pinned; trim to cutoff preserving the pinned places
            deduped = pinned + deduped[:max(0, cutoff - len(pinned))]
            print(f"\n[Pipeline] Pinned {len(pinned)} must-visit attraction(s) at top of schedule")

        # 4. Attach location + opening hours to ranked list
        coord_map = {a["name"]: a for a in attractions}
        for r in deduped:
            src = coord_map.get(r["name"], {})
            r.setdefault("lat",                   src.get("lat"))
            r.setdefault("lon",                   src.get("lon"))
            r.setdefault("opening_hours_periods", src.get("opening_hours_periods", []))
            r.setdefault("formatted_address",     src.get("formatted_address", ""))
            r.setdefault("open_now",              src.get("open_now", True))
            r.setdefault("user_ratings_total",    src.get("user_ratings_total", 0))
            r.setdefault("editorial_summary",     src.get("editorial_summary", ""))
            r.setdefault("price_level",           src.get("price_level"))

        # 5. Build itinerary
        t = time.time()
        itinerary  = self.planner.plan(
            attractions    = deduped,
            days            = days,
            start_date      = start_date,
            city            = city,
            state           = state,
            travel_style    = styles_label,
            interests       = interests,
            group           = group,
            month           = month,
            transport_mode  = transport_mode,
            wake_hour       = wake_hour,
            pace            = pace,
            dietary         = dietary,
            budget_per_meal = budget_per_meal,
            fitness         = fitness,
            group_ages      = group_ages,
            hotel           = hotel,
            weather_per_day = weather_ready,
        )
        T["5_plan"] = time.time() - t

        total = sum(T.values())
        print("\n" + "─"*50)
        print(f"  PIPELINE TIMING (total: {total:.1f}s)")
        for k, v in T.items():
            bar = "█" * int(v / total * 20)
            print(f"  {k:<20} {v:5.1f}s  {bar}")
        print("─"*50)

        return {"ranked": deduped, "itinerary": itinerary, "hotel": hotel}


# ── Display Results ───────────────────────────────────────────────────
def display_results(ranked: list, city: str, month: int,
                    year: int, travel_style: str):
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    print("\n" + "=" * 60)
    print(f"  TravelIQ Recommendations")
    print(f"  {city.title()} · {month_names[month]} {year} · {travel_style.title()}")
    print("=" * 60)
    for i, r in enumerate(ranked):
        crowd_label   = "Low" if r["crowd_index"] < 0.4 else "Medium" if r["crowd_index"] < 0.7 else "High"
        weather_label = "Great" if r["weather_score"] > 0.8 else "Good" if r["weather_score"] > 0.6 else "Fair"
        sent_label    = "Excellent" if r["sentiment_score"] >= 0.9 else "Good" if r["sentiment_score"] >= 0.7 else "Mixed"
        n_ratings     = r.get("user_ratings_total", 0)
        print(f"\n  #{i+1}  {r['name']}")
        print(f"       Type      : {r['attraction_type'].replace('_',' ').title()}")
        print(f"       Rating    : {'★'*round(r['google_rating'])} ({r['google_rating']}) — {n_ratings:,} reviews")
        print(f"       Sentiment : {sent_label}  |  Weather: {weather_label}  |  Crowds: {crowd_label}")
        print(f"       Score     : {r['utility_score']:.3f}")
        if r.get("ranking_reason"):
            print(f"       Why       : {r['ranking_reason']}")
    print("\n" + "=" * 60)


# ── Display Hotel ─────────────────────────────────────────────────────
def display_hotel(hotel: dict):
    if not hotel:
        return
    price_sym = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
    stars     = "★" * round(hotel.get("rating", 0)) + "☆" * (5 - round(hotel.get("rating", 0)))
    price     = price_sym.get(hotel.get("price_level", 2), "$$")
    print("\n" + "=" * 65)
    print("  Your Base Hotel")
    print("=" * 65)
    print(f"  {hotel['name']}")
    print(f"  {hotel.get('formatted_address', '')}")
    print(f"  {stars} ({hotel.get('rating', 'N/A')})  |  {price}")
    if hotel.get("reason"):
        print(f"  Why chosen: {hotel['reason']}")
    print("=" * 65)


# ── Display Itinerary ─────────────────────────────────────────────────
def display_itinerary(itinerary: dict, city: str, travel_style: str,
                      hotel: dict = None):
    from modules.itinerary import mins_to_time
    if not itinerary:
        print("\n  No itinerary generated.")
        return

    hotel_name = hotel.get("name", "Hotel") if hotel else "Hotel"

    print("\n" + "=" * 65)
    print(f"  TravelIQ Itinerary — {city.title()} · {travel_style.title()}")
    if hotel:
        print(f"  Staying at: {hotel_name}")
    print("=" * 65)

    for day_key, day_data in itinerary.items():
        events   = day_data.get("events", [])
        day_tip  = day_data.get("day_tip", "")
        weather  = day_data.get("weather_summary", "")
        day_note = day_data.get("day_note", "")

        print(f"\n  ── {day_key} " + "─" * (55 - len(day_key)))
        if weather:
            print(f"  Weather: {weather}")
        if day_note:
            print(f"  Note   : {day_note}")
        if day_tip:
            print(f"  Tip    : {day_tip}")

        for e in events:
            t_s    = mins_to_time(e["arrival_min"])
            t_e    = mins_to_time(e["departure_min"])
            dur    = e["departure_min"] - e["arrival_min"]
            dur_str = f"{dur//60}h {dur%60:02d}m" if dur >= 60 else f"{dur}m"
            travel = e.get("travel_to_next", "")

            etype  = e.get("event_type")

            if etype == "hotel_return":
                t_mins = e.get("travel_mins", 0)
                mode   = e.get("mode_label", "drive")
                print(f"\n  {t_s}  🏨 {e['name']}  ({t_mins} min {mode})")

            elif etype == "style_event":
                icon = e.get("icon", "✨")
                print(f"\n  {t_s}  {icon} {e['name']}  ({dur_str})")
                if e.get("suggestion"):
                    print(f"           → {e['suggestion']}")
                if e.get("tip"):
                    print(f"           Tip: {e['tip']}")

            elif etype == "meal":
                icon = e.get("icon", "🍴")
                print(f"\n  {t_s}  {icon} {e['name']}  ({dur_str})")
                if e.get("suggestion"):
                    print(f"           → {e['suggestion']}")
                if e.get("tip"):
                    print(f"           Tip: {e['tip']}")

            else:
                atype      = e.get("attraction_type", "").replace("_", " ").title()
                rating     = e.get("google_rating", 0)
                stars      = "★" * round(rating) + "☆" * (5 - round(rating))
                energy     = e.get("energy_level", "")
                energy_tag = f" [{energy} energy]" if energy else ""
                print(f"\n  {t_s}  {e['name']}{energy_tag}  ({dur_str})")
                print(f"           {atype}  |  {stars} ({rating})  |  until {t_e}")
                if e.get("highlights"):
                    print(f"           {e['highlights']}")
                if travel:
                    print(f"           → Next stop: {travel}")

        print()

    print("=" * 65)
    print("  Tip: Book restaurants in advance where suggested!")
    print("=" * 65)


# ── User Input ────────────────────────────────────────────────────────
def get_user_input() -> dict:
    print("=" * 60)
    print("         Welcome to TravelIQ")
    print("=" * 60)

    city      = input("\nEnter city name           (e.g. Phoenix):    ").strip()
    state     = input("Enter state code           (e.g. AZ):          ").strip()
    month     = int(input("Enter travel month         (1-12):             ").strip())
    year      = int(input("Enter travel year          (e.g. 2026):        ").strip())
    start_day = int(input("Enter start day of month   (1-31):             ").strip())
    days      = int(input("Number of days:                                ").strip())

    print(f"\nTravel styles: {', '.join(VALID_STYLES)}")
    style = input("Enter travel style:                            ").strip().lower()
    if style not in VALID_STYLES:
        print(f"  Unknown style '{style}' — defaulting to 'adventure'")
        style = "adventure"

    interests = input("Your interests (e.g. history, art, nature):    ").strip()
    group     = input("Traveling as (solo/couple/family/friends):      ").strip().lower()

    group_ages = ""
    if group == "family":
        group_ages = input("Ages of group members (e.g. 35,38,8,12):       ").strip()

    print("\nTransport modes: driving, transit, walking, cycling")
    transport = input("Transport mode:                                ").strip().lower()
    if transport not in ["driving", "transit", "walking", "cycling"]:
        print("  Defaulting to 'driving'")
        transport = "driving"

    wake_input = input("Preferred wake-up time (e.g. 7 for 7 AM):     ").strip()
    wake_hour  = int(wake_input) if wake_input.isdigit() else 8

    print("\nTrip pace: relaxed (few activities, long meals), moderate, packed (max activities)")
    pace = input("Trip pace:                                     ").strip().lower()
    if pace not in ["relaxed", "moderate", "packed"]:
        pace = "moderate"

    dietary        = input("Dietary restrictions (e.g. vegetarian, none):  ").strip() or "none"
    budget_per_meal= input("Meal budget per person (e.g. $20):             ").strip() or "$20"

    print("\nFitness level affects activity intensity and walking distances")
    fitness = input("Physical fitness (low/moderate/high):          ").strip().lower()
    if fitness not in ["low", "moderate", "high"]:
        fitness = "moderate"

    return {
        "city": city, "state": state, "month": month, "year": year,
        "start_day": start_day, "days": days, "travel_style": style,
        "interests": interests, "group": group, "group_ages": group_ages,
        "transport_mode": transport, "wake_hour": wake_hour, "pace": pace,
        "dietary": dietary, "budget_per_meal": budget_per_meal, "fitness": fitness,
    }


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user     = get_user_input()
    pipeline = TravelIQPipeline()
    result   = pipeline.run(
        city           = user["city"],
        state          = user["state"],
        month          = user["month"],
        year           = user["year"],
        travel_style   = user["travel_style"],
        interests      = user["interests"],
        group          = user["group"],
        days           = user["days"],
        start_day      = user["start_day"],
        transport_mode = user["transport_mode"],
        wake_hour      = user["wake_hour"],
        pace           = user["pace"],
        dietary        = user["dietary"],
        budget_per_meal= user["budget_per_meal"],
        fitness        = user["fitness"],
        group_ages     = user["group_ages"],
    )
    display_results(result["ranked"], user["city"], user["month"],
                    user["year"], user["travel_style"])
    display_hotel(result["hotel"])
    display_itinerary(result["itinerary"], user["city"], user["travel_style"],
                      hotel=result["hotel"])
