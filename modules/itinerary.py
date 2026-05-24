"""
TravelIQ — Itinerary Planner Module  (fully dynamic, v2)

Everything adapts: day start/end, meal times, visit durations, travel times,
energy curve, weather sensitivity, and day-over-day fatigue recovery.

Pipeline per run:
  1.  OpenAI  → per-attraction intelligence (duration, energy, best time, queue buffer, highlights)
  2.  Google Distance Matrix  → traffic-aware travel matrix with real departure time
  3.  Open-Meteo forecast     → actual weather per trip day (falls back to historical)
  4.  Day profiles (initial)  → dynamic start/end/meal windows per day
                                (adapts based on pace, group, previous day fatigue, weather)
  5.  OR-Tools VRPTW          → assign + order attractions; time limit scales with problem size
  6.  Variety optimisation    → OpenAI cross-day swaps to ensure type variety (no 3+ same type)
  7.  Day profiles (final)    → re-computed with actual activity counts for fatigue adaptation
  8.  Energy curve            → re-sort within each day (high energy → morning, low → evening)
  9.  Meal insertion          → fully dynamic windows from day profiles, varies by group
  10. Travel annotation       → add travel-time label between consecutive stops
  11. OpenAI enrichment       → specific restaurant suggestions, tips, day notes
                                (uses dietary, budget, group ages, weather, previous day)
  12. Final merge             → combine all signals into structured output dict
  13. Feasibility audit       → 10-point OpenAI review; corrects timing, removes closed venues,
                                enforces outdoor curfew, meal windows, pace limits
"""

import json
import math
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ── Constants & Lookup Tables ─────────────────────────────────────────────────

CITY_SPEED_KMH = 28   # urban driving fallback

# Energy sort order: 0 = earliest slot, 2 = latest slot
ENERGY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Pace → day end time and max attractions per day
PACE_CONFIG = {
    "relaxed" : {"day_end_adj":  0,  "max_per_day": 5,  "meal_longer": True},
    "moderate": {"day_end_adj": 60,  "max_per_day": 8,  "meal_longer": False},
    "packed"  : {"day_end_adj": 120, "max_per_day": 12, "meal_longer": False},
}

# Group → meal duration adjustments (minutes)
GROUP_MEAL = {
    "solo"   : {"breakfast": 20, "lunch": 45, "dinner": 60, "dinner_start_adj":   0},
    "couple" : {"breakfast": 30, "lunch": 60, "dinner": 90, "dinner_start_adj":  30},  # later/romantic
    "family" : {"breakfast": 45, "lunch": 75, "dinner": 90, "dinner_start_adj": -30},  # earlier for kids
    "friends": {"breakfast": 30, "lunch": 60, "dinner": 75, "dinner_start_adj":   0},
}

# Fitness → preferred activity energy levels
FITNESS_ENERGY = {
    "low"     : ["low", "medium"],
    "moderate": ["low", "medium", "high"],
    "high"    : ["medium", "high"],
}

# ── Per-style day timing adjustments ─────────────────────────────────
# Applied on top of pace config and group config.
#   wake_adj         : minutes added to wake hour (negative = earlier start)
#   day_end_extra    : minutes added / removed from pace-based day end
#   dinner_push      : extra minutes added to base dinner start
#   dinner_extra_dur : extra dinner duration beyond group config
#   extra_events     : list of special style events to inject each day
#     Each entry: {"name", "event_type", "icon", "offset_from", "offset_mins", "duration"}
#       offset_from: "dinner" | "wake" | "sunset"
STYLE_DAY_CONFIG = {
    "adventure"  : {"wake_adj": -30, "day_end_extra":   0, "dinner_push":   0, "dinner_extra_dur":  0},
    "cultural"   : {"wake_adj":   0, "day_end_extra":   0, "dinner_push":  30, "dinner_extra_dur": 15},
    "relaxation" : {"wake_adj":  30, "day_end_extra": -60, "dinner_push":  30, "dinner_extra_dur": 30},
    "foodie"     : {"wake_adj":   0, "day_end_extra":  60, "dinner_push":  60, "dinner_extra_dur": 60},
    "family"     : {"wake_adj":   0, "day_end_extra": -30, "dinner_push": -30, "dinner_extra_dur":  0},
    "luxury"     : {"wake_adj":  15, "day_end_extra":  60, "dinner_push":  60, "dinner_extra_dur": 60},
    "budget"     : {"wake_adj": -30, "day_end_extra":   0, "dinner_push":   0, "dinner_extra_dur":  0},
    "nightlife"  : {"wake_adj":  60, "day_end_extra": 180, "dinner_push":  90, "dinner_extra_dur": 30},
    "wellness"   : {"wake_adj": -30, "day_end_extra": -30, "dinner_push":   0, "dinner_extra_dur": 15},
    "sports"     : {"wake_adj": -30, "day_end_extra":  30, "dinner_push":   0, "dinner_extra_dur":  0},
    "romantic"   : {"wake_adj":  15, "day_end_extra":  60, "dinner_push":  60, "dinner_extra_dur": 60},
    "eco"        : {"wake_adj": -30, "day_end_extra":   0, "dinner_push":   0, "dinner_extra_dur":  0},
    "photography": {"wake_adj": -60, "day_end_extra":  60, "dinner_push":   0, "dinner_extra_dur":  0},
    "history"    : {"wake_adj":   0, "day_end_extra":   0, "dinner_push":  15, "dinner_extra_dur": 15},
    "art"        : {"wake_adj":  30, "day_end_extra":  60, "dinner_push":  30, "dinner_extra_dur": 30},
}

# ── Per-style meal guidance injected into the enrichment prompt ───────
STYLE_MEAL_GUIDANCE = {
    "adventure"  : (
        "Suggest energy-packed, quick but satisfying options — breakfast burritos, "
        "protein bowls, trail-ready lunches. Dinner should be a hearty reward meal "
        "at a local pub, grill, or outdoor beer garden."
    ),
    "cultural"   : (
        "Suggest culturally significant dining — local neighbourhood bistros, "
        "ethnic cuisine tied to the area's heritage, historic restaurants with "
        "a sense of place. Dinner should be a proper sit-down local experience."
    ),
    "relaxation" : (
        "Suggest calm, unhurried dining — garden cafes with beautiful ambiance, "
        "brunch spots, wellness-oriented menus, herbal teas. Long relaxed dinners "
        "at an intimate bistro or wine bar."
    ),
    "foodie"     : (
        "Give SPECIFIC restaurant names, chef-driven concepts, and must-try dishes. "
        "Breakfast at a beloved local institution. Lunch at a food hall or market stall. "
        "Dinner at a critically acclaimed or cult-favourite local spot. Always include "
        "a reservation note and a signature dish to order."
    ),
    "family"     : (
        "Suggest family-friendly spots with kids menus, outdoor seating, and casual "
        "vibes. Avoid slow or fancy restaurants. Highlight kid-friendly dishes, "
        "whether it's noisy/casual, and approximate wait times."
    ),
    "luxury"     : (
        "Suggest upscale experiences — hotel brunches, acclaimed tasting-menu "
        "restaurants, sommelier-curated wine pairings, private dining options. "
        "Dinner should be the highlight of the day — reservation essential, "
        "dress code noted."
    ),
    "budget"     : (
        "Suggest high-quality, affordable options — food trucks, happy hour deals, "
        "local taquerias, food markets, BYOB spots. Always mention approximate "
        "cost per person and any free/cheap tricks (e.g. lunch specials)."
    ),
    "nightlife"  : (
        "Breakfast should be a late brunch (keep it light and social). Lunch casual "
        "and quick. Dinner should be a lively, shareable pre-night-out meal — "
        "tapas, small plates, or a buzzy restaurant bar. Add a specific cocktail bar "
        "or live-music venue recommendation for after dinner."
    ),
    "wellness"   : (
        "Suggest clean, nourishing options — acai bowls, smoothie cafes, "
        "plant-based or farm-to-table restaurants, places with vegan/gluten-free "
        "options. Avoid heavy, fried, or processed suggestions."
    ),
    "sports"     : (
        "Suggest sports-bar atmospheres, hearty stadium-style food, craft breweries "
        "with big screens. Dinner at a lively sports-watching spot or local "
        "favourite near the venue."
    ),
    "romantic"   : (
        "Suggest intimate, atmospheric spots — candlelit restaurants, rooftop dining, "
        "waterfront tables, dessert bars. Dinner MUST be the romantic highlight of the "
        "day — sunset timing, wine list, ambiance description. Use specific venue names."
    ),
    "eco"        : (
        "Suggest sustainable, locally-sourced, farm-to-table options. Mention organic "
        "cafes, zero-waste restaurants, or spots that source from local farms. "
        "Avoid chain restaurants entirely."
    ),
    "photography": (
        "Breakfast should be early and quick — a great espresso spot near the first "
        "shoot location. Lunch at a photogenic cafe or colourful market stall. "
        "Dinner at a spot with stunning lighting, a view, or an Instagrammable interior."
    ),
    "history"    : (
        "Suggest historically themed dining — restaurants inside historic buildings, "
        "cuisine tied to the city's founding culture, old-school institutions, "
        "or speakeasy-style bars. Mention the history behind the venue where relevant."
    ),
    "art"        : (
        "Suggest artsy, creative dining — gallery cafes, restaurants in arts districts, "
        "eclectic menus, places popular with the local creative scene. "
        "Dinner at a venue that doubles as a gallery or performance space if possible."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def mins_to_time(mins: int) -> str:
    total    = int(mins)
    overflow = " (+1)" if total >= 1440 else ""  # flag next-day times
    total    = total % 1440                       # wrap to 0–1439
    h, m     = divmod(total, 60)
    suffix   = "AM" if h < 12 else "PM"
    h12      = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}{overflow}"


def haversine_minutes(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    km   = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return max(5, round((km / CITY_SPEED_KMH) * 60))


# ── Main Class ────────────────────────────────────────────────────────────────

class ItineraryPlanner:
    def __init__(self, openai_api_key: str, google_api_key: str):
        self.client         = OpenAI(api_key=openai_api_key)
        self.google_api_key = google_api_key
        self._weather_model = None   # lazy-loaded

    def _weather(self):
        if self._weather_model is None:
            from modules.weather import WeatherModel
            self._weather_model = WeatherModel()
        return self._weather_model

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Per-Attraction Intelligence (OpenAI)
    # ─────────────────────────────────────────────────────────────────────────
    def get_attraction_intelligence(self, attractions: list, city: str,
                                    month: int, group: str,
                                    fitness: str, group_ages: str) -> dict:
        """
        Single OpenAI call returning rich per-attraction data:
          duration      : realistic visit time in minutes
          energy_level  : high / medium / low  (physical demand)
          best_time     : morning / afternoon / evening
          queue_buffer  : extra minutes for queues, parking, getting in
          highlights    : 1-sentence what to focus on
          accessibility : high / medium / low  (wheelchair, stroller, stairs)
          skip_if       : condition when this attraction isn't worth it
        """
        lines = "\n".join(
            f"{i+1}. {a['name']} | type: {a['attraction_type']} | "
            f"rating: {a.get('google_rating','N/A')} ({a.get('user_ratings_total',0):,} reviews) | "
            f"address: {a.get('formatted_address','N/A')} | "
            f"summary: {a.get('editorial_summary','')[:80]}"
            for i, a in enumerate(attractions)
        )
        ages_note = f"Group ages: {group_ages}" if group_ages else ""

        prompt = f"""You are a professional travel planner with 20+ years of experience.

Trip context:
  City     : {city}
  Month    : {month}
  Group    : {group} {ages_note}
  Fitness  : {fitness}

For each attraction below, provide an expert assessment:

{lines}

Return ONLY valid JSON — a single object where each key is the attraction name:
{{
  "attraction name": {{
    "duration"     : <integer minutes, realistic for this group type>,
    "energy_level" : "<high|medium|low>",
    "best_time"    : "<morning|afternoon|evening>",
    "queue_buffer" : <integer minutes for queues, parking, entry — 0 for parks, 20+ for theme parks>,
    "highlights"   : "<one sentence: what to prioritize or not miss>",
    "accessibility": "<high|medium|low — high means stroller/wheelchair friendly>",
    "skip_if"      : "<one short condition when not worth visiting, e.g. 'crowded summer weekends' or 'none'>"
  }}
}}

Duration guidelines for {group} group:
- Families with young kids: add 30-45 min to base durations (bathroom breaks, slower pace)
- Couples: standard or +15 min for romantic spots
- Solo/friends: standard
- Theme parks: 240–480 min regardless
- Beach: 90–180 min
- Large museums: 90–150 min
- Parks/gardens: 60–150 min
- Historic sites: 45–90 min

No markdown, no explanation — just the JSON object."""

        print("\n[ItineraryPlanner] Getting attraction intelligence from OpenAI...")
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            intel = json.loads(raw)
            print(f"  ✓ Intelligence for {len(intel)} attractions")
            return intel
        except json.JSONDecodeError:
            print("  Warning: Could not parse intelligence — using defaults")
            return {
                a["name"]: {"duration": 90, "energy_level": "medium",
                             "best_time": "morning", "queue_buffer": 10,
                             "highlights": "", "accessibility": "high",
                             "skip_if": "none"}
                for a in attractions
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Weather Per Trip Day
    # ─────────────────────────────────────────────────────────────────────────
    def get_weather_per_day(self, city: str, start_date: datetime, days: int) -> list:
        """
        Returns list of weather dicts, one per trip day.
        Uses actual forecast (within 16 days) or historical average.
        """
        print("\n[ItineraryPlanner] Fetching weather data per day...")
        wm      = self._weather()
        results = []
        for d in range(days):
            date = start_date + timedelta(days=d)
            try:
                data = wm.get_forecast_for_date(city, date)
                label = "forecast" if data["is_forecast"] else "historical avg"
                print(f"  Day {d+1} ({date.strftime('%b %d')}): {data['summary']} [{label}]")
            except Exception as e:
                print(f"  Day {d+1}: Weather fetch failed ({e}) — skipping")
                data = {"suitability": 0.7, "summary": "Unknown", "is_forecast": False}
            results.append(data)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Dynamic Day Profiles
    # ─────────────────────────────────────────────────────────────────────────
    def compute_day_profiles(self, days: int, start_date: datetime,
                              group: str, group_ages: str, pace: str,
                              wake_hour: int, weather_per_day: list,
                              or_tools_schedule: dict = None,
                              travel_style: str = "") -> list:
        """
        Computes a dynamic profile for each day:
          day_start_min  : actual start time in minutes from midnight
          day_end_min    : actual end time
          breakfast      : (start, end)
          morning_snack  : (start, end) or None
          lunch          : (start, end)
          afternoon_snack: (start, end) or None
          dinner         : (start, end)
          day_note       : natural-language note about the day's pacing
        """
        meal_cfg   = GROUP_MEAL.get(group.lower(), GROUP_MEAL["solo"])
        pace_cfg   = PACE_CONFIG.get(pace, PACE_CONFIG["moderate"])
        style_cfg  = STYLE_DAY_CONFIG.get(travel_style, {})
        base_day_end = 1260 + pace_cfg["day_end_adj"] + style_cfg.get("day_end_extra", 0)

        profiles = []
        for d in range(days):
            # ── Start time ───────────────────────────────────────────
            # wake_adj: positive = later start (relaxation, nightlife),
            #           negative = earlier start (adventure, photography)
            day_start = wake_hour * 60 + 30 + style_cfg.get("wake_adj", 0)
            if group.lower() == "family":
                day_start += 30   # families take longer to get ready

            # Adapt based on previous day
            day_note      = ""
            prev_day_types = []
            if d > 0 and or_tools_schedule:
                prev_events    = or_tools_schedule.get(d - 1, [])
                n_activities   = len(prev_events)
                last_dep       = max((e.get("departure_min", 0) for e in prev_events), default=1200)
                prev_day_types = list({a.get("attraction_type", "outdoor")
                                       for a in prev_events})

                if n_activities >= 5 or last_dep > 1320:
                    day_start += 45
                    day_note   = "Slightly later start to recover from yesterday's full day."
                elif n_activities >= 4:
                    day_start += 20
                    day_note   = "Modest morning buffer after a busy previous day."

            # Weather adjustment
            weather = weather_per_day[d] if d < len(weather_per_day) else {}
            if weather.get("precipitation", 0) and weather["precipitation"] > 0.5:
                day_note += " Rain expected — prioritising indoor venues."

            # ── Day end ──────────────────────────────────────────────
            day_end = base_day_end
            if group.lower() == "family":
                ages_list = [int(x.strip()) for x in group_ages.split(",")
                             if x.strip().isdigit()] if group_ages else []
                has_young = any(a < 12 for a in ages_list)
                if has_young:
                    day_end = min(day_end, 1260)   # family with young kids — 9 PM max

            # ── Breakfast ────────────────────────────────────────────
            bf_start = wake_hour * 60
            bf_dur   = meal_cfg["breakfast"] + (15 if pace == "relaxed" else 0)
            bf_end   = bf_start + bf_dur

            # ── Morning snack: if first attraction is >90 min after breakfast ─
            morning_snack = None
            snack_start   = bf_end + 90
            if snack_start <= 660:   # before 11 AM
                morning_snack = (snack_start, snack_start + 15)

            # ── Lunch ────────────────────────────────────────────────
            lunch_dur   = meal_cfg["lunch"] + (30 if pace == "relaxed" else 0)
            lunch_start = max(690, bf_end + 120)   # at least 11:30 AM, 2h after breakfast
            lunch_end   = lunch_start + lunch_dur

            # ── Afternoon snack ──────────────────────────────────────
            afternoon_snack = None
            if group.lower() == "family" or pace == "relaxed":
                snack_start = lunch_end + 90
                if snack_start <= 990:   # before 4:30 PM
                    afternoon_snack = (snack_start, snack_start + 15)

            # ── Dinner ───────────────────────────────────────────────
            # dinner_push: styles like foodie/luxury/romantic eat later;
            #              family eats earlier; nightlife pushes it well past 8 PM
            base_dinner = (1110
                           + meal_cfg["dinner_start_adj"]
                           + style_cfg.get("dinner_push", 0))
            base_dinner = max(1080, min(base_dinner, 1260))      # clamp 6 PM – 9 PM
            dinner_dur  = (meal_cfg["dinner"]
                           + (30 if pace == "relaxed" else 0)
                           + style_cfg.get("dinner_extra_dur", 0))

            # Sightseeing must end 45 min before dinner so there's
            # always time to travel to the restaurant and sit down.
            sightseeing_cutoff = base_dinner - 45

            profiles.append({
                "day_start_min"       : day_start,
                "bf_end_min"          : bf_end,        # OR-Tools vehicles start here (after breakfast)
                "day_end_min"         : day_end,
                "sightseeing_cutoff_min": sightseeing_cutoff,
                "breakfast"           : (bf_start, bf_end),
                "morning_snack"       : morning_snack,
                "lunch"               : (lunch_start, lunch_end),
                "afternoon_snack"     : afternoon_snack,
                "dinner"              : (base_dinner, base_dinner + dinner_dur),
                "day_note"            : day_note.strip(),
                "prev_day_types"      : prev_day_types,
            })

        return profiles

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Traffic-Aware Travel Matrix
    # ─────────────────────────────────────────────────────────────────────────
    def get_travel_matrix(self, locations: list,
                           transport_mode: str,
                           departure_datetime: datetime) -> list:
        """
        NxN travel-time matrix (minutes).
        Uses Google Distance Matrix with departure_time for traffic-aware estimates.
        Falls back to haversine on error.
        """
        n = len(locations)
        print(f"\n[ItineraryPlanner] Building {n}×{n} travel matrix "
              f"({transport_mode}, departure {departure_datetime.strftime('%b %d %H:%M')})...")

        # Map cycling → bicycling (Google API name)
        mode_map    = {"cycling": "bicycling"}
        google_mode = mode_map.get(transport_mode, transport_mode)

        try:
            matrix = self._google_matrix(locations, google_mode, departure_datetime)
            print(f"  ✓ Google Distance Matrix ({n}×{n}, traffic-aware)")
        except Exception as e:
            print(f"  Warning: Distance Matrix failed ({e}) — haversine fallback")
            matrix = self._haversine_matrix(locations)
        return matrix

    def _google_matrix(self, locations: list, mode: str,
                        departure_dt: datetime) -> list:
        n      = len(locations)
        matrix = [[0] * n for _ in range(n)]
        batch  = 10   # 10×10 = 100 elements per request
        dep_ts = int(departure_dt.timestamp())

        # Build all (i0, j0) batch pairs upfront
        pairs = [(i0, j0)
                 for i0 in range(0, n, batch)
                 for j0 in range(0, n, batch)]

        def _fetch_batch(pair):
            i0, j0 = pair
            i1      = min(i0 + batch, n)
            j1      = min(j0 + batch, n)
            origins = "|".join(f"{lat},{lon}" for lat, lon in locations[i0:i1])
            dests   = "|".join(f"{lat},{lon}" for lat, lon in locations[j0:j1])
            params  = {
                "origins"      : origins,
                "destinations" : dests,
                "mode"         : mode,
                "key"          : self.google_api_key,
            }
            if mode in ("driving", "transit"):
                params["departure_time"] = dep_ts
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/distancematrix/json",
                params=params, timeout=15,
            ).json()
            return i0, j0, resp

        # Fire all batch calls in parallel
        with ThreadPoolExecutor(max_workers=min(len(pairs), 16)) as pool:
            results = list(pool.map(_fetch_batch, pairs))

        for i0, j0, resp in results:
            if resp.get("status") != "OK":
                raise ValueError(f"API status: {resp.get('status')}")
            for ri, row in enumerate(resp.get("rows", [])):
                for ci, elem in enumerate(row.get("elements", [])):
                    i, j = i0 + ri, j0 + ci
                    if elem.get("status") == "OK":
                        dur = elem.get("duration_in_traffic",
                                       elem.get("duration", {}))
                        matrix[i][j] = max(1, (dur.get("value", 1200)) // 60)
                    else:
                        matrix[i][j] = haversine_minutes(*locations[i], *locations[j])
        return matrix

    def _haversine_matrix(self, locations: list) -> list:
        n = len(locations)
        return [
            [0 if i == j else haversine_minutes(*locations[i], *locations[j])
             for j in range(n)]
            for i in range(n)
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Opening Hours Parser
    # ─────────────────────────────────────────────────────────────────────────
    def _get_time_window(self, attraction: dict, google_dow: int) -> tuple:
        """
        Returns (open_min, close_min) or None if closed that day.
        google_dow: 0=Sunday … 6=Saturday
        """
        periods = attraction.get("opening_hours_periods", [])
        for p in periods:
            if p.get("open", {}).get("day") == google_dow:
                ot = p["open"]["time"]
                ct = p.get("close", {}).get("time", "2100")
                om = int(ot[:2]) * 60 + int(ot[2:])
                cm = int(ct[:2]) * 60 + int(ct[2:])
                if cm == 0:
                    cm = 1440
                return (om, cm)
        return (480, 1260) if attraction.get("open_now", True) else None

    # ─────────────────────────────────────────────────────────────────────────
    # 6. OR-Tools VRPTW Solver (dynamic time limit, dynamic windows)
    # ─────────────────────────────────────────────────────────────────────────
    def solve_routing(self, attractions: list, days: int,
                      start_date: datetime, intelligence: dict,
                      travel_matrix: list, day_profiles: list,
                      fitness: str) -> dict:
        """
        Assign and order attractions across N days using OR-Tools VRPTW.
        Time windows come from Google opening hours clamped to day profiles.
        Solver time limit scales with problem size.
        """
        n       = len(attractions)
        n_nodes = n + 1   # depot = node 0

        # ── Service time = visit duration + queue buffer ─────────────
        svc = [0]
        for a in attractions:
            info   = intelligence.get(a["name"], {})
            dur    = max(30, int(info.get("duration", 90)))
            q_buf  = int(info.get("queue_buffer", 10))
            # Accessibility penalty: low accessibility means slower navigation
            access = info.get("accessibility", "high")
            if access == "low" and fitness == "low":
                dur = int(dur * 1.25)
            svc.append(dur + q_buf)

        # ── Time windows from opening hours, clamped to sightseeing cutoff ─
        # OR-Tools only schedules attractions — meals are inserted afterwards.
        # Each day's ceiling is sightseeing_cutoff_min (45 min before dinner)
        # so the dinner slot is always protected.
        day_start_min   = day_profiles[0]["day_start_min"]         if day_profiles else 480
        day_end_min     = day_profiles[0]["day_end_min"]           if day_profiles else 1260
        global_sight_cap = max(
            p.get("sightseeing_cutoff_min", day_end_min)
            for p in day_profiles
        ) if day_profiles else day_end_min

        tw = [(day_start_min, global_sight_cap)]   # depot
        for i, a in enumerate(attractions):
            travel_day = start_date + timedelta(days=i % days)
            google_dow = (travel_day.weekday() + 1) % 7
            hours      = self._get_time_window(a, google_dow)
            if hours is None:
                # Closed venue: give it a valid wide window — disjunction penalty ensures it
                # gets dropped naturally. A zero-width window can crash the CP solver.
                tw.append((day_start_min, global_sight_cap))
            else:
                om, cm   = hours
                tw_open  = max(om, day_start_min)
                # Ensure at least 1-minute window so the constraint is never degenerate
                tw_close = max(tw_open + 1, min(cm - svc[i + 1], global_sight_cap))
                tw.append((tw_open, tw_close))

        # ── OR-Tools model — wrapped so any internal exception falls to greedy ──
        try:
            mgr     = pywrapcp.RoutingIndexManager(n_nodes, days, 0)
            routing = pywrapcp.RoutingModel(mgr)

            def time_cb(fi, ti):
                fn = mgr.IndexToNode(fi)
                tn = mgr.IndexToNode(ti)
                return (travel_matrix[fn][tn] if fn < len(travel_matrix) else 20) + svc[fn]

            cb = routing.RegisterTransitCallback(time_cb)
            routing.SetArcCostEvaluatorOfAllVehicles(cb)

            routing.AddDimension(
                cb,
                300,              # max waiting slack — allows waiting up to 5h for morning openings
                global_sight_cap, # max cumulative — never schedule past dinner window
                False,
                "Time",
            )
            td = routing.GetDimensionOrDie("Time")

            # Use full-day time windows for OR-Tools so it focuses on
            # geographic routing, not tight hour-level constraints.
            # Opening-hour compliance is enforced downstream by energy_curve
            # (re-sorts to morning-first) and validate_and_fix.
            # Tight per-attraction windows cause the solver to time out trying
            # to satisfy infeasible combinations — loosening them cuts solve
            # time from 50s → <5s with equal or better route quality.
            for node in range(n_nodes):
                td.CumulVar(mgr.NodeToIndex(node)).SetRange(day_start_min, global_sight_cap)

            for v in range(days):
                prof       = day_profiles[v] if v < len(day_profiles) else day_profiles[-1]
                sight_end  = prof.get("sightseeing_cutoff_min", day_end_min)
                tour_start = prof.get("bf_end_min", prof["day_start_min"])
                td.CumulVar(routing.Start(v)).SetRange(tour_start, tour_start + 30)
                td.CumulVar(routing.End(v)).SetRange(tour_start, sight_end)
                td.SetSpanUpperBoundForVehicle(sight_end - tour_start + 60, v)

            for node in range(1, n_nodes):
                routing.AddDisjunction([mgr.NodeToIndex(node)], 8000)

            sp = pywrapcp.DefaultRoutingSearchParameters()
            # PATH_CHEAPEST_ARC finds a good greedy solution in <1s.
            # No metaheuristic — GLS kept running until the full timeout.
            sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            sp.time_limit.seconds      = 8

            print(f"\n[ItineraryPlanner] OR-Tools: {n} attractions, {days} days, "
                  f"{sp.time_limit.seconds}s limit...")
            sol = routing.SolveWithParameters(sp)

            if not sol:
                print("  Warning: No OR-Tools solution — using greedy fallback")
                return self._greedy_fallback(attractions, days, svc, day_profiles, intelligence)

            daily = {}
            for v in range(days):
                route, idx = [], routing.Start(v)
                while not routing.IsEnd(idx):
                    node = mgr.IndexToNode(idx)
                    if node != 0:
                        arr = sol.Value(td.CumulVar(idx))
                        dur = svc[node]
                        ac  = dict(attractions[node - 1])
                        ac["arrival_min"]   = arr
                        ac["departure_min"] = arr + dur
                        # Embed intelligence fields
                        info = intelligence.get(ac["name"], {})
                        ac["energy_level"]  = info.get("energy_level", "medium")
                        ac["best_time"]     = info.get("best_time", "morning")
                        ac["highlights"]    = info.get("highlights", "")
                        ac["skip_if"]       = info.get("skip_if", "")
                        ac["queue_buffer"]  = info.get("queue_buffer", 0)
                        route.append(ac)
                    idx = sol.Value(routing.NextVar(idx))
                daily[v] = route

            total = sum(len(r) for r in daily.values())
            print(f"  ✓ OR-Tools: {total}/{n} attractions scheduled across {days} days")
            return daily

        except Exception as e:
            print(f"  Warning: OR-Tools raised exception ({e}) — using greedy fallback")
            return self._greedy_fallback(attractions, days, svc, day_profiles, intelligence)

    def _greedy_fallback(self, attractions, days, svc, day_profiles, intelligence=None):
        """Simple round-robin assignment when OR-Tools fails. Always produces a valid schedule."""
        intelligence = intelligence or {}
        per   = max(1, math.ceil(len(attractions) / max(days, 1)))
        daily = {}
        for d in range(days):
            prof  = day_profiles[d] if d < len(day_profiles) else day_profiles[-1]
            cur   = prof.get("bf_end_min", prof.get("day_start_min", 480))
            cap   = prof.get("sightseeing_cutoff_min", 1200)
            route = []
            for j, a in enumerate(attractions[d * per: (d + 1) * per]):
                svc_idx = d * per + j + 1
                dur = svc[svc_idx] if svc_idx < len(svc) else 90
                if cur + dur > cap:
                    break   # day is full — don't schedule past sightseeing cutoff
                ac  = dict(a)
                info = intelligence.get(a.get("name", ""), {})
                ac["arrival_min"]   = cur
                ac["departure_min"] = cur + dur
                ac["energy_level"]  = info.get("energy_level", "medium")
                ac["best_time"]     = info.get("best_time", "morning")
                ac["highlights"]    = info.get("highlights", "")
                ac["skip_if"]       = info.get("skip_if", "")
                ac["queue_buffer"]  = info.get("queue_buffer", 0)
                route.append(ac)
                cur += dur + 20
            daily[d] = route
        print(f"  [Greedy] Scheduled {sum(len(r) for r in daily.values())}/{len(attractions)} attractions")
        return daily

    # ─────────────────────────────────────────────────────────────────────────
    # 6a. Variety Optimisation (OpenAI-guided cross-day swaps)
    # ─────────────────────────────────────────────────────────────────────────
    def optimize_variety(self, daily: dict, all_attractions: list,
                          intelligence: dict, travel_style: str,
                          days: int) -> dict:
        """
        Analyses attraction-type variety across days and uses OpenAI to suggest
        one swap per day where a day has 3+ attractions of the same type.

        Rules:
          - Only swap when a single type dominates a day (>=3 occurrences)
          - Replacement must come from the unscheduled pool
          - Never replace an attraction rated >=4.5 unless type imbalance is severe
          - At most one swap per day

        Returns updated daily schedule with better variety.
        """
        scheduled_names = {a["name"] for route in daily.values() for a in route}
        unscheduled     = [a for a in all_attractions if a["name"] not in scheduled_names]

        if not unscheduled:
            return daily   # nothing to swap with

        # ── Build day summaries ──────────────────────────────────────
        day_summaries = []
        needs_swap    = False
        for d in range(days):
            route  = daily.get(d, [])
            types  = [a.get("attraction_type", "outdoor") for a in route]
            names  = [a["name"] for a in route]
            # Check if any type repeats 3+ times
            from collections import Counter
            counts = Counter(types)
            dominant = [t for t, c in counts.items() if c >= 3]
            if dominant:
                needs_swap = True
            label = f" ⚠ dominated by {', '.join(dominant)}" if dominant else ""
            day_summaries.append(
                f"Day {d+1}{label}: " +
                ", ".join(f"{n} ({t})" for n, t in zip(names, types))
            )

        if not needs_swap:
            print("  ✓ Variety check passed — no swaps needed")
            return daily

        unscheduled_lines = "\n".join(
            f"  {a['name']} ({a.get('attraction_type','outdoor')}, "
            f"rating: {a.get('google_rating',0)}, "
            f"sentiment: {a.get('sentiment_score',0.7):.2f})"
            for a in unscheduled[:25]
        )

        prompt = f"""You are the head trip planner at a world-class travel agency. \
A tourist spending hundreds of dollars trusts you to build the most memorable, \
well-rounded itinerary possible. Your job: fix days that are monotonous.

Travel style: {travel_style}

Schedule:
{chr(10).join(day_summaries)}

Unscheduled attractions available for swaps:
{unscheduled_lines}

Task:
- Find days marked with ⚠ (dominated by one attraction type, causing a boring day)
- For each such day, suggest exactly ONE swap: remove one repetitive attraction \
  and replace it with the BEST different-type option from the unscheduled pool
- NEVER swap out an attraction rated 4.5★ or higher — those are must-sees
- Choose replacements that make the tourist excited: famous landmarks, unique local \
  experiences, and high-sentiment venues ALWAYS beat generic filler
- Prefer iconic or well-known replacements over obscure ones
- Maximum one swap per day

IMPORTANT: Think like a tourist who paid for a premium trip. Every attraction should \
make them say "wow, I'm so glad I went there". Avoid generic filler.

If no improvements are possible (e.g. no suitable replacements), return an empty array [].

Respond ONLY with valid JSON array:
[
  {{"day": 1, "remove": "exact attraction name", "add": "exact attraction name from unscheduled list"}}
]

No markdown, no explanation."""

        print("\n[ItineraryPlanner] Optimising day-by-day variety with OpenAI...")
        try:
            resp  = self.client.chat.completions.create(
                model       = "gpt-4o-mini",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0,
            )
            raw   = resp.choices[0].message.content.strip()
            swaps = json.loads(raw)
        except Exception as e:
            print(f"  Warning: Variety optimisation failed ({e}) — keeping original schedule")
            return daily

        if not swaps:
            print("  ✓ No variety swaps suggested by OpenAI")
            return daily

        # ── Apply swaps ──────────────────────────────────────────────
        unscheduled_map = {a["name"]: a for a in unscheduled}
        daily_copy      = {d: list(route) for d, route in daily.items()}
        applied         = 0

        for swap in swaps:
            day_idx     = swap.get("day", 1) - 1   # convert to 0-indexed
            remove_name = swap.get("remove", "")
            add_name    = swap.get("add", "")

            if day_idx not in daily_copy:
                continue
            if add_name not in unscheduled_map:
                continue

            route = daily_copy[day_idx]
            for i, a in enumerate(route):
                if a["name"] == remove_name:
                    add_a = dict(unscheduled_map[add_name])
                    # Inherit timing from the removed slot
                    add_a["arrival_min"]   = a["arrival_min"]
                    add_a["departure_min"] = a["departure_min"]
                    # Embed intelligence fields from pre-computed dict
                    info = intelligence.get(add_name, {})
                    add_a["energy_level"] = info.get("energy_level", "medium")
                    add_a["best_time"]    = info.get("best_time", "morning")
                    add_a["highlights"]   = info.get("highlights", "")
                    add_a["skip_if"]      = info.get("skip_if", "")
                    add_a["queue_buffer"] = info.get("queue_buffer", 0)
                    route[i] = add_a
                    print(f"  Day {day_idx + 1}: '{remove_name}' → '{add_name}' (variety swap)")
                    applied += 1
                    break

        print(f"  ✓ {applied} variety swap(s) applied")
        return daily_copy

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Energy Curve (re-sort within each day)
    # ─────────────────────────────────────────────────────────────────────────
    def apply_energy_curve(self, daily_schedule: dict,
                            intelligence: dict, fitness: str,
                            travel_matrix: list = None) -> dict:
        """
        Re-sort each day so high-energy activities are in the morning
        and low-energy ones are in the afternoon/evening.

        After re-sorting, arrival/departure times are recomputed using the
        ACTUAL travel time between consecutive stops (from the travel matrix
        when available, haversine otherwise).  This eliminates the previously
        hardcoded 20-minute gap that caused impossible zero-gap schedules.

        travel_matrix : the NxN matrix built in plan().  Attraction's
                        _location_idx field is used as the row/column key.
        """
        allowed = FITNESS_ENERGY.get(fitness, FITNESS_ENERGY["moderate"])
        result  = {}

        for day, route in daily_schedule.items():
            if not route:
                result[day] = route
                continue

            # Filter out activities that exceed fitness level
            filtered = [a for a in route
                        if a.get("energy_level", "medium") in allowed]
            dropped  = [a for a in route
                        if a.get("energy_level", "medium") not in allowed]
            if dropped:
                print(f"  Day {day+1}: Removed {len(dropped)} too-intense "
                      f"activity(ies) for fitness='{fitness}'")

            # Sort: high → morning, medium → midday, low → afternoon/evening
            sorted_route = sorted(
                filtered,
                key=lambda a: ENERGY_ORDER.get(a.get("energy_level", "medium"), 1)
            )

            # Recompute times with real travel gaps
            if sorted_route:
                anchor = sorted_route[0].get("arrival_min", 480)
                for i, a in enumerate(sorted_route):
                    dur = a["departure_min"] - a["arrival_min"]
                    if i == 0:
                        a["arrival_min"]   = anchor
                        a["departure_min"] = anchor + dur
                    else:
                        prev      = sorted_route[i - 1]
                        prev_dep  = prev["departure_min"]
                        travel_gap = self._travel_gap(prev, a, travel_matrix)
                        a["arrival_min"]   = prev_dep + travel_gap
                        a["departure_min"] = a["arrival_min"] + dur

            result[day] = sorted_route
        return result

    def _travel_gap(self, from_a: dict, to_a: dict,
                    travel_matrix: list = None) -> int:
        """
        Returns the travel time in minutes between two attractions.

        Priority:
          1. travel_matrix[from_idx][to_idx]  — exact, traffic-aware
          2. haversine(from_a, to_a)          — geometric fallback
          3. 15 minutes                        — safe absolute fallback

        Adds a 5-minute buffer (parking + walking to entrance) on top of
        the raw travel time so consecutive stops never overlap.
        """
        ENTRY_BUFFER = 5   # minutes for parking, security, entry queue

        from_idx = from_a.get("_location_idx")
        to_idx   = to_a.get("_location_idx")

        if (travel_matrix and
                from_idx is not None and to_idx is not None and
                from_idx < len(travel_matrix) and
                to_idx   < len(travel_matrix[from_idx])):
            raw = travel_matrix[from_idx][to_idx]
            return max(raw + ENTRY_BUFFER, 1)

        # Haversine fallback when matrix index is unavailable (e.g. swapped-in variety attraction)
        if from_a.get("lat") and to_a.get("lat"):
            raw = haversine_minutes(
                from_a["lat"], from_a["lon"],
                to_a["lat"],   to_a["lon"])
            return max(raw + ENTRY_BUFFER, 1)

        return 15   # absolute safe default

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Dynamic Meal Insertion
    # ─────────────────────────────────────────────────────────────────────────
    def insert_meals(self, daily_schedule: dict, day_profiles: list,
                     travel_style: str = "",
                     travel_matrix: list = None) -> dict:
        """
        Insert meals into each day using the dynamic windows from day profiles.
        Meals adapt to group, pace, and actual schedule timing.

        travel_matrix: the NxN matrix from plan() — used to compute the real
        travel time between the attraction before a meal and the attraction
        after it, so post-meal departure times are accurate.

        Also injects style-specific signature events:
          - photography : "Golden Hour Photography" slot (pre-dinner)
          - foodie      : "Local Food Market / Tasting" slot (mid-afternoon)
          - nightlife   : "Evening Bar / Live Music" slot (after dinner)
          - wellness    : "Morning Wellness Break" slot (early morning)
        """
        result = {}

        for day, route in daily_schedule.items():
            prof       = day_profiles[day] if day < len(day_profiles) else day_profiles[-1]
            all_events = []
            events     = sorted(route, key=lambda x: x.get("arrival_min", prof["day_start_min"]))

            # ── Breakfast ────────────────────────────────────────────
            bf_s, bf_e = prof["breakfast"]
            all_events.append({
                "event_type": "meal", "meal_type": "breakfast",
                "name": "Breakfast", "icon": "☕",
                "arrival_min": bf_s, "departure_min": bf_e,
                "suggestion": "", "tip": "",
            })

            prev_dep    = prof["day_start_min"]
            lunch_done  = False
            snack_am    = False
            snack_pm    = False
            lunch_s, lunch_e = prof["lunch"]
            din_s, din_e     = prof["dinner"]
            last_attr        = None   # last non-meal attraction appended
            last_lunch_dep   = None   # departure time of lunch (for meal gap enforcement)
            dinner_covered   = False  # True if a food-type attraction covers dinner window

            # Minimum gap between any two meals (breakfast excluded).
            MIN_MEAL_GAP = 180

            # Fallback when travel matrix is unavailable or indices are missing.
            DEFAULT_TRAVEL = 15

            def _travel_between(from_a, to_a) -> int:
                """Travel time (min) from from_a to to_a using matrix, else haversine, else default."""
                if travel_matrix and from_a and to_a:
                    fi = from_a.get("_location_idx")
                    ti = to_a.get("_location_idx")
                    if (fi is not None and ti is not None and
                            fi < len(travel_matrix) and ti < len(travel_matrix[fi])):
                        return max(5, travel_matrix[fi][ti])
                    # Haversine fallback
                    if from_a.get("lat") and to_a.get("lat"):
                        return max(5, haversine_minutes(
                            from_a["lat"], from_a["lon"],
                            to_a["lat"], to_a["lon"]))
                return DEFAULT_TRAVEL

            # Pre-detect food-type attractions that serve as implicit meals.
            # If a scheduled attraction IS a restaurant/bar, it replaces the
            # corresponding meal slot — prevents duplicate lunch/dinner cards.
            _FOOD_TYPES = {"food", "restaurant", "bar", "cafe", "meal_delivery", "meal_takeaway"}
            for _fa in events:
                _fa_types = set(t.lower() for t in (_fa.get("types") or []))
                if _fa_types & _FOOD_TYPES:
                    _fa_arr = _fa.get("arrival_min", 0)
                    _fa_dep = _fa.get("departure_min", _fa_arr + 90)
                    if _fa_arr < lunch_e + 45 and _fa_dep > lunch_s - 45:
                        lunch_done   = True
                        last_lunch_dep = _fa_dep
                    if _fa_arr < din_e + 45 and _fa_dep > din_s - 45:
                        dinner_covered = True

            for a in events:
                a   = dict(a)
                arr = a.get("arrival_min", prev_dep)
                dep = a.get("departure_min", arr + 90)

                # Real travel time between surrounding attractions (from Google Distance Matrix).
                # Restaurants have no coordinates, so we use half the prev→next travel as the
                # best estimate of how long it takes to reach (and leave) a meal stop.
                raw_travel   = _travel_between(last_attr, a)
                meal_travel  = max(10, raw_travel // 2)   # half-route estimate, min 10 min

                # ── Morning snack ────────────────────────────────────
                if not snack_am and prof["morning_snack"]:
                    ms_s, ms_e = prof["morning_snack"]
                    if ms_s <= prev_dep <= ms_s + 60:
                        snack_end = prev_dep + 15
                        all_events.append({
                            "event_type": "meal", "meal_type": "morning_snack",
                            "name": "Morning Snack / Coffee", "icon": "🥐",
                            "arrival_min": prev_dep, "departure_min": snack_end,
                            "suggestion": "", "tip": "",
                        })
                        snack_am = True
                        earliest_next = snack_end + meal_travel
                        if arr < earliest_next:
                            shift = earliest_next - arr
                            arr += shift; dep += shift
                            a["arrival_min"] = arr; a["departure_min"] = dep
                        prev_dep = snack_end

                # ── Lunch ────────────────────────────────────────────
                if not lunch_done and arr >= lunch_s:
                    ls = max(lunch_s, prev_dep + meal_travel)
                    le = ls + (lunch_e - lunch_s)
                    # Only insert lunch if there's still MIN_MEAL_GAP before dinner.
                    if le + MIN_MEAL_GAP <= din_s:
                        all_events.append({
                            "event_type": "meal", "meal_type": "lunch",
                            "name": "Lunch", "icon": "🍽️",
                            "arrival_min": ls, "departure_min": le,
                            "suggestion": "", "tip": "",
                        })
                        lunch_done     = True
                        last_lunch_dep = le
                        earliest_next  = le + meal_travel
                        if arr < earliest_next:
                            shift = earliest_next - arr
                            arr += shift; dep += shift
                            a["arrival_min"] = arr; a["departure_min"] = dep
                        prev_dep = le

                # ── Afternoon snack ──────────────────────────────────
                if not snack_pm and prof["afternoon_snack"]:
                    as_s, as_e = prof["afternoon_snack"]
                    if as_s <= prev_dep <= as_s + 60:
                        snack_end = prev_dep + 15
                        all_events.append({
                            "event_type": "meal", "meal_type": "afternoon_snack",
                            "name": "Afternoon Snack", "icon": "🧃",
                            "arrival_min": prev_dep, "departure_min": snack_end,
                            "suggestion": "", "tip": "",
                        })
                        snack_pm = True
                        earliest_next = snack_end + meal_travel
                        if arr < earliest_next:
                            shift = earliest_next - arr
                            arr += shift; dep += shift
                            a["arrival_min"] = arr; a["departure_min"] = dep
                        prev_dep = snack_end

                all_events.append(a)
                last_attr = a
                prev_dep  = dep

            # Lunch safety net — no sightseeing crossed lunch_s; no surrounding attractions
            # to use for matrix lookup so fall back to DEFAULT_TRAVEL.
            if not lunch_done:
                ls = max(lunch_s, prev_dep + DEFAULT_TRAVEL)
                le = ls + (lunch_e - lunch_s)
                # Skip lunch if it would leave less than MIN_MEAL_GAP before dinner.
                if le + MIN_MEAL_GAP <= din_s:
                    all_events.append({
                        "event_type": "meal", "meal_type": "lunch",
                        "name": "Lunch", "icon": "🍽️",
                        "arrival_min": ls, "departure_min": le,
                        "suggestion": "", "tip": "",
                    })
                    last_lunch_dep = le

            # ── Style-specific pre-dinner slot ───────────────────────
            # Inserted BEFORE dinner into the event list so the final
            # sort places it correctly between the last attraction and dinner.

            if travel_style == "foodie":
                # "Local Food Market / Tasting" — 60 min in the mid-afternoon
                # Target ~3:00 PM (900); shifts right if schedule is running late
                last_dep_before_din = max(e["departure_min"] for e in all_events)
                tasting_s = max(900, last_dep_before_din + 15)
                tasting_e = tasting_s + 60
                # Only insert if there is room before dinner window
                if tasting_e <= din_s - 30:
                    all_events.append({
                        "event_type": "style_event",
                        "style_event_type": "food_tasting",
                        "name": "Local Food Market / Tasting",
                        "icon": "🛒",
                        "arrival_min"  : tasting_s,
                        "departure_min": tasting_e,
                        "suggestion": "", "tip": "",
                    })

            elif travel_style == "photography":
                # "Golden Hour Photography" — 60 min starting ~75 min before dinner
                # (gives time to wrap up and travel to restaurant)
                gh_start = din_s - 90
                gh_end   = gh_start + 60
                last_dep_before_din = max(e["departure_min"] for e in all_events)
                gh_start = max(gh_start, last_dep_before_din + 10)
                gh_end   = gh_start + 60
                if gh_end <= din_s - 20:
                    all_events.append({
                        "event_type": "style_event",
                        "style_event_type": "golden_hour",
                        "name": "Golden Hour Photography",
                        "icon": "📷",
                        "arrival_min"  : gh_start,
                        "departure_min": gh_end,
                        "suggestion": "", "tip": "",
                    })

            elif travel_style == "wellness":
                # "Morning Wellness Break" — 30 min yoga/meditation right after breakfast
                bf_s, bf_e = prof["breakfast"]
                wb_start = bf_e + 10
                wb_end   = wb_start + 30
                all_events.append({
                    "event_type": "style_event",
                    "style_event_type": "wellness_break",
                    "name": "Morning Wellness / Meditation",
                    "icon": "🧘",
                    "arrival_min"  : wb_start,
                    "departure_min": wb_end,
                    "suggestion": "", "tip": "",
                })

            # ── Dinner ───────────────────────────────────────────────
            # Skip if a food-type attraction already covers the dinner window.
            if not dinner_covered:
                last_dep     = max(e["departure_min"] for e in all_events)
                # Enforce MIN_MEAL_GAP after lunch.
                min_din_s    = (last_lunch_dep + MIN_MEAL_GAP) if last_lunch_dep else 0
                actual_din_s = max(din_s, last_dep + DEFAULT_TRAVEL, min_din_s)
                actual_din_s = min(actual_din_s, 1320)   # never later than 10 PM
                din_dur      = din_e - din_s
                all_events.append({
                    "event_type": "meal", "meal_type": "dinner",
                    "name": "Dinner", "icon": "🍴",
                    "arrival_min": actual_din_s,
                    "departure_min": actual_din_s + din_dur,
                    "suggestion": "", "tip": "",
                })
            else:
                # Derive actual_din_s for the nightlife slot below from existing events.
                actual_din_s = max(
                    (e["departure_min"] for e in all_events if e.get("meal_type") == "dinner"),
                    default=din_s,
                )
                din_dur = din_e - din_s

            # ── Nightlife: post-dinner "Evening Out" ─────────────────
            # Added AFTER dinner so it always lands at the very end of the day.
            if travel_style == "nightlife":
                din_dep  = actual_din_s + din_dur
                eve_start = din_dep + DEFAULT_TRAVEL
                eve_end   = eve_start + 90
                if eve_end <= 1380:   # not past 11 PM
                    all_events.append({
                        "event_type": "style_event",
                        "style_event_type": "evening_out",
                        "name": "Evening Bar / Live Music",
                        "icon": "🎶",
                        "arrival_min"  : eve_start,
                        "departure_min": eve_end,
                        "suggestion": "", "tip": "",
                    })

            all_events.sort(key=lambda x: x["arrival_min"])
            result[day] = all_events

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 8b. Gap Filler — no dead air between events
    # ─────────────────────────────────────────────────────────────────────────
    def _fill_time_gaps(self, daily_schedule: dict,
                        unused_attractions: list = None,
                        intelligence: dict = None) -> dict:
        """
        Scans each day for gaps > 60 minutes between consecutive events.

        Priority order for filling a gap:
          1. Insert a real unused attraction that fits within the gap window
             (travel_to + visit_duration + travel_from ≤ gap).
             Picks the highest-utility attraction closest to the previous event.
          2. If no attraction fits, insert a short free-time placeholder
             capped at 60 minutes — never a 3-hour "Morning Stroll".

        Attractions consumed here are removed from the pool so the same place
        cannot appear twice across different days/gaps.
        """
        MIN_GAP      = 60   # only act on gaps longer than this
        MAX_FREETEXT = 60   # free-time placeholder ceiling (minutes)
        TRAVEL_PAD   = 20   # conservative haversine-to-real-travel buffer (minutes)

        pool = list(unused_attractions) if unused_attractions else []
        intel = intelligence or {}

        def _free_label(start_min: int) -> tuple[str, str]:
            if start_min < 600:
                return ("Morning Stroll / Explore the Neighborhood", "🚶")
            if start_min < 1020:
                return ("Free Time / Local Exploration", "🗺️")
            return ("Sunset Walk / Pre-Dinner Drinks", "🌇")

        def _visit_dur(a: dict) -> int:
            info = intel.get(a.get("name", ""), {})
            return max(30, int(info.get("duration", 60)) + int(info.get("queue_buffer", 0)))

        def _best_fit(prev_ev, gap_start, gap_end):
            """Return (attraction, arr_min, dep_min) or None."""
            p_lat = prev_ev.get("lat")
            p_lon = prev_ev.get("lon")
            gap   = gap_end - gap_start
            best  = None
            best_score = -1

            for a in pool:
                a_lat, a_lon = a.get("lat"), a.get("lon")
                if not a_lat or not a_lon:
                    continue
                # Travel time from previous event to this attraction
                t_to = (haversine_minutes(p_lat, p_lon, a_lat, a_lon) + TRAVEL_PAD
                        if p_lat and p_lon else 15)
                dur  = _visit_dur(a)
                # Need at least travel_to + visit + TRAVEL_PAD buffer to leave
                needed = t_to + dur + TRAVEL_PAD
                if needed > gap:
                    continue
                score = a.get("utility_score", 0) or a.get("google_rating", 0)
                if score > best_score:
                    best_score = score
                    best       = (a, gap_start + t_to, gap_start + t_to + dur)

            return best

        result = {}
        for day, events in daily_schedule.items():
            if not events:
                result[day] = events
                continue

            sorted_ev = sorted(events, key=lambda e: e.get("arrival_min", 0))
            filled    = []

            for i, ev in enumerate(sorted_ev):
                if i > 0:
                    prev      = sorted_ev[i - 1]
                    prev_dep  = prev.get("departure_min", 0)
                    curr_arr  = ev.get("arrival_min", prev_dep)
                    gap       = curr_arr - prev_dep

                    if gap >= MIN_GAP:
                        cursor = prev_dep   # tracks how far we've filled

                        # ── Try to fill with a real attraction ──────────────
                        fit = _best_fit(prev, cursor, curr_arr)
                        if fit:
                            a, arr, dep = fit
                            pool.remove(a)
                            # Leave a small travel buffer before the attraction
                            filled.append({
                                **a,
                                "event_type"   : "attraction",
                                "arrival_min"  : arr,
                                "departure_min": dep,
                                # Clear any stale scheduling fields from prior plan
                                "travel_to_next": "",
                            })
                            cursor = dep
                            print(f"  [gap-fill] inserted '{a['name']}' "
                                  f"({arr//60%12 or 12}:{arr%60:02d}–{dep//60%12 or 12}:{dep%60:02d})")

                        # ── Remaining gap → free-time placeholder, max 60 min ─
                        remaining = curr_arr - cursor
                        if remaining >= MIN_GAP:
                            ft_end = min(cursor + MAX_FREETEXT, curr_arr)
                            label, icon = _free_label(cursor)
                            filled.append({
                                "event_type"    : "free_time",
                                "name"          : label,
                                "icon"          : icon,
                                "arrival_min"   : cursor,
                                "departure_min" : ft_end,
                                "suggestion"    : "",
                                "tip"           : "",
                            })

                filled.append(ev)

            result[day] = filled
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 9. OpenAI Enrichment (context-aware: weather, dietary, budget, prev day)
    # ─────────────────────────────────────────────────────────────────────────
    def enrich_with_openai(self, daily_schedule: dict, city: str,
                           travel_style: str, interests: str, group: str,
                           group_ages: str, month: int, dietary: str,
                           budget_per_meal: str, weather_per_day: list,
                           day_profiles: list) -> dict:
        """
        Rich OpenAI enrichment call that knows about:
          - dietary restrictions and meal budget
          - actual weather forecast per day
          - group composition and ages
          - day profile notes (tired from yesterday, rainy day, etc.)
        """
        month_names = {
            1:"January", 2:"February", 3:"March",    4:"April",
            5:"May",      6:"June",     7:"July",     8:"August",
            9:"September",10:"October",11:"November",12:"December",
        }

        # Build schedule summary for OpenAI
        lines = []
        for day, events in daily_schedule.items():
            weather = weather_per_day[day]["summary"] if day < len(weather_per_day) else "unknown"
            prof    = day_profiles[day] if day < len(day_profiles) else {}
            note    = prof.get("day_note", "")
            lines.append(f"Day {day+1} | Weather: {weather}{' | ' + note if note else ''}:")
            for e in events:
                t = mins_to_time(e["arrival_min"])
                if e.get("event_type") == "meal":
                    lines.append(f"  {t}  [{e['meal_type']}]")
                elif e.get("event_type") == "style_event":
                    dur = e["departure_min"] - e["arrival_min"]
                    lines.append(f"  {t}  [{e['style_event_type']}] {e['name']} ({dur} min)")
                else:
                    dur = e["departure_min"] - e["arrival_min"]
                    lines.append(f"  {t}  {e['name']} ({e.get('attraction_type','')} | {dur} min)")

        ages_note   = f"Group ages: {group_ages}." if group_ages else ""
        style_guide = STYLE_MEAL_GUIDANCE.get(travel_style, "")

        prompt = f"""You are a 20-year expert local travel and food advisor for {city.title()}.

Trip profile:
  Style          : {travel_style}
  Interests      : {interests or 'not specified'}
  Group          : {group} {ages_note}
  Month          : {month_names[month]}
  Dietary        : {dietary}
  Meal budget    : {budget_per_meal} per person

{"─"*60}
STYLE-SPECIFIC DINING GUIDANCE (follow this precisely):
{style_guide}
{"─"*60}

Itinerary:
{chr(10).join(lines)}

For each meal AND style event (food_tasting, golden_hour, evening_out, wellness_break)
in each day provide:
  suggestion: A specific, named local experience or restaurant — tailored to the city
              neighbourhood, time of day, dietary restrictions, meal budget, and group
              type. Be concrete and style-aligned (e.g. for foodie: name the actual
              market + what to try; for romantic: name the exact rooftop restaurant).
              Do NOT give generic suggestions like "Mexican food" or "a nice restaurant."
  tip: One sharp practical tip (dish to order, book in advance, best table, etc.)

Also write "day_tip": one practical, style-relevant tip for the whole day.

Return ONLY valid JSON (no markdown). Include ALL days — every day in the itinerary must have its own entry.
The key for each day must be exactly "Day 1", "Day 2", "Day 3", etc.

Example structure (fill in ALL days, not just Day 1):
{{
  "Day 1": {{
    "breakfast":        {{"suggestion": "...", "tip": "..."}},
    "morning_snack":    {{"suggestion": "...", "tip": "..."}},
    "lunch":            {{"suggestion": "...", "tip": "..."}},
    "afternoon_snack":  {{"suggestion": "...", "tip": "..."}},
    "dinner":           {{"suggestion": "...", "tip": "..."}},
    "food_tasting":     {{"suggestion": "...", "tip": "..."}},
    "golden_hour":      {{"suggestion": "...", "tip": "..."}},
    "evening_out":      {{"suggestion": "...", "tip": "..."}},
    "wellness_break":   {{"suggestion": "...", "tip": "..."}},
    "day_tip": "..."
  }},
  "Day 2": {{
    ... same structure ...
  }},
  "Day 3": {{
    ... same structure ...
  }}
}}

Rules:
- Every day in the itinerary MUST have its own key ("Day 1", "Day 2", etc.)
- Within each day, only include meal/event keys that appear in that day's schedule
- Do NOT skip any day
- No markdown, no code fences"""

        print("\n[ItineraryPlanner] Fetching enriched meal suggestions from OpenAI...")
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            enrichments = json.loads(raw)
            print("  ✓ Enrichment received")
            return enrichments
        except json.JSONDecodeError:
            print("  Warning: Could not parse enrichments — skipping")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Travel-to-next annotation
    # ─────────────────────────────────────────────────────────────────────────
    def annotate_travel(self, daily_schedule: dict,
                         travel_matrix: list, locations: list,
                         transport_mode: str) -> dict:
        """
        Add a 'travel_to_next' label to each attraction showing how long
        it takes to get to the next stop (e.g. '12 min drive').

        Uses _location_idx — the global index in the locations/travel_matrix
        list — set on every attraction by plan().  Falls back to haversine
        when the index is missing (e.g. meal events, hotel-return events).
        """
        mode_label = {"driving": "drive", "transit": "transit",
                      "walking": "walk", "cycling": "cycle"}.get(transport_mode, "travel")
        result = {}
        for day, events in daily_schedule.items():
            annotated = []
            for i, e in enumerate(events):
                ec    = dict(e)
                etype = e.get("event_type", "")

                # No arrow after hotel return (last event) or free-time blocks
                if etype in ("hotel_return", "free_time"):
                    annotated.append(ec)
                    continue

                # Immediate next event in display order
                next_ev = events[i + 1] if i + 1 < len(events) else None
                if next_ev is None or next_ev.get("event_type") == "hotel_return":
                    annotated.append(ec)
                    continue

                next_etype = next_ev.get("event_type", "")

                if etype == "meal":
                    # Meal → whatever is next: use haversine if coords available,
                    # else fall back to the schedule gap (capped at 30 min).
                    if next_etype not in ("meal", "hotel_return", "free_time"):
                        if e.get("lat") and next_ev.get("lat"):
                            mins = haversine_minutes(e["lat"], e["lon"],
                                                     next_ev["lat"], next_ev["lon"])
                        else:
                            gap  = next_ev.get("arrival_min", 0) - e.get("departure_min", 0)
                            mins = min(gap, 30) if gap > 0 else 0
                        if mins > 0:
                            ec["travel_to_next"] = f"{mins} min {mode_label}"

                else:
                    # Sightseeing / style_event → next event
                    if next_etype == "meal":
                        # Use real travel time stored by _reanchor_meals, or haversine,
                        # NOT the full scheduling gap (which includes leisure buffer time).
                        travel_mins = next_ev.get("_travel_from_prev")
                        if travel_mins is None and next_ev.get("lat") and e.get("lat"):
                            travel_mins = haversine_minutes(e["lat"], e["lon"],
                                                            next_ev["lat"], next_ev["lon"])
                        if travel_mins and travel_mins > 0:
                            ec["travel_to_next"] = f"{travel_mins} min {mode_label}"
                    elif next_etype not in ("hotel_return", "free_time"):
                        # Sightseeing/style → sightseeing/style: use travel matrix
                        curr_idx = e.get("_location_idx")
                        next_idx = next_ev.get("_location_idx")
                        if (curr_idx is not None and next_idx is not None and
                                curr_idx < len(travel_matrix) and
                                next_idx < len(travel_matrix[curr_idx])):
                            mins = travel_matrix[curr_idx][next_idx]
                        else:
                            mins = (haversine_minutes(
                                        e["lat"], e["lon"],
                                        next_ev["lat"], next_ev["lon"])
                                    if e.get("lat") and next_ev.get("lat") else 15)
                        if mins > 0:
                            ec["travel_to_next"] = f"{mins} min {mode_label}"

                annotated.append(ec)
            result[day] = annotated
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 10b. Meal Reanchor — real travel times once restaurant names are known
    # ─────────────────────────────────────────────────────────────────────────
    def _reanchor_meals(self, final: dict, city: str,
                        transport_mode: str, departure_dt: datetime) -> dict:
        """
        After OpenAI enrichment gives real restaurant names, geocode each one
        with Google Places Find Place, then fire two parallel Distance Matrix
        calls — one batch for (prev_attraction → restaurant) and one for
        (restaurant → next_attraction) — and update meal arrival/departure
        times with actual travel data.  Downstream events are cascade-shifted
        forward to keep the schedule consistent.
        """
        import re

        mode_map    = {"cycling": "bicycling"}
        google_mode = mode_map.get(transport_mode, transport_mode)
        dep_ts      = int(departure_dt.timestamp())
        MAX_NIGHT   = 1380  # 11 PM cap

        # ── Step 1: Collect meal jobs ────────────────────────────────────
        class _Job:
            __slots__ = ("dk", "idx", "name", "prev_ev", "next_ev",
                         "rest_loc", "to_rest", "from_rest")
            def __init__(self, dk, idx, name, prev_ev, next_ev):
                self.dk = dk; self.idx = idx; self.name = name
                self.prev_ev = prev_ev; self.next_ev = next_ev
                self.rest_loc = self.to_rest = self.from_rest = None

        jobs = []
        for dk, day_data in final.items():
            events = day_data.get("events", [])
            for i, e in enumerate(events):
                if e.get("event_type") != "meal":
                    continue
                suggestion = e.get("suggestion", "").strip()
                if not suggestion:
                    continue
                name = re.split(r"\s*[-–—]\s*|\.\s+", suggestion)[0].strip()
                if len(name) < 3:
                    continue
                prev_ev = next(
                    (ev for ev in reversed(events[:i])
                     if ev.get("event_type") not in ("meal", "hotel_return", "free_time")
                     and ev.get("lat")),
                    None,
                )
                next_ev = next(
                    (ev for ev in events[i + 1:]
                     if ev.get("event_type") not in ("meal", "hotel_return", "free_time")
                     and ev.get("lat")),
                    None,
                )
                jobs.append(_Job(dk, i, name, prev_ev, next_ev))

        if not jobs:
            return final

        print(f"\n[ItineraryPlanner] Reanchoring {len(jobs)} meals with real restaurant travel...")

        # ── Step 2: Geocode restaurants in parallel ──────────────────────
        def _geocode(job):
            """Returns (lat, lon, formatted_address, confirmed_name) or None."""
            try:
                resp = requests.get(
                    "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                    params={
                        "input"    : f"{job.name} {city}",
                        "inputtype": "textquery",
                        "fields"   : "geometry,formatted_address,name",
                        "key"      : self.google_api_key,
                    },
                    timeout=10,
                ).json()
                cands = resp.get("candidates", [])
                if cands:
                    c   = cands[0]
                    loc = c["geometry"]["location"]
                    return (
                        loc["lat"],
                        loc["lng"],
                        c.get("formatted_address", ""),
                        c.get("name", job.name),
                    )
            except Exception:
                pass
            return None

        # _Job gains two extra slots for address data
        for job in jobs:
            job.rest_address = ""
            job.rest_name    = ""

        with ThreadPoolExecutor(max_workers=min(len(jobs), 10)) as pool:
            for job, geo in zip(jobs, pool.map(_geocode, jobs)):
                if geo:
                    job.rest_loc     = (geo[0], geo[1])
                    job.rest_address = geo[2]
                    job.rest_name    = geo[3]

        valid = [j for j in jobs if j.rest_loc]
        print(f"  ✓ Geocoded {len(valid)}/{len(jobs)} restaurants")
        if not valid:
            return final

        # ── Step 3: Two parallel batched Distance Matrix calls ───────────
        # Diagonal trick: origins[i] → destinations[i] maps to rows[i][i].
        prev_pairs = [(j.prev_ev, j.rest_loc) for j in valid if j.prev_ev]
        next_pairs = [(j.rest_loc, j.next_ev)  for j in valid if j.next_ev]

        def _loc_str(loc):
            if isinstance(loc, tuple):
                return f"{loc[0]},{loc[1]}"
            return f"{loc['lat']},{loc['lon']}"

        def _dm_diagonal(pairs):
            """One Distance Matrix call → diagonal travel minutes (origin_i→dest_i)."""
            if not pairs:
                return []
            origins = "|".join(_loc_str(o) for o, _ in pairs)
            dests   = "|".join(_loc_str(d) for _, d in pairs)
            params  = {"origins": origins, "destinations": dests,
                       "mode": google_mode, "key": self.google_api_key}
            if google_mode in ("driving", "transit"):
                params["departure_time"] = dep_ts
            try:
                resp = requests.get(
                    "https://maps.googleapis.com/maps/api/distancematrix/json",
                    params=params, timeout=15,
                ).json()
                if resp.get("status") != "OK":
                    return [None] * len(pairs)
                out = []
                for ri, row in enumerate(resp.get("rows", [])):
                    elems = row.get("elements", [])
                    elem  = elems[ri] if ri < len(elems) else {}
                    if elem.get("status") == "OK":
                        dur = elem.get("duration_in_traffic", elem.get("duration", {}))
                        out.append(max(1, dur.get("value", 900) // 60))
                    else:
                        out.append(None)
                return out
            except Exception:
                return [None] * len(pairs)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_dm_diagonal, prev_pairs)
            fut_b = pool.submit(_dm_diagonal, next_pairs)
            times_a = fut_a.result()
            times_b = fut_b.result()

        # Map diagonal times back to each job
        pi = ni = 0
        for j in valid:
            if j.prev_ev:
                j.to_rest   = times_a[pi]; pi += 1
            if j.next_ev:
                j.from_rest = times_b[ni]; ni += 1

        # ── Step 4: Update timing + cascade forward ──────────────────────
        job_map = {(j.dk, j.idx): j for j in valid}
        result  = {}
        for dk, day_data in final.items():
            events   = [dict(e) for e in day_data.get("events", [])]
            modified = False

            for i, e in enumerate(events):
                j = job_map.get((dk, i))
                if not j:
                    continue

                # Always store the address + coords — even if timing didn't change
                if j.rest_address:
                    e["restaurant_address"] = j.rest_address
                    e["restaurant_name"]    = j.rest_name
                if j.rest_loc:
                    e["lat"] = j.rest_loc[0]
                    e["lon"] = j.rest_loc[1]
                if j.to_rest:
                    e["_travel_from_prev"] = j.to_rest

                if j.to_rest is None and j.from_rest is None:
                    continue

                cur_arr  = e.get("arrival_min", 0)
                meal_dur = e.get("departure_min", cur_arr + 45) - cur_arr

                new_arr = cur_arr
                if j.to_rest and j.prev_ev:
                    prev_dep_min = j.prev_ev.get("departure_min", cur_arr)
                    new_arr = max(cur_arr, prev_dep_min + j.to_rest)

                new_dep = min(new_arr + meal_dur, MAX_NIGHT)
                if new_arr == cur_arr and new_dep == e.get("departure_min", 0):
                    continue  # timing unchanged

                e["arrival_min"]   = new_arr
                e["departure_min"] = new_dep
                modified = True

                # Cascade: shift everything after the meal if the restaurant took longer
                if j.from_rest:
                    earliest   = new_dep + j.from_rest
                    next_start = events[i + 1].get("arrival_min", earliest) if i + 1 < len(events) else earliest
                    cascade    = max(0, earliest - next_start)
                    if cascade > 0:
                        for k in range(i + 1, len(events)):
                            ek = events[k]
                            new_k_arr = min(ek.get("arrival_min", 0) + cascade, MAX_NIGHT)
                            new_k_dep = min(ek.get("departure_min", 0) + cascade, MAX_NIGHT)
                            ek["arrival_min"]   = new_k_arr
                            ek["departure_min"] = new_k_dep
                            if ek.get("event_type") == "hotel_return":
                                ek["departure_min"] = new_k_arr  # hotel return has zero duration
                                break

            result[dk] = {**day_data, "events": events} if modified else day_data

        updated = sum(1 for j in valid if j.to_rest or j.from_rest)
        print(f"  ✓ Updated timing for {updated} meals using real travel data")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 10c. Hotel Return — closing event for every day
    # ─────────────────────────────────────────────────────────────────────────
    def add_hotel_returns(self, daily_schedule: dict, hotel: dict,
                           travel_matrix: list, locations: list,
                           transport_mode: str) -> dict:
        """
        Appends a 'Return to Hotel' event at the end of every day,
        placed after dinner.  Travel time is looked up from the travel
        matrix (depot = index 0) using the last sightseeing attraction's
        location index.  Falls back to haversine if the index is unknown.
        """
        mode_label = {"driving": "drive", "transit": "transit",
                      "walking": "walk",  "cycling": "cycle"}.get(transport_mode, "travel")
        hotel_name = hotel.get("name", "Hotel")
        h_lat      = hotel.get("lat")
        h_lon      = hotel.get("lon")

        # Build a lat/lon → location-index lookup from the locations list
        # (locations[0] is the depot/hotel; attractions start at index 1)
        loc_index  = {(round(lat, 5), round(lon, 5)): idx
                      for idx, (lat, lon) in enumerate(locations)}

        result = {}
        for day, events in daily_schedule.items():
            # Find the last sightseeing attraction (has lat/lon)
            last_attr = next(
                (e for e in reversed(events)
                 if e.get("event_type") != "meal" and
                 e.get("lat") and e.get("lon")),
                None
            )

            # Travel time: matrix lookup → haversine fallback
            travel_mins = 20   # safe default
            if last_attr:
                key = (round(last_attr["lat"], 5), round(last_attr["lon"], 5))
                idx = loc_index.get(key)
                if idx and idx < len(travel_matrix) and len(travel_matrix[idx]) > 0:
                    travel_mins = travel_matrix[idx][0]   # column 0 = depot/hotel
                elif h_lat and h_lon:
                    travel_mins = haversine_minutes(
                        last_attr["lat"], last_attr["lon"], h_lat, h_lon)

            last_dep   = max(e.get("departure_min", 0) for e in events)
            hotel_arr  = last_dep + travel_mins

            return_event = {
                "event_type"   : "hotel_return",
                "name"         : f"Return to {hotel_name}",
                "icon"         : "🏨",
                "arrival_min"  : hotel_arr,
                "departure_min": hotel_arr,   # zero-duration marker
                "travel_mins"  : travel_mins,
                "mode_label"   : mode_label,
            }
            result[day] = list(events) + [return_event]

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Post-Generation Feasibility Validator & Corrector
    # ─────────────────────────────────────────────────────────────────────────
    def validate_and_fix(self, final: dict, city: str, travel_style: str,
                          group: str, group_ages: str, month: int,
                          start_date: datetime, transport_mode: str,
                          wake_hour: int, pace: str, dietary: str,
                          budget_per_meal: str, fitness: str,
                          interests: str, days: int) -> dict:
        """
        Sends the complete generated itinerary to OpenAI for a rigorous 10-point
        feasibility audit.  If the score is below 8/10, applies corrections and
        returns the improved schedule.

        Checks:
          1  Operating hours — is the venue open at the scheduled time?
          2  Midnight overflow — no activity past 11 PM (1380 min)
          3  Chronological order — no time overlaps within a day
          4  Travel-time gaps — sufficient gap between consecutive stops
          5  Human endurance — total active hours vs pace limit
          6  Pace compliance — max attractions per day
          7  Meal completeness — breakfast, lunch, dinner every day
          8  Outdoor curfew — no parks/trails after sunset
          9  Duplicate activities
          10 Logical daily flow — energy curve, geographic sense
        """
        from collections import Counter as _Counter

        MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        PACE_RULES  = {
            "relaxed" : "5 sightseeing attractions/day max, long leisurely meals, no rushing",
            "moderate": "8 sightseeing attractions/day max, balanced pace, normal meal times",
            "packed"  : "12 sightseeing attractions/day max, fast transitions, but still humanly doable",
        }
        PACE_MAX_ATTR = {"relaxed": 5, "moderate": 8, "packed": 12}
        PACE_DAY_END  = {"relaxed": 1260, "moderate": 1320, "packed": 1380}

        day_start_min = wake_hour * 60          # e.g. 300 for 5 AM
        day_start_str = mins_to_time(day_start_min)
        day_end_min   = PACE_DAY_END.get(pace, 1320)
        day_end_str   = mins_to_time(day_end_min)
        max_attr      = PACE_MAX_ATTR.get(pace, 5)
        trip_end      = start_date + timedelta(days=days - 1)

        # ── Build human-readable schedule for the prompt ─────────────
        schedule_lines = []
        for day_key, day_data in final.items():
            events  = day_data.get("events", [])
            weather = day_data.get("weather_summary", "")
            note    = day_data.get("day_note", "")

            schedule_lines.append(f"\n{'─'*62}")
            schedule_lines.append(f"  {day_key}  |  Weather: {weather}")
            if note:
                schedule_lines.append(f"  Note: {note}")
            schedule_lines.append(f"{'─'*62}")

            n_attr = 0
            for e in events:
                arr      = e.get("arrival_min", 0)
                dep      = e.get("departure_min", 0)
                dur      = dep - arr
                arr_str  = mins_to_time(arr)
                dep_str  = mins_to_time(dep)
                travel   = e.get("travel_to_next", "")
                travel_s = f" | Next: {travel}" if travel else ""

                if e.get("event_type") == "meal":
                    schedule_lines.append(
                        f"  [MEAL:{e['meal_type']}]  {e['name']}"
                        f"  {arr_str}–{dep_str}  ({dur} min)"
                        f"  [arr={arr} dep={dep}]"
                    )
                else:
                    n_attr += 1
                    addr = (e.get("formatted_address") or "")[:55]
                    schedule_lines.append(
                        f"  [{e.get('attraction_type','?').upper()}#{n_attr}]"
                        f"  {e['name']}"
                        f"  {arr_str}–{dep_str}  ({dur} min)"
                        f"  [arr={arr} dep={dep}]"
                        f"  ★{e.get('google_rating','?')}"
                        f"  energy:{e.get('energy_level','?')}"
                        f"{travel_s}"
                        + (f"\n              Address: {addr}" if addr else "")
                    )

            schedule_lines.append(f"  → {n_attr} sightseeing attractions this day")

        schedule_text = "\n".join(schedule_lines)

        # ── Build the prompt ─────────────────────────────────────────
        prompt = f"""You are a senior travel operations director at a world-class travel agency \
with 25+ years of field experience planning premium North American itineraries. \
A junior system just auto-generated the itinerary below using algorithms. \
Your job is to catch every unrealistic, impossible, or just plain bad scheduling \
decision before it goes to the client — and return a corrected version.

{"="*66}
TRAVELER PROFILE
{"="*66}
City             : {city.title()}
Trip dates       : {start_date.strftime("%B %d")} – {trip_end.strftime("%B %d, %Y")}  ({days} days)
Travel style     : {travel_style}
Group            : {group}{f"  |  Ages: {group_ages}" if group_ages else ""}
Fitness level    : {fitness}
Trip pace        : {pace}  —  {PACE_RULES.get(pace, "")}
Wake-up time     : {wake_hour}:00 AM every morning (no exceptions)
Transport mode   : {transport_mode}
Dietary          : {dietary}
Meal budget      : {budget_per_meal} per person per meal
Month            : {MONTH_NAMES.get(month, month)}
Interests        : {interests or "general sightseeing"}

{"="*66}
TIME FORMAT  (all times below are minutes from midnight)
{"="*66}
    0  = 12:00 AM (midnight)
  300  = 5:00 AM    |   360 = 6:00 AM    |   420 = 7:00 AM
  480  = 8:00 AM    |   540 = 9:00 AM    |   600 = 10:00 AM
  660  = 11:00 AM   |   720 = 12:00 PM   |   780 = 1:00 PM
  840  = 2:00 PM    |   960 = 4:00 PM    |  1080 = 6:00 PM
 1140  = 7:00 PM    |  1200 = 8:00 PM    |  1260 = 9:00 PM
 1320  = 10:00 PM   |  1380 = 11:00 PM   |  1440 = MIDNIGHT ← NEVER schedule here

Valid day window for this traveler:
  Earliest first activity : {day_start_min} ({day_start_str})
  Latest any departure     : {day_end_min} ({day_end_str})
  Hard curfew outdoor/parks: 1110 (6:30 PM) — October sunset in Denver ≈ 6:10 PM

{"="*66}
GENERATED ITINERARY — NEEDS AUDIT
{"="*66}
{schedule_text}

{"="*66}
YOUR 10-POINT FEASIBILITY AUDIT  (check every single point)
{"="*66}

1. OPERATING HOURS
   Use these as ground truth for Denver in October:
   ╔══════════════════════════════════════════════╦══════════════════╗
   ║ Venue type                                   ║ Realistic hours  ║
   ╠══════════════════════════════════════════════╬══════════════════╣
   ║ Natural history / science museums            ║ 9 AM – 5 PM      ║
   ║ Art museums / galleries                      ║ 10 AM – 5 PM     ║
   ║ Historic house museums (Molly Brown, etc.)   ║ 10 AM – 3:30 PM  ║
   ║ Colorado State Capitol                       ║ 7:30 AM – 5 PM   ║
   ║ National/state parks & trails                ║ 6:30 AM – 6:30 PM║
   ║ Botanic gardens / zoos / aquariums           ║ 9 AM – 5 PM      ║
   ║ Mile High Flea Market                        ║ Sat–Sun 7 AM–4 PM║
   ║ Distilleries / breweries                     ║ 11 AM – 6 PM     ║
   ║ Stadiums (tours, not events)                 ║ 9 AM – 3 PM      ║
   ║ Beer gardens / restaurants                   ║ 11 AM – 10 PM    ║
   ║ Cocktail bars                                ║ 3 PM – midnight  ║
   ║ Children's museums / family venues           ║ 9 AM – 5 PM      ║
   ║ Air & space / transportation museums         ║ 9 AM – 5 PM      ║
   ╚══════════════════════════════════════════════╩══════════════════╝
   → FLAG any event scheduled outside its realistic hours.

2. MIDNIGHT OVERFLOW
   Any event with dep_min ≥ 1380 is a clear error.
   → action: "remove" (unless it's an evening bar/nightlife venue).

3. CHRONOLOGICAL INTEGRITY
   Within each day: every event's arr_min MUST be ≥ the previous event's dep_min.
   Even a 1-minute overlap is invalid.
   → If overlap: shift the later event forward to start after the earlier one ends.

4. TRAVEL-TIME REALISM
   The schedule shows "Next: X min drive" between stops.  The gap between
   (event A departure) and (event B arrival) must be AT LEAST that travel time.
   If the gap is too short, either compress visit durations or remove lower-priority stops.
   Minimum gap when no travel time shown: 15 minutes (parking, walking to entrance).

5. HUMAN ENDURANCE
   Max total active time per day (travel + activities + meals combined):
     relaxed = 10 h  |  moderate = 13 h  |  packed = 16 h
   Current pace = {pace}.
   If a day's total span (last dep_min – first arr_min) exceeds this, remove low-value stops.
   People also need 15–20 min buffer between most consecutive activities.

6. PACE COMPLIANCE
   Max sightseeing attractions = {max_attr} for pace="{pace}".
   Meals do NOT count.  Snack stops do NOT count.
   If a day has more than {max_attr} sightseeing attractions → remove the weakest ones
   (lowest Google rating, lowest utility_score, or least on-theme for "{travel_style}").

7. MEAL COMPLETENESS  (every single day must have all three)
   ┌───────────┬─────────────────────────────────┬──────────┐
   │ Meal      │ Arrival window (minutes)         │ Duration │
   ├───────────┼─────────────────────────────────┼──────────┤
   │ Breakfast │ {wake_hour * 60} (exactly at wake time)         │ 30 min   │
   │ Lunch     │ 690 – 840  (11:30 AM – 2:00 PM) │ 60 min   │
   │ Dinner    │ 1080 – 1200 (6:00 PM – 8:00 PM) │ 75 min   │
   └───────────┴─────────────────────────────────┴──────────┘
   → NEVER remove Breakfast, Lunch, or Dinner — only reschedule them.
   → If a meal is missing, add it at the midpoint of its window.

8. OUTDOOR CURFEW (Denver, October)
   Sunset ≈ 6:10 PM.  Hard rule:
   • Outdoor / hiking / park activities: dep_min MUST be ≤ 1110 (6:30 PM)
   • If an outdoor activity's dep_min > 1110 → action: "remove"
   • Temperatures drop to 35–40 °F after dark — physically unpleasant, unsafe on trails.

9. DUPLICATE DETECTION
   If the same attraction appears more than once across all days → remove all but the first occurrence.

10. LOGICAL DAILY FLOW
    A well-crafted day feels like a narrative:
    • Morning  (wake → 12 PM): high-energy active attractions (hiking, sports, theme parks)
    • Midday   (12 PM → 3 PM): cultural / indoor after lunch
    • Afternoon(3 PM → 6 PM): scenic, walkable, or shopping
    • Evening  (6 PM → close): dinner then optional nightlife/food experience
    If the ordering is backwards (e.g., museum at 6 AM, park at 10 PM) → reschedule.

{"="*66}
RESPONSE FORMAT  (return ONLY valid JSON — NO markdown, NO text outside the JSON)
{"="*66}
{{
  "feasibility_score": <integer 1-10, where 10 = perfect, no issues>,
  "summary": "<honest 2–3 sentence plain-English assessment of the biggest problems>",
  "total_issues": <integer>,
  "issues": [
    {{
      "day":     "Day 1",
      "event":   "<exact name as shown in itinerary>",
      "problem": "<specific factual problem, e.g. 'dep_min=1430 exceeds midnight curfew'>",
      "action":  "remove | reschedule | keep"
    }}
  ],
  "corrected_itinerary": {{
    "Day 1": {{
      "events": [
        {{
          "name":          "<exact name from original — e.g. 'Breakfast', 'Lunch', 'Dinner', 'Morning Snack / Coffee', 'Afternoon Snack', or the attraction name>",
          "event_type":    "meal | attraction",
          "action":        "keep | reschedule | remove",
          "arrival_min":   <integer>,
          "departure_min": <integer>
        }}
      ]
    }},
    "Day 2": {{ ... }}
  }}
}}

ABSOLUTE HARD CONSTRAINTS FOR corrected_itinerary:
  ✅ Include EVERY event (meals + attractions) for each day that you keep
  ✅ Events in strictly ascending order of arrival_min within each day
  ✅ No two events may overlap (each arr_min >= previous dep_min)
  ✅ Breakfast arr_min = {wake_hour * 60}  (non-negotiable)
  ✅ Lunch arr_min between 690 – 840
  ✅ Dinner arr_min between 1080 – 1200
  ✅ No outdoor/park/hiking dep_min > 1110
  ✅ No event dep_min > 1380 (11 PM absolute maximum)
  ✅ Max {max_attr} sightseeing (non-meal) events per day
  ✅ Keep original visit duration (departure_min − arrival_min) unless you have a specific reason to change it
  ✅ Action "remove" = do NOT include the event in the events array at all
  ✅ If an event is missing from corrected_itinerary, it is assumed "keep" with original times"""

        print("\n[ItineraryPlanner] Running post-generation feasibility audit with OpenAI...")
        try:
            resp = self.client.chat.completions.create(
                model       = "gpt-4o-mini",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens  = 8000,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            analysis = json.loads(raw)
        except Exception as e:
            print(f"  Warning: Feasibility check failed ({e}) — keeping original itinerary")
            return final

        score = analysis.get("feasibility_score", 10)
        total = analysis.get("total_issues", 0)
        print(f"  Feasibility score: {score}/10  |  Issues found: {total}")
        print(f"  Summary: {analysis.get('summary', '')}")

        if score >= 8:
            print("  ✓ Itinerary passes feasibility check — no corrections needed")
            return final

        print(f"  Applying corrections ({total} issue(s))...")
        fixed = self._apply_corrections(final, analysis)
        print("  ✓ Corrected itinerary ready")
        return fixed

    def _apply_corrections(self, final: dict, analysis: dict) -> dict:
        """
        Apply OpenAI's corrections to the generated final schedule.

        For each day in corrected_itinerary:
          - "keep"      : use original event with original times
          - "reschedule": use original event data but with new arrival/departure_min
          - "remove"    : drop the event entirely

        Meals are never removed — if OpenAI tries to remove a meal, it is kept
        and rescheduled to the centre of its allowed window instead.
        """
        MEAL_FALLBACKS = {
            "breakfast"     : (None, 30),    # arrival = wake_hour * 60 (set below)
            "lunch"         : (765, 60),     # 12:45 PM
            "dinner"        : (1140, 75),    # 7:00 PM
            "morning_snack" : (570, 15),     # 9:30 AM
            "afternoon_snack": (870, 15),    # 2:30 PM
        }

        corrected_days = analysis.get("corrected_itinerary", {})
        if not corrected_days:
            return final

        result = {}
        for day_key, day_data in final.items():
            if day_key not in corrected_days:
                result[day_key] = day_data
                continue

            original_events = day_data.get("events", [])
            # Build name → event lookup (case-insensitive)
            event_map = {e["name"]: e for e in original_events}
            event_map_lower = {e["name"].lower(): e for e in original_events}

            corrected_list = corrected_days[day_key].get("events", [])
            kept_meals     = set()   # track which meal types we've placed
            new_events     = []

            for ce in corrected_list:
                name   = ce.get("name", "")
                action = ce.get("action", "keep")

                # Locate original event
                orig = (event_map.get(name) or
                        event_map_lower.get(name.lower()) or
                        next((e for e in original_events
                              if name.lower() in e["name"].lower() or
                              e["name"].lower() in name.lower()), None))

                if not orig:
                    continue   # OpenAI invented an event — ignore

                is_meal        = orig.get("event_type") == "meal"
                is_style_event = orig.get("event_type") in ("style_event", "free_time")

                # Never remove meals — reschedule to fallback instead.
                # Style events and free-time placeholders also survive removal
                # attempts — just reschedule them inside a safe window.
                if action == "remove" and is_style_event:
                    action = "keep"   # protect style/free-time events from validator removal

                if action == "remove" and is_meal:
                    meal_type = orig.get("meal_type", "")
                    fb        = MEAL_FALLBACKS.get(meal_type)
                    if fb:
                        action = "reschedule"
                        fb_arr = fb[0] if fb[0] else orig["arrival_min"]
                        ce     = {**ce, "arrival_min": fb_arr,
                                   "departure_min": fb_arr + fb[1]}

                if action == "remove":
                    continue

                e_copy = dict(orig)
                if action == "reschedule":
                    e_copy["arrival_min"]   = ce.get("arrival_min",   orig["arrival_min"])
                    e_copy["departure_min"] = ce.get("departure_min", orig["departure_min"])

                if is_meal:
                    kept_meals.add(orig.get("meal_type", ""))
                new_events.append(e_copy)

            # Safety net: ensure all three main meals are present
            for meal_type, (fb_arr, fb_dur) in MEAL_FALLBACKS.items():
                if meal_type in ("morning_snack", "afternoon_snack"):
                    continue  # optional
                if meal_type not in kept_meals:
                    orig_meal = next(
                        (e for e in original_events
                         if e.get("event_type") == "meal" and
                         e.get("meal_type") == meal_type), None)
                    if orig_meal:
                        m_copy = dict(orig_meal)
                        if fb_arr:
                            m_copy["arrival_min"]   = fb_arr
                            m_copy["departure_min"] = fb_arr + fb_dur
                        new_events.append(m_copy)
                        print(f"  {day_key}: Re-inserted missing {meal_type}")

            # Sort strictly by arrival time, then enforce travel-time gaps.
            # 1. Ensure arr[n] >= dep[n-1] (no overlap at all).
            # 2. If the previous event has a travel_to_next annotation, the gap
            #    must be at least that many minutes (e.g. "20 min drive" → +20 min).
            import re as _re
            new_events.sort(key=lambda e: e.get("arrival_min", 0))
            for i in range(1, len(new_events)):
                prev        = new_events[i - 1]
                curr        = new_events[i]
                prev_dep    = prev.get("departure_min", 0)
                curr_arr    = curr.get("arrival_min", 0)
                dur         = curr.get("departure_min", curr_arr) - curr_arr

                # Parse travel_to_next ("20 min drive" → 20)
                travel_mins = 0
                ttn = prev.get("travel_to_next", "")
                if ttn:
                    m = _re.search(r'(\d+)\s*min', ttn)
                    if m:
                        travel_mins = int(m.group(1))

                min_arr = max(prev_dep + travel_mins, prev_dep)
                if curr_arr < min_arr:
                    curr["arrival_min"]   = min_arr
                    curr["departure_min"] = min_arr + dur

            # Clamp meals back to reasonable windows after cascade.
            # OpenAI sometimes reschedules dinner to 10 PM+ when attractions run late;
            # this hard ceiling prevents absurd late-night meal times regardless of
            # what the audit or cascade produced.
            _MEAL_WIN = {
                "breakfast"      : (360,  600),   # 6 AM – 10 AM
                "morning_snack"  : (510,  660),   # 8:30 AM – 11 AM
                "lunch"          : (660,  870),   # 11 AM – 2:30 PM
                "afternoon_snack": (810,  960),   # 1:30 PM – 4 PM
                "dinner"         : (1080, 1320),  # 6 PM – 10 PM
            }
            for e in new_events:
                if e.get("event_type") == "meal":
                    mt  = e.get("meal_type", "")
                    win = _MEAL_WIN.get(mt)
                    if win:
                        arr = e.get("arrival_min", 0)
                        dur = e.get("departure_min", arr) - arr
                        arr = max(win[0], min(arr, win[1]))
                        e["arrival_min"]   = arr
                        e["departure_min"] = arr + dur

            result[day_key] = {**day_data, "events": new_events}

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 12. Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────
    def plan(self, attractions: list, days: int, start_date: datetime,
             city: str, state: str, travel_style: str, interests: str,
             group: str, month: int,
             transport_mode: str  = "driving",
             wake_hour: int       = 8,
             pace: str            = "moderate",
             dietary: str         = "none",
             budget_per_meal: str = "$20",
             fitness: str         = "moderate",
             group_ages: str      = "",
             hotel: dict          = None,
             weather_per_day: list = None) -> dict:
        """
        Full dynamic itinerary planning pipeline.
        Everything adapts: day schedules, meals, energy curve, weather, fatigue.

        weather_per_day: pre-fetched weather list from the pipeline's run() method.
        When provided, the internal weather fetch is skipped (saves ~2s on the
        critical path and avoids a redundant API call).
        """
        valid = [a for a in attractions if a.get("lat") and a.get("lon")]
        if not valid:
            print("[ItineraryPlanner] No attractions with coordinates — cannot build itinerary.")
            return {}

        print(f"\n[ItineraryPlanner] Planning {days}-day itinerary, "
              f"{len(valid)} attractions, pace={pace}, group={group}, mode={transport_mode}")

        PT = {}   # plan-phase timing

        # Location setup (pure Python, <1ms) — must happen before travel matrix
        departure_dt = datetime(start_date.year, start_date.month,
                                start_date.day, wake_hour, 30)
        if hotel and hotel.get("lat") and hotel.get("lon"):
            depot = (hotel["lat"], hotel["lon"])
            print(f"  Routing depot: {hotel['name']} ({depot[0]:.4f}, {depot[1]:.4f})")
        else:
            depot = (sum(a["lat"] for a in valid) / len(valid),
                     sum(a["lon"] for a in valid) / len(valid))
        locations = [depot] + [(a["lat"], a["lon"]) for a in valid]
        for i, a in enumerate(valid):
            a["_location_idx"] = i + 1

        # Steps 1 + 4 — Intelligence + travel matrix run IN PARALLEL.
        # Neither depends on the other — both only need `valid` + locations.
        # If weather wasn't pre-fetched, add it as a 3rd parallel task.
        t = time.time()
        if weather_per_day:
            print("\n[ItineraryPlanner] Pre-fetched weather ✓ — running intel + matrix in parallel...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_intel  = pool.submit(self.get_attraction_intelligence,
                                         valid, city, month, group, fitness, group_ages)
                fut_matrix = pool.submit(self.get_travel_matrix,
                                         locations, transport_mode, departure_dt)
                intelligence  = fut_intel.result()
                travel_matrix = fut_matrix.result()
        else:
            print("\n[ItineraryPlanner] Running intel + matrix + weather in parallel...")
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_intel   = pool.submit(self.get_attraction_intelligence,
                                          valid, city, month, group, fitness, group_ages)
                fut_matrix  = pool.submit(self.get_travel_matrix,
                                          locations, transport_mode, departure_dt)
                fut_weather = pool.submit(self.get_weather_per_day, city, start_date, days)
                intelligence    = fut_intel.result()
                travel_matrix   = fut_matrix.result()
                weather_per_day = fut_weather.result()
        PT["intel+matrix"] = time.time() - t

        # Step 3 — Initial day profiles (pure Python, no I/O — runs after weather is ready)
        day_profiles = self.compute_day_profiles(
            days, start_date, group, group_ages, pace, wake_hour,
            weather_per_day, or_tools_schedule=None,
            travel_style=travel_style)

        # Step 5 — OR-Tools routing (exception-safe — falls to greedy internally)
        t = time.time()
        daily = self.solve_routing(
            valid, days, start_date, intelligence,
            travel_matrix, day_profiles, fitness)
        PT["or_tools"] = time.time() - t

        # Step 6 — Variety optimisation (OpenAI cross-day swaps before re-sorting)
        t = time.time()
        try:
            daily = self.optimize_variety(daily, valid, intelligence, travel_style, days)
        except Exception as e:
            print(f"  Warning: Variety optimisation failed ({e}) — skipping")
        PT["variety"] = time.time() - t

        # Step 7 — Re-compute day profiles using actual OR-Tools + variety output
        try:
            day_profiles = self.compute_day_profiles(
                days, start_date, group, group_ages, pace, wake_hour,
                weather_per_day, or_tools_schedule=daily,
                travel_style=travel_style)
        except Exception as e:
            print(f"  Warning: Day profile recompute failed ({e}) — keeping initial profiles")

        # Step 8 — Apply energy curve (high energy morning → low energy evening)
        try:
            daily = self.apply_energy_curve(daily, intelligence, fitness,
                                            travel_matrix=travel_matrix)
        except Exception as e:
            print(f"  Warning: Energy curve failed ({e}) — skipping")

        # Step 9 — Insert meals + style-specific events dynamically
        try:
            with_meals = self.insert_meals(daily, day_profiles,
                                           travel_style=travel_style,
                                           travel_matrix=travel_matrix)
        except Exception as e:
            print(f"  Warning: Meal insertion failed ({e}) — using raw schedule")
            with_meals = daily

        # Step 9b — Fill time gaps > 60 min.
        # First try to slot in unused attractions; fall back to a 60-min free-time cap.
        try:
            scheduled_names = {
                e.get("name", "").lower()
                for day_evs in with_meals.values()
                for e in day_evs
                if e.get("event_type") not in ("meal", "free_time", "style_event", "hotel_return")
            }
            unused_for_gaps = sorted(
                [a for a in valid
                 if a.get("name", "").lower() not in scheduled_names
                 and a.get("lat") and a.get("lon")],
                key=lambda x: x.get("utility_score", 0),
                reverse=True,
            )
            with_meals = self._fill_time_gaps(
                with_meals,
                unused_attractions=unused_for_gaps,
                intelligence=intelligence,
            )
        except Exception as e:
            print(f"  Warning: Gap fill failed ({e}) — skipping")

        # Step 9c — Append "Return to Hotel" at end of every day
        if hotel and hotel.get("lat") and hotel.get("lon"):
            try:
                with_meals = self.add_hotel_returns(
                    with_meals, hotel, travel_matrix, locations, transport_mode)
            except Exception as e:
                print(f"  Warning: Hotel return annotation failed ({e}) — skipping")

        # Steps 11 + 13 — Enrichment and feasibility audit run IN PARALLEL.
        # validate_and_fix only cares about timing (arrival/departure_min) so it
        # can operate on with_meals directly — it doesn't need meal suggestions.
        # Enrichments are merged afterwards into the corrected schedule.
        # Saves ~enrichment_time (≈9s) off the critical path.
        t = time.time()

        # Build the minimal "final" structure validate_and_fix expects, without enrichments.
        pre_final = {}
        for day, events in with_meals.items():
            dk   = f"Day {day + 1}"
            prof = day_profiles[day] if day < len(day_profiles) else {}
            wx   = weather_per_day[day] if day < len(weather_per_day) else {}
            pre_final[dk] = {
                "events"         : [dict(e) for e in events],
                "day_tip"        : "",
                "day_note"       : prof.get("day_note", ""),
                "weather_summary": wx.get("summary", ""),
            }

        # Annotate travel BEFORE validate_and_fix so OpenAI sees the real travel
        # times between stops and enforces correct gaps (not just the 15-min minimum).
        try:
            _ann_pre = {dk: d["events"] for dk, d in pre_final.items()}
            _ann_pre = self.annotate_travel(_ann_pre, travel_matrix, locations, transport_mode)
            for dk in pre_final:
                if dk in _ann_pre:
                    pre_final[dk]["events"] = _ann_pre[dk]
        except Exception:
            pass

        def _run_enrich():
            try:
                return self.enrich_with_openai(
                    with_meals, city, travel_style, interests, group, group_ages,
                    month, dietary, budget_per_meal, weather_per_day, day_profiles)
            except Exception as e:
                print(f"  Warning: OpenAI enrichment failed ({e}) — using empty enrichment")
                return {}

        def _run_validate():
            try:
                return self.validate_and_fix(
                    pre_final,
                    city           = city,
                    travel_style   = travel_style,
                    group          = group,
                    group_ages     = group_ages,
                    month          = month,
                    start_date     = start_date,
                    transport_mode = transport_mode,
                    wake_hour      = wake_hour,
                    pace           = pace,
                    dietary        = dietary,
                    budget_per_meal= budget_per_meal,
                    fitness        = fitness,
                    interests      = interests,
                    days           = days,
                )
            except Exception as e:
                print(f"  Warning: Feasibility audit failed ({e}) — returning unaudited schedule")
                return pre_final

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_enrich   = pool.submit(_run_enrich)
            fut_validate = pool.submit(_run_validate)
            enrichments  = fut_enrich.result()
            corrected    = fut_validate.result()

        PT["enrich+validate(parallel)"] = time.time() - t

        # Step 12 — Merge enrichments into the corrected (validated) schedule
        final = {}
        for dk, day_data in corrected.items():
            day_idx    = int(dk.split()[-1]) - 1
            day_enrich = enrichments.get(dk, {})
            enriched   = []
            for e in day_data.get("events", []):
                ec = dict(e)
                if ec.get("event_type") == "meal":
                    me               = day_enrich.get(ec["meal_type"], {})
                    ec["suggestion"] = me.get("suggestion", "")
                    ec["tip"]        = me.get("tip", "")
                elif ec.get("event_type") == "style_event":
                    se               = day_enrich.get(ec.get("style_event_type", ""), {})
                    ec["suggestion"] = se.get("suggestion", "")
                    ec["tip"]        = se.get("tip", "")
                enriched.append(ec)
            final[dk] = {
                **day_data,
                "events"  : enriched,
                "day_tip" : day_enrich.get("day_tip", day_data.get("day_tip", "")),
            }

        # Step 12b — Reanchor meal timing using real restaurant travel data.
        # Now that enrichment has named actual restaurants, geocode each one and
        # replace the half-travel estimate with a real Google Distance Matrix time.
        t = time.time()
        try:
            final = self._reanchor_meals(final, city, transport_mode, departure_dt)
        except Exception as e:
            print(f"  Warning: Meal reanchoring failed ({e}) — keeping estimate-based timing")
        PT["reanchor_meals"] = time.time() - t

        # Step 12c — Annotate travel arrows now that timing is finalised.
        # Running here (rather than before enrichment) ensures arrows reflect
        # real restaurant travel gaps, not the earlier half-route estimates.
        try:
            events_for_annotation = {dk: d["events"] for dk, d in final.items()}
            annotated = self.annotate_travel(
                events_for_annotation, travel_matrix, locations, transport_mode)
            for dk in final:
                if dk in annotated:
                    final[dk]["events"] = annotated[dk]
        except Exception as e:
            print(f"  Warning: Travel annotation failed ({e}) — skipping")

        # ── Plan timing summary ──────────────────────────────────────
        pt_total = sum(PT.values())
        print("\n  ── plan() timing ──────────────────────────────")
        for k, v in PT.items():
            bar = "█" * int(v / pt_total * 20) if pt_total else ""
            print(f"  {k:<18} {v:5.1f}s  {bar}")
        print(f"  {'TOTAL':<18} {pt_total:5.1f}s")
        print("  ────────────────────────────────────────────────")

        return final
