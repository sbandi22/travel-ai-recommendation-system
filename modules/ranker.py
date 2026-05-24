"""
TravelIQ — LLM Ranker Module
Uses OpenAI to rank attractions based on user travel style and all model signals.

Usage:
    from modules.ranker import LLMRanker
    ranker = LLMRanker(api_key="your_key")
    ranked = ranker.rank(results, city="Phoenix", month=11, travel_style="relaxation", year=2026)
"""

import json
from openai import OpenAI


class LLMRanker:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def rank(self, results: list, city: str, month: int,
         travel_style: str, year: int,
         interests: str = "", group: str = "solo", days: int = 3) -> list:
        """
        Rank attractions using OpenAI based on all model signals.

        Args:
            results      : list of attraction dicts from pipeline.process_attractions()
            city         : city name
            month        : travel month (1-12)
            travel_style : user's travel style
            year         : travel year

        Returns:
            ranked list of attraction dicts (highest utility first)
            each dict has an added 'utility_score' and 'ranking_reason'
        """
        month_names = {1:"January",2:"February",3:"March",4:"April",
                       5:"May",6:"June",7:"July",8:"August",9:"September",
                       10:"October",11:"November",12:"December"}

        attraction_lines = []
        for i, r in enumerate(results):
            attraction_lines.append(
                f"{i+1}. {r['name']} | "
                f"type: {r['attraction_type']} | "
                f"google_rating: {r['google_rating']} | "
                f"sentiment: {r['sentiment_score']} | "
                f"weather_suitability: {r['weather_score']} | "
                f"crowd_index: {r['crowd_index']} (lower = less crowded)"
            )
        attraction_text = "\n".join(attraction_lines)

        # Build explicit exclusion list from interests + style
        combined = f"{travel_style} {interests}".lower()
        exclude_wellness  = not any(w in combined for w in ["wellness","relaxation","spa","yoga"])
        exclude_nightlife = not any(w in combined for w in ["nightlife","bar","club","party"])
        exclude_shopping  = not any(w in combined for w in ["shopping","mall","luxury","budget"])

        exclusion_rules = []
        if exclude_wellness:
            exclusion_rules.append("- Spas, wellness centers, yoga studios, massage parlors → RANK LAST (user did not ask for wellness)")
        if exclude_nightlife:
            exclusion_rules.append("- Bars, nightclubs, adult entertainment → RANK LAST (user did not ask for nightlife)")
        if exclude_shopping:
            exclusion_rules.append("- Shopping malls, outlet stores → RANK LAST (user did not ask for shopping)")
        exclusion_block = "\n".join(exclusion_rules) if exclusion_rules else "None"

        prompt = f"""You are the head curator at a world-class travel agency. A tourist is spending \
hundreds of dollars on this trip and expects an unforgettable, expertly planned experience. \
Your job is to rank these attractions so the BEST, most relevant ones appear first — \
because the itinerary planner only uses the top-ranked attractions.

═══════════════════════════════════════════════════
TRAVELER PROFILE
═══════════════════════════════════════════════════
City          : {city.title()}
Month         : {month_names[month]} {year}
Travel styles : {travel_style}
Interests     : {interests if interests else "general sightseeing — prioritise most famous / iconic spots"}
Trip duration : {days} days
Group         : {group}

═══════════════════════════════════════════════════
MANDATORY RANKING RULES (apply in this exact order)
═══════════════════════════════════════════════════

RULE 1 — ICONIC CITY LANDMARKS ALWAYS RANK IN THE TOP 10
  World-famous attractions that every tourist visiting {city.title()} would expect to see MUST
  appear in the top 10 regardless of travel style. A tourist visiting New York always expects
  to see Times Square, Statue of Liberty, Central Park, Brooklyn Bridge, Empire State Building.
  A tourist in San Francisco expects the Golden Gate Bridge, Alcatraz, Fisherman's Wharf.
  If you recognise a landmark as globally or nationally famous for {city.title()}, it goes near the top.

RULE 2 — USER INTERESTS ARE NON-NEGOTIABLE (override style when in conflict)
  Stated interests: "{interests if interests else 'none — use famous landmarks + top-rated'}"
  Every attraction that directly matches an interest goes to the TOP.
  Every attraction with NO connection to any stated interest gets demoted.

RULE 3 — TRAVEL STYLE REFINES THE RANKING (does not override Rules 1 & 2)
  Use the style guidance below to break ties between equally-relevant attractions.

RULE 4 — MANDATORY EXCLUSIONS (these categories go to the VERY BOTTOM):
{exclusion_block}

RULE 5 — QUALITY SIGNAL
  Between two equally-relevant attractions, prefer:
  - Higher google_rating (a 4.8★ museum beats a 3.9★ museum of the same type)
  - More user_ratings (50,000 reviews is more reliable than 40 reviews)
  - Higher sentiment score

═══════════════════════════════════════════════════
STYLE GUIDANCE (use as tiebreaker, never to demote iconic landmarks)
═══════════════════════════════════════════════════

  adventure    : Prefer → hiking, kayaking, rock climbing, active outdoor parks
  cultural     : Prefer → museums, galleries, historic landmarks, heritage districts
  relaxation   : Prefer → peaceful gardens, scenic viewpoints, quiet nature
  foodie       : Prefer → food markets, renowned restaurants, culinary landmarks
  family       : Prefer → theme parks, zoos, aquariums, interactive museums
  luxury       : Prefer → high-rated exclusive experiences, iconic upscale venues
  budget       : Prefer → free parks, public landmarks, free-entry museums
  nightlife    : Prefer → cocktail bars, rooftop bars, jazz clubs, live music
  wellness     : Prefer → spas, botanical gardens, peaceful nature
  sports       : Prefer → stadiums, arenas, sports museums
  romantic     : Prefer → sunset viewpoints, rooftop restaurants, waterfront parks
  eco          : Prefer → nature reserves, wildlife sanctuaries, botanical gardens
  photography  : Prefer → iconic skylines, unique architecture, murals, viewpoints, colorful markets
  history      : Prefer → battlefields, historic districts, monuments, heritage museums
  art          : Prefer → art galleries, street art, sculpture gardens, creative districts
  popular      : Prefer → highest-rated, most-reviewed, most-visited attractions in {city.title()}

═══════════════════════════════════════════════════
ATTRACTIONS TO RANK
═══════════════════════════════════════════════════
{attraction_text}

Respond ONLY with a valid JSON array of EXACTLY {len(results)} objects — one per attraction, no more, no fewer:
[
  {{"name": "attraction name", "reason": "one sentence: why this ranks here for THIS traveler"}},
  ...
]
Every attraction from the list above must appear exactly once. Do NOT invent names. No markdown, no extra text."""

        print(f"\n[LLMRanker] Ranking {len(results)} attractions for '{travel_style}' traveler...")

        response = self.client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
        )

        raw = response.choices[0].message.content.strip()

        try:
            ranked_list = json.loads(raw)
            if not isinstance(ranked_list, list):
                raise ValueError("Response is not a JSON array")
        except (json.JSONDecodeError, ValueError):
            print("[LLMRanker] Warning: Could not parse response — returning original order")
            for i, r in enumerate(results):
                r["utility_score"]  = round(1.0 - i / len(results), 4)
                r["ranking_reason"] = "N/A"
            return results

        if len(ranked_list) != len(results):
            print(f"[LLMRanker] Warning: LLM returned {len(ranked_list)} items for {len(results)} inputs — truncating/padding")
            # Truncate hallucinated extras; missing items are handled by the append-missed loop below
            ranked_list = ranked_list[:len(results)]

        # Map ranked names back to full result dicts
        results_map = {r["name"]: r for r in results}
        ranked = []
        for i, item in enumerate(ranked_list):
            name = item.get("name", "")
            if name in results_map:
                r_copy = dict(results_map[name])
                r_copy["utility_score"]  = round(1.0 - i / len(ranked_list), 4)
                r_copy["ranking_reason"] = item.get("reason", "")
                ranked.append(r_copy)

        # Append any attractions the LLM missed
        ranked_names = {r["name"] for r in ranked}
        for r in results:
            if r["name"] not in ranked_names:
                r_copy = dict(r)
                r_copy["utility_score"]  = 0.0
                r_copy["ranking_reason"] = "Not ranked by LLM"
                ranked.append(r_copy)

        print(f"[LLMRanker] Ranked {len(ranked)} attractions successfully")
        return ranked