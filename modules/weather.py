"""
TravelIQ — Weather Suitability Module
Fetches historical climate data from Open-Meteo and computes
attraction-type suitability scores (0-1).

Usage:
    from modules.weather import WeatherModel
    wm = WeatherModel()
    score = wm.get_suitability(city="Miami", month=7, attraction_type="beach")
    print(score)  # e.g. 0.91
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# ── Attraction type categories ────────────────────────────────────────
ATTRACTION_TYPES = ["outdoor", "indoor", "beach", "hiking", "theme_park", "cultural"]


class WeatherModel:
    def __init__(self, openai_api_key: str = None):
        self._cache         = {}   # cache climate data so we don't re-fetch same city twice
        self._weights_cache = {}   # cache dynamic weights keyed by (travel_style, month, group)
        self._openai_key    = openai_api_key
        print("[WeatherModel] Ready.")

    # ── Geocoding ────────────────────────────────────────────────────
    def get_coordinates(self, city_name: str) -> tuple:
        url    = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
        r      = requests.get(url, params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            raise ValueError(f"City '{city_name}' not found.")
        res = results[0]
        return res["latitude"], res["longitude"]

    # ── Historical fetch ─────────────────────────────────────────────
    def fetch_historical_weather(self, lat: float, lon: float, years: int = 5) -> pd.DataFrame:
        end_date   = datetime.today() - timedelta(days=7)
        start_date = end_date - relativedelta(years=years)

        params = {
            "latitude"           : lat,
            "longitude"          : lon,
            "start_date"         : start_date.strftime("%Y-%m-%d"),
            "end_date"           : end_date.strftime("%Y-%m-%d"),
            "daily"              : [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "rain_sum", "windspeed_10m_max", "sunshine_duration",
            ],
            "timezone"           : "auto",
            "temperature_unit"   : "fahrenheit",
            "windspeed_unit"     : "mph",
            "precipitation_unit" : "inch",
        }
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
        r.raise_for_status()
        daily = r.json()["daily"]

        df = pd.DataFrame({
            "date"             : pd.to_datetime(daily["time"]),
            "temp_max"         : daily["temperature_2m_max"],
            "temp_min"         : daily["temperature_2m_min"],
            "temp_mean"        : daily["temperature_2m_mean"],
            "precipitation"    : daily["precipitation_sum"],
            "rain_sum"         : daily["rain_sum"],
            "windspeed_max"    : daily["windspeed_10m_max"],
            "sunshine_duration": daily["sunshine_duration"],
        })
        df["month"]           = df["date"].dt.month
        df["sunshine_hours"]  = df["sunshine_duration"] / 3600
        df["is_rainy"]        = (df["precipitation"] > 0.1).astype(int)
        df["is_extreme_heat"] = (df["temp_max"] > 95).astype(int)
        df["is_extreme_cold"] = (df["temp_min"] < 20).astype(int)
        return df

    def get_monthly_climate(self, df: pd.DataFrame) -> pd.DataFrame:
        monthly = df.groupby("month").agg(
            avg_temp_max      = ("temp_max",        "mean"),
            avg_temp_min      = ("temp_min",        "mean"),
            avg_temp_mean     = ("temp_mean",       "mean"),
            avg_precipitation = ("precipitation",   "mean"),
            avg_sunshine_hrs  = ("sunshine_hours",  "mean"),
            avg_windspeed     = ("windspeed_max",   "mean"),
            rain_days_pct     = ("is_rainy",        "mean"),
            extreme_heat_pct  = ("is_extreme_heat", "mean"),
            extreme_cold_pct  = ("is_extreme_cold", "mean"),
        ).reset_index()
        return monthly

    def _get_climate(self, city: str) -> pd.DataFrame:
        """Fetch + cache monthly climate for a city."""
        key = city.lower().strip()
        if key not in self._cache:
            lat, lon = self.get_coordinates(city)
            daily    = self.fetch_historical_weather(lat, lon)
            monthly  = self.get_monthly_climate(daily)
            self._cache[key] = monthly
            print(f"[WeatherModel] Cached climate data for {city}")
        return self._cache[key]

    # ── Dynamic Weight Generation (OpenAI) ──────────────────────────
    def get_dynamic_weights(self, travel_style: str, month: int,
                             group: str = "solo", interests: str = "") -> dict:
        """
        Uses OpenAI to generate personalised weather-scoring weights for each
        attraction type based on the user's travel profile.

        Returns a dict:
          {
            "outdoor":    {"temp": 0.35, "rain": 0.30, "sun": 0.25, "wind": 0.10},
            "beach":      { ... },
            ...
          }

        Weights sum to 1.0 per attraction type.
        Results are cached by (travel_style, month, group) so OpenAI is called
        at most once per pipeline run.
        """
        if not self._openai_key:
            return {}   # no key → caller falls back to hardcoded weights

        cache_key = (travel_style.lower(), month, group.lower())
        if cache_key in self._weights_cache:
            return self._weights_cache[cache_key]

        from openai import OpenAI
        client = OpenAI(api_key=self._openai_key)

        month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

        prompt = f"""You are a travel weather analyst. For a traveler with this profile:
  Travel style : {travel_style}
  Month        : {month_names.get(month, month)}
  Group        : {group}
  Interests    : {interests or 'not specified'}

Generate weather-scoring weights for each attraction type. Weights control how much
each weather factor matters when evaluating if conditions are suitable.

Factors (must sum to 1.0 for each type):
  temp  – temperature comfort (ideal ~72°F)
  rain  – dryness (low rain probability)
  sun   – sunshine hours
  wind  – low wind speed

Attraction types to score: outdoor, indoor, beach, hiking, theme_park, cultural

Rules:
- indoor and cultural are mostly unaffected by weather → keep temp/rain/sun/wind low,
  but do not zero them out completely
- beach: temperature and sun matter most
- hiking: rain avoidance matters most (slippery trails), wind moderate
- adventure travelers care more about outdoor conditions than relaxation travelers
- relaxation travelers penalise bad weather more for outdoor types
- month context: {month_names.get(month, month)} — adjust accordingly (e.g. summer → heat matters more)

Respond ONLY with valid JSON:
{{
  "outdoor"   : {{"temp": 0.0, "rain": 0.0, "sun": 0.0, "wind": 0.0}},
  "indoor"    : {{"temp": 0.0, "rain": 0.0, "sun": 0.0, "wind": 0.0}},
  "beach"     : {{"temp": 0.0, "rain": 0.0, "sun": 0.0, "wind": 0.0}},
  "hiking"    : {{"temp": 0.0, "rain": 0.0, "sun": 0.0, "wind": 0.0}},
  "theme_park": {{"temp": 0.0, "rain": 0.0, "sun": 0.0, "wind": 0.0}},
  "cultural"  : {{"temp": 0.0, "rain": 0.0, "sun": 0.0, "wind": 0.0}}
}}

Each row must sum to exactly 1.0. No markdown, no explanation."""

        try:
            resp = client.chat.completions.create(
                model       = "gpt-4o-mini",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0,
            )
            raw     = resp.choices[0].message.content.strip()
            weights = json.loads(raw)

            # Validate: normalise each type so weights always sum to 1.0
            for atype, w in weights.items():
                total = sum(w.values())
                if total > 0:
                    weights[atype] = {k: round(v / total, 4) for k, v in w.items()}

            self._weights_cache[cache_key] = weights
            print(f"[WeatherModel] Dynamic weights cached for ({travel_style}, {month_names.get(month)}, {group})")
            return weights

        except Exception as e:
            print(f"[WeatherModel] Dynamic weight generation failed ({e}) — using defaults")
            return {}

    # ── Suitability scorer ───────────────────────────────────────────
    def _compute_suitability(self, row: pd.Series, attraction_type: str,
                              weights: dict = None) -> float:
        """
        Suitability score (0-1) given monthly climate stats and attraction type.

        weights : optional dict {"temp": w, "rain": w, "sun": w, "wind": w}
                  from get_dynamic_weights().  When provided, overrides the
                  hard-coded per-type weights.  Physical hard penalties
                  (beach cold, hiking heat) are always applied regardless.
        """
        temp        = row["avg_temp_mean"]
        rain_pct    = row["rain_days_pct"]       # 0-1, % of days with rain
        sunshine    = row["avg_sunshine_hrs"]    # hours per day
        wind        = row["avg_windspeed"]       # mph
        ext_heat    = row["extreme_heat_pct"]    # % of days > 95°F
        ext_cold    = row["extreme_cold_pct"]    # % of days < 20°F

        # ── Component scores ─────────────────────────────────────────
        temp_score    = 1.0 - min(abs(temp - 72) / 40, 1.0)  # ideal 65-80°F
        rain_score    = 1.0 - rain_pct
        sun_score     = min(sunshine / 12.0, 1.0)
        extreme_penalty = (ext_heat + ext_cold) * 0.5

        # ── Dynamic weights path ─────────────────────────────────────
        if weights and all(k in weights for k in ("temp", "rain", "sun", "wind")):
            wt = weights["temp"]
            wr = weights["rain"]
            ws = weights["sun"]
            ww = weights["wind"]
            wind_score = 1 - min(wind / 30, 1.0)
            score = (wt * temp_score +
                     wr * rain_score +
                     ws * sun_score  +
                     ww * wind_score)

            # Physical hard penalties remain regardless of weights
            if attraction_type == "beach" and temp < 65:
                score *= 0.4
            elif attraction_type == "hiking":
                score *= (1 - ext_heat * 0.5)
            elif attraction_type in ("indoor", "cultural"):
                # Blend with extreme-weather base for sheltered venues
                score = 0.7 * score + 0.3 * (0.85 + 0.15 * (1 - extreme_penalty))

            return round(float(np.clip(score, 0.0, 1.0)), 4)

        # ── Hardcoded weights fallback ────────────────────────────────
        if attraction_type == "outdoor":
            score = (0.35 * temp_score +
                     0.35 * rain_score +
                     0.20 * sun_score  +
                     0.10 * (1 - min(wind / 30, 1.0)))

        elif attraction_type == "indoor":
            score = 0.85 + 0.15 * (1 - extreme_penalty)

        elif attraction_type == "beach":
            score = (0.40 * temp_score +
                     0.30 * rain_score +
                     0.20 * sun_score  +
                     0.10 * (1 - min(wind / 25, 1.0)))
            if temp < 65:
                score *= 0.4

        elif attraction_type == "hiking":
            score = (0.30 * temp_score +
                     0.40 * rain_score +
                     0.20 * sun_score  +
                     0.10 * (1 - min(wind / 30, 1.0)))
            score *= (1 - ext_heat * 0.5)

        elif attraction_type == "theme_park":
            score = (0.30 * temp_score +
                     0.40 * rain_score +
                     0.20 * sun_score  +
                     0.10 * (1 - extreme_penalty))

        elif attraction_type == "cultural":
            score = 0.75 + 0.25 * rain_score

        else:
            score = (0.40 * temp_score + 0.40 * rain_score + 0.20 * sun_score)

        return round(float(np.clip(score, 0.0, 1.0)), 4)

    # ── Public API ───────────────────────────────────────────────────
    def get_suitability(self, city: str, month: int,
                        attraction_type: str = "outdoor",
                        weights: dict = None) -> float:
        """
        Main function. Returns suitability score (0-1).

        Args:
            city            : e.g. "Miami", "Chicago"
            month           : 1-12
            attraction_type : one of ATTRACTION_TYPES
            weights         : optional full weights dict from get_dynamic_weights()
                              e.g. {"outdoor": {"temp":0.35,"rain":0.35,...}, ...}
                              The per-type sub-dict is extracted automatically.

        Returns:
            float between 0 (unsuitable) and 1 (ideal)
        """
        monthly = self._get_climate(city)
        row     = monthly[monthly["month"] == month]
        if row.empty:
            raise ValueError(f"No climate data for month {month} in {city}")
        # Extract per-type weights if the full dict was passed in
        type_weights = (weights or {}).get(attraction_type) if weights else None
        return self._compute_suitability(row.iloc[0], attraction_type, weights=type_weights)

    def get_suitability_all_types(self, city: str, month: int,
                                   weights: dict = None) -> dict:
        """Returns suitability scores for all attraction types at once.

        weights : optional full weights dict from get_dynamic_weights()
        """
        return {
            atype: self.get_suitability(city, month, atype, weights=weights)
            for atype in ATTRACTION_TYPES
        }

    def get_dynamic_suitability(self, city: str, month: int,
                                 attraction_type: str,
                                 travel_style: str,
                                 group: str = "solo",
                                 interests: str = "") -> float:
        """
        Convenience method: compute dynamic weights from user profile then score.
        Useful when you only need one score and don't want to manage the weights dict.
        """
        weights = self.get_dynamic_weights(travel_style, month, group, interests)
        return self.get_suitability(city, month, attraction_type, weights=weights)

    # ── Actual Forecast for Trip Dates ───────────────────────────────
    def get_forecast_for_date(self, city: str, date: "datetime.date",
                               attraction_type: str = "outdoor") -> dict:
        """
        Returns actual weather forecast + suitability score for a specific date.
        Uses Open-Meteo forecast API (works up to ~16 days ahead).
        Falls back to historical averages if date is too far out.

        Returns dict with keys: suitability, summary, temp_max, temp_min,
                                 precipitation, sunshine_hrs, is_forecast
        """
        from datetime import datetime, timedelta

        today    = datetime.today().date()
        horizon  = today + timedelta(days=16)

        if isinstance(date, datetime):
            trip_date = date.date()
        else:
            trip_date = date

        # ── Attempt live forecast ────────────────────────────────────
        if trip_date <= horizon:
            try:
                lat, lon = self.get_coordinates(city)
                resp = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude"           : lat,
                        "longitude"          : lon,
                        "daily"              : [
                            "temperature_2m_max", "temperature_2m_min",
                            "temperature_2m_mean", "precipitation_sum",
                            "rain_sum", "windspeed_10m_max", "sunshine_duration",
                        ],
                        "temperature_unit"   : "fahrenheit",
                        "windspeed_unit"     : "mph",
                        "precipitation_unit" : "inch",
                        "timezone"           : "auto",
                        "start_date"         : trip_date.strftime("%Y-%m-%d"),
                        "end_date"           : trip_date.strftime("%Y-%m-%d"),
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                daily = resp.json().get("daily", {})

                if daily.get("time"):
                    row = {
                        "avg_temp_max"      : daily["temperature_2m_max"][0]   or 70,
                        "avg_temp_min"      : daily["temperature_2m_min"][0]   or 55,
                        "avg_temp_mean"     : daily["temperature_2m_mean"][0]  or 65,
                        "avg_precipitation" : daily["precipitation_sum"][0]    or 0,
                        "avg_sunshine_hrs"  : (daily["sunshine_duration"][0] or 0) / 3600,
                        "avg_windspeed"     : daily["windspeed_10m_max"][0]    or 10,
                        "rain_days_pct"     : 1.0 if (daily["rain_sum"][0] or 0) > 0.1 else 0.0,
                        "extreme_heat_pct"  : 1.0 if (daily["temperature_2m_max"][0] or 0) > 95 else 0.0,
                        "extreme_cold_pct"  : 1.0 if (daily["temperature_2m_min"][0] or 0) < 20 else 0.0,
                    }
                    import pandas as pd
                    suitability = self._compute_suitability(pd.Series(row), attraction_type)
                    summary     = self._weather_summary(row)
                    return {
                        "suitability"    : suitability,
                        "summary"        : summary,
                        "temp_max"       : round(row["avg_temp_max"]),
                        "temp_min"       : round(row["avg_temp_min"]),
                        "precipitation"  : round(row["avg_precipitation"], 2),
                        "sunshine_hrs"   : round(row["avg_sunshine_hrs"], 1),
                        "is_forecast"    : True,
                    }
            except Exception as e:
                print(f"  [WeatherModel] Forecast failed ({e}) — using historical")

        # ── Fallback: historical average for that month ──────────────
        month       = trip_date.month
        suitability = self.get_suitability(city, month, attraction_type)
        return {
            "suitability" : suitability,
            "summary"     : f"Typical {trip_date.strftime('%B')} weather",
            "temp_max"    : None,
            "temp_min"    : None,
            "precipitation": None,
            "sunshine_hrs": None,
            "is_forecast" : False,
        }

    def get_forecast_all_days(self, city: str, start_date: "datetime",
                               days: int) -> list:
        """
        Returns a list of forecast dicts (one per trip day).
        Used by ItineraryPlanner to assign weather-sensitive activities.
        """
        from datetime import timedelta
        return [
            self.get_forecast_for_date(city, start_date + timedelta(days=d))
            for d in range(days)
        ]

    @staticmethod
    def _weather_summary(row: dict) -> str:
        """Generate a short human-readable weather summary."""
        temp    = row.get("avg_temp_mean", 65)
        rain    = row.get("rain_days_pct", 0)
        sun     = row.get("avg_sunshine_hrs", 6)
        extreme = row.get("extreme_heat_pct", 0) + row.get("extreme_cold_pct", 0)

        if extreme > 0.5:
            cond = "Extreme conditions"
        elif rain > 0.6:
            cond = "Rainy / overcast"
        elif sun > 8:
            cond = "Sunny"
        elif sun > 5:
            cond = "Partly cloudy"
        else:
            cond = "Mostly cloudy"

        return f"{cond}, ~{round(temp)}°F"


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    wm = WeatherModel()

    print("\n── Miami, July ──")
    scores = wm.get_suitability_all_types("Miami", month=7)
    for k, v in scores.items():
        print(f"  {k:<15} {v}")

    print("\n── Chicago, January ──")
    scores = wm.get_suitability_all_types("Chicago", month=1)
    for k, v in scores.items():
        print(f"  {k:<15} {v}")
