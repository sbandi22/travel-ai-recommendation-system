"""
TravelIQ — Crowd Score Module
SARIMAX-based tourism demand forecasting → crowd index (0-1).

Usage:
    from modules.crowd import CrowdModel
    cm = CrowdModel(model_path="models/sarimax_models.pkl")
    score = cm.get_crowd_index(city="chicago", state="IL", year=2026, month=8)
    print(score)  # e.g. 0.73
"""

import os
import pickle
import sys

import pandas as pd


class CrowdModel:
    def __init__(self, model_path: str = "models/sarimax_models.pkl"):
        if not os.path.exists(model_path):
            print(f"[CrowdModel] Model file not found: {model_path}")
            print("Run your SARIMAX training script first.")
            sys.exit(1)
        with open(model_path, "rb") as f:
            self.models = pickle.load(f)
        print(f"[CrowdModel] Loaded {len(self.models)} city models from {model_path}")

    def list_cities(self):
        """Print all available (city, state) keys."""
        for city, state in sorted(self.models.keys()):
            print(f"  {city}, {state}")

    def _forecast_df(self, city: str, state: str, until_year: int = 2030) -> pd.DataFrame:
        """Internal: get raw forecast DataFrame for a city."""
        key = (city.lower().strip(), state.upper().strip())
        if key not in self.models:
            raise ValueError(f"No model found for {key}. Use list_cities() to see options.")

        res      = self.models[key]
        fc_start = res.forecast(steps=1).index[0]
        target   = pd.Period(year=until_year, month=12, freq="M")
        steps    = max(1, (target.year - fc_start.year) * 12
                       + (target.month - fc_start.month) + 1)
        fc = res.forecast(steps=steps)

        rows = []
        for period, value in fc.items():
            rows.append({
                "year" : period.year,
                "month": period.month,
                "pressure_ratio": round(max(0.0, float(value)), 6),
            })
        return pd.DataFrame(rows)

    def get_crowd_index(self, city: str, state: str, year: int, month: int) -> float:
        """
        Returns a crowd index (0-1) for a given city and travel month.

        The SARIMAX model outputs a 'pressure ratio' (demand proxy).
        We normalize it to 0-1 across all forecasted values for that city
        so it's directly usable as a feature in the ranker.

        0 = very low crowd, 1 = very high crowd
        """
        df = self._forecast_df(city, state, until_year=max(year, 2030))

        # Normalize pressure ratio to 0-1 across the full forecast range
        min_val = df["pressure_ratio"].min()
        max_val = df["pressure_ratio"].max()

        row = df[(df["year"] == year) & (df["month"] == month)]
        if row.empty:
            raise ValueError(f"No forecast available for {city}, {state} — {year}/{month:02d}")

        raw = row["pressure_ratio"].values[0]

        if max_val == min_val:
            return 0.5   # flat forecast edge case

        crowd_index = (raw - min_val) / (max_val - min_val)
        return round(float(crowd_index), 4)

    def get_crowd_index_range(self, city: str, state: str,
                               start_year: int, start_month: int,
                               num_months: int = 1) -> list:
        """
        Returns crowd indices for a range of months.
        Useful for multi-day trip planning.
        """
        results = []
        for i in range(num_months):
            period = pd.Period(year=start_year, month=start_month, freq="M") + i
            try:
                score = self.get_crowd_index(city, state, period.year, period.month)
                results.append({"year": period.year, "month": period.month, "crowd_index": score})
            except ValueError as e:
                print(f"Warning: {e}")
        return results


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    cm = CrowdModel()

    # Single month
    score = cm.get_crowd_index(city="chicago", state="IL", year=2026, month=8)
    print(f"Chicago crowd index for Aug 2026: {score}")

    # Range of months (e.g. a 3-month trip window)
    scores = cm.get_crowd_index_range("chicago", "IL", 2026, 6, num_months=3)
    for s in scores:
        print(s)
