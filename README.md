# TravelIQ — AI-Powered Personalized Travel Itinerary Generator

TravelIQ is an end-to-end intelligent travel planning system that takes a destination, travel dates, and personal preferences and generates a complete, optimized day-by-day itinerary — with real travel-time enforcement, weather-aware scheduling, restaurant recommendations, and AI-generated outfit suggestions — in under 90 seconds.

---

## Features

- **OR-Tools VRPTW routing** — optimal multi-day attraction scheduling with time windows and traffic-aware travel times
- **DistilBERT sentiment analysis** — review quality scoring across thousands of Google Places reviews
- **SARIMAX crowd prediction** — historical crowd-level forecasting per attraction type and time slot
- **GPT-4o feasibility audit** — post-scheduling LLM verification that catches timing conflicts and enriches descriptions
- **Weather-aware scheduling** — OpenMeteo forecast + historical averages; rain/cold adjusts activity order and meal timing
- **Real travel-time enforcement** — Google Distance Matrix API used for an N×N matrix; every transition enforces actual drive/walk/transit minutes
- **Meal insertion** — breakfast, lunch, and dinner slotted around attractions with 3-hour minimum gap enforcement and duplicate detection
- **Gap filling** — unused high-rated attractions are automatically inserted into schedule gaps > 60 min before falling back to a free-time placeholder (capped at 60 min)
- **Outfit recommendation engine** — scikit-learn classifier predicts outfit category from weather + activity; FLUX.1-schnell (Hugging Face) generates photorealistic outfit images per day/slot
- **Temperature-aware outfit items** — cold/cool/warm/hot bands ensure NYC in January never shows shorts
- **Must-visit places** — users can specify places that are validated against the destination city via Google Places before being pinned into the itinerary
- **Server-Sent Events streaming** — real-time progress updates during the ~90-second generation pipeline
- **Deterministic image caching** — outfit images are cached by prompt hash so regenerating the same trip reuses existing images

---

## Project Structure

```
traveliq/
├── app.py                          # Flask app — all API endpoints
├── pipeline.py                     # Orchestrates the full planning pipeline
│
├── modules/
│   ├── itinerary.py                # Core planner: OR-Tools, meal insertion, gap fill,
│   │                               #   travel matrix, annotate_travel, validate_and_fix
│   ├── review.py                   # DistilBERT sentiment + complaint scoring
│   ├── crowd.py                    # SARIMAX crowd-level forecasting
│   ├── weather.py                  # OpenWeatherMap forecast + historical averages
│   └── ranker.py                   # Utility score calculation and attraction ranking
│
├── models/
│   ├── best_model.pt               # Fine-tuned DistilBERT weights (Git LFS, 253 MB)
│   ├── traveliq_tokenizer/         # DistilBERT tokenizer files
│   ├── sarimax_models.pkl          # SARIMAX crowd models (3.4 GB — see download below)
│   ├── saved_model.pkl             # Outfit category classifier (scikit-learn pipeline)
│   └── label_encoder.pkl           # Outfit category label encoder
│
├── travel_outfit_engine/
│   ├── src/
│   │   ├── image_generator.py      # FLUX.1-schnell image generation via HF Inference API
│   │   ├── prompt_builder.py       # Structured 5-part outfit prompt construction
│   │   └── outfit_mapper.py        # Temperature-band outfit item lookup
│   ├── models/
│   │   ├── saved_model.pkl         # Outfit category classifier
│   │   └── label_encoder.pkl       # Label encoder
│   └── requirements.txt
│
├── static/
│   ├── css/style.css               # Full UI stylesheet
│   ├── js/app.js                   # Frontend logic, SSE handling, outfit gallery
│   └── generated_outfits/          # Runtime outfit images (gitignored)
│
├── templates/
│   └── index.html                  # Single-page app template
│
├── requirements.txt
└── .env                            # API keys (not committed — see Environment Variables)
```

---

## Full Pipeline Overview

```
User Input
  city, state, travel dates, duration, travel style, group type,
  pace, fitness level, transport mode, dietary preferences,
  meal budget, wake hour, interests, must-visit places, gender
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  INPUT VALIDATION                                        │
│  • Google Places city validation                         │
│  • Must-visit place validation (city-scoped)             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  PARALLEL DATA FETCH  (ThreadPoolExecutor)               │
│  • Google Places — hotel search                          │
│  • Google Places — attraction candidate pool             │
│  • OpenWeatherMap — 5-day forecast / historical avg      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  PARALLEL INTELLIGENCE LAYER  (ThreadPoolExecutor)       │
│  • DistilBERT — batch sentiment + complaint scoring      │
│  • SARIMAX — crowd index per attraction × time slot      │
│  • Utility score = f(rating, sentiment, crowd, relevance)│
│  • Google Distance Matrix — N×N traffic-aware matrix     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ATTRACTION RANKING & PINNING                            │
│  • Sort by utility score                                 │
│  • Pin must-visit places at top (promote or synthesise)  │
│  • Apply cutoff for days × pace capacity                 │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  OR-TOOLS VRPTW ROUTING                                  │
│  • Vehicle Routing Problem with Time Windows             │
│  • One vehicle per day, depot = hotel                    │
│  • Service times from intelligence (duration + queue)    │
│  • Time windows from Google opening hours                │
│  • Greedy fallback if CP-SAT solver finds no solution    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  POST-ROUTING REFINEMENT                                 │
│  • Variety optimisation (cross-day swaps via GPT-4o)     │
│  • Energy curve (high-energy morning → relaxed evening)  │
│  • Day profile computation (wake, meal windows, cutoffs) │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  MEAL INSERTION                                          │
│  • Breakfast, lunch, dinner slotted around attractions   │
│  • 3-hour minimum gap between meals                      │
│  • Pre-detection of food-type attractions (no duplicates)│
│  • Style-specific events: food tasting, golden hour,     │
│    wellness break, evening out (based on travel style)   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  GAP FILLING                                             │
│  • Gaps > 60 min → try inserting unused attractions      │
│    (highest utility that fits with haversine travel est) │
│  • Remaining gap capped at 60-min free-time placeholder  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  HOTEL RETURNS                                           │
│  • "Return to hotel" appended to each day's schedule     │
│  • Travel time computed from last attraction via matrix  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  PARALLEL AUDIT + ENRICHMENT  (ThreadPoolExecutor)       │
│  • GPT-4o feasibility audit — checks timing conflicts,   │
│    returns corrected arrival/departure_min per event     │
│  • GPT-4o enrichment — restaurant names, tips, day notes │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  POST-PROCESSING                                         │
│  • Apply GPT-4o corrections with meal window clamping    │
│    (dinner hard-capped 6 PM – 10 PM)                     │
│  • Meal reanchoring — geocode restaurants, real travel   │
│    times via Distance Matrix, cascade-shift downstream   │
│  • Travel annotations (→ X min drive/walk between stops) │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  OUTFIT ENGINE  (travel_outfit_engine/)                  │
│  • Scikit-learn classifier → outfit category per slot    │
│  • Temperature band → contextual clothing item labels    │
│  • FLUX.1-schnell via HF Inference API → outfit image    │
│  • Deterministic cache: prompt hash → filename           │
│    (same trip = same image, different city = new image)  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   Rendered Itinerary streamed via SSE to browser
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask |
| Routing | Google OR-Tools (CP-SAT / VRPTW) |
| Sentiment | DistilBERT (fine-tuned, HuggingFace Transformers) |
| Crowd prediction | SARIMAX (statsmodels) |
| Outfit classifier | scikit-learn pipeline (Random Forest / Gradient Boosting) |
| Outfit images | FLUX.1-schnell via Hugging Face Inference API |
| Maps & travel | Google Places API, Google Distance Matrix API |
| Weather | OpenMeteo API |
| LLM audit/enrich | OpenAI GPT-4o |
| Frontend | Vanilla JS, SSE streaming, CSS custom properties |

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
OPENWEATHER_API_KEY=...
HF_API_KEY=hf_...
```

| Variable | Where to get it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) — enable Places API + Distance Matrix API |
| `OPENWEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) |
| `HF_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## Large Model File

`models/sarimax_models.pkl` (3.4 GB) exceeds GitHub's file size limits and cannot be stored in this repository.

**Download it from Google Drive:**
[https://drive.google.com/file/d/1PNjZvC9SSSovZj_Q9Xt437Gl8QMRu4HY/view?usp=sharing](https://drive.google.com/file/d/1PNjZvC9SSSovZj_Q9Xt437Gl8QMRu4HY/view?usp=sharing)

Place the downloaded file at `models/sarimax_models.pkl` before running the app.

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/MithunShivakoti/TravelIQ.git
cd TravelIQ

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the large model file (see above) and place at models/sarimax_models.pkl

# 4. Create your .env file with the API keys listed above

# 5. Run the app
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Notes

- The outfit image generator requires a valid `HF_API_KEY` with access to `black-forest-labs/FLUX.1-schnell` via the Hugging Face Inference API.
- Generated outfit images are saved to `static/generated_outfits/` and are gitignored.
- The `models/best_model.pt` (DistilBERT weights, 253 MB) is stored via Git LFS and will be downloaded automatically on `git clone` if Git LFS is installed (`git lfs install`).
