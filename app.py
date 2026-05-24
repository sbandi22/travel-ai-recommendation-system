"""TravelIQ — Flask web server with SSE progress streaming."""

import json
import sys
import time
import uuid
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)

_pipeline      = None
_pipeline_lock = threading.Lock()
_jobs: dict    = {}
_jobs_lock     = threading.Lock()


def get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from pipeline import TravelIQPipeline
            _pipeline = TravelIQPipeline()
        return _pipeline


def _clean(obj):
    """Recursively strip non-JSON-serializable values (numpy arrays, etc.)."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ('features',)}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


class _Capture:
    """Redirects stdout lines into the job's message list."""
    def __init__(self, jid: str):
        self._jid  = jid
        self._real = sys.__stdout__

    def write(self, s: str):
        if s and s.strip():
            with _jobs_lock:
                if self._jid in _jobs:
                    _jobs[self._jid]['messages'].append(s.strip())
        try:
            self._real.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/validate-city', methods=['POST'])
def validate_city():
    data  = request.get_json(force=True, silent=True) or {}
    city  = (data.get('city') or '').strip()
    state = (data.get('state') or '').strip()
    if not city or not state:
        return jsonify({'valid': False, 'message': 'Please enter a city and select a state.'})
    try:
        import requests as _req
        from dotenv import dotenv_values
        google_key = dotenv_values('.env').get('GOOGLE_API_KEY', '')
        resp = _req.get(
            'https://maps.googleapis.com/maps/api/geocode/json',
            params={'address': f'{city}, {state}', 'key': google_key},
        ).json()

        results = resp.get('results', [])
        if not results:
            return jsonify({'valid': False,
                            'message': f'"{city}" could not be found. Please check the spelling.'})

        top     = results[0]
        # Extract locality (city-level) component from the geocoded result
        locality = next(
            (c['long_name'] for c in top.get('address_components', [])
             if 'locality' in c['types']),
            None
        )

        if locality is None:
            # No city-level match — geocoded to a region, state, or country
            return jsonify({'valid': False,
                            'message': f'"{city}" doesn\'t appear to be a real city. Please enter a valid city name.'})

        # Check that what Google found resembles what the user typed
        if city.lower() not in locality.lower() and locality.lower() not in city.lower():
            return jsonify({'valid': False,
                            'message': f'"{city}" wasn\'t found — did you mean {locality}?'})

        return jsonify({'valid': True})
    except Exception:
        # On any error, let the user through rather than blocking
        return jsonify({'valid': True})


@app.route('/api/outfit', methods=['POST'])
def get_outfit():
    import re, os, sys, random
    data            = request.get_json(force=True, silent=True) or {}
    gender          = data.get('gender', 'male')
    attractions     = data.get('attractions', [])
    weather_summary = data.get('weather_summary', '')

    # ── Parse weather from summary string ────────────────────
    temp = 75
    m = re.search(r'(\d+)\s*[°]?\s*F', weather_summary, re.IGNORECASE)
    if m:
        temp = int(m.group(1))

    condition = 'Sunny'
    for c in ['Rain', 'Snowy', 'Cloudy', 'Windy', 'Sunny']:
        if c.lower() in weather_summary.lower():
            condition = c
            break
    rain = 65 if condition == 'Rain' else 20 if condition == 'Cloudy' else 5

    # ── Map attraction_type → engine activity ─────────────────
    TYPE_MAP = {
        'museum': 'Museum', 'art_gallery': 'Museum', 'aquarium': 'Museum',
        'zoo': 'Museum', 'amusement_park': 'Museum', 'tourist_attraction': 'Museum',
        'park': 'Hiking', 'natural_feature': 'Hiking', 'campground': 'Hiking',
        'stadium': 'Hiking',
        'restaurant': 'Dinner', 'bar': 'Dinner',
        'cafe': 'Walking', 'spa': 'Walking', 'shopping_mall': 'Walking',
    }
    daytime_activities = [
        TYPE_MAP.get(a.get('attraction_type', ''), 'Museum')
        for a in attractions
        if a.get('event_type', 'attraction') == 'attraction'
        and a.get('arrival_min', 0) < 1020   # before 5 PM
    ]
    from collections import Counter
    dominant = Counter(daytime_activities).most_common(1)[0][0] if daytime_activities else 'Museum'

    # ── Load outfit engine (paths relative to engine folder) ──
    engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'travel_outfit_engine')
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)

    try:
        import joblib
        import pandas as pd

        _model = joblib.load(os.path.join(engine_dir, 'models', 'saved_model.pkl'))
        _enc   = joblib.load(os.path.join(engine_dir, 'models', 'label_encoder.pkl'))

        from src.outfit_mapper  import get_items
        from src.prompt_builder import STYLE_VARIANTS

        def _predict(activity, time_of_day):
            df  = pd.DataFrame([{'temp': temp, 'rain': rain, 'activity': activity,
                                  'weather': condition, 'time': time_of_day}])
            raw = _model.predict(df)
            return _enc.inverse_transform(raw)[0]

        def _slot(category, activity, time_of_day):
            variants = STYLE_VARIANTS.get(category, [])
            v        = random.choice(variants) if variants else None
            is_f     = gender == 'female'
            items    = get_items(category, temp)
            return {
                'category'   : category,
                'activity'   : activity,
                'items'      : items,
                'style_name' : v.name        if v else '',
                'colors'     : v.colors      if v else '',
                'footwear'   : (v.women_footwear    or v.footwear)    if (v and is_f) else (v.footwear    if v else ''),
                'accessories': (v.women_accessories or v.accessories) if (v and is_f) else (v.accessories if v else ''),
            }

        day_cat = _predict(dominant,  'morning')
        eve_cat = _predict('Dinner',  'evening')
        day_slot = _slot(day_cat, dominant, 'morning')
        eve_slot = _slot(eve_cat, 'Dinner',  'evening')

        all_items    = list({i for s in [day_slot, eve_slot] for i in s['items']})
        return jsonify({'daytime': day_slot, 'evening': eve_slot, 'packing_list': all_items})

    except Exception:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'Outfit engine failed. Please try again.'}), 500


@app.route('/api/outfit-images', methods=['POST'])
def get_outfit_images():
    import re, os, sys, random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from collections import Counter

    data   = request.get_json(force=True, silent=True) or {}
    gender = data.get('gender', 'male')
    days   = data.get('days', [])
    city   = data.get('city', '')

    engine_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'travel_outfit_engine')
    outfits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_outfits')
    os.makedirs(outfits_dir, exist_ok=True)

    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)

    try:
        from dotenv import dotenv_values
        hf_key = dotenv_values('.env').get('HF_API_KEY', '')
        if not hf_key:
            return jsonify({'error': 'HF_API_KEY not set in .env'}), 500
        os.environ['HF_API_KEY'] = hf_key

        import joblib, pandas as pd
        import src.image_generator as ig
        from src.outfit_mapper  import get_items
        from src.prompt_builder import STYLE_VARIANTS

        # Override engine defaults for speed + correct output path
        ig.CANDIDATES  = 1
        ig.OUTPUT_DIR  = Path(outfits_dir)

        _model = joblib.load(os.path.join(engine_dir, 'models', 'saved_model.pkl'))
        _enc   = joblib.load(os.path.join(engine_dir, 'models', 'label_encoder.pkl'))

        TYPE_MAP = {
            'museum': 'Museum', 'art_gallery': 'Museum', 'aquarium': 'Museum',
            'zoo': 'Museum', 'amusement_park': 'Museum', 'tourist_attraction': 'Museum',
            'park': 'Hiking', 'natural_feature': 'Hiking', 'campground': 'Hiking',
            'stadium': 'Hiking',
            'restaurant': 'Dinner', 'bar': 'Dinner',
            'cafe': 'Walking', 'spa': 'Walking', 'shopping_mall': 'Walking',
        }

        def _parse_weather(summary):
            temp = 75
            m = re.search(r'(\d+)\s*[°]?\s*F', summary, re.IGNORECASE)
            if m: temp = int(m.group(1))
            cond = 'Sunny'
            for c in ['Rain', 'Snowy', 'Cloudy', 'Windy', 'Sunny']:
                if c.lower() in summary.lower(): cond = c; break
            rain = 65 if cond == 'Rain' else 20 if cond == 'Cloudy' else 5
            return temp, rain, cond

        def _predict(temp, rain, cond, activity, time_of_day):
            df = pd.DataFrame([{'temp': temp, 'rain': rain, 'activity': activity,
                                 'weather': cond, 'time': time_of_day}])
            return _enc.inverse_transform(_model.predict(df))[0]

        def _gen_slot(day_key, slot_label, category, event_features):
            variants = STYLE_VARIANTS.get(category, [])
            v        = random.choice(variants) if variants else None
            # Inject day + slot so each day/slot gets a unique variant seed and prompt
            ef_unique = {**event_features, 'day': day_key, 'slot': slot_label}
            # Pass outfit_items=None — let prompt_builder variants handle clothing.
            # They are richer and already weather-context-aware via event_features.
            path     = ig.generate_outfit_image(category=category,
                                                event_features=ef_unique,
                                                outfit_items=None, gender=gender)
            rel  = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')

            # Build a description from the variant's actual clothing fields so the
            # tags shown beneath the image describe what's actually in it.
            is_f = gender == 'female'
            if v:
                parts = []
                top       = (v.women_top       or v.top)       if is_f else v.top
                bottom    = (v.women_bottom     or v.bottom)    if is_f else v.bottom
                outerwear = (v.women_outerwear  or v.outerwear) if is_f else v.outerwear
                footwear  = (v.women_footwear   or v.footwear)  if is_f else v.footwear
                for piece in (top, bottom, outerwear, footwear):
                    if piece and piece.lower() not in ('none', ''):
                        # Capitalise first letter and trim long descriptions to ~40 chars
                        label = piece.strip().capitalize()
                        if len(label) > 48:
                            label = label[:45].rsplit(' ', 1)[0] + '…'
                        parts.append(label)
                description = parts
            else:
                description = []

            return {
                'day_key':     day_key,
                'slot':        slot_label,
                'category':    category,
                'style_name':  v.name   if v else '',
                'colors':      v.colors if v else '',
                'description': description,
                'image_url':   f'/{rel}',
            }

        # Build task list
        tasks = []
        for day in days:
            dk      = day.get('day_key', '')
            events  = day.get('events', [])
            temp, rain, cond = _parse_weather(day.get('weather_summary', ''))
            daytime_types = [
                TYPE_MAP.get(a.get('attraction_type', ''), 'Museum')
                for a in events
                if a.get('event_type', 'attraction') == 'attraction'
                and a.get('arrival_min', 0) < 1020
            ]
            dominant = Counter(daytime_types).most_common(1)[0][0] if daytime_types else 'Museum'
            tasks.append((dk, 'daytime', _predict(temp, rain, cond, dominant, 'morning'),
                          {'temp': temp, 'rain': rain, 'activity': dominant, 'weather': cond, 'time': 'morning', 'city': city}))
            tasks.append((dk, 'evening', _predict(temp, rain, cond, 'Dinner',  'evening'),
                          {'temp': temp, 'rain': rain, 'activity': 'Dinner',  'weather': cond, 'time': 'evening', 'city': city}))

        # Fire all in parallel
        results_by_day = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_gen_slot, *t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    r  = fut.result()
                    dk = r['day_key']
                    results_by_day.setdefault(dk, {})[r['slot']] = r
                except Exception as e:
                    print(f"  [outfit] image failed: {e}")

        outfits = [{'day_key': d['day_key'],
                    'daytime': results_by_day.get(d['day_key'], {}).get('daytime'),
                    'evening': results_by_day.get(d['day_key'], {}).get('evening')}
                   for d in days]
        return jsonify({'outfits': outfits})

    except Exception:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'Outfit image generation failed. Please try again.'}), 500


@app.route('/api/validate-places', methods=['POST'])
def validate_places():
    data   = request.get_json(force=True, silent=True) or {}
    city   = (data.get('city') or '').strip()
    state  = (data.get('state') or '').strip()
    places = data.get('places', [])

    if not places or not city:
        return jsonify({'valid': True, 'validated': []})

    try:
        import requests as _req
        from dotenv import dotenv_values
        google_key = dotenv_values('.env').get('GOOGLE_API_KEY', '')

        invalid, validated = [], []
        for name in places:
            resp = _req.get(
                'https://maps.googleapis.com/maps/api/place/textsearch/json',
                params={'query': f'{name} {city} {state}', 'key': google_key},
                timeout=10,
            ).json()
            results = resp.get('results', [])
            if not results:
                invalid.append(name)
                continue
            top  = results[0]
            addr = top.get('formatted_address', '').lower()
            # Confirm the result is in the requested city or state
            if city.lower() not in addr and state.lower() not in addr:
                invalid.append(name)
            else:
                loc = top.get('geometry', {}).get('location', {})
                validated.append({
                    'name':    top.get('name', name),
                    'lat':     loc.get('lat'),
                    'lon':     loc.get('lng'),
                    'address': top.get('formatted_address', ''),
                    'types':   top.get('types', ['tourist_attraction']),
                    'rating':  top.get('rating', 4.0),
                })

        if invalid:
            return jsonify({
                'valid':         False,
                'message':       f'Not found in {city}: ' + ', '.join(invalid),
                'invalid_places': invalid,
                'validated':     validated,
            })
        return jsonify({'valid': True, 'validated': validated})
    except Exception:
        return jsonify({'valid': True, 'validated': []})


@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True, silent=True) or {}
    if not data.get('city'):
        return jsonify({'error': 'city is required'}), 400

    jid = uuid.uuid4().hex[:10]
    with _jobs_lock:
        _jobs[jid] = {'status': 'running', 'messages': [], 'result': None, 'error': None}

    def _run():
        cap = _Capture(jid)
        old = sys.stdout
        sys.stdout = cap
        try:
            p = get_pipeline()
            result = p.run(
                city            = data['city'],
                state           = data['state'],
                month           = int(data['month']),
                year            = int(data['year']),
                travel_style    = data.get('travel_styles') or data.get('travel_style', 'popular'),
                interests       = data.get('interests', ''),
                group           = data.get('group', 'solo'),
                days            = int(data['days']),
                start_day       = int(data['start_day']),
                transport_mode  = data.get('transport_mode', 'driving'),
                wake_hour       = int(data.get('wake_hour', 8)),
                pace            = data.get('pace', 'moderate'),
                dietary         = data.get('dietary', 'none'),
                budget_per_meal = data.get('budget_per_meal', '$20'),
                fitness         = data.get('fitness', 'moderate'),
                group_ages      = data.get('group_ages', ''),
                must_visit      = data.get('must_visit', []),
            )
            with _jobs_lock:
                _jobs[jid]['result'] = _clean(result)
                _jobs[jid]['status'] = 'done'
        except Exception as exc:
            import traceback
            traceback.print_exc()   # log full trace to server console
            with _jobs_lock:
                _jobs[jid]['error']  = (
                    "Something went wrong while building your itinerary. "
                    "Please try again — this sometimes happens with unusual city/date combinations."
                )
                _jobs[jid]['status'] = 'error'
        finally:
            sys.stdout = old

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': jid})


@app.route('/api/stream/<jid>')
def stream_events(jid):
    def _gen():
        sent = 0
        while True:
            with _jobs_lock:
                job = dict(_jobs.get(jid, {}))
            if not job:
                yield f"data: {json.dumps({'type':'error','msg':'Job not found'})}\n\n"
                return
            new_msgs = job['messages'][sent:]
            for m in new_msgs:
                yield f"data: {json.dumps({'type':'progress','msg':m})}\n\n"
            sent += len(new_msgs)
            if job['status'] == 'done':
                yield f"data: {json.dumps({'type':'done','result':job['result']})}\n\n"
                with _jobs_lock:
                    _jobs.pop(jid, None)
                return
            if job['status'] == 'error':
                yield f"data: {json.dumps({'type':'error','msg':job.get('error','Unknown error')})}\n\n"
                with _jobs_lock:
                    _jobs.pop(jid, None)
                return
            time.sleep(0.3)

    return Response(
        stream_with_context(_gen()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


if __name__ == '__main__':
    print("TravelIQ web server starting on http://localhost:5000")
    try:
        get_pipeline()
        print("Pipeline ready.")
    except Exception as e:
        print(f"Warning: Pipeline pre-load skipped: {e}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
