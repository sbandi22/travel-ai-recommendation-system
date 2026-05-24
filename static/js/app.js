/* ═══════════════════════════════════════════════════════════
   TravelIQ — Frontend Application
   Particle canvas · Wizard state · SSE streaming · Results
═══════════════════════════════════════════════════════════ */

'use strict';

/* ── Style metadata ──────────────────────────────────────── */
const STYLES = [
  { id:'adventure',   emoji:'🏔️', name:'Adventure',   tag:'Outdoor thrills',   grad:'#f7971e,#ffd200', accent:'#ffd200' },
  { id:'cultural',    emoji:'🏛️', name:'Cultural',    tag:'Art & heritage',    grad:'#8360c3,#2ebf91', accent:'#2ebf91' },
  { id:'relaxation',  emoji:'🌅', name:'Relaxation',  tag:'Slow & serene',     grad:'#c9d6ff,#e2e2e2', accent:'#c9d6ff' },
  { id:'foodie',      emoji:'🍜', name:'Foodie',      tag:'Eat everything',    grad:'#f093fb,#f5576c', accent:'#f093fb' },
  { id:'family',      emoji:'👨‍👩‍👧', name:'Family',    tag:'Fun for all',       grad:'#4facfe,#00f2fe', accent:'#4facfe' },
  { id:'luxury',      emoji:'💎', name:'Luxury',      tag:'Premium & refined', grad:'#d4af37,#a07020', accent:'#d4af37' },
  { id:'budget',      emoji:'🎒', name:'Budget',      tag:'Smart spending',    grad:'#56ab2f,#a8e063', accent:'#a8e063' },
  { id:'nightlife',   emoji:'🎆', name:'Nightlife',   tag:'After-dark vibes',  grad:'#e94560,#0f3460', accent:'#e94560' },
  { id:'wellness',    emoji:'🌿', name:'Wellness',    tag:'Mind & body',       grad:'#a8edea,#fed6e3', accent:'#a8edea' },
  { id:'sports',      emoji:'🏃', name:'Sports',      tag:'Active & athletic', grad:'#ff416c,#ff4b2b', accent:'#ff4b2b' },
  { id:'romantic',    emoji:'🌹', name:'Romantic',    tag:'For two',           grad:'#fd79a8,#6c5ce7', accent:'#fd79a8' },
  { id:'eco',         emoji:'🌍', name:'Eco',         tag:'Earth-first',       grad:'#11998e,#38ef7d', accent:'#38ef7d' },
  { id:'photography', emoji:'📷', name:'Photography', tag:'Chase the light',   grad:'#636363,#a2ab58', accent:'#a2ab58' },
  { id:'history',     emoji:'🏺', name:'History',     tag:'Through the ages',  grad:'#c79a4a,#5c3d11', accent:'#c79a4a' },
  { id:'art',         emoji:'🎨', name:'Art',         tag:'Creativity first',  grad:'#f953c6,#b91d73', accent:'#f953c6' },
];

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

/* ── Tiny state store ────────────────────────────────────── */
const State = {
  step: 1,
  month: new Date().getMonth() + 1,
  year: 2026,
  styles: [],      // up to 3 selected style IDs
  group: 'solo',
  pace: 'moderate',
  fitness: 'moderate',
  transport: 'driving',
  budget: '$20',
  wakeHour: 8,
  days: 3,
  activeDay: 0,
};

/* ── Helpers ─────────────────────────────────────────────── */
function minsToTime(mins) {
  if (mins == null) return '';
  const total  = Math.floor(mins % 1440);
  const h      = Math.floor(total / 60);
  const m      = total % 60;
  const suffix = h < 12 ? 'AM' : 'PM';
  const h12    = h % 12 || 12;
  const ov     = mins >= 1440 ? ' +1' : '';
  return `${h12}:${m.toString().padStart(2,'0')} ${suffix}${ov}`;
}

function fmtDur(mins) {
  if (!mins || mins <= 0) return '';
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function starsHtml(n) {
  const full = Math.round(n || 0);
  return Array.from({length:5}, (_,i) =>
    `<span style="color:${i<full?'#ffc857':'rgba(255,255,255,0.2)'}">${i<full?'★':'☆'}</span>`
  ).join('');
}

function qs(sel, ctx=document) { return ctx.querySelector(sel); }
function qsa(sel, ctx=document) { return [...ctx.querySelectorAll(sel)]; }

/* ══════════════════════════════════════════════════════════
   CANVAS PARTICLE SYSTEM
══════════════════════════════════════════════════════════ */
(function initCanvas() {
  const canvas = qs('#bg-canvas');
  const ctx    = canvas.getContext('2d');
  const mouse  = { x: null, y: null };
  let   W, H, particles;
  let   particleColor = '124,108,245';

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function Particle() {
    this.x  = Math.random() * W;
    this.y  = Math.random() * H;
    this.vx = (Math.random() - 0.5) * 0.5;
    this.vy = (Math.random() - 0.5) * 0.5;
    this.r  = Math.random() * 1.4 + 0.4;
    this.a  = Math.random() * 0.45 + 0.1;
  }
  Particle.prototype.update = function() {
    this.x += this.vx;
    this.y += this.vy;
    if (this.x < 0 || this.x > W) this.vx *= -1;
    if (this.y < 0 || this.y > H) this.vy *= -1;
    if (mouse.x != null) {
      const dx = mouse.x - this.x, dy = mouse.y - this.y;
      const d  = Math.sqrt(dx*dx + dy*dy);
      if (d < 180) {
        this.vx += dx * 0.00006;
        this.vy += dy * 0.00006;
      }
    }
    const sp = Math.sqrt(this.vx*this.vx + this.vy*this.vy);
    if (sp > 1.8) { this.vx = this.vx/sp*1.8; this.vy = this.vy/sp*1.8; }
  };

  function init() {
    resize();
    particles = Array.from({length:160}, () => new Particle());
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.update();
      for (let j = i+1; j < particles.length; j++) {
        const q  = particles[j];
        const dx = p.x-q.x, dy = p.y-q.y;
        const d  = Math.sqrt(dx*dx+dy*dy);
        if (d < 110) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = `rgba(${particleColor},${0.12*(1-d/110)})`;
          ctx.lineWidth   = 0.6;
          ctx.stroke();
        }
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(${particleColor},${p.a})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });

  init();
  draw();

  // Expose color changer
  window.setParticleColor = function(hex) {
    // hex like '#f7971e' → '247,151,30'
    const r = parseInt(hex.slice(1,3),16);
    const g = parseInt(hex.slice(3,5),16);
    const b = parseInt(hex.slice(5,7),16);
    particleColor = `${r},${g},${b}`;
  };
})();

/* ══════════════════════════════════════════════════════════
   CURSOR GLOW
══════════════════════════════════════════════════════════ */
(function initCursor() {
  const glow = qs('#cursor-glow');
  document.addEventListener('mousemove', e => {
    glow.style.left = e.clientX + 'px';
    glow.style.top  = e.clientY + 'px';
  });
})();

/* ══════════════════════════════════════════════════════════
   SCREEN TRANSITIONS
══════════════════════════════════════════════════════════ */
const Screens = {
  current: 'screen-hero',
  go(id) {
    if (this.current === id) return;
    const from = document.getElementById(this.current);
    const to   = document.getElementById(id);
    if (!to) return;

    // Play exit animation on current screen
    if (from) {
      from.classList.add('screen-out');
      setTimeout(() => {
        from.classList.remove('active', 'screen-out');
      }, 400);
    }

    // Bring in new screen
    to.classList.add('active', 'screen-in');
    setTimeout(() => {
      to.classList.remove('screen-in');
    }, 600);

    this.current = id;
  }
};

/* ══════════════════════════════════════════════════════════
   WIZARD INIT (month grid, year picker, style cards)
══════════════════════════════════════════════════════════ */
function buildMonthGrid() {
  const grid = qs('#month-grid');
  MONTHS.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'mth-btn' + (i+1 === State.month ? ' active' : '');
    btn.textContent = m;
    btn.onclick = () => {
      qsa('.mth-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State.month = i + 1;
    };
    grid.appendChild(btn);
  });
}

function buildYearPicker() {
  const row = qs('#year-picker');
  [2025,2026,2027].forEach(y => {
    const btn = document.createElement('button');
    btn.className = 'yr-btn' + (y === State.year ? ' active' : '');
    btn.textContent = y;
    btn.onclick = () => {
      qsa('.yr-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State.year = y;
    };
    row.appendChild(btn);
  });
}

function buildStyleGrid() {
  const grid = qs('#style-grid');

  // Inject selection counter above the grid
  const head = qs('#wstep-2 .ws-head');
  if (head && !qs('#style-counter')) {
    const counter = document.createElement('div');
    counter.id = 'style-counter';
    counter.className = 'style-counter';
    counter.textContent = 'Pick 1–3 vibes (optional)';
    head.appendChild(counter);
  }

  STYLES.forEach(s => {
    const [c1, c2] = s.grad.split(',');
    const card = document.createElement('div');
    card.className   = 'style-card';
    card.dataset.id  = s.id;
    card.style.setProperty('--card-grad',   `linear-gradient(145deg, ${c1}, ${c2})`);
    card.style.setProperty('--card-accent', s.accent);
    card.innerHTML = `
      <div class="sc-emoji">${s.emoji}</div>
      <div class="sc-selected-ring">✓</div>
      <div class="sc-name">${s.name}</div>
      <div class="sc-tag">${s.tag}</div>`;

    // 3D tilt on mouse move
    card.addEventListener('mousemove', e => {
      const r   = card.getBoundingClientRect();
      const x   = (e.clientX - r.left) / r.width  - 0.5;
      const y   = (e.clientY - r.top)  / r.height - 0.5;
      card.style.transform = `translateY(-8px) scale(1.02) rotateX(${-y*8}deg) rotateY(${x*10}deg)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });

    card.onclick = () => {
      const MAX = 3;
      const alreadySelected = card.classList.contains('selected');

      if (alreadySelected) {
        // Deselect
        card.classList.remove('selected');
        State.styles = State.styles.filter(id => id !== s.id);
      } else {
        if (State.styles.length >= MAX) {
          // Shake the counter to signal limit
          const ctr = qs('#style-counter');
          if (ctr) { ctr.classList.remove('shake'); void ctr.offsetWidth; ctr.classList.add('shake'); }
          return;
        }
        card.classList.add('selected');
        State.styles.push(s.id);
        setParticleColor('#' + c1.replace('#','').trim().substring(0,6));
      }

      // Update dimming — unselected cards dim only when at least 1 is selected
      qsa('.style-card').forEach(c => {
        const sel = c.classList.contains('selected');
        c.classList.toggle('dimmed', State.styles.length > 0 && !sel);
      });

      // Update counter
      const ctr = qs('#style-counter');
      if (ctr) {
        const n = State.styles.length;
        ctr.textContent = n === 0 ? 'Pick 1–3 vibes (optional — skip to show top sights)'
                        : n === 1 ? '1 vibe selected — add up to 2 more'
                        : n === 2 ? '2 vibes selected — add 1 more or continue'
                        :           '3 vibes selected ✓';
        ctr.classList.toggle('counter-done', n >= 1);
      }
    };
    grid.appendChild(card);
  });
}

/* ── Toggle group helper ─────────────────────────────────── */
function bindToggle(groupId, stateProp) {
  qsa(`#${groupId} .tgl`).forEach(btn => {
    btn.onclick = () => {
      qsa(`#${groupId} .tgl`).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State[stateProp] = btn.dataset.v;
    };
  });
}

/* ── Budget grid ─────────────────────────────────────────── */
function bindBudget() {
  qsa('#budget-grid .bdg').forEach(btn => {
    btn.onclick = () => {
      qsa('#budget-grid .bdg').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State.budget = btn.dataset.v;
    };
  });
}

/* ── Sliders ─────────────────────────────────────────────── */
function bindSliders() {
  const dSlider = qs('#inp-days');
  const wSlider = qs('#inp-wake');
  dSlider.oninput = () => { State.days = +dSlider.value; qs('#days-lbl').textContent = `${State.days} day${State.days>1?'s':''}`; };
  wSlider.oninput = () => {
    State.wakeHour = +wSlider.value;
    const h = State.wakeHour; const s = h < 12 ? 'AM' : 'PM'; const h12 = h % 12 || 12;
    qs('#wake-lbl').textContent = `${h12}:00 ${s}`;
  };
}

/* ── Family ages row toggle ──────────────────────────────── */
function bindGroupToggle() {
  qsa('#grp-toggle .tgl').forEach(btn => {
    btn.onclick = () => {
      qsa('#grp-toggle .tgl').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State.group = btn.dataset.v;
      qs('#ages-row').style.display = State.group === 'family' ? 'flex' : 'none';
    };
  });
}

/* ══════════════════════════════════════════════════════════
   STEP NAVIGATION
══════════════════════════════════════════════════════════ */
const App = {
  prevStep() {
    if (State.step <= 1) return;
    this._transition(State.step, State.step - 1);
    State.step--;
    updateProgress();
  },

  nextStep(targetStep) {
    const next = targetStep || State.step + 1;
    if (next > 4) return;
    this._transition(State.step, next);
    State.step = next;
    updateProgress();
  },

  _transition(fromN, toN) {
    const cur = document.getElementById(`wstep-${fromN}`);
    const nxt = document.getElementById(`wstep-${toN}`);
    if (!cur || !nxt) return;

    // Slide out current
    cur.classList.add('slide-out');
    setTimeout(() => {
      cur.classList.remove('active', 'slide-out');
    }, 320);

    // Slide in next after a tiny overlap
    setTimeout(() => {
      nxt.classList.add('active');
    }, 80);
  },

  retry() {
    // Reset loading screen state and go back to wizard step 4
    const wrap = qs('.load-wrap');
    if (wrap) {
      wrap.innerHTML = `
        <div class="load-ring-wrap">
          <svg class="load-ring" viewBox="0 0 140 140">
            <circle class="lr-track" cx="70" cy="70" r="62"/>
            <circle class="lr-fill" id="lr-fill" cx="70" cy="70" r="62"/>
          </svg>
          <div class="load-inner">
            <div class="load-plane">✈</div>
            <div class="load-pct" id="load-pct">0%</div>
          </div>
        </div>
        <h3 class="load-title">Crafting your perfect journey</h3>
        <p class="load-sub" id="load-sub">Initialising AI systems…</p>
        <div class="load-pipeline">
          <div class="lp-step" id="lp-hotel"><span class="lp-icon">🏨</span><span class="lp-txt">Finding your hotel</span><span class="lp-check">✓</span></div>
          <div class="lp-step" id="lp-attrs"><span class="lp-icon">🗺️</span><span class="lp-txt">Discovering attractions</span><span class="lp-check">✓</span></div>
          <div class="lp-step" id="lp-feat"><span class="lp-icon">🧠</span><span class="lp-txt">Analysing reviews & sentiment</span><span class="lp-check">✓</span></div>
          <div class="lp-step" id="lp-weather"><span class="lp-icon">⛅</span><span class="lp-txt">Checking weather forecast</span><span class="lp-check">✓</span></div>
          <div class="lp-step" id="lp-route"><span class="lp-icon">📍</span><span class="lp-txt">Building optimal route</span><span class="lp-check">✓</span></div>
          <div class="lp-step" id="lp-meals"><span class="lp-icon">🍽️</span><span class="lp-txt">Curating dining experiences</span><span class="lp-check">✓</span></div>
          <div class="lp-step" id="lp-audit"><span class="lp-icon">✅</span><span class="lp-txt">Final feasibility check</span><span class="lp-check">✓</span></div>
        </div>
      `;
    }
    Screens.go('screen-wizard');
  },
};

// Expose prevStep globally for inline onclick
window.App = App;

function updateProgress() {
  qsa('.wp-step').forEach((el, i) => {
    const n = i + 1;
    el.classList.toggle('active', n === State.step);
    el.classList.toggle('done',   n < State.step);
  });
  qsa('.wp-line').forEach((el, i) => {
    el.classList.toggle('done', i + 1 < State.step);
  });
}

/* ── Step validation ─────────────────────────────────────── */
async function validateStep1() {
  const city  = qs('#inp-city').value.trim();
  const state = qs('#inp-state').value;
  if (!city) { shake(qs('#inp-city')); return false; }
  if (!state) { shake(qs('#inp-state')); return false; }

  const btn = qs('#s1-next');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin-icon">⏳</span> Checking…';
  clearCityError();

  try {
    const resp = await fetch('/api/validate-city', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ city, state }),
    });
    const data = await resp.json();
    if (!data.valid) {
      shake(qs('#inp-city'));
      showCityError(data.message || `"${city}" doesn't appear to be a real city in that state. Please check the spelling.`);
      return false;
    }
    return true;
  } catch {
    return true; // network error — don't block user
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Continue <span>→</span>';
  }
}

function showCityError(msg) {
  clearCityError();
  const err = document.createElement('p');
  err.id        = 'city-error';
  err.className = 'field-error';
  err.textContent = msg;
  qs('#inp-city').insertAdjacentElement('afterend', err);
}

function clearCityError() {
  const el = qs('#city-error');
  if (el) el.remove();
}
function validateStep2() {
  // Vibe selection is optional — empty styles → pipeline defaults to popular/top sights
  if (false && State.styles.length === 0) {
    qs('#style-grid').style.animation = 'none';
    requestAnimationFrame(() => { qs('#style-grid').style.animation = 'shake 0.4s ease'; });
    const ctr = qs('#style-counter');
    if (ctr) { ctr.classList.remove('shake'); void ctr.offsetWidth; ctr.classList.add('shake'); }
    return false;
  }
  return true;
}
function shake(el) {
  el.style.animation = 'none';
  requestAnimationFrame(() => {
    el.style.animation = 'shakeField 0.4s ease';
    el.focus();
  });
}

/* ── Step button bindings ────────────────────────────────── */
function bindStepButtons() {
  qs('#s1-next').onclick = async () => { if (await validateStep1()) App.nextStep(); };
  qs('#s2-next').onclick = () => { if (validateStep2()) App.nextStep(); };
  qs('#s3-next').onclick = () => App.nextStep();
  qs('#btn-gen').onclick = generate;
  qs('#btn-start').onclick = () => Screens.go('screen-wizard');
  qs('#btn-new').onclick = () => Screens.go('screen-hero');
}

/* ══════════════════════════════════════════════════════════
   GENERATE — API CALL + SSE
══════════════════════════════════════════════════════════ */
function setMustVisitStatus(html, cls) {
  const el = qs('#must-visit-status');
  if (!el) return;
  el.className = cls;
  el.innerHTML = html;
}

async function generate() {
  const city  = qs('#inp-city').value.trim();
  const state = qs('#inp-state').value.trim();
  const raw   = (qs('#inp-must-visit').value || '').trim();
  const mustVisitNames = raw ? raw.split(',').map(s => s.trim()).filter(Boolean) : [];

  let validatedPlaces = [];

  if (mustVisitNames.length) {
    const btn = qs('#btn-gen');
    btn.disabled = true;
    btn.querySelectorAll('span')[1].textContent = 'Validating places…';
    setMustVisitStatus('⏳ Checking places are in ' + city + '…', 'must-visit-chk');

    try {
      const res = await fetch('/api/validate-places', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({city, state, places: mustVisitNames}),
      }).then(r => r.json());

      btn.disabled = false;
      btn.querySelectorAll('span')[1].textContent = 'Generate My Itinerary';

      if (!res.valid) {
        setMustVisitStatus(
          '⚠ ' + (res.message || 'Some places could not be found in ' + city),
          'must-visit-err'
        );
        return;
      }
      validatedPlaces = res.validated || [];
      const names = validatedPlaces.map(p => p.name).join(', ');
      setMustVisitStatus('✓ Confirmed: ' + names, 'must-visit-ok');
    } catch {
      btn.disabled = false;
      btn.querySelectorAll('span')[1].textContent = 'Generate My Itinerary';
      // Network error — proceed without must-visit
      setMustVisitStatus('', '');
    }
  }

  const payload = {
    city,
    state,
    month:          State.month,
    year:           State.year,
    start_day:      parseInt(qs('#inp-start-day').value) || 1,
    days:           State.days,
    travel_styles:  State.styles,
    interests:      qs('#inp-interests').value.trim(),
    group:          State.group,
    group_ages:     qs('#inp-ages').value.trim(),
    transport_mode: State.transport,
    wake_hour:      State.wakeHour,
    pace:           State.pace,
    dietary:        qs('#inp-dietary').value.trim() || 'none',
    budget_per_meal:State.budget,
    fitness:        State.fitness,
    must_visit:     validatedPlaces,
  };

  Screens.go('screen-loading');
  resetLoadingUI();

  fetch('/api/generate', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload),
  })
  .then(r => r.json())
  .then(({job_id}) => {
    if (!job_id) throw new Error('No job ID returned');
    listenSSE(job_id);
  })
  .catch(err => showError(err.message));
}

/* ── Loading UI helpers ──────────────────────────────────── */
let _pct = 0;

// Keys are matched against lowercase progress messages from the backend.
// Use specific substrings to avoid early false-matches from parallel threads.
const PROGRESS_MAP = {
  'searching lodging':  {el:'lp-hotel',   pct:10},
  'hotel selected':     {el:'lp-hotel',   pct:14},
  'fetching compreh':   {el:'lp-attrs',   pct:22},
  'categoriz':          {el:'lp-attrs',   pct:32},
  'extracting feature': {el:'lp-feat',    pct:40},
  'sentiment':          {el:'lp-feat',    pct:44},
  'distilbert':         {el:'lp-feat',    pct:48},
  'pre-fetched weather':{el:'lp-weather', pct:54},
  'travel matrix':      {el:'lp-route',   pct:62},
  'or-tools':           {el:'lp-route',   pct:70},
  'variety':            {el:'lp-route',   pct:75},
  'energy curve':       {el:'lp-route',   pct:78},
  'meal suggestion':    {el:'lp-meals',   pct:84},
  'enrichment':         {el:'lp-meals',   pct:87},
  'feasibility':        {el:'lp-audit',   pct:94},
  'audit':              {el:'lp-audit',   pct:95},
};

const STEP_FRIENDLY = {
  'lp-hotel':   'Finding your perfect hotel…',
  'lp-attrs':   'Discovering top attractions…',
  'lp-feat':    'Analysing reviews & sentiment…',
  'lp-weather': 'Checking weather forecast…',
  'lp-route':   'Optimising your route…',
  'lp-meals':   'Curating dining experiences…',
  'lp-audit':   'Running final feasibility check…',
};

function resetLoadingUI() {
  _pct = 0;
  updateRing(0);
  qs('#load-pct').textContent = '0%';
  qs('#load-sub').textContent = 'Initialising AI systems…';
  qsa('.lp-step').forEach(el => { el.classList.remove('active','done'); });
}

function updateRing(pct) {
  const circ = 2 * Math.PI * 62;
  const fill = qs('#lr-fill');
  if (fill) fill.style.strokeDashoffset = circ * (1 - pct / 100);
  qs('#load-pct').textContent = Math.round(pct) + '%';
}

function addRingGrad() {
  // Inject SVG gradient into the SVG element once
  const svg = qs('.load-ring');
  if (svg && !qs('#ringGrad')) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
    defs.innerHTML = `
      <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%"   stop-color="#4facfe"/>
        <stop offset="50%"  stop-color="#7c6cf5"/>
        <stop offset="100%" stop-color="#f5576c"/>
      </linearGradient>`;
    svg.insertBefore(defs, svg.firstChild);
  }
}

function handleProgress(msg) {
  const lower = msg.toLowerCase();
  for (const [key, info] of Object.entries(PROGRESS_MAP)) {
    if (lower.includes(key) && info.pct > _pct) {
      _pct = info.pct;
      updateRing(_pct);
      qs('#load-sub').textContent = STEP_FRIENDLY[info.el] || 'Crafting your itinerary…';
      const el = qs('#' + info.el);
      if (el) {
        qsa('.lp-step').forEach(s => {
          if (s.classList.contains('active')) s.classList.replace('active','done');
        });
        el.classList.add('active');
      }
      break;
    }
  }
}

/* ── SSE listener ────────────────────────────────────────── */
function listenSSE(jobId) {
  const es = new EventSource(`/api/stream/${jobId}`);
  es.onmessage = function(e) {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    if (data.type === 'progress') {
      handleProgress(data.msg);
    } else if (data.type === 'done') {
      es.close();
      updateRing(100);
      // Mark all steps done
      qsa('.lp-step').forEach(s => s.classList.replace('active','done') || s.classList.add('done'));
      qs('#load-sub').textContent = 'Itinerary ready!';
      setTimeout(() => {
        renderResults(data.result);
        Screens.go('screen-results');
      }, 800);
    } else if (data.type === 'error') {
      es.close();
      showError(data.msg);
    }
  };
  es.onerror = () => {
    es.close();
    showError('Connection lost. Please try again.');
  };
}

function showError(msg) {
  const wrap = qs('.load-wrap');
  wrap.innerHTML = `
    <div class="error-card">
      <div class="err-icon">⚠️</div>
      <h3 class="err-title">Something went wrong</h3>
      <p class="err-msg">${msg}</p>
      <button class="btn-retry" onclick="App.retry()">↩ Try Again</button>
    </div>
  `;
}

/* ══════════════════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════════════════ */
function renderResults(result) {
  const itinerary = result.itinerary || {};
  const hotel     = result.hotel || null;

  // ── Titles ──────────────────────────────────────────────
  const city = qs('#inp-city').value.trim();
  const stylesLabel = State.styles.map(id => STYLES.find(s => s.id === id)?.name || id).join(' + ');
  qs('#res-title').textContent = `${city} · ${stylesLabel} Trip`;
  qs('#res-sub').textContent   = `${State.days} day${State.days>1?'s':''} · ${MONTHS[State.month-1]} ${State.year}`;

  // ── Hotel banner ─────────────────────────────────────────
  renderHotel(hotel);

  // ── Day tabs & panes ─────────────────────────────────────
  const tabsEl  = qs('#day-tabs');
  const panesEl = qs('#day-panes');
  tabsEl.innerHTML  = '';
  panesEl.innerHTML = '';

  State.dayData = [];   // store per-day data for outfit API

  const days = Object.entries(itinerary);
  days.forEach(([dayKey, dayData], idx) => {
    State.dayData.push(dayData);

    // Tab
    const tab = document.createElement('button');
    tab.className   = 'day-tab' + (idx === 0 ? ' active' : '');
    tab.textContent = dayKey;
    tab.onclick     = () => switchDay(idx);
    tabsEl.appendChild(tab);

    // Pane
    const pane = document.createElement('div');
    pane.className = 'day-pane' + (idx === 0 ? ' active' : '');
    pane.id        = `day-pane-${idx}`;
    pane.innerHTML = buildDayPane(dayData, idx);
    panesEl.appendChild(pane);
  });

  State.activeDay = 0;

  // Outfit gallery trigger
  const existingGallery = qs('#outfit-gallery-section');
  if (existingGallery) existingGallery.remove();
  const gallerySection = document.createElement('div');
  gallerySection.id = 'outfit-gallery-section';
  gallerySection.innerHTML = `
    <div class="outfit-trigger-bar">
      <div class="otb-text">
        <span class="otb-icon">👗</span>
        <div>
          <div class="otb-title">What should I wear?</div>
          <div class="otb-sub">Get AI-generated outfit images for every day of your trip</div>
        </div>
      </div>
      <button class="btn-see-outfits" id="btn-see-outfits" onclick="showOutfitModal()">
        See Outfits ✨
      </button>
    </div>
    <div id="outfit-gallery" style="display:none"></div>`;
  panesEl.insertAdjacentElement('afterend', gallerySection);
}

function switchDay(idx) {
  qsa('.day-tab').forEach((t,i)  => t.classList.toggle('active', i===idx));
  qsa('.day-pane').forEach((p,i) => p.classList.toggle('active', i===idx));
  State.activeDay = idx;
}

function renderHotel(hotel) {
  const el = qs('#hotel-banner');
  if (!hotel || !hotel.name) { el.style.display='none'; return; }
  const prices = ['','$','$$','$$$','$$$$'];
  const price  = prices[hotel.price_level || 2] || '$$';
  const rating = parseFloat(hotel.rating || 0);
  el.style.display = '';
  el.innerHTML = `
    <div class="hb-icon">🏨</div>
    <div class="hb-info">
      <div class="hb-name">${hotel.name}</div>
      <div class="hb-addr">${hotel.formatted_address || ''}</div>
      <div class="hb-meta">
        <div class="hb-stars">${starsHtml(rating)}</div>
        <span style="font-size:.75rem;color:var(--txt3)">${rating}</span>
        <div class="hb-price">${price}</div>
      </div>
      ${hotel.reason ? `<div class="hb-reason">"${hotel.reason}"</div>` : ''}
    </div>`;
}

/* ── Day pane builder ────────────────────────────────────── */
function buildDayPane(dayData, dayIdx) {
  const events  = dayData.events || [];
  const weather = dayData.weather_summary || '';
  const note    = dayData.day_note || '';
  const tip     = dayData.day_tip  || '';

  let html = '<div class="day-meta">';
  if (weather) html += `<div class="day-weather">⛅ ${weather}</div>`;
  if (note)    html += `<div class="day-note-badge">💡 ${note}</div>`;
  html += '</div>';
  if (tip)     html += `<div class="day-tip-bar"><strong>Tip:</strong> ${tip}</div>`;

  html += '<div class="timeline">';
  events.forEach(e => { html += buildEventRow(e); });
  html += '</div>';

  return html;
}

/* ── Event row builder ───────────────────────────────────── */
function buildEventRow(e) {
  const type    = e.event_type || 'attraction';
  const arrMins = e.arrival_min;
  const depMins = e.departure_min;
  const dur     = fmtDur((depMins || 0) - (arrMins || 0));
  const timeStr = minsToTime(arrMins);

  // Dot color per event type
  const dotColor = {
    meal:         '#ffc857',
    hotel_return: '#00f2fe',
    style_event:  '#7c6cf5',
    free_time:    '#38ef7d',
    attraction:   '#4facfe',
  }[type] || '#4facfe';

  const cardHtml = buildEventCard(e, dur, type);

  return `
    <div class="tl-item">
      <div class="tl-time">${timeStr}</div>
      <div class="tl-spine">
        <div class="tl-dot" style="background:${dotColor}; --dot-color:${dotColor}88;"></div>
        <div class="tl-line"></div>
      </div>
      <div class="tl-card ${type}">
        ${cardHtml}
      </div>
    </div>`;
}

function buildEventCard(e, dur, type) {
  if (type === 'hotel_return') {
    return `
      <div class="tc-head">
        <span class="tc-icon">🏨</span>
        <span class="tc-name">${e.name || 'Return to Hotel'}</span>
        <span class="tc-dur">${e.travel_mins || 0} min ${e.mode_label || 'drive'}</span>
      </div>`;
  }

  if (type === 'meal') {
    const mealIcons = {breakfast:'☕', lunch:'🍽️', dinner:'🍴', morning_snack:'🥐', afternoon_snack:'🧃'};
    const icon      = e.icon || mealIcons[e.meal_type] || '🍴';
    const mealLabel = (e.meal_type || '').replace('_', ' ');
    return `
      <div class="tc-head">
        <span class="tc-icon">${icon}</span>
        <span class="tc-name">${e.name || mealLabel}</span>
        ${dur ? `<span class="tc-dur">${dur}</span>` : ''}
      </div>
      ${e.suggestion        ? `<div class="tc-body">→ ${e.suggestion}</div>`                   : ''}
      ${e.restaurant_address? `<div class="tc-address">📍 ${e.restaurant_address}</div>`      : ''}
      ${e.tip               ? `<div class="tc-tip">💡 ${e.tip}</div>`                          : ''}
      ${e.travel_to_next    ? `<div class="tc-travel">→ ${e.travel_to_next} to next stop</div>`: ''}`;
  }

  if (type === 'free_time') {
    const icon = e.icon || '🗺️';
    return `
      <div class="tc-head">
        <span class="tc-icon">${icon}</span>
        <span class="tc-name">${e.name}</span>
        ${dur ? `<span class="tc-dur">${dur}</span>` : ''}
      </div>
      <div class="tc-body" style="opacity:0.7">Unstructured time — wander, rest, or discover something spontaneous.</div>`;
  }

  if (type === 'style_event') {
    const seIcons = {food_tasting:'🛒', golden_hour:'📷', evening_out:'🎶', wellness_break:'🧘'};
    const icon    = e.icon || seIcons[e.style_event_type] || '✨';
    return `
      <div class="tc-head">
        <span class="tc-icon">${icon}</span>
        <span class="tc-name">${e.name}</span>
        ${dur ? `<span class="tc-dur">${dur}</span>` : ''}
      </div>
      ${e.suggestion ? `<div class="tc-body">→ ${e.suggestion}</div>` : ''}
      ${e.tip        ? `<div class="tc-tip">💡 ${e.tip}</div>`        : ''}`;
  }

  // Attraction (default)
  const rating  = parseFloat(e.google_rating || 0);
  const energy  = e.energy_level || '';
  const atype   = (e.attraction_type || '').replace(/_/g,' ');
  const energyClass = { high:'badge-energy-high', medium:'badge-energy-medium', low:'badge-energy-low' }[energy] || '';

  // user_ratings_total — format as "12.4K" or "1.2M"
  const nRatings = e.user_ratings_total || 0;
  const nFmt = nRatings >= 1e6 ? `${(nRatings/1e6).toFixed(1)}M`
             : nRatings >= 1e3 ? `${(nRatings/1e3).toFixed(1)}K`
             : nRatings > 0    ? `${nRatings}` : '';

  // price_level — 1=$  2=$$  3=$$$  4=$$$$  (null/0 = not applicable)
  const pl = e.price_level;
  const priceHtml = (pl && pl > 0) ? `<span class="badge badge-price">${'$'.repeat(pl)}</span>` : '';

  // queue_buffer — show only if meaningful (> 10 min)
  const qbuf = parseInt(e.queue_buffer || 0);
  const queueHtml = qbuf > 10 ? `<div class="tc-queue">⏱ ~${qbuf} min queue</div>` : '';

  // skip_if — show only when it's a real condition, not "none"
  const skipIf = (e.skip_if || '').toLowerCase();
  const skipHtml = (skipIf && skipIf !== 'none')
    ? `<div class="tc-skipif">⚠ Skip if: ${e.skip_if}</div>` : '';

  return `
    <div class="tc-head">
      <span class="tc-icon">${attrIcon(atype)}</span>
      <span class="tc-name">${e.name}</span>
      ${dur ? `<span class="tc-dur">${dur}</span>` : ''}
    </div>
    <div class="tc-badges">
      ${atype    ? `<span class="badge badge-type">${atype}</span>` : ''}
      ${energy   ? `<span class="badge ${energyClass}">${energy} energy</span>` : ''}
      ${priceHtml}
    </div>
    ${rating > 0 ? `<div style="margin-bottom:4px">${starsHtml(rating)}<span class="tc-rating">${rating}${nFmt ? ` · ${nFmt} reviews` : ''}</span></div>` : ''}
    ${e.highlights  ? `<div class="tc-highlight">${e.highlights}</div>` : ''}
    ${queueHtml}
    ${skipHtml}
    ${e.travel_to_next ? `<div class="tc-travel">→ ${e.travel_to_next}</div>` : ''}`;
}

/* ── Icon map for attraction types ──────────────────────── */
function attrIcon(type) {
  const map = {
    outdoor:'🌳', indoor:'🏢', cultural:'🏛️', hiking:'⛰️',
    theme_park:'🎢', beach:'🏖️', nightlife:'🎵', food:'🍜',
    wellness:'🌿', museum:'🏛️', park:'🌳', gallery:'🎨',
    zoo:'🦁', aquarium:'🐠', stadium:'🏟️',
  };
  return map[type] || '📍';
}

/* ══════════════════════════════════════════════════════════
   CSS KEYFRAME INJECTION (shake animation)
══════════════════════════════════════════════════════════ */
(function injectKeyframes() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes shakeField {
      0%,100%{transform:translateX(0)}
      20%{transform:translateX(-8px)}
      40%{transform:translateX(8px)}
      60%{transform:translateX(-5px)}
      80%{transform:translateX(5px)}
    }
    @keyframes shake {
      0%,100%{transform:translateX(0)}
      20%{transform:translateX(-6px)}
      40%{transform:translateX(6px)}
      60%{transform:translateX(-4px)}
      80%{transform:translateX(4px)}
    }
  `;
  document.head.appendChild(style);
})();

/* ══════════════════════════════════════════════════════════
   OUTFIT ENGINE — IMAGE GALLERY
══════════════════════════════════════════════════════════ */
function showOutfitModal() {
  // Prevent double-fire while modal is open
  const btn = qs('#btn-see-outfits');
  if (btn) btn.disabled = true;

  // Remove any existing modal
  const old = qs('#outfit-modal');
  if (old) old.remove();

  const modal = document.createElement('div');
  modal.id = 'outfit-modal';
  modal.className = 'outfit-modal-overlay';
  modal.innerHTML = `
    <div class="outfit-modal">
      <div class="om-title">👗 Who are you dressing?</div>
      <div class="om-sub">We'll generate outfit images tailored to each day of your trip</div>
      <div class="om-btns">
        <button class="om-btn" onclick="startOutfitGeneration('male')">
          <span class="om-btn-icon">👔</span>
          <span>Male</span>
        </button>
        <button class="om-btn" onclick="startOutfitGeneration('female')">
          <span class="om-btn-icon">👗</span>
          <span>Female</span>
        </button>
      </div>
      <button class="om-close" onclick="qs('#outfit-modal').remove(); const b=qs('#btn-see-outfits'); if(b) b.disabled=false;">✕</button>
    </div>`;
  document.body.appendChild(modal);
}

async function startOutfitGeneration(gender) {
  const modal = qs('#outfit-modal');
  if (modal) modal.remove();

  const gallery = qs('#outfit-gallery');
  const triggerBar = qs('.outfit-trigger-bar');
  if (!gallery) return;

  gallery.style.display = 'block';
  gallery.innerHTML = `
    <div class="outfit-gen-loading">
      <div class="ogl-spinner">✨</div>
      <div class="ogl-title">Generating outfit images…</div>
      <div class="ogl-sub">Firing all AI calls in parallel — usually takes 10–20 seconds</div>
    </div>`;
  if (triggerBar) triggerBar.style.display = 'none';

  const days = (State.dayData || []).map((d, i) => ({
    day_key:         `Day ${i + 1}`,
    day_idx:         i,
    events:          d.events || [],
    weather_summary: d.weather_summary || '',
  }));

  try {
    const resp = await fetch('/api/outfit-images', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ gender, days, month: State.month, city: qs('#inp-city').value.trim() }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    gallery.innerHTML = buildOutfitGallery(data.outfits || [], gender);
  } catch (err) {
    gallery.innerHTML = `
      <div class="outfit-gen-error">
        ⚠️ ${err.message || 'Image generation failed.'}
        <button onclick="resetOutfitTrigger()">Try again</button>
      </div>`;
    if (triggerBar) triggerBar.style.display = '';
  }
}

function resetOutfitTrigger() {
  const gallery = qs('#outfit-gallery');
  if (gallery) { gallery.style.display = 'none'; gallery.innerHTML = ''; }
  const triggerBar = qs('.outfit-trigger-bar');
  if (triggerBar) triggerBar.style.display = '';
  const btn = qs('#btn-see-outfits');
  if (btn) btn.disabled = false;
}

function buildOutfitGallery(outfits, gender) {
  const genderIcon = gender === 'female' ? '👗' : '👔';
  let html = `
    <div class="og-gallery-header">
      <span>${genderIcon} Your Trip Outfits</span>
      <button class="og-rerun" onclick="showOutfitModal()">Regenerate ↺</button>
    </div>`;

  outfits.forEach(day => {
    html += `<div class="og-day-block"><div class="og-day-label">${day.day_key}</div><div class="og-slots">`;
    ['daytime', 'evening'].forEach(slot => {
      const s    = day[slot];
      const icon = slot === 'daytime' ? '☀️' : '🌙';
      const lbl  = slot === 'daytime' ? 'Daytime' : 'Evening';
      if (!s) return;
      const pieces = (s.description || s.items || []).map(i => `<span class="og-item">${i}</span>`).join('');
      const fname  = s.image_url ? s.image_url.split('/').pop() : 'outfit.jpg';
      html += `
        <div class="og-slot-card">
          <div class="ogsc-label">${icon} ${lbl}</div>
          <div class="ogsc-category">${s.category || ''} · <em>${s.style_name || ''}</em></div>
          ${s.image_url
            ? `<div class="ogsc-img-wrap">
                <img src="${s.image_url}" alt="${lbl} outfit" class="ogsc-img" loading="lazy"/>
               </div>
               <a href="${s.image_url}" download="${fname}" class="btn-dl">⬇ Download</a>`
            : `<div class="ogsc-no-img">Image unavailable</div>`}
          ${pieces ? `<div class="ogsc-items">${pieces}</div>` : ''}
        </div>`;
    });
    html += '</div></div>';
  });

  return html;
}

/* ══════════════════════════════════════════════════════════
   BOOT
══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  buildMonthGrid();
  buildYearPicker();
  buildStyleGrid();
  bindToggle('pace-toggle',  'pace');
  bindToggle('fit-toggle',   'fitness');
  bindToggle('trans-toggle', 'transport');
  bindGroupToggle();
  bindBudget();
  bindSliders();
  bindStepButtons();
  addRingGrad();

  // Hero scroll hint fade on scroll
  qs('#screen-hero').addEventListener('scroll', function() {
    const hint = qs('.hero-scroll-hint');
    if (hint) hint.style.opacity = Math.max(0, 1 - this.scrollTop / 80);
  });
});
