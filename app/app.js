/* ════════════════════════════════════════════════════════
   Rawalpindi Dengue Alert — App Logic
   Renders forecast data, map, chart, and auto-refresh
   ════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ──────────────────────────────────────────
const ALERT_COLORS = {
  Green:  '#059669',
  Yellow: '#d97706',
  Orange: '#ea580c',
  Red:    '#dc2626',
};

const ALERT_PULSE = {
  Green:  'rgba(5,150,105,0.2)',
  Yellow: 'rgba(217,119,6,0.2)',
  Orange: 'rgba(234,88,12,0.25)',
  Red:    'rgba(220,38,38,0.28)',
};

const REFRESH_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes
const DATA_PATH_FORECAST  = 'data/latest_forecast.json';
const DATA_PATH_GEOJSON   = 'data/rawalpindi_uc_forecast.geojson';

// City viewport bounds for Rawalpindi
const CITY_BOUNDS = { south: 33.48, west: 72.94, north: 33.67, east: 73.16 };

// ── Shared state ──────────────────────────────────────
let _chartInstance = null;
let _mapInstance   = null;
let _geojsonLayer  = null;
let _refreshTimer  = null;
let _lastForecast  = null;

// ── Helpers ────────────────────────────────────────────
function fmt(v, digits = 1) {
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function alertColor(alert) {
  return ALERT_COLORS[alert] || '#4d6a8a';
}

function alertPulse(alert) {
  return ALERT_PULSE[alert] || 'rgba(100,120,140,0.15)';
}

function isoWeekRange(year, week) {
  // Get the Monday of ISO week
  const simple = new Date(Date.UTC(year, 0, 1 + (week - 1) * 7));
  const day = simple.getUTCDay() || 7;
  const monday = new Date(simple);
  monday.setUTCDate(simple.getUTCDate() - day + 1);
  const sunday = new Date(monday);
  sunday.setUTCDate(monday.getUTCDate() + 6);
  const opts = { month: 'short', day: 'numeric' };
  return `${monday.toLocaleDateString(undefined, opts)} – ${sunday.toLocaleDateString(undefined, opts)}`;
}

function isoWeekLabel(year, week) {
  const simple = new Date(Date.UTC(year, 0, 1 + (week - 1) * 7));
  const day = simple.getUTCDay() || 7;
  const monday = new Date(simple);
  monday.setUTCDate(simple.getUTCDate() - day + 1);
  return monday.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

async function loadJson(path) {
  const res = await fetch(path + '?t=' + Date.now()); // cache-bust
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

function el(id) { return document.getElementById(id); }

// Animate a number counting up from 0 to target
function animateCount(element, target, duration = 600, decimals = 0) {
  const start = performance.now();
  const from = 0;
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (target - from) * eased;
    element.textContent = fmt(current, decimals);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Render: Header / Summary ───────────────────────────
function renderSummary(data) {
  const first = data.weekly_forecasts[0];
  const topUcs = data.top_ucs || [];

  // Weather pill
  const isLive = (data.weather_status || '').toLowerCase().includes('fresh');
  const dot = el('weatherDot');
  el('weatherStatus').textContent = isLive ? 'Live Weather' : 'Fallback Weather';
  dot.className = 'pill-dot ' + (isLive ? 'live' : 'warn');

  // Timestamp pill
  const ts = new Date(data.generated_at);
  el('generatedAt').textContent = `Updated ${ts.toLocaleDateString(undefined, { month:'short', day:'numeric' })} ${ts.toLocaleTimeString(undefined, { hour:'2-digit', minute:'2-digit' })}`;

  // Forecast period & cases
  el('forecastPeriodLabel').textContent = isoWeekRange(first.year, first.week);

  const casesEl = el('expectedCases');
  const targetCases = Number(first.expected_cases);
  animateCount(casesEl, targetCases, 700, 1);

  el('forecastRange').textContent = `${fmt(first.lower_cases, 1)} – ${fmt(first.upper_cases, 1)} cases range`;

  // Alert badge
  const redRows    = topUcs.filter(r => r.alert === 'Red');
  const orangeRows = topUcs.filter(r => r.alert === 'Orange');
  const alertRows  = topUcs.filter(r => ['Red','Orange','Yellow'].includes(r.alert));

  const badge    = el('alertBadge');
  const badgeLvl = el('badgeLevel');

  if (redRows.length > 0) {
    badge.className    = 'forecast-badge red';
    badgeLvl.textContent = `⚠ RED ALERT`;
  } else if (orangeRows.length > 0) {
    badge.className    = 'forecast-badge orange';
    badgeLvl.textContent = `▲ HIGH RISK`;
  } else if (alertRows.length > 0) {
    badge.className    = 'forecast-badge yellow';
    badgeLvl.textContent = `◆ WATCH`;
  } else {
    badge.className    = 'forecast-badge green';
    badgeLvl.textContent = `✓ LOW RISK`;
  }

  // KPIs
  const alertUcEl = el('alertUcCount');
  animateCount(alertUcEl, alertRows.length, 600, 0);

  const redUcEl = el('redUcCount');
  animateCount(redUcEl, redRows.length, 600, 0);

  // Model info
  const model = data.selected_model || {};
  el('modelR2').textContent = model.rolling_origin_mean_r2 != null
    ? fmt(model.rolling_origin_mean_r2, 3)
    : '—';
  el('modelName').textContent = model.name || 'Gradient Boosting';

  // Operational note
  el('surveillanceStatus').textContent =
    `${data.weather_status || ''}. ${data.surveillance_status || ''}`;
}

// ── Render: 4-Week Forecast Chart ─────────────────────
function renderChart(weeks) {
  const labels = weeks.map(w => isoWeekLabel(w.year, w.week));
  const expected = weeks.map(w => w.expected_cases);
  const lower    = weeks.map(w => w.lower_cases);
  const upper    = weeks.map(w => w.upper_cases);

  // Error bar as +/- from expected
  const errorPlus  = upper.map((u, i) => u - expected[i]);
  const errorMinus = expected.map((e, i) => e - lower[i]);

  if (_chartInstance) {
    _chartInstance.data.labels = labels;
    _chartInstance.data.datasets[0].data = expected;
    _chartInstance.data.datasets[1].data = upper;
    _chartInstance.data.datasets[2].data = lower;
    _chartInstance.update('active');
    return;
  }

  const ctx = el('forecastChart').getContext('2d');

  // Gradient fill — blue accent matching light theme
  const grad = ctx.createLinearGradient(0, 0, 0, 166);
  grad.addColorStop(0, 'rgba(37,99,235,0.80)');
  grad.addColorStop(1, 'rgba(14,165,233,0.45)');

  _chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Expected',
          data: expected,
          backgroundColor: grad,
          borderColor: 'rgba(37,99,235,0.9)',
          borderWidth: 1,
          borderRadius: 5,
          borderSkipped: false,
          order: 2,
        },
        {
          label: 'Upper',
          data: upper,
          type: 'line',
          fill: '+1',
          borderColor: 'rgba(37,99,235,0.22)',
          backgroundColor: 'rgba(37,99,235,0.06)',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.4,
          order: 1,
        },
        {
          label: 'Lower',
          data: lower,
          type: 'line',
          fill: false,
          borderColor: 'rgba(34,211,238,0.28)',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.4,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      animation: { duration: 700, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.98)',
          titleColor: '#1a2740',
          bodyColor: '#4a6480',
          borderColor: 'rgba(148,174,208,0.4)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 0) return ` Expected: ${fmt(ctx.parsed.y, 1)} cases`;
              if (ctx.datasetIndex === 1) return ` Upper CI: ${fmt(ctx.parsed.y, 1)}`;
              return ` Lower CI: ${fmt(ctx.parsed.y, 1)}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(148,174,208,0.18)' },
          ticks: { color: '#8aa0bb', font: { size: 10 } },
          border: { color: 'rgba(148,174,208,0.25)' },
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(148,174,208,0.18)' },
          ticks: {
            color: '#8aa0bb',
            font: { size: 10 },
            maxTicksLimit: 5,
            callback: v => (Number.isInteger(v) ? v : ''),
          },
          border: { color: 'rgba(34,211,238,0.08)' },
        },
      },
    },
  });
}

// ── Render: UC List ───────────────────────────────────
function renderUcList(rows) {
  const container = el('topUcRows');
  const filtered = rows
    .filter(r => r.tehsil === 'Rawalpindi Tehsil')
    .slice(0, 12);

  container.innerHTML = filtered
    .map((row, i) => {
      const level = row.alert.toLowerCase();
      const color = alertColor(row.alert);
      const delay = i * 40; // stagger animation
      return `
        <div class="uc-row is-${level}" role="listitem" style="animation-delay:${delay}ms">
          <div class="uc-rank" style="background:${color}">${i + 1}</div>
          <div class="uc-info">
            <strong>${row.uc}</strong>
            <span>${fmt(row.expected_cases, 2)} exp · ${fmt(row.historical_cases, 0)} hist</span>
          </div>
          <span class="uc-badge badge-${level}">${row.alert}</span>
        </div>`;
    })
    .join('');
}

// ── Render: Map ───────────────────────────────────────
function renderMap(geojson) {
  if (_mapInstance) {
    // Update existing layers on auto-refresh without resetting map view
    if (_geojsonLayer) {
      _geojsonLayer.remove();
    }
  } else {
    _mapInstance = L.map('map', {
      center: [33.575, 73.045],
      zoom: 12,
      zoomControl: false,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      touchZoom: true,
      dragging: true,
      minZoom: 10,
      maxZoom: 18,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(_mapInstance);
    L.control.scale({ position: 'bottomright', imperial: false, maxWidth: 90 }).addTo(_mapInstance);

    // CartoDB Positron — clean light tiles, perfect for a light theme
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org">OSM</a> &copy; <a href="https://carto.com">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(_mapInstance);
  }

  // Filter to Rawalpindi Tehsil + within city bounds
  function featureInBounds(f) {
    const coords = [];
    const g = f.geometry;
    if (g.type === 'Polygon') g.coordinates.forEach(r => r.forEach(([lon,lat]) => coords.push({lon,lat})));
    else if (g.type === 'MultiPolygon') g.coordinates.forEach(p => p.forEach(r => r.forEach(([lon,lat]) => coords.push({lon,lat}))));
    if (!coords.length) return false;
    const w = Math.min(...coords.map(p => p.lon));
    const e = Math.max(...coords.map(p => p.lon));
    const s = Math.min(...coords.map(p => p.lat));
    const n = Math.max(...coords.map(p => p.lat));
    return e >= CITY_BOUNDS.west && w <= CITY_BOUNDS.east &&
           n >= CITY_BOUNDS.south && s <= CITY_BOUNDS.north;
  }

  const visibleFeatures = geojson.features.filter(
    f => f.properties.tehsil === 'Rawalpindi Tehsil' && featureInBounds(f)
  );

  const visibleGeoJson = { ...geojson, features: visibleFeatures };

  function featureStyle(feature) {
    const a = feature.properties.alert;
    const color = alertColor(a);
    if (a === 'Green') {
      return { color: '#a7f3d0', weight: 1, fillColor: '#d1fae5', fillOpacity: 0.55, opacity: 1 };
    }
    const fills    = { Yellow: '#fef3c7', Orange: '#ffedd5', Red: '#fee2e2' };
    const strokes  = { Yellow: '#fbbf24', Orange: '#fb923c', Red: '#f87171' };
    const opacities= { Yellow: 0.72, Orange: 0.78, Red: 0.82 };
    const weights  = { Yellow: 1.5, Orange: 2.0, Red: 2.5 };
    return {
      color:       strokes[a]   || color,
      weight:      weights[a]   || 1.5,
      fillColor:   fills[a]     || color,
      fillOpacity: opacities[a] || 0.5,
      opacity: 1,
    };
  }

  _geojsonLayer = L.geoJSON(visibleGeoJson, {
    style: featureStyle,
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      const color = alertColor(p.alert);
      layer.bindPopup(`
        <strong>${p.uc}</strong><br>
        ${p.tehsil}<br>
        Alert: <strong style="color:${color}">${p.alert}</strong><br>
        Expected: <strong>${fmt(p.expected_cases, 2)}</strong> reported cases<br>
        Historical burden: ${fmt(p.historical_cases, 0)} total cases
      `);
      layer.on('mouseover', () => layer.setStyle({ weight: 3, color: '#1e40af', fillOpacity: 0.92 }));
      layer.on('mouseout', () => _geojsonLayer.resetStyle(layer));
    },
  }).addTo(_mapInstance);

  // Add numbered pins for top Red/Orange UCs
  const ranked = visibleFeatures
    .filter(f => ['Red','Orange'].includes(f.properties.alert))
    .sort((a,b) => {
      const order = { Red:2, Orange:1 };
      return (order[b.properties.alert]||0) - (order[a.properties.alert]||0) ||
             (b.properties.expected_cases||0) - (a.properties.expected_cases||0);
    })
    .slice(0, 10);

  ranked.forEach((feature, i) => {
    const p = feature.properties;
    const color = alertColor(p.alert);
    const pulse = alertPulse(p.alert);
    const bounds = L.geoJSON(feature).getBounds();
    const center = bounds.getCenter();

    const icon = L.divIcon({
      className: '',
      html: `<div class="alert-pin" style="background:${color};--pulse-color:${pulse}" aria-label="Risk rank ${i+1}: ${p.uc}">${i+1}</div>`,
      iconSize: [30, 30],
      iconAnchor: [15, 15],
    });

    L.marker(center, { icon, zIndexOffset: 1000 })
      .bindPopup(`
        <strong>${i+1}. ${p.uc}</strong><br>
        Alert: <strong style="color:${color}">${p.alert}</strong><br>
        Expected: <strong>${fmt(p.expected_cases, 2)}</strong> reported cases
      `)
      .bindTooltip(`${p.uc} · ${fmt(p.expected_cases, 1)} exp`, {
        className: 'uc-map-tooltip',
        direction: 'top',
        offset: [0, -18],
      })
      .addTo(_mapInstance);
  });

  // ── Smart initial fit: frame Rawalpindi city perfectly on every screen ──
  // Only runs on first load — auto-refresh updates layers without resetting the view
  try {
    if (!_mapInstance._rawalpindiInitialFitDone && visibleFeatures.length > 0) {
      _mapInstance._rawalpindiInitialFitDone = true;

      const mapEl = document.getElementById('map');
      const mapW  = mapEl ? mapEl.offsetWidth : window.innerWidth;

      // Responsive padding: comfortable breathing room on all screen sizes
      let padTop, padRight, padBottom, padLeft;
      if (mapW >= 1100) {
        // Desktop: wide map panel, generous padding
        padTop = 44; padRight = 52; padBottom = 68; padLeft = 44;
      } else if (mapW >= 640) {
        // Tablet: full-width map above sidebar
        padTop = 30; padRight = 44; padBottom = 60; padLeft = 44;
      } else {
        // Mobile: tight but usable
        padTop = 14; padRight = 14; padBottom = 52; padLeft = 14;
      }

      _mapInstance.fitBounds(_geojsonLayer.getBounds(), {
        paddingTopLeft:     [padLeft,  padTop],
        paddingBottomRight: [padRight, padBottom],
        animate: false,
        maxZoom: 13,
      });

      // Clamp zoom: always show the whole city (≥11) but never too close (≤13)
      const z = _mapInstance.getZoom();
      if (z < 11) _mapInstance.setZoom(11, { animate: false });
      if (z > 13) _mapInstance.setZoom(13, { animate: false });

      // Restrict pan so user stays near Rawalpindi
      _mapInstance.setMaxBounds(_geojsonLayer.getBounds().pad(0.4));
    }
  } catch (_) { /* ignore if layer is empty */ }
}


// ── Refresh Ring ──────────────────────────────────────
function startRefreshRing() {
  const ring = el('refreshRing');
  if (!ring) return;
  ring.addEventListener('click', () => {
    ring.classList.add('spinning');
    loadAndRender().finally(() => {
      setTimeout(() => ring.classList.remove('spinning'), 600);
    });
  });
}

// ── Auto-Refresh ──────────────────────────────────────
function scheduleAutoRefresh() {
  if (_refreshTimer) clearInterval(_refreshTimer);
  _refreshTimer = setInterval(() => {
    loadAndRender().catch(console.warn);
  }, REFRESH_INTERVAL_MS);
}

// ── Main Load ─────────────────────────────────────────
async function loadAndRender() {
  const [forecast, geojson] = await Promise.all([
    loadJson(DATA_PATH_FORECAST),
    loadJson(DATA_PATH_GEOJSON),
  ]);
  _lastForecast = forecast;
  renderSummary(forecast);
  renderChart(forecast.weekly_forecasts);
  renderUcList(forecast.top_ucs);
  renderMap(geojson);
}

// ── Bootstrap ─────────────────────────────────────────
(async function main() {
  startRefreshRing();

  try {
    await loadAndRender();
    scheduleAutoRefresh();
  } catch (err) {
    console.error('Dengue Alert App failed to load:', err);
    document.getElementById('appMain').innerHTML = `
      <div style="grid-column:1/-1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:40px;text-align:center;">
        <div style="font-size:48px">⚠️</div>
        <h2 style="color:#ef4444;font-family:'Barlow Condensed',sans-serif;font-size:24px">Could not load forecast data</h2>
        <p style="color:#8da3c0;max-width:400px;line-height:1.6">
          The forecast data files could not be loaded.<br>
          Run <code style="color:#f97316">python src/build_alert_outputs.py</code> first,
          then serve from the <code style="color:#f97316">app/</code> folder.
        </p>
        <p style="color:#4d6a8a;font-size:12px">${err.message}</p>
      </div>`;
  }
})();
