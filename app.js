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

const ALERT_ORDER = { Red: 4, Orange: 3, Yellow: 2, Green: 1 };
const MAPPED_ALERTS = new Set(['Yellow', 'Orange', 'Red']);

const REFRESH_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes
const DATA_PATH_FORECAST  = 'data/latest_forecast.json';
const DATA_PATH_GEOJSON   = 'data/rawalpindi_uc_forecast.geojson';

// City viewport bounds for Rawalpindi
const CITY_BOUNDS = { south: 33.48, west: 72.94, north: 33.67, east: 73.16 };

// ── Shared state ──────────────────────────────────────
let _chartInstance = null;
let _mapInstance   = null;
let _geojsonLayer  = null;
let _markerLayer   = null;
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

function isMappedAlert(alert) {
  return MAPPED_ALERTS.has(alert);
}

function sortAlertRows(a, b) {
  return (ALERT_ORDER[b.alert] || 0) - (ALERT_ORDER[a.alert] || 0) ||
         (b.expected_cases || 0) - (a.expected_cases || 0);
}

// Monday of an ISO week. Anchored on Jan 4, which is always in ISO week 1 —
// anchoring on Jan 1 instead is off by a week in years starting Fri or Sat.
function isoWeekMonday(year, week) {
  const jan4 = new Date(Date.UTC(year, 0, 4));
  const week1Monday = new Date(jan4);
  week1Monday.setUTCDate(jan4.getUTCDate() - (jan4.getUTCDay() || 7) + 1);
  const monday = new Date(week1Monday);
  monday.setUTCDate(week1Monday.getUTCDate() + (week - 1) * 7);
  return monday;
}

function isoWeekRange(year, week) {
  const monday = isoWeekMonday(year, week);
  const sunday = new Date(monday);
  sunday.setUTCDate(monday.getUTCDate() + 6);
  const opts = { month: 'short', day: 'numeric', timeZone: 'UTC' };
  return `${monday.toLocaleDateString(undefined, opts)} – ${sunday.toLocaleDateString(undefined, opts)}`;
}

function isoWeekLabel(year, week) {
  return isoWeekMonday(year, week)
    .toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
}

// End of the forecast week, used to tell "stale data" from "expired forecast".
function isoWeekEnd(year, week) {
  const end = isoWeekMonday(year, week);
  end.setUTCDate(end.getUTCDate() + 7);
  return end;
}

async function loadJson(path) {
  const res = await fetch(path + '?t=' + Date.now()); // cache-bust
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

function el(id) { return document.getElementById(id); }

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

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
  const counts = data.alert_counts || {};
  const now = Date.now();

  // ── Freshness. The old pill read "Live Weather" straight from the payload
  // string, so a stale file kept claiming to be live. Age decides now.
  const ts = new Date(data.generated_at);
  const ageHours = (now - ts.getTime()) / 3.6e6;
  const expired  = now > isoWeekEnd(first.year, first.week).getTime();
  const weatherLive = (data.weather_status || '').toLowerCase().includes('fresh');

  const dot = el('weatherDot');
  let freshness;
  if (expired) {
    freshness = { text: 'Forecast expired', cls: 'stale' };
  } else if (ageHours > 168) {
    freshness = { text: 'Data over a week old', cls: 'stale' };
  } else if (ageHours > 48) {
    freshness = { text: `Data ${Math.floor(ageHours / 24)} days old`, cls: 'warn' };
  } else {
    freshness = { text: weatherLive ? 'Live Weather' : 'Fallback Weather', cls: weatherLive ? 'live' : 'warn' };
  }
  el('weatherStatus').textContent = freshness.text;
  dot.className = 'pill-dot ' + freshness.cls;
  el('weatherPill').classList.toggle('is-stale', freshness.cls === 'stale');

  el('generatedAt').textContent = `Updated ${ts.toLocaleDateString(undefined, { month:'short', day:'numeric' })} ${ts.toLocaleTimeString(undefined, { hour:'2-digit', minute:'2-digit' })}`;

  // Forecast period & cases
  el('forecastPeriodLabel').textContent = isoWeekRange(first.year, first.week);

  const casesEl = el('expectedCases');
  animateCount(casesEl, Number(first.expected_cases), 700, 1);

  el('forecastRange').textContent = `${fmt(first.lower_cases, 1)} – ${fmt(first.upper_cases, 1)} cases range`;

  // ── Alert badge, driven by the forecast across ALL UCs. Previously this read
  // top_ucs, whose Red entries came from historical burden, so the banner said
  // RED ALERT every day of the year regardless of the forecast.
  const nRed    = Number(counts.Red    || 0);
  const nOrange = Number(counts.Orange || 0);
  const nYellow = Number(counts.Yellow || 0);

  const badge    = el('alertBadge');
  const badgeLvl = el('badgeLevel');

  if (nRed > 0) {
    badge.className = 'forecast-badge red';
    badgeLvl.textContent = '⚠ RED ALERT';
  } else if (nOrange > 0) {
    badge.className = 'forecast-badge orange';
    badgeLvl.textContent = '▲ HIGH RISK';
  } else if (nYellow > 0) {
    badge.className = 'forecast-badge yellow';
    badgeLvl.textContent = '◆ WATCH';
  } else {
    badge.className = 'forecast-badge green';
    badgeLvl.textContent = '✓ LOW RISK';
  }

  animateCount(el('alertUcCount'), nRed + nOrange + nYellow, 600, 0);
  animateCount(el('redUcCount'), nRed, 600, 0);

  // ── Active model. Two engines: Study 1's XGBoost when case counts exist,
  // the weather-only model when they don't. Their accuracy differs by a lot,
  // so the page always shows which one produced these numbers.
  const model = data.selected_model || {};
  const weatherOnly = model.engine === 'weather_only';
  el('modelR2').textContent = model.r2 != null ? fmt(model.r2, 2) : '—';
  el('modelName').textContent = model.name || 'XGBoost';
  const extEl = el('modelExternal');
  if (extEl) {
    extEl.textContent = weatherOnly
      ? `weather-only · corr ${fmt(model.correlation, 2)}`
      : (model.mape != null ? `2025 held-out · MAPE ${fmt(model.mape, 1)}%` : '—');
    extEl.className = 'kpi-foot' + (weatherOnly || (model.r2 != null && model.r2 < 0) ? ' is-negative' : '');
  }

  // ── Case-data provenance: the single most important honesty signal.
  const cutoffEl = el('casesThrough');
  if (cutoffEl) {
    const through = data.cases_through;
    const basis = (first.cases_basis || '').toLowerCase();
    if (through) {
      cutoffEl.textContent = `Cases observed through ${through.year} wk ${through.week}`;
      cutoffEl.className = 'data-basis basis-ok';
    } else if (weatherOnly) {
      cutoffEl.textContent = 'Weather-only forecast — seasonal risk, not a case count';
      cutoffEl.className = 'data-basis basis-weak';
    } else {
      cutoffEl.textContent = 'No observed case data — seasonal baseline only';
      cutoffEl.className = 'data-basis basis-weak';
    }
  }

  el('surveillanceStatus').textContent =
    `${data.weather_status || ''}. ${data.surveillance_status || ''}`;
}

// ── Render: 4-Week Forecast Chart ─────────────────────
// Open-Meteo's free forecast reaches ~16 days, so the last weeks of the horizon
// carry no live weather. Those bars are drawn faded so the chart does not imply
// the same confidence across all four weeks.
function basisOf(week) {
  const w = (week.weather_basis || '').toLowerCase();
  if (w === 'forecast') return 'forecast';
  if (w === 'partial')  return 'partial';
  return 'climatology';
}

const BASIS_NOTE = {
  forecast:    'live forecast weather',
  partial:     'partly forecast weather',
  climatology: 'seasonal weather — no live data',
};

function renderChart(weeks) {
  const labels   = weeks.map(w => isoWeekLabel(w.year, w.week));
  const expected = weeks.map(w => w.expected_cases);
  const lower    = weeks.map(w => w.lower_cases);
  const upper    = weeks.map(w => w.upper_cases);
  const bases    = weeks.map(basisOf);

  const barFill = bases.map(b =>
    b === 'forecast'    ? 'rgba(37,99,235,0.78)'
    : b === 'partial'   ? 'rgba(37,99,235,0.46)'
                        : 'rgba(120,150,185,0.30)');
  const barEdge = bases.map(b =>
    b === 'climatology' ? 'rgba(120,150,185,0.55)' : 'rgba(37,99,235,0.9)');

  if (_chartInstance) {
    _chartInstance.data.labels = labels;
    _chartInstance.data.datasets[0].data = expected;
    _chartInstance.data.datasets[0].backgroundColor = barFill;
    _chartInstance.data.datasets[0].borderColor = barEdge;
    _chartInstance.data.datasets[1].data = upper;
    _chartInstance.data.datasets[2].data = lower;
    _chartInstance._weekBases = bases;
    _chartInstance.update('active');
    return;
  }

  const ctx = el('forecastChart').getContext('2d');

  _chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Expected',
          data: expected,
          backgroundColor: barFill,
          borderColor: barEdge,
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
            afterBody: items => {
              const b = (_chartInstance && _chartInstance._weekBases || [])[items[0].dataIndex];
              return b ? `basis: ${BASIS_NOTE[b]}` : '';
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
  _chartInstance._weekBases = bases;
}

// ── Render: UC List ───────────────────────────────────
function renderUcList(rows) {
  const container = el('topUcRows');
  const filtered = rows
    .filter(r => r.tehsil === 'Rawalpindi Tehsil')
    .filter(r => isMappedAlert(r.alert))
    .sort(sortAlertRows)
    .slice(0, 12);

  if (!filtered.length) {
    container.innerHTML = `
      <div class="uc-empty" role="listitem">
        <strong>No medium or high alert UCs</strong>
        <span>Hover any UC on the map for its expected cases.</span>
      </div>`;
    return;
  }

  container.innerHTML = filtered
    .map((row, i) => {
      const level = row.alert.toLowerCase();
      const color = alertColor(row.alert);
      const delay = i * 40; // stagger animation
      const share = row.share_pct != null ? ` · ${fmt(row.share_pct, 1)}% share` : '';
      return `
        <div class="uc-row is-${level}" role="listitem" style="animation-delay:${delay}ms">
          <div class="uc-rank" style="background:${color}">${i + 1}</div>
          <div class="uc-info">
            <strong>${row.uc}</strong>
            <span>${fmt(row.expected_cases, 2)} expected${share}</span>
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
    if (_markerLayer) {
      _markerLayer.remove();
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

  // Draw every Rawalpindi Tehsil UC. The city-bounds test is now only used to
  // frame the initial view — previously it also filtered the layer, silently
  // hiding 14 UCs from the map entirely.
  const visibleFeatures = geojson.features.filter(f => f.properties.tehsil === 'Rawalpindi Tehsil');
  const cityCoreFeatures = visibleFeatures.filter(featureInBounds);

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
      const uc = escapeHtml(p.uc);
      const tehsil = escapeHtml(p.tehsil);
      const alert = escapeHtml(p.alert);
      const expected = fmt(p.expected_cases, 2);
      const share = fmt(p.share_pct, 2);
      const historical = fmt(p.historical_cases, 0);
      layer.bindPopup(`
        <strong>${uc}</strong><br>
        ${tehsil}<br>
        Alert: <strong style="color:${color}">${alert}</strong><br>
        Expected next week: <strong>${expected}</strong> reported cases<br>
        Share of city forecast: ${share}%<br>
        Historical burden: ${historical} total cases
      `);
      layer.bindTooltip(`
        <div class="uc-hover-card">
          <strong>${uc}</strong>
          <span>${expected} expected next week</span>
          <em style="color:${color}">${alert} alert</em>
        </div>
      `, {
        className: 'uc-map-tooltip uc-polygon-tooltip',
        sticky: true,
        direction: 'top',
        opacity: 1,
      });
      layer.on('mouseover', () => layer.setStyle({ weight: 3, color: '#1e40af', fillOpacity: 0.92 }));
      layer.on('mouseout', () => _geojsonLayer.resetStyle(layer));
    },
  }).addTo(_mapInstance);

  _markerLayer = L.layerGroup().addTo(_mapInstance);

  // Add numbered pins only for mapped alert UCs. These numbers match the
  // alert-only ranking in the sidebar.
  const ranked = visibleFeatures
    .filter(f => isMappedAlert(f.properties.alert))
    .sort((a,b) => sortAlertRows(a.properties, b.properties))
    .slice(0, 12);

  ranked.forEach((feature, i) => {
    const p = feature.properties;
    const color = alertColor(p.alert);
    const pulse = alertPulse(p.alert);
    const uc = escapeHtml(p.uc);
    const alert = escapeHtml(p.alert);
    const expected = fmt(p.expected_cases, 2);
    const bounds = L.geoJSON(feature).getBounds();
    const center = bounds.getCenter();
    const label = String(i + 1);

    const icon = L.divIcon({
      className: '',
      html: `<div class="alert-pin" style="background:${color};--pulse-color:${pulse}" aria-label="Risk rank ${label}: ${uc}">${label}</div>`,
      iconSize: [34, 34],
      iconAnchor: [17, 17],
    });

    const marker = L.marker(center, {
      icon,
      keyboard: false,
      zIndexOffset: 2000,
    })
      .bindPopup(`
        <strong>${label}. ${uc}</strong><br>
        Alert: <strong style="color:${color}">${alert}</strong><br>
        Expected next week: <strong>${expected}</strong> reported cases
      `)
      .bindTooltip(`${label}. ${uc} · ${fmt(p.expected_cases, 1)} expected`, {
        className: 'uc-map-tooltip',
        direction: 'top',
        offset: [0, -18],
      })
      .addTo(_markerLayer);

    marker.getElement()?.classList.add('is-alert-label');
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

      // Frame on the city core, but let the user pan out to the rural UCs that
      // are now drawn as well.
      const frameLayer = cityCoreFeatures.length
        ? L.geoJSON({ ...geojson, features: cityCoreFeatures })
        : _geojsonLayer;

      _mapInstance.fitBounds(frameLayer.getBounds(), {
        paddingTopLeft:     [padLeft,  padTop],
        paddingBottomRight: [padRight, padBottom],
        animate: false,
        maxZoom: 13,
      });

      // Clamp zoom: always show the whole city (≥11) but never too close (≤13)
      const z = _mapInstance.getZoom();
      if (z < 11) _mapInstance.setZoom(11, { animate: false });
      if (z > 13) _mapInstance.setZoom(13, { animate: false });

      // Pan limit follows every drawn UC, not just the city core
      _mapInstance.setMaxBounds(_geojsonLayer.getBounds().pad(0.25));
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
          Run <code style="color:#f97316">python src/update_live_forecast.py</code> first,
          then serve this folder over HTTP.
        </p>
        <p style="color:#4d6a8a;font-size:12px">${err.message}</p>
      </div>`;
  }
})();
