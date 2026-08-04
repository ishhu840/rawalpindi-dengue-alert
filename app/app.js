const alertColors = {
  Green: "#2ca25f",
  Yellow: "#f4c542",
  Orange: "#ef7d32",
  Red: "#d73027",
};

function fmt(value, digits = 1) {
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function alertColor(alert) {
  return alertColors[alert] || "#94a3b8";
}

function isoWeekRange(year, week) {
  const simple = new Date(Date.UTC(year, 0, 1 + (week - 1) * 7));
  const day = simple.getUTCDay() || 7;
  const monday = new Date(simple);
  monday.setUTCDate(simple.getUTCDate() - day + 1);
  const sunday = new Date(monday);
  sunday.setUTCDate(monday.getUTCDate() + 6);
  const options = { month: "short", day: "numeric" };
  return `${monday.toLocaleDateString(undefined, options)} to ${sunday.toLocaleDateString(undefined, options)}, ${year}`;
}

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return res.json();
}

function renderSummary(data) {
  const first = data.weekly_forecasts[0];
  const period = isoWeekRange(first.year, first.week);
  const alertRows = (data.top_ucs || []).filter((row) => ["Red", "Orange", "Yellow"].includes(row.alert));
  const redRows = alertRows.filter((row) => row.alert === "Red");
  const orangeRows = alertRows.filter((row) => row.alert === "Orange");
  const alertNode = document.getElementById("modelAlert");
  document.getElementById("generatedAt").textContent =
    `Generated ${new Date(data.generated_at).toLocaleString()}`;
  document.getElementById("forecastPeriodLabel").textContent = `Next week: ${period}`;
  document.getElementById("expectedCases").textContent = fmt(first.expected_cases);
  document.getElementById("forecastRange").textContent =
    `${fmt(first.lower_cases)} to ${fmt(first.upper_cases)} reported cases expected overall`;
  document.getElementById("redCount").textContent = String(alertRows.length);
  if (redRows.length) {
    alertNode.textContent = `High alert: ${redRows.length} UC${redRows.length === 1 ? "" : "s"}`;
    alertNode.className = "model-alert red";
  } else if (orangeRows.length) {
    alertNode.textContent = `Elevated: ${orangeRows.length} UC${orangeRows.length === 1 ? "" : "s"}`;
    alertNode.className = "model-alert orange";
  } else if (alertRows.length) {
    alertNode.textContent = `Watch: ${alertRows.length} UC${alertRows.length === 1 ? "" : "s"}`;
    alertNode.className = "model-alert yellow";
  } else {
    alertNode.textContent = "No elevated UC alert this week";
    alertNode.className = "model-alert green";
  }
  document.getElementById("weatherStatus").textContent =
    data.weather_status.includes("used") ? "Connected" : "Fallback";
  document.getElementById("surveillanceStatus").textContent =
    `${data.weather_status}. ${data.surveillance_status}`;
  document.getElementById("modelName").textContent = data.selected_model.name;
  document.getElementById("modelScore").textContent =
    `Mean RMSE ${fmt(data.selected_model.rolling_origin_mean_rmse, 2)} · MAE ${fmt(data.selected_model.rolling_origin_mean_mae, 2)}`;
  document.getElementById("externalScore").textContent =
    fmt(data.external_2025_validation.rmse, 2);
}

function renderTopUcs(rows) {
  const container = document.getElementById("topUcRows");
  container.innerHTML = rows
    .filter((row) => row.tehsil === "Rawalpindi Tehsil")
    .slice(0, 10)
    .map(
      (row, index) => `
        <div class="uc-chip is-${row.alert.toLowerCase()}" style="border-left-color:${alertColor(row.alert)}">
          <div class="uc-rank" style="background:${alertColor(row.alert)}">${index + 1}</div>
          <div>
            <strong>${row.uc}</strong>
            <span>${row.alert} · ${fmt(row.expected_cases, 2)} expected cases</span>
          </div>
        </div>`
    )
    .join("");
}

function renderWeeks(rows) {
  document.getElementById("weekCards").innerHTML = rows
    .map(
      (row) => `
        <div class="week-card">
          <div>
            <strong>${isoWeekRange(row.year, row.week)}</strong><br>
            <span>${fmt(row.lower_cases)} to ${fmt(row.upper_cases)} range</span>
          </div>
          <strong>${fmt(row.expected_cases)}</strong>
        </div>`
    )
    .join("");
}

function renderModels(rows) {
  document.getElementById("modelGrid").innerHTML = rows
    .map(
      (row) => `
        <div class="model-card">
          <div>
            <strong>${row.model}</strong><br>
            <span>MAE ${fmt(row.mae, 2)} · sMAPE ${fmt(row.smape, 1)}%</span>
          </div>
          <strong>${fmt(row.rmse, 2)}</strong>
        </div>`
    )
    .join("");
}

function renderMap(geojson) {
  const cityBounds = {
    south: 33.48,
    west: 72.94,
    north: 33.67,
    east: 73.16,
  };

  function featureIntersectsCity(feature) {
    const points = [];
    const geometry = feature.geometry;
    if (geometry.type === "Polygon") {
      geometry.coordinates.forEach((ring) => ring.forEach(([lon, lat]) => points.push({ lon, lat })));
    } else if (geometry.type === "MultiPolygon") {
      geometry.coordinates.forEach((poly) =>
        poly.forEach((ring) => ring.forEach(([lon, lat]) => points.push({ lon, lat })))
      );
    }
    if (!points.length) return false;
    const west = Math.min(...points.map((p) => p.lon));
    const east = Math.max(...points.map((p) => p.lon));
    const south = Math.min(...points.map((p) => p.lat));
    const north = Math.max(...points.map((p) => p.lat));
    return east >= cityBounds.west && west <= cityBounds.east &&
      north >= cityBounds.south && south <= cityBounds.north;
  }

  const visibleGeojson = {
    ...geojson,
    features: geojson.features.filter((feature) =>
      feature.properties.tehsil === "Rawalpindi Tehsil" && featureIntersectsCity(feature)
    ),
  };
  const alertFeatures = visibleGeojson.features.filter(
    (feature) => feature.properties.alert !== "Green"
  );

  const map = L.map("map", {
    center: [33.58, 73.05],
    zoom: 12,
    zoomControl: false,
    scrollWheelZoom: false,
    doubleClickZoom: true,
    touchZoom: true,
    dragging: true,
    maxBounds: [
      [33.46, 72.91],
      [33.69, 73.18],
    ],
    maxBoundsViscosity: 0.7,
  });

  L.control.zoom({ position: "bottomright" }).addTo(map);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OSM &copy; CartoDB",
    subdomains: "abcd",
    maxZoom: 19,
  }).addTo(map);

  function style(feature) {
    const color = alertColor(feature.properties.alert);
    if (feature.properties.alert === "Green") {
      return {
        color: "#c8d2dc",
        weight: 0.65,
        fillColor: "#f7fafc",
        fillOpacity: 0.18,
        opacity: 0.8,
      };
    }
    return {
      color: feature.properties.alert === "Red" ? "#9f2922" : "#ffffff",
      weight: feature.properties.alert === "Red" ? 2.2 : 1.2,
      fillColor: color,
      fillOpacity: feature.properties.alert === "Yellow" ? 0.42 : 0.58,
      opacity: 0.95,
    };
  }

  const layer = L.geoJSON(visibleGeojson, {
    style,
    onEachFeature: (feature, lyr) => {
      const p = feature.properties;
      lyr.bindPopup(`
        <strong>${p.uc}</strong><br>
        ${p.tehsil}<br>
        Alert: <strong style="color:${alertColor(p.alert)}">${p.alert}</strong><br>
        Expected: <strong>${fmt(p.expected_cases, 2)}</strong> reported cases<br>
        Historical mapped cases: ${fmt(p.historical_cases, 0)}
      `);
      lyr.on("mouseover", () => lyr.setStyle({ weight: 2.4, color: "#1f2937", fillOpacity: 0.74 }));
      lyr.on("mouseout", () => layer.resetStyle(lyr));
    },
  }).addTo(map);

  const rankedLayers = [];
  layer.eachLayer((lyr) => {
    const p = lyr.feature.properties;
    rankedLayers.push({ lyr, expected: Number(p.expected_cases || 0), alert: p.alert });
  });

  const rankedAlertLayers = rankedLayers
    .filter(({ alert }) => alert === "Red" || alert === "Orange")
    .sort((a, b) => {
      const priority = { Red: 2, Orange: 1, Yellow: 0, Green: -1 };
      return priority[b.alert] - priority[a.alert] || b.expected - a.expected;
    })
    .slice(0, 10);

  rankedAlertLayers
    .forEach(({ lyr }, index) => {
      const p = lyr.feature.properties;
      const center = lyr.getBounds().getCenter();
      const color = alertColor(p.alert);
      const icon = L.divIcon({
        className: "",
        html: `<div class="alert-pin" style="background:${color};--pin-color:${color}" aria-label="Priority ${index + 1}: ${p.uc}">${index + 1}</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });
      const marker = L.marker(center, { icon, interactive: true, zIndexOffset: 600 })
        .bindPopup(`
          <strong>${index + 1}. ${p.uc}</strong><br>
          ${p.tehsil}<br>
          Alert: <strong style="color:${color}">${p.alert}</strong><br>
          Expected: <strong>${fmt(p.expected_cases, 2)}</strong> reported cases
        `)
        .addTo(map);
      marker.bindTooltip(`${p.uc} · ${fmt(p.expected_cases, 1)} expected`, {
        direction: "top",
        offset: [0, -16],
        className: "uc-map-label",
      });
    });

  const focusLayer = layer;
  const isSmall = window.matchMedia("(max-width: 720px)").matches;
  map.fitBounds(focusLayer.getBounds().pad(0.06), {
    paddingTopLeft: isSmall ? [24, 24] : [54, 38],
    paddingBottomRight: isSmall ? [24, 84] : [90, 72],
    animate: false,
  });
  // Keep the first view on the city itself; users can zoom out when they need context.
  map.setView([33.575, 73.045], isSmall ? 12.45 : 12.8, { animate: false });
  map.setMaxBounds(layer.getBounds().pad(0.2));
  L.control.scale({ position: "bottomright", imperial: false, maxWidth: 90 }).addTo(map);
}

async function main() {
  const [forecast, geojson] = await Promise.all([
    loadJson("data/latest_forecast.json"),
    loadJson("data/rawalpindi_uc_forecast.geojson"),
  ]);
  renderSummary(forecast);
  renderTopUcs(forecast.top_ucs);
  renderWeeks(forecast.weekly_forecasts);
  renderModels(forecast.model_comparison);
  renderMap(geojson);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#991b1b">${err.message}</pre>`;
});
