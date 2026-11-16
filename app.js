let SITE_DATA = null;
let MODEL_DATA = null;
let currentFeatures = [];
let activeSample = null;

const PALETTE_19 = [
  "#38bdf8", "#34d399", "#f472b6", "#fbbf24", "#a78bfa",
  "#fb7185", "#2dd4bf", "#60a5fa", "#f97316", "#4ade80",
  "#e879f9", "#facc15", "#22d3ee", "#818cf8", "#fb923c",
  "#a3e635", "#c084fc", "#f87171", "#94a3b8"
];

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const [siteRes, modelRes] = await Promise.all([
      fetch("site_data.json"),
      fetch("models_export.json")
    ]);
    SITE_DATA = await siteRes.json();
    MODEL_DATA = await modelRes.json();

    document.getElementById("model-status-badge").textContent = "3 Models Loaded (Live JS)";
    initAll();
  } catch (err) {
    console.error("Error loading JSON assets:", err);
    document.getElementById("model-status-badge").textContent =
      "Serve via local HTTP / GitHub Pages to load JSON";
  }
});

function initAll() {
  renderMetricsTable();
  renderFoldChart();
  renderFeatureImportances();
  initPCAPlot();
  initConfusionMatrixTabs();
  initInteractiveLab();
}

/* ---------- 1. Metrics & Fold Stability ---------- */
function renderMetricsTable() {
  const tbody = document.getElementById("metrics-tbody");
  const configs = [
    {
      name: "SVM",
      label: "SVM (RBF Kernel)",
      prep: "StandardScaler",
      params: "C = 1, gamma = 'scale'",
      best: true
    },
    {
      name: "Random Forest",
      label: "Random Forest",
      prep: "Raw 17 Features",
      params: "n_estimators = 200, min_samples_split = 5, max_depth = None",
      best: false
    },
    {
      name: "Logistic Regression",
      label: "Logistic Regression (Baseline)",
      prep: "StandardScaler",
      params: "C = 1, penalty = 'l2', max_iter = 3000",
      best: false
    }
  ];

  tbody.innerHTML = configs.map(cfg => {
    const m = SITE_DATA.metrics[cfg.name];
    return `
      <tr class="${cfg.best ? "highlight-row" : ""}">
        <td><strong>${cfg.label}</strong></td>
        <td><code>${cfg.prep}</code></td>
        <td><code>${cfg.params}</code></td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td><strong>${(m.f1_macro * 100).toFixed(2)}%</strong></td>
        <td>&plusmn;${(m.fold_std * 100).toFixed(2)}%</td>
      </tr>
    `;
  }).join("");
}

function renderFoldChart() {
  const container = document.getElementById("fold-chart");
  const modelSpecs = [
    { key: "SVM", short: "SVM", color: "#34d399" },
    { key: "Random Forest", short: "RF", color: "#38bdf8" },
    { key: "Logistic Regression", short: "LR", color: "#a78bfa" }
  ];

  let html = "";
  for (let f = 0; f < 5; f++) {
    html += `<div class="fold-col"><div class="fold-title">Outer Fold ${f + 1}</div><div class="fold-bar-group">`;
    modelSpecs.forEach(ms => {
      const score = SITE_DATA.metrics[ms.key].fold_f1_scores[f];
      const pct = (score * 100).toFixed(1);
      html += `
        <div class="fold-mini">
          <div class="fold-mini-top"><span>${ms.short}</span><span>${pct}%</span></div>
          <div class="fold-mini-track">
            <div class="fold-mini-fill" style="width:${pct}%; background:${ms.color};"></div>
          </div>
        </div>
      `;
    });
    html += `</div></div>`;
  }
  container.innerHTML = html;
}

/* ---------- 2. Random Forest Feature Importance ---------- */
function renderFeatureImportances() {
  const container = document.getElementById("fi-bars");
  const importances = MODEL_DATA.random_forest.feature_importances;
  const names = SITE_DATA.feature_names;

  const paired = names.map((name, idx) => ({ name, val: importances[idx] }))
    .sort((a, b) => b.val - a.val);
  const maxVal = paired[0].val || 1;

  container.innerHTML = paired.map(item => `
    <div class="fi-row">
      <span class="fi-name" title="${item.name}">${item.name}</span>
      <div class="fi-track">
        <div class="fi-fill" style="width:${((item.val / maxVal) * 100).toFixed(1)}%"></div>
      </div>
      <span class="fi-val">${(item.val * 100).toFixed(1)}%</span>
    </div>
  `).join("");
}

/* ---------- 3. Interactive 2D PCA Scatter Plot ---------- */
function initPCAPlot() {
  const select = document.getElementById("pca-highlight");
  SITE_DATA.categories.forEach(cat => {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = `Highlight: ${cat}`;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => drawPCA());

  const canvas = document.getElementById("pca-canvas");
  const tooltip = document.getElementById("pca-tooltip");
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    let closest = null;
    let minDist = 14;
    for (const pt of SITE_DATA.pca_points) {
      if (pt._px === undefined) continue;
      const d = Math.hypot(pt._px - mx, pt._py - my);
      if (d < minDist) {
        minDist = d;
        closest = pt;
      }
    }
    if (closest) {
      tooltip.textContent = `${closest.category} (${closest.filename}, Fold ${closest.fold}) — PC1: ${closest.pc1}, PC2: ${closest.pc2}`;
    }
  });

  drawPCA();
}

function drawPCA() {
  const canvas = document.getElementById("pca-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const pad = 38;
  const selectedCat = document.getElementById("pca-highlight").value;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b111e";
  ctx.fillRect(0, 0, W, H);

  const pts = SITE_DATA.pca_points;
  const xs = pts.map(p => p.pc1);
  const ys = pts.map(p => p.pc2);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const mapX = x => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const mapY = y => H - pad - ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);

  // Grid axes
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, mapY(0));
  ctx.lineTo(W - pad, mapY(0));
  ctx.moveTo(mapX(0), pad);
  ctx.lineTo(mapX(0), H - pad);
  ctx.stroke();

  ctx.fillStyle = "#9aa8c7";
  ctx.font = "11px JetBrains Mono, monospace";
  ctx.fillText("PC1 (26.38% variance)", W / 2 - 65, H - 10);
  ctx.fillText("PC2 (16.49%)", 10, 20);

  pts.forEach(pt => {
    const px = mapX(pt.pc1);
    const py = mapY(pt.pc2);
    pt._px = px;
    pt._py = py;
    const catIdx = SITE_DATA.categories.indexOf(pt.category);
    const color = PALETTE_19[catIdx % PALETTE_19.length];
    const isMatch = selectedCat === "all" || pt.category === selectedCat;

    ctx.beginPath();
    ctx.arc(px, py, isMatch ? 4.3 : 2.6, 0, Math.PI * 2);
    ctx.fillStyle = isMatch ? color : "rgba(148, 163, 184, 0.16)";
    ctx.fill();
  });

  // Highlight active sample clip from Interactive Lab
  if (activeSample) {
    const ax = mapX(activeSample.pc1);
    const ay = mapY(activeSample.pc2);
    ctx.beginPath();
    ctx.arc(ax, ay, 9, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
}

/* ---------- 4. Confusion Matrix & Top Confused Pairs ---------- */
function initConfusionMatrixTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderConfusionMatrix(btn.dataset.model);
    });
  });
  renderConfusionMatrix("SVM");
}

function renderConfusionMatrix(modelName) {
  const cm = SITE_DATA.metrics[modelName].confusion_matrix;
  const cats = SITE_DATA.categories;
  const table = document.getElementById("cm-table");

  // Compute top 3 off-diagonal confusions
  const pairs = [];
  for (let i = 0; i < cats.length; i++) {
    for (let j = 0; j < cats.length; j++) {
      if (i !== j && cm[i][j] > 0) {
        pairs.push({ trueCat: cats[i], predCat: cats[j], count: cm[i][j] });
      }
    }
  }
  pairs.sort((a, b) => b.count - a.count);
  document.getElementById("top-confusions").innerHTML = pairs.slice(0, 3).map(p =>
    `<span class="conf-pill">${p.trueCat} &rarr; ${p.predCat} (${p.count}/40)</span>`
  ).join("");

  let html = "<thead><tr><th class='row-hdr'>True \\ Pred</th>";
  cats.forEach(c => { html += `<th class="col-hdr">${c}</th>`; });
  html += "</tr></thead><tbody>";

  for (let i = 0; i < cats.length; i++) {
    html += `<tr><th class="row-hdr">${cats[i]}</th>`;
    for (let j = 0; j < cats.length; j++) {
      const val = cm[i][j];
      const alpha = Math.min(val / 40, 1);
      const bg = i === j
        ? `rgba(52, 211, 153, ${0.12 + alpha * 0.65})`
        : val > 0
          ? `rgba(248, 113, 113, ${0.08 + alpha * 0.75})`
          : "transparent";
      const color = val >= 18 ? "#ffffff" : "#cbd5e1";
      html += `<td style="background:${bg}; color:${color}; font-weight:${i === j ? 700 : 400}">${val}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody>";
  table.innerHTML = html;
}

/* ---------- 5. Interactive Audio & Live 3-Model Inference ---------- */
function initInteractiveLab() {
  const select = document.getElementById("clip-select");
  SITE_DATA.sample_clips.forEach((clip, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${clip.category} — ${clip.filename} (Fold ${clip.fold})`;
    select.appendChild(opt);
  });

  buildSliders();

  select.addEventListener("change", () => loadSampleClip(Number(select.value)));
  document.getElementById("reset-clip-btn").addEventListener("click", () => {
    loadSampleClip(Number(select.value));
  });

  loadSampleClip(0);
}

function buildSliders() {
  const primaryIndices = [0, 1, 2, 3, 13, 14, 15, 16]; // MFCC 1-4 + 4 Spectral features
  const secondaryIndices = [4, 5, 6, 7, 8, 9, 10, 11, 12]; // MFCC 5-13

  const renderGroup = (indices, containerId) => {
    const el = document.getElementById(containerId);
    el.innerHTML = indices.map(i => {
      const name = SITE_DATA.feature_names[i];
      const min = SITE_DATA.feature_mins[i];
      const max = SITE_DATA.feature_maxs[i];
      const step = ((max - min) / 250) || 0.001;
      return `
        <div class="slider-item">
          <div class="slider-head">
            <span>${name}</span>
            <span id="val-f-${i}">0</span>
          </div>
          <input type="range" id="slider-f-${i}" min="${min}" max="${max}" step="${step}" />
        </div>
      `;
    }).join("");

    indices.forEach(i => {
      const input = document.getElementById(`slider-f-${i}`);
      input.addEventListener("input", () => {
        currentFeatures[i] = parseFloat(input.value);
        document.getElementById(`val-f-${i}`).textContent = formatFeat(i, currentFeatures[i]);
        runAllThreeModels();
      });
    });
  };

  renderGroup(primaryIndices, "sliders-primary");
  renderGroup(secondaryIndices, "sliders-secondary");
}

function formatFeat(idx, val) {
  if (idx === 13 || idx === 14) return val.toFixed(0);
  if (idx === 15 || idx === 16) return val.toFixed(4);
  return val.toFixed(1);
}

function loadSampleClip(idx) {
  const clip = SITE_DATA.sample_clips[idx];
  activeSample = clip;
  currentFeatures = [...clip.features];

  const audio = document.getElementById("audio-player");
  audio.src = `audio/${clip.filename}`;

  document.getElementById("clip-meta").textContent =
    `True Ground-Truth Label: ${clip.category.toUpperCase()} | File: ${clip.filename} | ESC-50 Fold ${clip.fold} | PCA: (${clip.pc1}, ${clip.pc2})`;

  currentFeatures.forEach((val, i) => {
    const slider = document.getElementById(`slider-f-${i}`);
    if (slider) {
      slider.value = val;
      document.getElementById(`val-f-${i}`).textContent = formatFeat(i, val);
    }
  });

  drawPCA();
  runAllThreeModels();
}

/* ---------- Exact Client-Side ML Inference ---------- */
function standardize(x) {
  const mean = MODEL_DATA.scaler.mean;
  const scale = MODEL_DATA.scaler.scale;
  return x.map((v, i) => (v - mean[i]) / scale[i]);
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExp);
}

function predictLogisticRegression(xScaled) {
  const { coef, intercept } = MODEL_DATA.logistic_regression;
  const logits = coef.map((row, c) => {
    let z = intercept[c];
    for (let j = 0; j < xScaled.length; j++) z += row[j] * xScaled[j];
    return z;
  });
  return softmax(logits);
}

function predictSVM(xScaled) {
  const { gamma, support_vectors, dual_coef, intercept, n_support } = MODEL_DATA.svm;
  const nClasses = n_support.length;
  const nSV = support_vectors.length;

  // Compute RBF kernel K(sv, x) for all support vectors
  const kVals = new Float64Array(nSV);
  for (let i = 0; i < nSV; i++) {
    let sqDist = 0;
    const sv = support_vectors[i];
    for (let j = 0; j < xScaled.length; j++) {
      const diff = sv[j] - xScaled[j];
      sqDist += diff * diff;
    }
    kVals[i] = Math.exp(-gamma * sqDist);
  }

  // Class SV offsets
  const start = new Array(nClasses).fill(0);
  for (let c = 1; c < nClasses; c++) start[c] = start[c - 1] + n_support[c - 1];

  // One-vs-One decision function + vote/margin aggregation
  const votes = new Array(nClasses).fill(0);
  const marginSum = new Array(nClasses).fill(0);
  let pairIdx = 0;

  for (let i = 0; i < nClasses; i++) {
    for (let j = i + 1; j < nClasses; j++) {
      let sum = intercept[pairIdx];
      for (let k = 0; k < n_support[i]; k++) {
        sum += dual_coef[j - 1][start[i] + k] * kVals[start[i] + k];
      }
      for (let k = 0; k < n_support[j]; k++) {
        sum += dual_coef[i][start[j] + k] * kVals[start[j] + k];
      }
      if (sum > 0) votes[i] += 1;
      else votes[j] += 1;
      marginSum[i] += sum;
      marginSum[j] -= sum;
      pairIdx++;
    }
  }

  // Combine One-vs-One votes with continuous margin tie-breaker (matches sklearn SVC.predict)
  const decisionScores = votes.map((v, idx) => v + marginSum[idx] / (3 * (Math.abs(marginSum[idx]) + 1)));
  return softmax(decisionScores);
}

function predictRandomForest(xRaw) {
  const trees = MODEL_DATA.random_forest.trees;
  const nClasses = MODEL_DATA.classes.length;
  const avgProbs = new Array(nClasses).fill(0);

  for (let t = 0; t < trees.length; t++) {
    const tree = trees[t];
    let node = 0;
    while (tree.children_left[node] !== -1) {
      const featIdx = tree.feature[node];
      node = xRaw[featIdx] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
    }
    const leafProbs = tree.value[node];
    for (let c = 0; c < nClasses; c++) avgProbs[c] += leafProbs[c];
  }

  return avgProbs.map(p => p / trees.length);
}

function runAllThreeModels() {
  const xScaled = standardize(currentFeatures);
  const classes = MODEL_DATA.classes;

  const lrProbs = predictLogisticRegression(xScaled);
  const svmProbs = predictSVM(xScaled);
  const rfProbs = predictRandomForest(currentFeatures);

  renderTop3("lr", classes, lrProbs);
  renderTop3("svm", classes, svmProbs);
  renderTop3("rf", classes, rfProbs);
}

function renderTop3(prefix, classes, probs) {
  const ranked = classes.map((cls, idx) => ({ cls, prob: probs[idx] }))
    .sort((a, b) => b.prob - a.prob);

  const top1 = ranked[0];
  document.getElementById(`${prefix}-top-label`).textContent =
    `${top1.cls} (${(top1.prob * 100).toFixed(1)}%)`;

  document.getElementById(`${prefix}-bars`).innerHTML = ranked.slice(0, 3).map(item => {
    const pct = (item.prob * 100).toFixed(1);
    return `
      <div class="prob-row">
        <span class="prob-label">${item.cls}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
        <span class="prob-pct">${pct}%</span>
      </div>
    `;
  }).join("");
}