// SmartAutoDJ showcase — renders the cut-demo gallery from manifest.json over the
// main-site visual style, plus the animated signal canvas and the interactive
// pipeline board. All manifest paths are RELATIVE so the site works under a
// project-Pages subpath as well as locally.

// ---------------------------------------------------------------------------
// tiny DOM helper
// ---------------------------------------------------------------------------
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

const PLOT_LABELS = {
  loudness: "Loudness continuity",
  keywheel: "Harmonic compatibility (key wheel)",
  ssm: "Self-similarity matrices",
  structure: "Structure & cue points",
  ribbon: "Arrangement ribbon",
  genre: "Genre classifier (distilHuBERT)",
  melbridge: "Mel-spectrogram + bridge",
  beatmatch: "Beat-alignment scatter",
  bassswap: "Bass-swap EQ handoff",
  spectrogram: "Spectrogram",
  fades: "Fade / EQ curves",
  waveforms: "Waveforms & beats",
};

const TOPICS = [
  ["Loudness & Amplitude", "The loudness-continuity plot and metric (max dB jump across the seam) plus the volume-fade curves."],
  ["Beat & Downbeat tracking", "The beat-alignment scatter and the live beat ticks that light up on the player."],
  ["Fourier & Spectrograms", "Per-render spectrogram and the mel-spectrogram with the bridge effect region shaded."],
  ["Convolution & Filtering", "Equal-power bass-swap EQ — the bass-swap handoff plot and the swept-resonant cut effects."],
  ["Self-Similarity & Structure", "Beat-synchronous self-similarity matrices, the arrangement ribbon, and fused-novelty cue points."],
  ["MFCCs, Chroma & Pitch", "Krumhansl key estimation visualized on the Camelot / circle-of-fifths key wheel."],
  ["Deep Learning & Embeddings", "Neural beat tracking (beat_this), the distilHuBERT genre classifier, and MusicGen bridges."],
];

// ---------------------------------------------------------------------------
// interactive WaveSurfer player (built as placeholders, initialised post-mount)
// ---------------------------------------------------------------------------
const PENDING_PLAYERS = [];

function buildPlayer(demo) {
  const playBtn = el("button", { class: "play-btn", type: "button", "aria-label": "Play" }, "▶");
  const ticks = el("div", { class: "ticks" });
  const host = el("div", { class: "wave-host" }, [ticks]);
  const wrap = el("div", { class: "player" }, [el("div", { class: "player-row" }, [playBtn, host])]);
  PENDING_PLAYERS.push({ host, ticks, playBtn, demo });
  return wrap;
}

function buildTicks(container, viz, duration) {
  container.innerHTML = "";
  if (!duration || !viz) return;
  const add = (t, cls) => {
    const tk = el("span", { class: cls });
    tk.style.left = `${(t / duration) * 100}%`;
    tk.dataset.time = t;
    container.appendChild(tk);
  };
  for (const t of viz.beat_times || []) add(t, "tick beat");
  for (const t of viz.downbeat_times || []) add(t, "tick downbeat");
}

function lightTicks(container, t) {
  for (const tk of container.children) {
    tk.classList.toggle("lit", Math.abs(parseFloat(tk.dataset.time) - t) < 0.08);
  }
}

function fallbackToAudio(wrap, demo) {
  wrap.replaceWith(el("audio", { controls: "", preload: "none", src: demo.audio }));
}

function initPlayers() {
  if (!window.WaveSurfer) {
    for (const { host, demo } of PENDING_PLAYERS) fallbackToAudio(host.closest(".player"), demo);
    PENDING_PLAYERS.length = 0;
    return;
  }
  for (const { host, ticks, playBtn, demo } of PENDING_PLAYERS) {
    const wrap = host.closest(".player");
    let ws;
    try {
      ws = WaveSurfer.create({
        container: host, height: 60, url: demo.audio,
        waveColor: "#46506a", progressColor: "#52d5ff", cursorColor: "#f7f4e8",
        barWidth: 2, barGap: 1, barRadius: 2,
      });
    } catch (err) {
      fallbackToAudio(wrap, demo);
      continue;
    }
    ws.on("error", () => fallbackToAudio(wrap, demo));
    const ov = demo.viz && demo.viz.overlap;
    if (window.WaveSurfer.Regions) {
      const regions = ws.registerPlugin(WaveSurfer.Regions.create());
      ws.on("decode", () => {
        if (ov) regions.addRegion({
          start: ov.start, end: ov.start + ov.dur,
          color: "rgba(255,207,90,0.16)", drag: false, resize: false,
        });
      });
    }
    ws.on("decode", (duration) => buildTicks(ticks, demo.viz, duration));
    ws.on("timeupdate", (t) => lightTicks(ticks, t));
    playBtn.addEventListener("click", () => ws.playPause());
    ws.on("play", () => (playBtn.textContent = "❚❚"));
    ws.on("pause", () => (playBtn.textContent = "▶"));
    ws.on("finish", () => (playBtn.textContent = "▶"));
  }
  PENDING_PLAYERS.length = 0;
}

// ---------------------------------------------------------------------------
// demo card
// ---------------------------------------------------------------------------
function chip(text) { return text ? el("span", {}, text) : null; }

function statCard(k, v, opts = {}) {
  return el("div", { class: "stat-card" }, [
    el("span", { class: "k" }, k),
    el("span", { class: `v${opts.good ? " good" : ""}` }, v),
    opts.note ? el("span", { class: "note" }, opts.note) : null,
  ].filter(Boolean));
}

function metricStrip(demo) {
  const m = demo.metrics || {};
  const cards = [];
  if (m.max_db_jump != null) {
    cards.push(statCard("Max loudness jump", `${m.max_db_jump} dB`,
      { good: m.max_db_jump < 3, note: m.max_db_jump < 3 ? "smooth handoff" : null }));
  }
  if (m.bpm_gap_after != null) {
    const matched = m.bpm_gap_after < 1;
    cards.push(statCard("Tempo gap (after)", `${m.bpm_gap_after} BPM`,
      { good: matched, note: matched ? "beat-matched" : "tempos kept (cut)" }));
  }
  cards.push(statCard("Drop on downbeat", "✓", { good: true, note: "by construction" }));
  return el("div", { class: "stat-strip" }, cards);
}

function plotsBlock(demo) {
  const plots = demo.plots || {};
  const keys = Object.keys(plots);
  if (!keys.length) return null;
  const figs = keys.map((k) =>
    el("figure", {}, [
      el("img", { src: plots[k], alt: PLOT_LABELS[k] || k, loading: "lazy" }),
      el("figcaption", {}, PLOT_LABELS[k] || k),
    ])
  );
  // "Full" depth: the gallery is shown by default; the toggle can collapse it.
  const gallery = el("div", { class: "plots open" }, figs);
  const toggle = el("button", { class: "plots-toggle", type: "button" },
    `Hide ${keys.length} analysis plots ▴`);
  toggle.addEventListener("click", () => {
    const open = gallery.classList.toggle("open");
    toggle.textContent = `${open ? "Hide" : "Show"} ${keys.length} analysis plots ${open ? "▴" : "▾"}`;
  });
  return el("div", {}, [toggle, gallery]);
}

function beatgridBlock(viz) {
  if (!viz || !window.BeatGrid) return null;
  const canvas = el("canvas", { class: "beatgrid-canvas" });
  const state = el("span", { class: "beatgrid-state" });
  const wrap = el("div", { class: "beatgrid" }, [
    el("div", { class: "beatgrid-legend" }, [
      el("span", { class: "bg-a" }, `A · ${viz.bpm_a ?? "?"} BPM`),
      el("span", { class: "bg-b" }, `B · ${viz.bpm_b ?? "?"} BPM`),
      state,
    ]),
    canvas,
  ]);
  requestAnimationFrame(() =>
    window.BeatGrid.mount(canvas, { bpmA: viz.bpm_a, bpmB: viz.bpm_b, stretch: viz.stretch_ratio })
  );
  return wrap;
}

// Small tier-1/tier-2 players: hear the same pair one or two tiers "dumber" than
// the featured cut (tier 1 = naive crossfade, tier 2 = beat-aligned blend).
function ladderBlock(demo) {
  const rungs = demo.ladder || [];
  if (!rungs.length) return null;
  const rows = rungs.map((r) =>
    el("div", { class: "ladder-rung" }, [
      el("div", { class: "ladder-meta" }, [
        el("span", { class: "ladder-tier" }, `Tier ${r.tier}`),
        el("span", { class: "ladder-label" }, r.label),
        r.max_db_jump != null
          ? el("span", { class: "ladder-stat" }, `Δ ${r.max_db_jump} dB`)
          : null,
      ].filter(Boolean)),
      el("audio", { controls: "", preload: "none", src: r.audio }),
    ])
  );
  return el("div", { class: "ladder" }, [
    el("span", { class: "ladder-head" }, "Same pair, earlier tiers"),
    el("div", { class: "ladder-rows" }, rows),
  ]);
}

function demoCard(demo) {
  const head = el("div", { class: "demo-head" }, [
    el("h3", {}, `${demo.title_a || "A"} → ${demo.title_b || "B"}`),
    demo.effect_label ? el("span", { class: "effect-badge" }, demo.effect_label) : null,
  ].filter(Boolean));

  const meta = el("div", { class: "demo-meta" }, [
    chip(demo.bpm_a != null ? `A · ${demo.bpm_a} BPM` : null),
    chip(demo.key_a),
    chip(demo.genre_a),
    chip(demo.bpm_b != null ? `B · ${demo.bpm_b} BPM` : null),
    chip(demo.key_b),
    chip(demo.genre_b),
    chip(`${demo.shape === "cut" ? "quick cut" : "blend"} · tier 3`),
  ].filter(Boolean));

  const children = [head, meta, buildPlayer(demo), ladderBlock(demo),
    beatgridBlock(demo.viz), metricStrip(demo)];
  if (demo.bridge_prompt) {
    const src = demo.bridge_source ? ` (${demo.bridge_source})` : "";
    children.push(el("p", { class: "bridge-prompt" }, `Bridge prompt${src}: “${demo.bridge_prompt}”`));
  }
  children.push(plotsBlock(demo));
  // Full untrimmed transition — a plain playbar at the bottom of the card.
  if (demo.audio_full) {
    children.push(el("div", { class: "full-audio" }, [
      el("span", { class: "full-audio-label" }, "Full transition (untrimmed)"),
      el("audio", { controls: "", preload: "none", src: demo.audio_full }),
    ]));
  }
  return el("article", { class: "demo-card" }, children.filter(Boolean));
}

function renderGallery(demos) {
  const container = document.getElementById("demo-gallery");
  container.innerHTML = "";
  if (!demos || !demos.length) {
    container.appendChild(el("p", { class: "loading" },
      "No demos yet — run scripts/render_cut_demos.py then scripts/build_site_assets.py."));
    return;
  }
  for (const d of demos) container.appendChild(demoCard(d));
  initPlayers();
}

function renderTopics() {
  const grid = document.getElementById("topic-grid");
  if (!grid) return;
  grid.innerHTML = "";
  for (const [t, body] of TOPICS) {
    grid.appendChild(el("div", { class: "topic-card" }, [
      el("span", { class: "t" }, t), el("p", {}, body),
    ]));
  }
}

// ---------------------------------------------------------------------------
// objective evaluation: per-pair tier 1/2/3 comparison (chart + metric table)
// ---------------------------------------------------------------------------
const fmtMs = (v) => (v == null ? "—" : `${Math.round(v * 1000)} ms`);

function evalTable(tiers) {
  const cols = ["1", "2", "3"].filter((t) => tiers[t]);
  const rows = [
    ["Beat-align error", (t) => fmtMs(tiers[t].beat_align_err_sec)],
    ["Tempo gap (after)", (t) => (tiers[t].bpm_gap_after == null ? "—" : `${tiers[t].bpm_gap_after} BPM`)],
    ["Max loudness jump", (t) => (tiers[t].max_db_jump == null ? "—" : `${tiers[t].max_db_jump} dB`)],
  ];
  const head = el("tr", {}, [el("th", {}, ""), ...cols.map((t) => el("th", {}, `Tier ${t}`))]);
  const body = rows.map(([label, f]) =>
    el("tr", {}, [el("th", {}, label), ...cols.map((t) => el("td", {}, f(t)))])
  );
  return el("table", { class: "eval-table" }, [el("thead", {}, [head]), el("tbody", {}, body)]);
}

function comparisonCard(c) {
  const children = [el("div", { class: "demo-head" }, [el("h3", {}, c.title)])];
  if (c.chart) {
    children.push(el("figure", { class: "comparison-fig" }, [
      el("img", { src: c.chart, alt: `${c.title} tier comparison`, loading: "lazy" }),
    ]));
  }
  if (c.tiers) children.push(evalTable(c.tiers));
  return el("article", { class: "demo-card eval-card" }, children);
}

function renderComparisons(comparisons) {
  const box = document.getElementById("eval-gallery");
  if (!box) return;
  box.innerHTML = "";
  if (!comparisons || !comparisons.length) {
    box.appendChild(el("p", { class: "loading" }, "No comparisons yet — run the build script."));
    return;
  }
  for (const c of comparisons) box.appendChild(comparisonCard(c));
}

// ---------------------------------------------------------------------------
// lightbox for plot zoom
// ---------------------------------------------------------------------------
function setupLightbox() {
  const lb = el("div", { class: "lightbox" }, [el("img", { alt: "" })]);
  document.body.appendChild(lb);
  const lbImg = lb.querySelector("img");
  document.addEventListener("click", (e) => {
    if (e.target.matches(".plots img, .comparison-fig img")) {
      lbImg.src = e.target.src;
      lb.classList.add("open");
    } else if (e.target === lb || e.target === lbImg) {
      lb.classList.remove("open");
    }
  });
}

// ---------------------------------------------------------------------------
// interactive pipeline board
// ---------------------------------------------------------------------------
const STAGES = {
  analyze: { kicker: "Stage 01", title: "Analyze musical structure",
    body: "SmartAutoDJ extracts tempo, beat grids, downbeats, key, and section cues for both tracks. It prefers the beat_this neural tracker and falls back to librosa so the demo stays reproducible.",
    heights: [44, 82, 56, 94, 68] },
  align: { kicker: "Stage 02", title: "Select and align the overlap",
    body: "The transition region is chosen around musically meaningful anchors. For tier 2/3 the incoming track is stretched toward the outgoing tempo so downbeats land together inside the seam.",
    heights: [38, 72, 92, 58, 86] },
  shape: { kicker: "Stage 03", title: "Shape energy through the handoff",
    body: "Equal-power fades keep the center of the transition from feeling hollow, while bass-swap filtering drops the outgoing low end before the incoming bass takes over.",
    heights: [92, 74, 54, 76, 96] },
  bridge: { kicker: "Stage 04", title: "Layer a genre-aware effect",
    body: "Tier 3 lays an effect over the seam — a riser, air-horn + vocal stab, or a melodic riff — chosen from the detected genre, then trimmed, normalized, and length-matched before mixing.",
    heights: [28, 52, 76, 92, 100] },
  render: { kicker: "Stage 05", title: "Render audio, plots, and metrics",
    body: "Each run exports a WAV, a JSON sidecar, a dozen analysis plots, and objective metrics for beat alignment, tempo matching, and loudness continuity.",
    heights: [66, 88, 42, 98, 72] },
};

function setupPipelineBoard() {
  const buttons = document.querySelectorAll(".stage");
  const kicker = document.querySelector("#stage-kicker");
  const title = document.querySelector("#stage-title");
  const body = document.querySelector("#stage-body");
  const bars = document.querySelectorAll("#stage-visual span");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("is-active"));
      button.classList.add("is-active");
      const s = STAGES[button.dataset.stage];
      if (!s) return;
      kicker.textContent = s.kicker;
      title.textContent = s.title;
      body.textContent = s.body;
      bars.forEach((bar, i) => bar.style.setProperty("--peak", `${s.heights[i]}%`));
    });
  });
}

// ---------------------------------------------------------------------------
// animated signal canvas background
// ---------------------------------------------------------------------------
function startCanvas() {
  const canvas = document.querySelector("#signal-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let width = 0, height = 0, frame = 0;
  function resize() {
    const ratio = window.devicePixelRatio || 1;
    width = window.innerWidth; height = window.innerHeight;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  function signal(offset, color, amp, speed) {
    ctx.beginPath();
    for (let x = -20; x <= width + 20; x += 12) {
      const t = x * 0.012 + frame * speed + offset;
      const y = height * 0.52 + Math.sin(t) * amp + Math.sin(t * 0.42 + offset) * amp * 0.72;
      x === -20 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.stroke();
  }
  function draw() {
    frame += 1;
    ctx.clearRect(0, 0, width, height);
    ctx.globalAlpha = 0.32;
    signal(0, "#52d5ff", 34, 0.014);
    signal(2.2, "#ffcf5a", 48, 0.011);
    signal(4.8, "#ff6b5f", 24, 0.018);
    ctx.globalAlpha = 1;
    window.requestAnimationFrame(draw);
  }
  resize();
  draw();
  window.addEventListener("resize", resize);
}

// ---------------------------------------------------------------------------
async function main() {
  setupLightbox();
  setupPipelineBoard();
  startCanvas();
  renderTopics();
  try {
    const res = await fetch("manifest.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`manifest.json ${res.status}`);
    const manifest = await res.json();
    renderGallery(manifest.demos);
    renderComparisons(manifest.comparisons);
  } catch (err) {
    document.getElementById("demo-gallery").innerHTML =
      `<p class="loading">Could not load manifest.json (${err.message}). ` +
      `Run <code>scripts/build_site_assets.py</code> and serve from the docs/ directory.</p>`;
  }
}

main();
