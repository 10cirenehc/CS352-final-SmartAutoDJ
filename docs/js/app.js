// SmartAutoDJ showcase — renders the demo gallery + survey from manifest.json.
// All paths in the manifest are RELATIVE so they resolve under the project-Pages
// subpath (https://<user>.github.io/CS352-final-SmartAutoDJ/) as well as locally.

const TIER_ORDER = ["1", "2", "3"];
const PLOT_LABELS = {
  spectrogram: "Spectrogram",
  structure: "Structure & cue points",
  fades: "Fade / EQ curves",
  region: "Output overlap region",
  waveforms: "Waveforms & beats",
};

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

function metricsTable(m) {
  if (!m) return null;
  const rows = [
    ["Beat-align error", `${(m.beat_align_err_sec * 1000).toFixed(0)} ms`],
    ["BPM gap (before → after)", `${m.bpm_gap_before} → ${m.bpm_gap_after}`],
    ["Max loudness jump", `${m.max_db_jump} dB`],
  ];
  const tbody = el("tbody", {}, rows.map(([k, v]) =>
    el("tr", {}, [el("th", {}, k), el("td", {}, v)])
  ));
  return el("table", { class: "metrics" }, [tbody]);
}

function plotFigures(plots) {
  if (!plots) return [];
  // stable, meaningful order
  const order = ["spectrogram", "structure", "fades", "region", "waveforms"];
  const keys = order.filter((k) => plots[k]).concat(
    Object.keys(plots).filter((k) => !order.includes(k))
  );
  return keys.map((k) =>
    el("figure", {}, [
      el("img", { src: plots[k], alt: PLOT_LABELS[k] || k, loading: "lazy" }),
      el("figcaption", {}, PLOT_LABELS[k] || k),
    ])
  );
}

function tierBlock(num, tier) {
  const block = el("div", { class: `tier-block t${num}` });
  block.appendChild(el("h4", {}, `Tier ${num} — ${tier.label}`));
  block.appendChild(el("audio", { controls: "", preload: "none", src: tier.audio }));
  const mt = metricsTable(tier.metrics);
  if (mt) block.appendChild(mt);

  if (tier.bridge_prompt) {
    const src = tier.bridge_source ? ` (${tier.bridge_source})` : "";
    block.appendChild(
      el("p", { class: "bridge-prompt" }, `AI bridge prompt${src}: “${tier.bridge_prompt}”`)
    );
  }

  const figs = plotFigures(tier.plots);
  if (figs.length) block.appendChild(el("div", { class: "plots" }, figs));
  return block;
}

function pairCard(pair) {
  const meta = [
    pair.bpm_a != null ? `A: ${pair.bpm_a} BPM` : null,
    pair.key_a ? pair.key_a : null,
    pair.bpm_b != null ? `B: ${pair.bpm_b} BPM` : null,
    pair.key_b ? pair.key_b : null,
  ].filter(Boolean).join(" · ");

  const title = `${pair.title_a || "A"} → ${pair.title_b || "B"}`;
  const head = el("div", { class: "card-head" }, [
    el("h3", {}, title),
    meta ? el("span", { class: "meta" }, meta) : null,
  ]);

  const grid = el("div", { class: "tier-grid" });
  for (const num of TIER_ORDER) {
    if (pair.tiers && pair.tiers[num]) grid.appendChild(tierBlock(num, pair.tiers[num]));
  }
  return el("div", { class: "card" }, [head, grid]);
}

function renderGallery(pairs) {
  const container = document.getElementById("gallery-cards");
  container.innerHTML = "";
  if (!pairs || !pairs.length) {
    container.appendChild(el("p", { class: "loading" },
      "No demos yet — run scripts/build_site_assets.py to generate them."));
    return;
  }
  for (const pair of pairs) container.appendChild(pairCard(pair));
}

function renderSurvey(url) {
  const box = document.getElementById("survey-embed");
  box.innerHTML = "";
  if (url) {
    box.appendChild(el("iframe", {
      src: url, title: "SmartAutoDJ listening survey", loading: "lazy",
    }));
    box.appendChild(el("p", {}, [
      "Trouble loading? ",
      el("a", { href: url, target: "_blank", rel: "noopener" }, "Open the survey in a new tab."),
    ]));
  } else {
    box.appendChild(el("p", { class: "loading" },
      "The listening survey will be linked here once the Google Form is created."));
  }
}

// Simple click-to-zoom lightbox for plot images.
function setupLightbox() {
  const lb = el("div", { class: "lightbox" }, [el("img", { alt: "" })]);
  document.body.appendChild(lb);
  const lbImg = lb.querySelector("img");
  document.addEventListener("click", (e) => {
    if (e.target.matches(".plots img")) {
      lbImg.src = e.target.src;
      lb.classList.add("open");
    } else if (e.target === lb || e.target === lbImg) {
      lb.classList.remove("open");
    }
  });
}

async function main() {
  setupLightbox();
  try {
    const res = await fetch("manifest.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`manifest.json ${res.status}`);
    const manifest = await res.json();
    renderGallery(manifest.pairs);
    renderSurvey(manifest.survey_url);
  } catch (err) {
    document.getElementById("gallery-cards").innerHTML =
      `<p class="loading">Could not load manifest.json (${err.message}). ` +
      `Run <code>scripts/build_site_assets.py</code> and serve from the docs/ directory.</p>`;
    renderSurvey(null);
  }
}

main();
