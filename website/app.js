const stages = {
  analyze: {
    kicker: "Stage 01",
    title: "Analyze musical structure",
    body:
      "SmartAutoDJ extracts tempo, beat grids, downbeats, and section cues for both tracks. The pipeline can use allin1 for richer structure or fall back to librosa so the demo stays reproducible.",
    heights: [44, 82, 56, 94, 68],
  },
  align: {
    kicker: "Stage 02",
    title: "Select and align the overlap",
    body:
      "The transition region is chosen around musically meaningful anchors. For tier 2 and tier 3, song B is stretched toward song A's tempo so downbeats land together inside the overlap.",
    heights: [38, 72, 92, 58, 86],
  },
  shape: {
    kicker: "Stage 03",
    title: "Shape energy through the handoff",
    body:
      "Equal-power fade curves keep the center of the transition from feeling hollow, while bass-swap filtering drops the outgoing low end before the incoming bass takes over.",
    heights: [92, 74, 54, 76, 96],
  },
  bridge: {
    kicker: "Stage 04",
    title: "Layer an optional generated bridge",
    body:
      "Tier 3 adds a short riser, drum fill, swell, or pad under the transition. Generated clips are static assets that are trimmed, normalized, and length-matched before mixing.",
    heights: [28, 52, 76, 92, 100],
  },
  render: {
    kicker: "Stage 05",
    title: "Render audio, plots, and metrics",
    body:
      "Each run exports a WAV, a JSON sidecar, waveform and spectrogram plots, and objective metrics for beat alignment, tempo matching, and loudness continuity.",
    heights: [66, 88, 42, 98, 72],
  },
};

const demoTracks = [
  {
    title: "Tier 1 baseline",
    tier: "Linear crossfade",
    file: "",
    note: "Placeholder for the naive A-tail into B-head overlap.",
  },
  {
    title: "Tier 2 beat-aligned",
    tier: "Aligned + EQ",
    file: "",
    note: "Placeholder for the structure-aware transition render.",
  },
  {
    title: "Tier 3 AI-enhanced",
    tier: "Bridge layer",
    file: "",
    note: "Placeholder for the transition with a generated riser or fill.",
  },
];

const stageButtons = document.querySelectorAll(".stage");
const stageKicker = document.querySelector("#stage-kicker");
const stageTitle = document.querySelector("#stage-title");
const stageBody = document.querySelector("#stage-body");
const stageBars = document.querySelectorAll("#stage-visual span");

stageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    stageButtons.forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");

    const stage = stages[button.dataset.stage];
    stageKicker.textContent = stage.kicker;
    stageTitle.textContent = stage.title;
    stageBody.textContent = stage.body;
    stageBars.forEach((bar, index) => {
      bar.style.setProperty("--peak", `${stage.heights[index]}%`);
    });
  });
});

function renderAudioCards() {
  const stack = document.querySelector("#audio-stack");
  stack.innerHTML = demoTracks
    .map((track, trackIndex) => {
      const bars = Array.from({ length: 18 }, (_, index) => {
        const height = 18 + ((index * 17 + trackIndex * 23) % 46);
        return `<span style="--bar:${height}px"></span>`;
      }).join("");

      const player = track.file
        ? `<audio controls preload="metadata"><source src="${track.file}" /></audio>`
        : `<div class="audio-missing">Add an audio path in <code>demoTracks</code> when this render is ready.</div>`;

      return `
        <article class="audio-card">
          <div>
            <div class="audio-meta">
              <span>${track.tier}</span>
              <span>Demo ${trackIndex + 1}</span>
            </div>
            <h3>${track.title}</h3>
            <p>${track.note}</p>
          </div>
          <div>
            <div class="fake-wave" aria-hidden="true">${bars}</div>
            ${player}
          </div>
        </article>
      `;
    })
    .join("");
}

function startCanvas() {
  const canvas = document.querySelector("#signal-canvas");
  const context = canvas.getContext("2d");
  let width = 0;
  let height = 0;
  let frame = 0;

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function drawSignal(offset, color, amplitude, speed) {
    context.beginPath();
    for (let x = -20; x <= width + 20; x += 12) {
      const t = x * 0.012 + frame * speed + offset;
      const y =
        height * 0.52 +
        Math.sin(t) * amplitude +
        Math.sin(t * 0.42 + offset) * amplitude * 0.72;
      if (x === -20) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
    context.strokeStyle = color;
    context.lineWidth = 1.2;
    context.stroke();
  }

  function draw() {
    frame += 1;
    context.clearRect(0, 0, width, height);
    context.globalAlpha = 0.32;
    drawSignal(0, "#52d5ff", 34, 0.014);
    drawSignal(2.2, "#ffcf5a", 48, 0.011);
    drawSignal(4.8, "#ff6b5f", 24, 0.018);
    context.globalAlpha = 1;
    window.requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener("resize", resize);
}

renderAudioCards();
startCanvas();
