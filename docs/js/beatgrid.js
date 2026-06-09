// Beat-grid "snap" animation — illustrates tempo matching by time-stretch.
//
// Two lanes of beat ticks: A at its tempo (fixed reference) and B, whose grid
// starts at B's own tempo and eases toward `bpm_b * stretch_ratio` (its tempo
// AFTER the pipeline's time-stretch). When the pipeline beat-matched the pair,
// that target equals A's tempo, so B's ticks visibly slide into phase under A's
// ("locked!"). When the tempo gap was too large to stretch (stretch_ratio == 1),
// the target is B's own tempo, so the grids stay offset — faithfully showing the
// tempo-tolerance gate (DJs don't stretch tempos that are too far apart).
//
// Exposes window.BeatGrid.mount(canvas, {bpmA, bpmB, stretch}).

(function () {
  const VISIBLE_SEC = 2.6; // seconds of grid shown across the canvas width
  const CYCLE_MS = 5200; // one drift->lock->hold->reset loop

  function easeInOut(p) {
    return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
  }

  function drawLane(ctx, y, h, periodSec, pxPerSec, color, phasePx) {
    const w = ctx.canvas.clientWidth;
    const stepPx = periodSec * pxPerSec;
    if (stepPx < 4) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (let x = phasePx % stepPx; x < w; x += stepPx) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + h);
      ctx.stroke();
    }
  }

  function mount(canvas, opts) {
    const bpmA = opts.bpmA || 120;
    const bpmB = opts.bpmB || 120;
    const stretch = opts.stretch || 1.0;
    const locked = Math.abs(stretch - 1.0) > 1e-3; // pipeline applied a stretch
    const periodA = 60 / bpmA;
    const periodB = 60 / bpmB;
    const periodTarget = 60 / (bpmB * stretch); // B's period after stretch
    const label = canvas.parentElement.querySelector(".beatgrid-state");

    const ctx = canvas.getContext("2d");
    let raf = null;
    let start = null;

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 300;
      const h = canvas.clientHeight || 90;
      canvas.width = w * ratio;
      canvas.height = h * ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function frame(ts) {
      if (start == null) start = ts;
      const t = (ts - start) % CYCLE_MS;
      // 0..40% drift, 40..70% settle to target, 70..100% hold locked.
      let p = t < CYCLE_MS * 0.7 ? easeInOut(Math.min(t / (CYCLE_MS * 0.7), 1)) : 1;
      const periodNow = periodB + (periodTarget - periodB) * p;

      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const pxPerSec = w / VISIBLE_SEC;
      ctx.clearRect(0, 0, w, h);

      // Lane A (reference) and lane B (animated). Phase 0 so both start at the left
      // edge — when periods match, ticks coincide.
      drawLane(ctx, h * 0.12, h * 0.3, periodA, pxPerSec, "#52d5ff", 0);
      drawLane(ctx, h * 0.58, h * 0.3, periodNow, pxPerSec, "#ff6b5f", 0);

      if (label) {
        if (!locked) {
          label.textContent = "tempo gap too large — no stretch (grids stay offset)";
          label.className = "beatgrid-state warn";
        } else {
          label.textContent = p > 0.98 ? "locked — B stretched onto A's tempo" : "stretching B…";
          label.className = "beatgrid-state ok";
        }
      }
      raf = requestAnimationFrame(frame);
    }

    resize();
    window.addEventListener("resize", resize);
    // Only animate when visible (saves CPU on long pages).
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && raf == null) {
          start = null;
          raf = requestAnimationFrame(frame);
        } else if (!e.isIntersecting && raf != null) {
          cancelAnimationFrame(raf);
          raf = null;
        }
      }
    });
    io.observe(canvas);
  }

  window.BeatGrid = { mount };
})();
