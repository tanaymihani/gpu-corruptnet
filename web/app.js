// In-browser inference for GPU-CorruptNet: preprocessing matches the Python pipeline
// (squash-resize to the model input, ImageNet normalisation), then temperature scaling,
// the conformal threshold and a PCA-Mahalanobis OOD score on the penultimate embedding.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const $ = (id) => document.getElementById(id);

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

function f32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function preprocess(img, size) {
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    for (let ch = 0; ch < 3; ch++) {
      out[ch * plane + i] = (px[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
    }
  }
  return new ort.Tensor("float32", out, [1, 3, size, size]);
}

function oodScore(emb, ref) {
  let d2 = 0;
  for (let j = 0; j < ref.k; j++) {
    let p = 0;
    const row = j * ref.dim;
    for (let i = 0; i < ref.dim; i++) p += (emb[i] - ref.mean[i]) * ref.comps[row + i];
    d2 += (p * p) / ref.variance[j];
  }
  return Math.sqrt(d2);
}

function render(cfg, probs, ood, ms) {
  const corrupted = probs.some((p) => p >= 0.5);
  const top = Math.max(...probs);
  const v = $("verdict");
  v.textContent = corrupted
    ? `Artifact detected (max p = ${top.toFixed(2)})`
    : `Looks clean (max p = ${top.toFixed(2)})`;
  v.className = "verdict " + (corrupted ? "bad" : "good");
  $("latency").textContent = `Inference ${ms.toFixed(0)} ms in this browser (WASM, single thread).`;

  const rows = cfg.classes
    .map((name, i) => ({ name, p: probs[i] }))
    .sort((a, b) => b.p - a.p)
    .map(({ name, p }) =>
      `<tr><td class="name">${name.replace(/_/g, " ")}</td>` +
      `<td><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i><b></b></div></td>` +
      `<td class="val">${p.toFixed(2)}</td></tr>`)
    .join("");
  $("probs").innerHTML = rows;

  const set = cfg.classes.filter((_, i) => probs[i] >= cfg.conformal_tau);
  $("conformal").innerHTML = set.length
    ? set.map((n) => `<span class="chip">${n.replace(/_/g, " ")}</span>`).join("")
    : '<span class="meta">empty set: no artifact type is plausible</span>';

  const box = $("ood");
  box.hidden = ood === null;
  if (ood === null) {
    box.textContent = "";
  } else if (ood > cfg.ood.threshold) {
    box.className = "ood flag";
    box.textContent = `Out of distribution (score ${ood.toFixed(1)} > ${cfg.ood.threshold.toFixed(1)}): ` +
      "this frame is unlike the training data, so treat the prediction as unreliable.";
  } else {
    box.className = "ood";
    box.textContent = `In distribution (score ${ood.toFixed(1)} ≤ ${cfg.ood.threshold.toFixed(1)}).`;
  }
}

async function main() {
  const cfg = await (await fetch("config.json")).json();
  let ref = null;
  try {
    const r = await (await fetch("ood_reference.json")).json();
    ref = { dim: r.dim, k: r.k, mean: f32(r.mean_b64), comps: f32(r.components_b64),
            variance: f32(r.variance_b64) };
    cfg.ood = { threshold: r.threshold };
  } catch (e) {
    ref = null;
  }
  const session = await ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  $("status").textContent = "Model ready. Pick a sample or upload a frame.";

  async function run(img) {
    const x = preprocess(img, cfg.img_size);
    const t0 = performance.now();
    const out = await session.run({ input: x });
    const ms = performance.now() - t0;
    const logits = out.logits.data;
    const probs = Array.from(logits, (z) => sigmoid(z / cfg.temperature));
    const ood = ref ? oodScore(out.embedding.data, ref) : null;
    render(cfg, probs, ood, ms);
  }

  function show(src) {
    const img = $("preview");
    img.onload = () => run(img);
    img.src = src;
  }

  const gallery = $("gallery");
  cfg.samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<img src="samples/${s.file}" alt="${s.caption}" loading="lazy"><span>${s.caption}</span>`;
    b.onclick = () => {
      gallery.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      show(`samples/${s.file}`);
    };
    gallery.appendChild(b);
    if (i === 0) b.click();
  });

  $("file").onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    gallery.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    show(URL.createObjectURL(f));
  };
}

main().catch((e) => {
  $("status").textContent = "Could not load the model: " + e.message;
});
