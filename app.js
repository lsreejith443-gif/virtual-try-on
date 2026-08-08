// Virtual Dress Try-On - plain JS, no build step.
// MediaPipe is loaded straight from a CDN below; everything else here
// is vanilla DOM/canvas code. Open this file's folder with any static
// host (or index.html directly - see README.md for the one caveat on
// that) and it runs as-is.

// ---------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------
const video = document.getElementById('camera-video');
const canvas = document.getElementById('camera-canvas');
const ctx = canvas.getContext('2d');

const stageSpinner = document.getElementById('stage-spinner');
const stageSpinnerLabel = document.getElementById('stage-spinner-label');
const dragHint = document.getElementById('drag-hint');

const cameraErrorBanner = document.getElementById('camera-error-banner');
const uploadErrorBanner = document.getElementById('upload-error-banner');
const poseWarningBanner = document.getElementById('pose-warning-banner');

const dropzone = document.getElementById('dropzone');
const dropzoneDefault = document.getElementById('dropzone-default');
const uploadSpinner = document.getElementById('upload-spinner');
const fileInput = document.getElementById('file-input');
const garmentStrip = document.getElementById('garment-strip');

const sizeSlider = document.getElementById('size-slider');
const flipCheckbox = document.getElementById('flip-checkbox');
const resetBtn = document.getElementById('reset-btn');
const snapshotBtn = document.getElementById('snapshot-btn');

// ---------------------------------------------------------------
// App state (the plain-JS equivalent of the React version's useState)
// ---------------------------------------------------------------
const state = {
  isCameraReady: false,
  isPoseLoading: true,
  garments: [], // { id, name, canvas, anchors, thumbnailUrl }
  activeGarmentId: null,
  manualOffset: { x: 0, y: 0 },
  manualScale: 1,
  isFlipped: false
};

let nextGarmentId = 1;
let latestLandmarks = null; // updated by the pose-detection loop
let smoothed = null; // exponentially-smoothed anchor triangle, for jitter-free tracking
let dragState = null;

function activeGarment() {
  return state.garments.find((g) => g.id === state.activeGarmentId) || null;
}

// ---------------------------------------------------------------
// Pose landmark helpers
// ---------------------------------------------------------------
const POSE_LANDMARKS = { LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_HIP: 23, RIGHT_HIP: 24 };
const MIN_VISIBILITY = 0.3;

// Extracts normalized (0-1) shoulder/hip-center coordinates from a raw
// MediaPipe landmark array, or null if a person wasn't confidently
// detected. Coordinates are in the *raw, unmirrored* video's
// coordinate space - mirroring for the on-screen "selfie" view happens
// later, in drawGarment().
function getBodyAnchors(landmarks) {
  if (!landmarks) return null;
  const ls = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
  const rs = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
  const lh = landmarks[POSE_LANDMARKS.LEFT_HIP];
  const rh = landmarks[POSE_LANDMARKS.RIGHT_HIP];
  if (!ls || !rs || !lh || !rh) return null;
  if ((ls.visibility ?? 1) < MIN_VISIBILITY || (rs.visibility ?? 1) < MIN_VISIBILITY) return null;
  return {
    leftShoulder: { x: ls.x, y: ls.y },
    rightShoulder: { x: rs.x, y: rs.y },
    hipCenter: { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 }
  };
}

function lerpPoint(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// ---------------------------------------------------------------
// Affine transform: solves rotation + scale + translation + shear in
// one matrix from three point correspondences. This is what lets the
// garment scale, rotate, and move as a single operation driven by
// where three body landmarks currently are.
//
// Canvas2D's ctx.transform(a, b, c, d, e, f) applies:
//   x' = a*x + c*y + e
//   y' = b*x + d*y + f
// so given three (x, y) -> (x', y') pairs, we solve two independent
// 3x3 linear systems (one for a/c/e, one for b/d/f).
// ---------------------------------------------------------------
function solve3x3(rows, b) {
  const m = rows.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    if (Math.abs(m[col][col]) < 1e-9) return null; // degenerate/collinear points
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const factor = m[r][col] / m[col][col];
      for (let c = col; c <= 3; c++) m[r][c] -= factor * m[col][c];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

function computeAffineMatrix(src, dst) {
  const rows = src.map((p) => [p.x, p.y, 1]);
  const solX = solve3x3(rows, dst.map((p) => p.x));
  const solY = solve3x3(rows, dst.map((p) => p.y));
  if (!solX || !solY) return null;
  const [a, c, e] = solX;
  const [b, d, f] = solY;
  return { a, b, c, d, e, f };
}

// ---------------------------------------------------------------
// Shared prep: decode the uploaded file and draw it onto a
// size-capped canvas before either cutout method touches it. This is
// the fix for the crash you hit - a modern phone photo can be 12-48
// megapixels, which explodes into 100+ MB of raw pixel data once
// decoded, multiplied by several copies during ML inference. Capping
// the working size keeps memory use trivial on any device, and
// doesn't meaningfully hurt quality since the garment is displayed at
// torso-size in the live view either way.
// ---------------------------------------------------------------
const MAX_DIMENSION = 1024;

async function loadDownscaledCanvas(file, maxDimension = MAX_DIMENSION) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxDimension / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  c.getContext('2d', { willReadFrequently: true }).drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  return c;
}

// ---------------------------------------------------------------
// Client-side background removal (fallback path) - no server, no ML
// model. Flood-fills inward from the image border: any
// border-connected region close to the sampled background color
// becomes transparent. Works well for product photos shot on a plain
// backdrop; will not cleanly cut out busy/textured backgrounds.
// Operates in place on the canvas it's given and returns it.
// ---------------------------------------------------------------
const COLOR_THRESHOLD = 34;

async function removeBackgroundFloodFill(sourceCanvas) {
  const width = sourceCanvas.width;
  const height = sourceCanvas.height;
  const cctx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = cctx.getImageData(0, 0, width, height);
  const { data } = imageData;

  const sample = (x, y) => {
    const i = (y * width + x) * 4;
    return [data[i], data[i + 1], data[i + 2]];
  };

  const corners = [sample(0, 0), sample(width - 1, 0), sample(0, height - 1), sample(width - 1, height - 1)];
  const bg = corners.reduce(
    (acc, s) => [acc[0] + s[0] / corners.length, acc[1] + s[1] / corners.length, acc[2] + s[2] / corners.length],
    [0, 0, 0]
  );

  const isBackground = (x, y) => {
    const i = (y * width + x) * 4;
    const dr = data[i] - bg[0];
    const dg = data[i + 1] - bg[1];
    const db = data[i + 2] - bg[2];
    return Math.sqrt(dr * dr + dg * dg + db * db) < COLOR_THRESHOLD;
  };

  const visited = new Uint8Array(width * height);
  const stack = [];
  const trySeed = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const idx = y * width + x;
    if (visited[idx] || !isBackground(x, y)) return;
    visited[idx] = 1;
    stack.push(idx);
  };

  for (let x = 0; x < width; x++) {
    trySeed(x, 0);
    trySeed(x, height - 1);
  }
  for (let y = 0; y < height; y++) {
    trySeed(0, y);
    trySeed(width - 1, y);
  }

  while (stack.length) {
    const idx = stack.pop();
    const x = idx % width;
    const y = (idx / width) | 0;
    data[idx * 4 + 3] = 0;
    trySeed(x + 1, y);
    trySeed(x - 1, y);
    trySeed(x, y + 1);
    trySeed(x, y - 1);
  }

  cctx.putImageData(imageData, 0, 0);
  return sourceCanvas;
}

// ---------------------------------------------------------------
// ML-based background removal (primary path) - a real trained
// segmentation model instead of a color heuristic, so it understands
// "the main subject of this photo" regardless of background
// complexity. Runs via Transformers.js (Hugging Face) + ONNX Runtime
// Web, entirely client-side. Model: ormbg (IS-Net architecture,
// Apache-2.0 licensed - no AGPL entanglement). ~30-40MB, downloaded
// once and cached by the browser from then on.
// ---------------------------------------------------------------
const TRANSFORMERS_CDN_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
const ML_MODEL_ID = 'onnx-community/ormbg-ONNX';

let segmenterPromise = null;
function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = (async () => {
      const { pipeline } = await import(TRANSFORMERS_CDN_URL);
      return pipeline('background-removal', ML_MODEL_ID);
    })();
  }
  return segmenterPromise;
}

// Returns a canvas with the background made transparent, or throws if
// the model can't be loaded/run (caller falls back to the flood fill).
// Takes an already-downscaled canvas, not the raw file.
async function removeBackgroundML(sourceCanvas) {
  const segmenter = await getSegmenter();
  const output = await segmenter(sourceCanvas);
  const source = output.toCanvas(); // actually an OffscreenCanvas, not <canvas>

  // Normalize to a real <canvas> element - OffscreenCanvas has no
  // toDataURL(), which the rest of the app relies on for thumbnails,
  // so without this the upload would succeed but crash one line later.
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  canvas.getContext('2d').drawImage(source, 0, 0);
  return canvas;
}

function computeAnchors(garmentCanvas) {
  const gctx = garmentCanvas.getContext('2d');
  const { width, height } = garmentCanvas;
  const { data } = gctx.getImageData(0, 0, width, height);

  let minX = width, maxX = 0, minY = height, maxY = 0, found = false;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4 + 3] > 10) {
        found = true;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (!found) {
    minX = 0; minY = 0; maxX = width; maxY = height;
  }
  return {
    left: { x: minX, y: minY },
    right: { x: maxX, y: minY },
    bottomCenter: { x: (minX + maxX) / 2, y: maxY }
  };
}

// ---------------------------------------------------------------
// Camera
// ---------------------------------------------------------------
async function initCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showBanner(cameraErrorBanner, 'This browser does not support camera access. Try Chrome or Edge.');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    state.isCameraReady = true;
    updateSpinner();
    startRenderLoop();
  } catch (err) {
    console.error(err);
    showBanner(
      cameraErrorBanner,
      'Camera access was denied or unavailable. Please allow camera permissions in your browser settings and reload the page.'
    );
    updateSpinner();
  }
}

// ---------------------------------------------------------------
// Pose tracking - MediaPipe is loaded straight from jsDelivr's CDN as
// an ES module. Importing the bare package URL (rather than a
// specific bundle file) is deliberate: jsDelivr resolves it to the
// correct ESM entry point, which is the one combination that reliably
// exposes named exports (PoseLandmarker, FilesetResolver) without a
// bundler. Pinned to a specific version so this doesn't break under
// you if a future release changes the API.
// ---------------------------------------------------------------
const MEDIAPIPE_CDN_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

let poseLandmarker = null;

async function initPose() {
  try {
    const { PoseLandmarker, FilesetResolver } = await import(MEDIAPIPE_CDN_URL);
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    async function create(delegate) {
      return PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        numPoses: 1
      });
    }

    try {
      poseLandmarker = await create('GPU');
    } catch {
      // Some devices/browsers don't support the WebGL delegate - fall
      // back to the CPU delegate rather than failing outright.
      poseLandmarker = await create('CPU');
    }

    state.isPoseLoading = false;
    updateSpinner();
    detectLoop();
  } catch (err) {
    console.error(err);
    state.isPoseLoading = false;
    updateSpinner();
    showBanner(poseWarningBanner, null, true);
  }
}

function detectLoop() {
  let lastVideoTime = -1;
  function tick() {
    if (poseLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = poseLandmarker.detectForVideo(video, performance.now());
      latestLandmarks = result.landmarks?.[0] ?? null;
    }
    requestAnimationFrame(tick);
  }
  tick();
}

// ---------------------------------------------------------------
// Render loop: draws the mirrored camera feed to canvas, then
// composites the active garment on top via the affine transform.
// ---------------------------------------------------------------
function startRenderLoop() {
  function drawFrame() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw && vh) {
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      const g = activeGarment();
      if (g) drawGarment(canvas.width, canvas.height, g);
    }
    requestAnimationFrame(drawFrame);
  }
  drawFrame();
}

function drawGarment(cw, ch, garment) {
  const bodyAnchors = getBodyAnchors(latestLandmarks);

  const fallback = {
    leftShoulder: { x: cw * 0.63, y: ch * 0.32 },
    rightShoulder: { x: cw * 0.37, y: ch * 0.32 },
    hipCenter: { x: cw * 0.5, y: ch * 0.58 }
  };

  // Mirror the landmarks' x coordinate to match the mirrored video we
  // just drew (MediaPipe runs on the raw, unmirrored frame).
  const target = bodyAnchors
    ? {
        leftShoulder: { x: cw - bodyAnchors.leftShoulder.x * cw, y: bodyAnchors.leftShoulder.y * ch },
        rightShoulder: { x: cw - bodyAnchors.rightShoulder.x * cw, y: bodyAnchors.rightShoulder.y * ch },
        hipCenter: { x: cw - bodyAnchors.hipCenter.x * cw, y: bodyAnchors.hipCenter.y * ch }
      }
    : smoothed || fallback;

  const prev = smoothed || target;
  const smoothing = 0.75;
  smoothed = {
    leftShoulder: lerpPoint(prev.leftShoulder, target.leftShoulder, 1 - smoothing),
    rightShoulder: lerpPoint(prev.rightShoulder, target.rightShoulder, 1 - smoothing),
    hipCenter: lerpPoint(prev.hipCenter, target.hipCenter, 1 - smoothing)
  };

  const centroid = {
    x: (smoothed.leftShoulder.x + smoothed.rightShoulder.x + smoothed.hipCenter.x) / 3,
    y: (smoothed.leftShoulder.y + smoothed.rightShoulder.y + smoothed.hipCenter.y) / 3
  };
  const applyManualAdjust = (p) => ({
    x: centroid.x + (p.x - centroid.x) * state.manualScale + state.manualOffset.x,
    y: centroid.y + (p.y - centroid.y) * state.manualScale + state.manualOffset.y
  });

  const dstPoints = [
    applyManualAdjust(smoothed.leftShoulder),
    applyManualAdjust(smoothed.rightShoulder),
    applyManualAdjust(smoothed.hipCenter)
  ];

  // A front-facing garment photo has the wearer's LEFT side on the
  // photo's RIGHT (just like looking at a person face-on) - so
  // anchors.right pairs with the person's left shoulder, and vice
  // versa. The flip toggle swaps this for garments that come out
  // mirrored (e.g. an asymmetric logo).
  const srcPoints = state.isFlipped
    ? [garment.anchors.left, garment.anchors.right, garment.anchors.bottomCenter]
    : [garment.anchors.right, garment.anchors.left, garment.anchors.bottomCenter];

  const matrix = computeAffineMatrix(srcPoints, dstPoints);
  if (!matrix) return;

  ctx.save();
  ctx.transform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
  ctx.drawImage(garment.canvas, 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------
// Manual drag-to-reposition on the canvas
// ---------------------------------------------------------------
function toCanvasPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
}

canvas.addEventListener('pointerdown', (e) => {
  if (!activeGarment()) return;
  canvas.setPointerCapture(e.pointerId);
  dragState = { start: toCanvasPoint(e.clientX, e.clientY), offset: { ...state.manualOffset } };
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragState) return;
  const p = toCanvasPoint(e.clientX, e.clientY);
  state.manualOffset = {
    x: dragState.offset.x + (p.x - dragState.start.x),
    y: dragState.offset.y + (p.y - dragState.start.y)
  };
});
function endDrag() {
  dragState = null;
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

// ---------------------------------------------------------------
// Upload handling
// ---------------------------------------------------------------
async function handleFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith('image/'));
  if (files.length === 0) {
    showBanner(uploadErrorBanner, 'Please upload a JPG or PNG image of a dress.');
    return;
  }
  hideBanner(uploadErrorBanner);
  setUploading(true, !segmenterPromise);
  try {
    for (const file of files) {
      const sourceCanvas = await loadDownscaledCanvas(file);
      let garmentCanvas;
      try {
        garmentCanvas = await removeBackgroundML(sourceCanvas);
      } catch (mlErr) {
        console.warn('ML background removal unavailable, using flood-fill fallback:', mlErr);
        garmentCanvas = await removeBackgroundFloodFill(sourceCanvas);
      }
      const anchors = computeAnchors(garmentCanvas);
      const thumbnailUrl = garmentCanvas.toDataURL('image/png');
      const id = nextGarmentId++;
      state.garments.push({ id, name: file.name, canvas: garmentCanvas, anchors, thumbnailUrl });
      setActiveGarment(id);
      state.manualOffset = { x: 0, y: 0 };
      state.manualScale = 1;
      state.isFlipped = false;
      sizeSlider.value = '1';
      flipCheckbox.checked = false;
    }
    renderGarmentStrip();
    updateControlsEnabled();
  } catch (err) {
    console.error(err);
    showBanner(uploadErrorBanner, `Something went wrong processing that image (${err.message || err}). Try a different photo.`);
  } finally {
    setUploading(false);
  }
}

function setActiveGarment(id) {
  state.activeGarmentId = id;
  renderGarmentStrip();
  updateControlsEnabled();
  dragHint.hidden = !activeGarment();
}

function renderGarmentStrip() {
  garmentStrip.innerHTML = '';
  for (const g of state.garments) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'garment-thumb' + (g.id === state.activeGarmentId ? ' garment-thumb--active' : '');
    btn.title = g.name;
    btn.addEventListener('click', () => setActiveGarment(g.id));
    const img = document.createElement('img');
    img.src = g.thumbnailUrl;
    img.alt = g.name;
    btn.appendChild(img);
    garmentStrip.appendChild(btn);
  }
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') fileInput.click();
});
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dropzone--active');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dropzone--active'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dropzone--active');
  if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', (e) => {
  if (e.target.files?.length) handleFiles(e.target.files);
  fileInput.value = '';
});

// ---------------------------------------------------------------
// Fit controls
// ---------------------------------------------------------------
sizeSlider.addEventListener('input', (e) => {
  state.manualScale = parseFloat(e.target.value);
});
flipCheckbox.addEventListener('change', (e) => {
  state.isFlipped = e.target.checked;
});
resetBtn.addEventListener('click', () => {
  state.manualOffset = { x: 0, y: 0 };
  state.manualScale = 1;
  sizeSlider.value = '1';
});
snapshotBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = `try-on-${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
});

function updateControlsEnabled() {
  const has = !!activeGarment();
  sizeSlider.disabled = !has;
  flipCheckbox.disabled = !has;
  resetBtn.disabled = !has;
  snapshotBtn.disabled = !has;
}

// ---------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------
function showBanner(el, message, keepDefaultText = false) {
  if (!keepDefaultText) el.textContent = message;
  el.hidden = false;
}
function hideBanner(el) {
  el.hidden = true;
}
function setUploading(isUploading, isFirstModelLoad = false) {
  dropzoneDefault.hidden = isUploading;
  uploadSpinner.hidden = !isUploading;
  const label = uploadSpinner.querySelector('.spinner__label');
  if (label && isUploading) {
    label.textContent = isFirstModelLoad
      ? 'Downloading segmentation model (one-time)…'
      : 'Analyzing image…';
  }
}
function updateSpinner() {
  if (!state.isCameraReady) {
    stageSpinnerLabel.textContent = 'Starting camera…';
    stageSpinner.hidden = false;
  } else if (state.isPoseLoading) {
    stageSpinnerLabel.textContent = 'Loading pose model…';
    stageSpinner.hidden = false;
  } else {
    stageSpinner.hidden = true;
  }
}

// ---------------------------------------------------------------
// Boot
// ---------------------------------------------------------------
initCamera();
initPose();
