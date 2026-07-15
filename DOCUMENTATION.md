# Action-Aware Video Pipeline — Technical Documentation

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage 1 — Action-Aware Filter](#2-stage-1--action-aware-filter)
   - [Model: R3D-18 (Primary)](#model-r3d-18-primary)
   - [Model: MotionEnergy (Fallback)](#model-motionenergy-fallback)
   - [Model: EnsembleActionModel (--ensemble)](#model-ensembleactionmodel---ensemble)
   - [Audio Energy Spike Detection (--audio-spikes)](#audio-energy-spike-detection---audio-spikes)
   - [Streaming Inference](#streaming-inference)
   - [Frame Selection Logic](#frame-selection-logic)
   - [Correlation Method](#correlation-method)
   - [Parallel Filtering](#parallel-filtering)
3. [Stage 2 — Person Detection & Face Recognition](#3-stage-2--person-detection--face-recognition)
   - [Person Detector: YOLOv8n](#person-detector-yolov8n)
   - [Face Detector: YuNet (default)](#face-detector-yunet-default)
   - [Face Detector: SCRFD det_10g (alternative)](#face-detector-scrfd-det_10g-alternative)
   - [Face Alignment: Procrustes](#face-alignment-procrustes)
   - [Face Embedder: ArcFace R50](#face-embedder-arcface-r50)
   - [Face Registry: Milvus](#face-registry-milvus)
   - [IoU Tracker](#iou-tracker)
4. [Stage 3 — Chunking & Kafka](#4-stage-3--chunking--kafka)
   - [Chunking](#chunking)
   - [Frame-Level Metadata](#frame-level-metadata)
   - [Kafka Message Schema](#kafka-message-schema)
   - [File Storage Paths](#file-storage-paths)
5. [GPU Acceleration](#5-gpu-acceleration)
6. [Output Files](#6-output-files)
7. [Configuration Reference](#7-configuration-reference)
8. [CLI Reference](#8-cli-reference)
9. [RTSP Streaming Mode](#9-rtsp-streaming-mode)

---

## 1. Pipeline Overview

Three independent stages are executed in sequence. Each stage is opt-in; its output becomes the input of the next.

```
Input Video
    │
    ▼  --filter true
┌─────────────────────────────┐
│  Stage 1 — Filter           │  R3D-18 action model → keep frames with
│  selector.py / action_model │  meaningful action changes only
└─────────────────────────────┘
    │  filtered clip
    ▼  --detect true
┌─────────────────────────────┐
│  Stage 2 — Detect           │  YOLOv8n → YuNet/SCRFD → ArcFace →
│  fr_annotate.py / fr_core   │  Milvus → annotated clip with names
└─────────────────────────────┘
    │  annotated clip
    ▼  --chunk true
┌─────────────────────────────┐
│  Stage 3 — Chunk            │  Fixed-duration MP4 chunks +
│  chunk_exporter.py          │  per-frame metadata + Kafka publish
└─────────────────────────────┘
```

**Key identifiers** used throughout:

| Identifier | Purpose |
|---|---|
| `site_id` | Physical site / location (e.g. `site-001`) |
| `camera_id` | Camera within the site (e.g. `cam-001`) |
| `run_id` | UUID for this processing run (e.g. `120abc-test123`) |
| `chunk_id` | UUID auto-generated per chunk (e.g. `4a8df726-…`) |
| `event_id` | UUID auto-generated per Kafka event |

---

## 2. Stage 1 — Action-Aware Filter

**Entry point:** `selector.py → ActionAwareSelector.select()`

The filter scores every `sample_stride`-th frame using a sliding-window action recognition model, then keeps only the frames around detected action changes — discarding visually static or repetitive segments.

---

### Model: R3D-18 (Primary)

**File:** `action_model.py → TorchvisionActionModel`

| Property | Value |
|---|---|
| Architecture | ResNet-3D 18-layer (R3D-18) |
| Source | `torchvision.models.video.r3d_18` |
| Weights | `R3D_18_Weights.DEFAULT` (Kinetics-400 pretrained) |
| Classes | 400 Kinetics action categories |
| Input | `(B, C, T, H, W)` = `(batch, 3, 16, 112, 112)` float32 |
| Normalization | mean `[0.43216, 0.394666, 0.37645]`, std `[0.22803, 0.22145, 0.216989]` |
| Output | 400-dim logit vector → softmax probability distribution |
| Mode | `torch.no_grad()`, `eval()` |

**Preprocessing pipeline per clip:**

```
BGR frames (list of T numpy arrays)
    │
    ├─ BGR → RGB via numpy slice reversal ([:, :, ::-1])
    │
    ├─ Stack to (T, H, W, C) → permute to (T, C, H, W)
    │
    ├─ Divide by 255.0
    │
    ├─ F.interpolate to (T, C, 112, 112) — all T frames in one call
    │   (was a per-frame Python loop; now vectorized on GPU/CPU tensor)
    │
    ├─ Subtract mean, divide by std (cached tensors, no re-allocation)
    │
    └─ Permute to (1, C, T, H, W) — model input format
```

**Batched inference** (`predict_batch`, batch size = `_INFER_BATCH = 8`):

```
8 clips pre-processed on CPU
    │
    ├─ torch.cat → (8, C, T, H, W)
    │
    ├─ Single .to(device) call — one Host→Device transfer
    │
    ├─ model(batch) → (8, 400) logits
    │
    └─ softmax → per-clip class_id, confidence, top_label
```

---

### Model: MotionEnergy (Fallback)

**File:** `action_model.py → MotionEnergyActionModel`

Used when `torch`/`torchvision` is unavailable or `--no-torch` is passed.

| Property | Value |
|---|---|
| Input | Raw BGR frames |
| Method | Frame-difference energy + Canny edge density |
| Classes | 4 pseudo-classes: `static_low`, `static_high`, `motion_low`, `motion_high` |

**Algorithm:**

```
For each consecutive frame pair:
    diff = absdiff(gray[i-1], gray[i])
    energy = mean(diff)          # motion energy
    edge   = mean(Canny(gray[i], 50, 150))  # edge density

e_bin = 1 if energy > 8.0 else 0
s_bin = 1 if edge > 12.0 else 0
class_id = e_bin * 2 + s_bin     # 0-3
confidence = min(1.0, 0.5 + energy/40.0)
```

---

### Model: EnsembleActionModel (--ensemble)

**File:** `action_model.py → EnsembleActionModel`

Combines R3D-18 and MotionEnergy with OR-logic: a frame is kept when **either** sub-model detects a change. Enabled with `--ensemble`.

| Property | Value |
|---|---|
| Sub-models | `TorchvisionActionModel` (R3D-18) + `MotionEnergyActionModel` |
| Combined class space | 400 × 4 = 1600 virtual classes (`r3d_class * 4 + motion_class`) |
| Trigger | `label_changed` fires when either sub-model's class changes |
| Confidence | Weighted blend: `0.7 × r3d_conf + 0.3 × motion_conf` |
| Logits | R3D-18 logits passed through (used for correlation scoring) |

**How the OR-logic works without extra code in `selector.py`:**

```
combined_class_id = r3d_class * 4 + motion_class

If R3D-18 changes (r3d_class changes) but MotionEnergy doesn't:
    combined_class_id changes → selector sees label_changed = True ✓

If MotionEnergy changes (motion_class changes) but R3D-18 doesn't:
    combined_class_id changes → selector sees label_changed = True ✓

If neither changes:
    combined_class_id unchanged → no trigger (as expected)
```

The encoding `r3d * 4 + motion` keeps the comparison in the existing `class_id != prev_class_id` check — no changes needed in the selection logic.

---

### Audio Energy Spike Detection (--audio-spikes)

**File:** `audio_filter.py → AudioSpikeFinder`

An additional frame-keep signal that runs **alongside** the visual model — frames near audio energy spikes are unioned into the `keep` set before anchor gap enforcement. Enabled with `--audio-spikes true`.

**Why:** The action model operates on visual features only. A loud sound (gunshot, shout, breaking glass, door slam) may coincide with minimal visual change at the R3D-18 level. Audio spikes catch these events independently.

**Implementation:** Uses only `ffmpeg` (already required by the pipeline) + `numpy` — no extra Python dependencies.

**Audio extraction:**

```
ffmpeg -i <video> -ac 1 -ar 22050 -f f32le -vn -
  → raw float32 mono PCM at 22050 Hz piped to numpy
```

**RMS energy per frame:**

```
hop = floor(22050 / video_fps)   # samples per video frame
rms[i] = sqrt(mean(audio[i*hop : (i+1)*hop]^2))
```

**Two detection signals:**

| Signal | Trigger | Catches |
|---|---|---|
| `rms_z_thresh` | `RMS[i] > mean(RMS) + z·std(RMS)` | Loud events (shouts, impacts, alarms) |
| `delta_z_thresh` | `\|ΔRMS[i]\| > mean(\|ΔRMS\|) + z·std(\|ΔRMS\|)` | Sudden transitions (onset/offset of activity) |

Both signals are OR-ed; their union forms the raw spike set.

**Spike suppression and expansion:**

```
1. Sort all raw spike indices
2. Suppress spikes closer than min_gap_sec=0.5s (avoids flooding from one long sound)
3. Expand each surviving spike by ±neighbor_pad frames (default 2)
4. Union the resulting frame set into the visual-model keep set
```

**Configuration:**

| Parameter | Default | CLI flag | Description |
|---|---|---|---|
| `rms_z_thresh` | 2.5 | `--audio-rms-z` | Z-score for loud events |
| `delta_z_thresh` | 2.0 | `--audio-delta-z` | Z-score for sudden changes |
| `neighbor_pad` | same as `--neighbor-pad` | — | Frames around each spike |
| `min_gap_sec` | 0.5 | — | Min seconds between spike events |

Lower thresholds (e.g. `--audio-rms-z 1.5`) capture quieter events; raise them to reduce false positives in noisy environments. A video with no audio track is handled gracefully — detection is skipped with a log message.

---

### Streaming Inference

**File:** `selector.py`

Memory-efficient ring buffer approach — constant `O(clip_len)` memory regardless of video length (previously loaded entire video into RAM).

```python
ring = deque(maxlen=clip_len)   # holds last 16 resized frames

for frame_idx, frame in iter_frames(video):
    resized = resize_for_inference(frame, scale, max_side)
    ring.append(resized)

    if frame_idx % sample_stride == 0:
        clip = list(ring)                 # last clip_len frames
        # pad to clip_len by repeating first frame if ring not full yet
        while len(clip) < clip_len:
            clip.insert(0, clip[0])
        pending.append((frame_idx, clip))

        if len(pending) >= _INFER_BATCH:
            flush()                       # GPU batch inference
```

**Inference resolution** is controlled by:
- `--scale 0.5` — scale down by factor (e.g. 0.5 = half resolution)
- `--max-side 480` — cap longest dimension (keeps aspect ratio)

Default is `max_side=480`, meaning a 1920×1080 source is downscaled to `854×480` before R3D-18 input (R3D-18 further resizes to 112×112 internally).

---

### Frame Selection Logic

After all predictions are collected, frames are selected using three rules:

**Rule 1 — Action change trigger (keep frames around changes):**

```python
label_changed = curr.class_id != prev.class_id
conf_jump     = |curr.confidence - prev.confidence| > conf_delta  # default 0.15

if label_changed OR conf_jump:
    keep frames [curr.frame_index - neighbor_pad, curr.frame_index + neighbor_pad]
    # default neighbor_pad = 2, so 5 frames around each change
```

**Rule 2 — Boundary frames:**
- First frame (index 0) always kept
- Last frame always kept

**Rule 3 — Anchor frames (max gap enforcement):**
```python
for consecutive kept frames [a, b]:
    if b - a > max_gap:          # default max_gap = 30
        keep midpoint (a + b) // 2
```

This prevents the filtered clip from having gaps larger than `max_gap` frames, which preserves temporal continuity for downstream detection.

**Rule 4 — Audio energy spikes (optional, `--audio-spikes true`):**

```
For each detected audio spike (RMS or ΔRMS above z-score threshold):
    keep frames [spike_frame - neighbor_pad, spike_frame + neighbor_pad]

Audio-flagged frames are unioned with visual-flagged frames BEFORE anchor
gap enforcement, so anchor midpoints respect the combined keep set.
```

**Result:** `kept_indices` — sorted list of frame indices to write to the filtered clip.

---

### Correlation Method

**File:** `correlation_plot.py`

Correlation measures how **similar** two consecutive model predictions are. A low correlation score (close to 0) indicates an action change; a high score (close to 1) indicates a stable scene.

**Primary method — Cosine similarity of logit vectors:**

```
score = dot(logits_prev, logits_curr) / (||logits_prev|| × ||logits_curr||)
```

- Both vectors are the full 400-class softmax probability distributions from R3D-18.
- Score = **1.0** → identical distributions (no change)
- Score = **0.0** → orthogonal distributions (maximum change)

**Fallback method** (when logits unavailable, e.g. MotionEnergy model):

```
if class_id_prev == class_id_curr:
    score = 1.0 - |confidence_prev - confidence_curr|
else:
    score = 0.0
```

**Correlation timeline** (`build_correlation_timeline`):

Each entry in the timeline represents one prediction sample:

```json
{
  "frame": 48,
  "time_sec": 1.920,
  "correlation": 0.9213,
  "trigger": false,
  "label_changed": false,
  "confidence_delta": 0.0321,
  "prev_label": "walking",
  "curr_label": "walking",
  "detail": null
}
```

`trigger = true` when the frame was kept (either `label_changed` or `confidence_delta > conf_delta`).

The correlation timeline is saved in the filter metadata JSON and can be plotted with `--plot-correlation`.

---

### Parallel Filtering

**File:** `parallel_filter.py`

When `--workers N` (N > 1) is passed and `--device cpu`, the video is split into N equal segments via `ffmpeg -c copy` (no re-encode), processed in parallel using `ProcessPoolExecutor` with `spawn` start method (avoids CUDA-after-fork bugs), then concatenated with a single H.264 encode pass.

> **GPU + parallel workers:** When `device=auto` or `device=cuda`, the worker count is automatically capped to 1. Each worker would otherwise create an independent CUDA context and load R3D-18 into VRAM simultaneously, exhausting GPU memory. Use `--device cpu` to enable multi-worker parallelism.

---

## 3. Stage 2 — Person Detection & Face Recognition

**Entry point:** `fr_annotate.py → annotate_video()`

Per-frame pipeline for each person detected:

```
Frame
  │
  ▼ YOLOv8n person detector
Person bounding boxes (full body)
  │
  ▼ Expand ROI by 5% padding, crop
Person ROI crop
  │
  ▼ YuNet or SCRFD face detector
Face bounding box + 5 keypoints (eyes, nose, mouth corners)
  │
  ▼ Procrustes alignment
112×112 aligned face crop
  │
  ▼ Ellipse background mask + ArcFace R50
512-d unit-norm embedding
  │
  ▼ Milvus cosine similarity search
Name + confidence score (or "Unknown")
  │
  ▼ IoU tracker (temporal smoothing)
Stable label drawn on frame
```

---

### Person Detector: YOLOv8n

**File:** `fr_core.py → PersonDetector`

| Property | Value |
|---|---|
| Model | `yolov8n.onnx` (Ultralytics, COCO) |
| Input | 640×640 RGB float32 [0,1] with letterbox padding (fill=114) |
| Output | `(1, 84, 8400)` — 8400 anchors × (4 bbox + 80 class scores) |
| Person class | Class 0 (COCO) |
| Confidence threshold | `person_conf` (default 0.4) |
| NMS IoU threshold | 0.45 |
| Runtime | ONNX Runtime (CUDA if available) |

**Decode:** Anchors filtered by `score >= conf`, CX/CY/W/H converted to X1/Y1/X2/Y2, de-padded and de-scaled back to original frame coordinates.

---

### Face Detector: YuNet (default)

**File:** `fr_core.py → YuNetDetector`

| Property | Value |
|---|---|
| Model | `face_detection_yunet_2023mar.onnx` (~373 KB) |
| Source | opencv/opencv_zoo |
| Runtime | `cv2.FaceDetectorYN` — **CPU only** (OpenCV DNN has no CUDA in pip builds) |
| Input | Letterboxed to 640×640 |
| Output | `(N, 15)` — `[x, y, w, h, kp0_x, kp0_y, ..., kp4_x, kp4_y, score]` |
| Score threshold | 0.5 |
| NMS threshold | 0.3 |
| Keypoints | right-eye, left-eye, nose-tip, right-mouth, left-mouth |

Preferred for its robustness across all ages (toddlers to elderly).

---

### Face Detector: SCRFD det_10g (alternative)

**File:** `fr_core.py → FaceDetector`

Set `detector_type: "scrfd"` in config to use this instead of YuNet.

| Property | Value |
|---|---|
| Model | `det_10g.onnx` |
| Input | 640×640 float32 normalized [0,1] |
| Strides | [8, 16, 32] (multi-scale anchors) |
| Anchors per cell | 2 per stride |
| Output heads | scores, bboxes, landmarks per stride |
| NMS IoU threshold | 0.4 |
| Runtime | ONNX Runtime (CUDA if available) |

**Decode:** For each stride, grid anchor centres are computed, bboxes and landmarks are decoded from stride-relative offsets, then NMS is applied across all strides.

---

### Face Alignment: Procrustes

**File:** `fr_core.py → align_face()`

Aligns detected face keypoints to a 112×112 canonical coordinate system used by ArcFace.

**Reference keypoints (InsightFace standard):**

```
Right-eye:        [38.29, 51.70]
Left-eye:         [73.53, 51.50]
Nose tip:         [56.03, 71.74]
Right-mouth:      [41.55, 92.37]
Left-mouth:       [70.73, 92.20]
```

**Method:**

```
1. Centre-subtract source and reference keypoints
2. Compute cross-covariance matrix C = src_c.T @ dst_c
3. SVD decomposition: U, S, Vt = svd(C)
4. Rotation matrix: R = Vt.T @ U.T
5. Scale: s = sum(dst_c * (src_c @ R)) / sum(src_c^2)
6. Translation: t = dst_mean - s * (src_mean @ R)
7. Apply affine warp: cv2.warpAffine(frame, M_inv, (112, 112))
```

After alignment, a soft ellipse mask is applied (`preprocess_face`) to suppress background pixels — the region outside the ellipse is filled with neutral gray (127.5) so ArcFace ignores it.

---

### Face Embedder: ArcFace R50

**File:** `fr_core.py → FaceEmbedder`

| Property | Value |
|---|---|
| Model | `w600k_r50.onnx` (ArcFace ResNet-50, trained on WebFace600K) |
| Input | 112×112 RGB float32, normalized to `[-1, 1]` via `(pixel - 127.5) / 127.5` |
| Output | 512-dimensional embedding vector |
| Post-processing | L2-normalize to unit norm (`out / ||out||`) |
| Runtime | ONNX Runtime (CUDA if available) |

The resulting vector is a point on the 512-d unit hypersphere. Angular distance (or equivalently cosine similarity) between two such vectors measures face similarity.

---

### Face Registry: Milvus

**File:** `fr_core.py → FaceDB`

| Property | Value |
|---|---|
| Backend | Milvus vector database |
| Host | `10.178.120.159:19530` (from config) |
| Collection | `face_registry` |
| Index type | `FLAT` (exact, no approximation) |
| Metric | `COSINE` similarity |
| Embedding dim | 512 |
| Scoping | Every query is filtered by `site_id` |

**Schema:**

| Field | Type | Notes |
|---|---|---|
| `id` | VARCHAR(64) | UUID, primary key |
| `site_id` | VARCHAR(128) | Site-scoped isolation |
| `person_name` | VARCHAR(256) | Display name |
| `embedding` | FLOAT_VECTOR(512) | ArcFace unit-norm vector |
| `notes` | VARCHAR(512) | Optional metadata |

**Search:** Top-1 cosine similarity search within `site_id`. Match is accepted only if `score >= similarity_threshold` (default 0.45). Returns `{name, score, id}` or `None`.

---

### IoU Tracker

**File:** `fr_core.py → IoUTracker`

Temporally smooths face recognition labels across frames so a person's name doesn't flicker when face detection is skipped or fails.

| Parameter | Default | Description |
|---|---|---|
| `iou_thresh` | 0.4 | Minimum box overlap to match a detection to a track |
| `max_age` | 30 | Frames before a track is discarded if not seen |
| `history_len` | 7 | Sliding window of past name predictions per track |

**Name smoothing:** The track's displayed name is the **majority vote** over the last `history_len` predictions. A person who is briefly mis-identified in one frame won't change their displayed label.

**Update cycle per frame:**

```
1. Age all active tracks (age += 1)
2. For each detection, find best IoU match among existing tracks
3. If IoU >= threshold → update track (update box, push name to history ring)
4. Else → spawn new track
5. Remove tracks with age > max_age
6. Return only age-0 tracks (visible this frame)
```

The detection stage runs every `frame_skip` frames (default 5); in between, the last `tracked_state` is redrawn on each frame without re-running inference.

---

## 4. Stage 3 — Chunking & Kafka

**Entry point:** `chunk_exporter.py → split_and_publish_chunks()`

---

### Chunking

The annotated (or filtered, if detect was skipped) clip is split into fixed-duration chunks by reading frames sequentially with OpenCV:

```
frames_per_chunk = round(clip_fps × chunk_duration_sec)  # e.g. 25fps × 10s = 250 frames

buffer = []
for frame in cap.read():
    buffer.append(frame)
    if len(buffer) >= frames_per_chunk:
        flush_chunk(buffer)     # write chunk, build metadata, publish
        buffer = []

flush_chunk(buffer)             # final partial chunk
```

Each chunk is assigned a UUID (`chunk_id`) and written to disk, optionally resized (`--chunk-width`, `--chunk-height`), and re-encoded to H.264 for web compatibility.

---

### Frame-Level Metadata

Every frame in every chunk gets a metadata record:

```json
{
  "frame_id": "3f8e1a2b-...",
  "source_frame_number": 412,
  "filtered_index": 87,
  "position_in_chunk": 12,
  "source_time_sec": 16.48,
  "timestamp_ms": 1750000000412,
  "chunk_id": "4a8df726-...",
  "chunk_index": 1
}
```

| Field | Description |
|---|---|
| `source_frame_number` | Frame index in the original input video |
| `filtered_index` | Frame index in the filtered clip |
| `position_in_chunk` | 0-based position within this chunk |
| `source_time_sec` | Timestamp relative to original video start |
| `timestamp_ms` | Absolute epoch milliseconds (`base_ts + source_time_sec × 1000`) |

Metadata is saved in two places:
- **Per-chunk sidecar:** `<chunk_id>.frames.json` next to each chunk file
- **Run-level JSON:** `<run_id>_frames_metadata.json` — all frames for the entire run

---

### Kafka Message Schema

One message is published per chunk to topic `semantic-chunks-data`:

```json
{
  "event_id": "uuid",
  "camera_id": "cam-001",
  "site_id": "site-001",
  "chunk_id": "4a8df726-c302-40ea-8df7-804a96d35139",
  "run_id": "120abc-test123",
  "start_timestamp": 1750000000000,
  "end_timestamp": 1750000010000,
  "metadata": {
    "chunk_format": "mp4",
    "path": "/jvadata/vst/assets/site-001/cam-001/4a8df726-....mp4",
    "sp_enabled": "true",
    "critic_enabled": "true",
    "alert_level": {
      "sp": "true",
      "critic": "true"
    }
  },
  "event_metadata": {
    "chunk_id": "4a8df726-...",
    "chunk_index": 0,
    "run_id": "120abc-test123",
    "site_id": "site-001",
    "camera_id": "cam-001",
    "start_timestamp": 1750000000000,
    "end_timestamp": 1750000010000,
    "source_fps": 25.0,
    "frame_count": 250,
    "frames_sidecar": "/jvadata/vst/assets/site-001/cam-001/4a8df726-....frames.json",
    "frames_metadata_file": "/jvadata/vst/assets/site-001/cam-001/full/<run_id>_frames_metadata.json",
    "frames": [ ... ]
  }
}
```

**`alert_level` logic:**
- `sp` / `critic` in `alert_level` use the explicit `--sp` / `--critic` CLI value if provided
- Otherwise they inherit the corresponding `--sp-enabled` / `--critic-enabled` flag value
- This allows `sp_enabled=true` (feature is on) but `sp="false"` (this specific clip is not actionable)

**`embed_frame_metadata`** (config `kafka.embed_frame_metadata`, default `true`):
- `true` → full `frames` array in every message
- `false` → only `first_frame` + `last_frame` summary; full data in sidecar files

---

### File Storage Paths

| File | Path |
|---|---|
| Filtered clip | `/jvadata/vst/assets/<site>/<camera>/<run_id>.mp4` |
| Filter metadata | `/jvadata/vst/assets/<site>/<camera>/<run_id>_metadata.json` |
| Chunk video | `/jvadata/vst/assets/<site>/<camera>/<chunk_id>.mp4` |
| Chunk sidecar | `/jvadata/vst/assets/<site>/<camera>/<chunk_id>.frames.json` |
| Run frames metadata | `/jvadata/vst/assets/<site>/<camera>/full/<run_id>_frames_metadata.json` |
| Local output dir | `<video_dir>/action_aware_output/` |
| Kafka debug log | `output/kafka_debug.log` |
| Kafka pending spool | `output/kafka_pending.jsonl` (when broker unreachable) |

All paths use `assets_base` from `config.json → kafka.assets_base` (default `/jvadata/vst/assets`).

---

## 5. GPU Acceleration

| Component | Device | Notes |
|---|---|---|
| R3D-18 (filter) | CUDA / CPU | `--device auto` uses CUDA if available |
| YOLOv8n (person detect) | CUDA / CPU | Via `onnxruntime-gpu` |
| SCRFD det_10g (face detect) | CUDA / CPU | Via `onnxruntime-gpu` |
| ArcFace R50 (face embed) | CUDA / CPU | Via `onnxruntime-gpu` |
| YuNet (face detect) | **CPU only** | `cv2.FaceDetectorYN` has no CUDA in pip OpenCV |

**Device resolution for R3D-18:**

```
device="auto"  →  "cuda" if torch.cuda.is_available() else "cpu"
device="cuda"  →  "cuda" (error if CUDA unavailable)
device="cpu"   →  "cpu"
```

**Device resolution for ONNX models:**

```
device="auto"  →  [CUDAExecutionProvider, CPUExecutionProvider]  if onnxruntime-gpu installed
               →  [CPUExecutionProvider]  otherwise
device="cuda"  →  [CUDAExecutionProvider, CPUExecutionProvider]  (raises if not found)
device="cpu"   →  [CPUExecutionProvider]
```

**Install for GPU (CUDA 12.1 / H200):**

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu>=1.17
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. Output Files

After a full pipeline run (`--filter --detect --chunk --kafka`) the following files are created under `<video_dir>/action_aware_output/`:

| File | Description |
|---|---|
| `<stem>_filtered.mp4` | Action-filtered clip (kept frames only) |
| `<stem>_detection.mp4` | Filtered clip with person boxes + name overlays |
| `<stem>_filter_metadata.json` | Filter stats, kept/dropped indices, predictions, correlation timeline |
| `<stem>_kept_indices.json` | Raw list of kept frame indices |
| `<stem>_frames_metadata.json` | Per-frame metadata for all chunks |
| `<stem>_benchmark.json` | CPU/GPU usage, processing time, cost estimate |
| `<stem>_report.json` | Complete pipeline report (all stages) |
| `<stem>_correlation.png` | Correlation score vs time plot (if `--plot-correlation`) |

---

## 7. Configuration Reference

**`config.json` — key fields:**

```jsonc
{
  "device": "auto",              // R3D-18 device: "auto" | "cpu" | "cuda"
  "clip_len": 16,                // frames per sliding window clip
  "sample_stride": 8,            // infer every Nth frame
  "conf_delta": 0.15,            // confidence jump threshold to trigger keep
  "max_gap": 30,                 // max frames between kept frames (anchor rule)
  "neighbor_pad": 2,             // frames to keep around each trigger point
  "inference_max_side": 480,     // downscale long side before R3D-18
  "inference_scale": 1.0,        // additional scale factor
  "output_resolution": { "width": 640, "height": 480 },

  "kafka": {
    "enabled": true,
    "brokers": "10.178.120.135:9092",
    "topic": "semantic-chunks-data",
    "assets_base": "/jvadata/vst/assets",
    "sp_enabled": "true",        // smart-processor flag in message
    "critic_enabled": "true",    // critic flag in message
    "embed_frame_metadata": true // embed full frames array in Kafka message
  },

  "face_recognition": {
    "enabled": false,
    "device": "auto",            // ONNX models device
    "detector_type": "yunet",    // "yunet" (recommended) or "scrfd"
    "similarity_threshold": 0.45,
    "frame_skip": 5,             // run detection every Nth frame
    "track_iou_thresh": 0.4,
    "track_max_age": 30,
    "track_history": 7
  }
}
```

---

## 8. CLI Reference

```
python3 main.py <video> [options]
```

### Stage flags

| Flag | Type | Description |
|---|---|---|
| `--filter true\|false` | bool | Stage 1 — action-aware filter |
| `--detect true\|false` | bool | Stage 2 — person detect + face recognition |
| `--chunk  true\|false` | bool | Stage 3 — split into chunks |
| `--kafka  true\|false` | bool | Publish chunks to Kafka (requires `--chunk true`) |

### Identity

| Flag | Description |
|---|---|
| `--site SITE_ID` | Site identifier |
| `--camera CAMERA_ID` | Camera identifier |
| `--run RUN_ID` | Run UUID |

### Filter options

| Flag | Default | Description |
|---|---|---|
| `--workers N` | 1 | Parallel filter workers (CPU only; auto-capped to 1 on GPU) |
| `--device auto\|cpu\|cuda` | auto | Inference device |
| `--stride N` | 8 | Frame sampling stride |
| `--clip-len N` | 16 | Frames per sliding window |
| `--conf-delta F` | 0.15 | Confidence jump threshold |
| `--max-gap N` | 30 | Max gap between kept frames |
| `--max-side N` | 480 | Downscale inference max side (px) |
| `--scale F` | 1.0 | Additional scale factor |
| `--no-torch` | false | Use MotionEnergy fallback |
| `--ensemble` | false | Run R3D-18 AND MotionEnergy together (OR-logic triggers) |
| `--audio-spikes true\|false` | false | Keep frames near audio energy spikes |
| `--audio-rms-z F` | 2.5 | Z-score for loud events (lower = more sensitive) |
| `--audio-delta-z F` | 2.0 | Z-score for sudden energy changes (lower = more sensitive) |

### Chunk options

| Flag | Default | Description |
|---|---|---|
| `--duration F` | 5 | Chunk length in seconds |
| `--chunk-width N` | source | Chunk output width |
| `--chunk-height N` | source | Chunk output height |
| `--chunks-dir PATH` | none | Also save chunks locally to this folder |
| `--save-clip PATH` | none | Copy entire filtered clip to this path |

### Kafka overrides

| Flag | Description |
|---|---|
| `--sp-enabled true\|false` | Override `metadata.sp_enabled` in message |
| `--critic-enabled true\|false` | Override `metadata.critic_enabled` in message |
| `--sp VALUE` | Override `alert_level.sp` (default: inherits `--sp-enabled`) |
| `--critic VALUE` | Override `alert_level.critic` (default: inherits `--critic-enabled`) |

### Example — full pipeline

```bash
python3 main.py ../baby_vids/N2.mp4 \
  --filter true --detect true --chunk true --kafka true \
  --site site-001 --camera cam-001 --run 120abc-test123 \
  --duration 10 --chunk-width 640 --chunk-height 480 \
  --workers 1
```

### Example — with ensemble model + audio spike detection

```bash
python3 main.py ../baby_vids/N2.mp4 \
  --filter true --detect true --chunk true --kafka true \
  --site site-001 --camera cam-001 --run 120abc-test123 \
  --ensemble --audio-spikes true \
  --duration 10 --chunk-width 640 --chunk-height 480
```

### Example — audio spikes only (no visual ensemble), tuned sensitivity

```bash
python3 main.py ../baby_vids/N2.mp4 \
  --filter true --chunk true --kafka true \
  --site site-001 --camera cam-001 --run 120abc-test123 \
  --audio-spikes true --audio-rms-z 1.5 --audio-delta-z 1.8
```

### Example — RTSP stream

```bash
python3 main.py rtsp://192.168.1.100:554/stream \
  --site site-001 --camera cam-001 --run stream-001 \
  --kafka true --duration 10 --chunk-width 640 --chunk-height 480 \
  --out-dir ./stream_out
```

---

## 9. RTSP Streaming Mode

Pass an `rtsp://` URL as the video input to activate streaming mode. The
pipeline reads live frames from the RTSP camera and merges the filter and
chunk stages into a single real-time loop. No intermediate filtered clip is
written to disk.

### Chunk creation logic

Let `N` = `chunk_duration_sec`, `half_n` = `fps × N / 2` frames.

**Single frame of interest `F`:**

```
chunk = [ F - half_n, F + half_n ]
```

An `N`-second clip with `F` exactly in the middle.

**Multiple frames of interest within a burst:**

Each new trigger extends the cluster end: `cluster_end = trigger + half_n`.
Once `half_n` frames pass with no new trigger the cluster is closed and one
chunk is flushed covering `[cluster_start, cluster_end]`.

All frames in the window are retained (including non-interesting ones) so the
output is a continuous, watchable video clip centred on the action burst.

### Architecture

```
RTSP camera
    │
    ▼  cv2.VideoCapture(rtsp://...)  — auto-reconnects on drop
ALL frames buffered in ring  (capacity ≈ 2.5 × N seconds)
    │
    ▼  action model every sample_stride frames
Trigger detected? (label change OR conf jump > conf_delta)
    │  YES
    ▼
Update event cluster
  • new cluster   → cluster_start = trigger - half_n
                    cluster_end   = trigger + half_n
  • extend cluster → cluster_end = max(cluster_end, trigger + half_n)
    │
    ▼  current_frame > cluster_end?  (= half_n frames of silence)
Extract frames [cluster_start, cluster_end] from ring buffer
    │
    ▼  write MP4 → publish Kafka message
```

### How it differs from file mode

| Property | File mode | RTSP mode |
|---|---|---|
| Input | Local `.mp4` / `.avi` etc. | `rtsp://` or `rtsps://` URL |
| Filter + chunk stages | Separate (filter first, then chunk) | Single merged loop |
| Filtered clip output | Written to disk | Not written (no intermediate file) |
| Detection stage | Supported | Not supported (too slow for real-time) |
| Audio spike detection | Supported | Not supported (no file to extract from) |
| Parallel workers | Supported | Not applicable |
| Stream disconnect | N/A | Auto-reconnects after 2 s |
| Termination | After video ends | `Ctrl-C` (partial chunk flushed on exit) |

### Output path priority

Same as chunk mode in file pipeline:

1. `--chunks-dir PATH` — write chunks here
2. `--out-dir PATH` — write chunks here (if `--chunks-dir` not set)
3. `/jvadata/vst/assets/<site_id>/<camera_id>/<chunk_id>.mp4` — kafka asset path (default)

### Configuration

All filter and chunk options work in RTSP mode:

| Flag | Description |
|---|---|
| `--duration F` | Chunk length in seconds (default 5) |
| `--stride N` | Inference stride (default 4) |
| `--clip-len N` | Sliding window size (default 16) |
| `--conf-delta F` | Trigger sensitivity (default 0.15) |
| `--device auto\|cpu\|cuda` | Inference device |
| `--ensemble` | Use R3D-18 + MotionEnergy OR-logic |
| `--chunk-width N` | Output chunk width |
| `--chunk-height N` | Output chunk height |
| `--out-dir PATH` | Directory for chunk output files |
| `--chunks-dir PATH` | Explicit chunk directory |
| `--kafka true\|false` | Publish chunks to Kafka |
| `--site SITE_ID` | Site identifier (required with `--kafka`) |
| `--camera CAMERA_ID` | Camera identifier (required with `--kafka`) |
| `--run RUN_ID` | Run identifier for Kafka messages (required with `--kafka`) |
