# Action-Aware Frame Selection — Implementation Guide

> **Note (July 2026 refactor):** the face stack was unified into the `faces/`
> package. Mapping: `fr_core.py` → `faces/persons.py` (PersonDetector),
> `faces/engine.py` (SCRFD/ArcFace/align), `faces/tracker.py` (FaceTracker),
> `faces/store.py` (MilvusFaceStore); `fr_annotate.py`/`face_recognizer.py` →
> `faces/annotate.py`; `fr_milvus.py` → `faces/store.py`. YuNet and the
> random-name annotator were removed. Kafka is OFF unless `--kafka true`
> (there is no `--skip-kafka`). A FastAPI server now lives in `server/`.
> Sections below may describe the pre-refactor layout; current architecture:
> `README.md` and `CLAUDE.md`.



This document explains **only what is implemented** in `07-action-aware`: how the code works, the theory behind it, and enough detail to rebuild the system from scratch.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Project Layout](#3-project-layout)
4. [Dependencies](#4-dependencies)
5. [Core Theory](#5-core-theory)
6. [End-to-End Code Flow](#6-end-to-end-code-flow)
7. [Frame Selection (Deep Dive)](#7-frame-selection-deep-dive)
8. [Action Model Backends](#8-action-model-backends)
9. [Video Outputs](#9-video-outputs)
10. [Kafka Chunk Export](#10-kafka-chunk-export)
11. [Face Recognition (Optional)](#11-face-recognition-optional)
12. [Benchmarking & Validation](#12-benchmarking--validation)
13. [Configuration Reference](#13-configuration-reference)
14. [CLI Reference](#14-cli-reference)
15. [Output Files](#15-output-files)
16. [How to Recreate From Scratch](#16-how-to-recreate-from-scratch)

---

## 1. What This System Does

Given an input video, this POC:

1. **Analyzes** short temporal clips with an action-recognition model.
2. **Decides** which frames to keep based on when the predicted action class or confidence changes.
3. **Produces** a shorter “compressed” video containing only kept frames.
4. **Optionally** writes an annotated review video, a correlation plot, benchmark metrics, Kafka chunk messages, and face-recognition overlays.

**Goal:** Reduce video size while keeping frames where “something is happening” (action changes), instead of keeping every frame.

---

## 2. High-Level Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  main.py    │────▶│    runner.py     │────▶│  ActionAwareSelector │
│  (CLI)      │     │  (orchestration) │     │    (selector.py)     │
└─────────────┘     └────────┬─────────┘     └──────────┬──────────┘
                             │                            │
                             │                   ┌────────▼────────┐
                             │                   │  action_model.py │
                             │                   │  R3D-18 or motion│
                             │                   └────────┬────────┘
                             │                            │
                    ┌────────▼─────────┐         ┌────────▼────────┐
                    │ common/video_io  │         │  preprocess.py  │
                    │ common/visualize │         │  (downscale)    │
                    │ common/benchmark │         └─────────────────┘
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
  ┌──────▼──────┐   ┌────────▼────────┐  ┌──────▼──────────┐
  │ Annotated   │   │ Compressed MP4  │  │ chunk_exporter  │
  │ video       │   │ (kept frames)   │  │ + kafka_producer│
  └─────────────┘   └────────┬────────┘  └─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  faces/annotate    │  (optional, if --detect)
                    │  OpenCV + FR    │
                    └─────────────────┘
```

**Shared library:** `../common/` provides video I/O, visualization, benchmarking, types, metrics, and synthetic test videos. `runner.py` imports from there.

---

## 3. Project Layout

| Path | Role |
|------|------|
| `main.py` | CLI entry point; loads config, builds `RunOptions`, calls `run_action_aware()` |
| `runner.py` | Orchestrates selection, outputs, Kafka, face recognition, benchmarking |
| `selector.py` | Sliding-window inference + keep/drop logic |
| `action_model.py` | R3D-18 (PyTorch) or motion-energy CPU fallback |
| `preprocess.py` | Downscale frames before inference |
| `correlation_plot.py` | Correlation timeline + matplotlib plot |
| `chunk_exporter.py` | Split compressed video into chunks + frame metadata |
| `kafka_producer.py` | Publish chunk events to Kafka (`confluent_kafka`) |
| `faces/engine.py` | SCRFD + ArcFace + Milvus (shared by FR tools) |
| `faces/annotate.py` | OpenCV person detection + face recognition → annotated clip (used by `--detect`) |
| `faces/ package` | YOLOv8n person detector + YuNet face detector + ArcFace embedder + IoUTracker |
| `fr_onboard.py` | Enroll a person into Milvus from a still/video |
| `fr_discover.py` | Scan video → cluster unique faces → register by UUID |
| `fr_ingest.py` | Ingest faces from video into Milvus |
| `fr_merge.py` | Cluster embeddings into person identities |
| `fr_tag.py` | Tag UUIDs with names/roles |
| `fr_inspect.py` | Visual review of untagged faces |
| `benchmark.py` | Benchmark-only CLI |
| `validate.py` | Synthetic cross-validation |
| `config.json` | Default settings |
| `requirements.txt` | Python packages |
| `../common/` | Shared video, viz, benchmark, types, metrics code |

**Model files (not in repo; paths in config):**

- `FREmbeddings/Model/yolov8n.onnx` — YOLOv8n person detector
- `FREmbeddings/Model/face_detection_yunet_2023mar.onnx` — YuNet face detector
- `FREmbeddings/Model/w600k_r50.onnx` — ArcFace R50 embeddings
- `FREmbeddings/Model/det_10g.onnx` — SCRFD face detector (alternative)

---

## 4. Dependencies

From `requirements.txt`:

| Package | Used for |
|---------|----------|
| `numpy` | Arrays, math |
| `opencv-python` | Video read/write, image ops |
| `torch`, `torchvision` | R3D-18 action model |
| `psutil` | RAM/CPU sampling in benchmarks |
| `matplotlib` | Correlation plot |
| `confluent-kafka` | Kafka publishing |

**Additional (face recognition — not in requirements.txt):**

- `onnxruntime` — YuNet, YOLOv8n, ArcFace ONNX inference
- `pymilvus` — vector DB for face embeddings
- `ffmpeg` (system) — H.264 re-encode + web-compatible faststart output

---

## 5. Core Theory

### 5.1 Problem

Surveillance and similar videos have long stretches where the scene is stable. Storing every frame wastes bandwidth and storage. This POC keeps frames where the **action semantics** (as seen by a classifier) change, and drops frames inside stable segments.

### 5.2 Selection Rule (Implemented)

For each sampled frame index `t`, the code runs an action model on a **16-frame clip centered on `t`** and gets:

- `class_id` — argmax class index
- `confidence` — softmax probability of the top class

Compared to the **previous sampled** prediction (stride `k` = `sample_stride`):

```
KEEP frames near t when:
  argmax(logits_t) ≠ argmax(logits_{t-k})
  OR |max_conf_t - max_conf_{t-k}| > conf_delta
```

This is implemented in `selector.py` (not as a separate threshold on “correlation”; correlation is computed for visualization in `correlation_plot.py`).

### 5.3 Extra Keep Rules (Implemented)

| Rule | Parameter | Behavior |
|------|-----------|----------|
| Boundary frames | — | First and last frame always kept |
| Neighbor padding | `neighbor_pad` | When a change is detected at `t`, also keep `t ± neighbor_pad` |
| Max gap anchors | `max_gap` | If two kept frames are farther apart than `max_gap`, insert a frame at the midpoint |

### 5.4 Model Choice

| Backend | Class | When used |
|---------|-------|-----------|
| `TorchvisionActionModel` | `torchvision.models.video.r3d_18` (Kinetics-400, 16 frames) | Default when `prefer_torch=true` and PyTorch is available |
| `MotionEnergyActionModel` | 4 pseudo-classes from motion + edge density | When `--no-torch` or torch unavailable |

R3D-18 is the practical stand-in documented in `action_model.py` comments. **X3D, VideoMAE, InternVideo are not implemented.**

### 5.5 Inference vs Output Resolution

- **Inference:** Frames are downscaled via `preprocess.resize_for_inference()` (`inference_scale`, `inference_max_side`) before being loaded into memory and fed to the model. This speeds up CPU runs.
- **Output videos:** Annotated and compressed outputs use **full source resolution** (compressed can be resized to `output_width` × `output_height`, default 640×480).

### 5.6 Correlation Score (Visualization Only)

`correlation_plot.py` computes a **cosine similarity** between consecutive prediction logit vectors (or a class/confidence proxy if logits are missing). Low correlation aligns with triggers but **does not drive** the keep/drop decision; the decision uses label change and `conf_delta` directly.

---

## 6. End-to-End Code Flow

### Step 1 — CLI (`main.py`)

1. Parse arguments (video path, config overrides, Kafka, face recognition, etc.).
2. Load `config.json` via `runner.load_config()`.
3. Build `RunOptions` via `runner.options_from_config()`.
4. Apply CLI overrides (device, strides, Kafka flags, etc.).
5. Validate: video must exist; if Kafka enabled, `camera_id` and `site_id` required.
6. Call `runner.run_action_aware(opts)` and print JSON report.

### Step 2 — Orchestration (`runner.run_action_aware`)

```
1. Create ActionAwareSelector with RunOptions params
2. BenchmarkSession.start("selection")
3. result = selector.select(video)          ← core algorithm
4. BenchmarkSession.end("selection")
5. read_video_meta(video)
6. resolve_output_paths(video)              ← sibling action_aware_output/ folder
7. write kept_indices JSON
8. Build report dict (stats, predictions, correlation_timeline)
9. Optional: plot_correlation_timeline()
10. Optional: write_annotated_video()       ← common/visualize.py
11. Optional: write_kept_video()            ← common/video_io.py
12. Optional: annotate_video()              ← faces/annotate.py  (--detect)
13. Optional: split_and_publish_chunks()    ← chunk_exporter.py
14. merge_benchmark_into_report()           ← common/benchmark.py
15. Write report JSON to disk
16. Return report
```

### Step 3 — When Outputs Are Written

| Condition | What gets written |
|-----------|-------------------|
| `annotate=true` or `--output-video` | Annotated MP4 + usually compressed MP4 |
| `write_video` or `--compressed-video` | Compressed MP4 only |
| `kafka_enabled` or `chunks_dir` set | Forces compressed video + chunk export |
| `face_recognition.enabled` | Forces compressed video + face-annotated MP4 |
| None of the above | Selection + benchmark only (no video files) |

---

## 7. Frame Selection (Deep Dive)

**File:** `selector.py` — class `ActionAwareSelector`

### 7.1 Initialization

```python
ActionAwareSelector(
    clip_len=16,           # temporal window for R3D-18
    sample_stride=4,       # run model every N frames
    conf_delta=0.15,       # confidence jump threshold
    max_gap=30,            # max frames between kept anchors
    neighbor_pad=2,        # frames kept around each trigger
    prefer_torch=True,
    device="auto",         # "cuda" if available, else "cpu"
    inference_scale=1.0,
    inference_max_side=None,
)
```

### 7.2 `select(video_path)` Algorithm

```
LOAD:
  For each frame in video:
    resize_for_inference(frame) → append to frames[]
  n = len(frames)

INITIALIZE:
  keep = {0, n-1}                    # first and last frame
  predictions = []
  events = []

SAMPLE:
  sample_indices = [0, stride, 2*stride, ...] plus last frame if missing
  For each idx in sample_indices:
    clip = extract 16 frames centered on idx (pad by repeating last frame)
    pred = model.predict_clip(clip, idx)
    predictions.append(pred)

DETECT CHANGES:
  For i = 1 .. len(predictions)-1:
    prev, curr = predictions[i-1], predictions[i]
    label_changed = curr.class_id != prev.class_id
    conf_jump = |curr.confidence - prev.confidence| > conf_delta
    If label_changed OR conf_jump:
      Add frames [curr.frame_index - pad .. curr.frame_index + pad] to keep
      Append DetectedEvent(type="action_change", ...)

ANCHOR GAPS:
  For each consecutive pair in sorted(keep):
    If gap > max_gap: keep.add(midpoint)

RETURN FrameSelectionResult(kept_indices, events, stats, metadata)
```

### 7.3 Clip Extraction (`_extract_clip`)

- `clip_len // 2` frames before center, rest after.
- Clamped to video bounds; padded to exactly `clip_len` by repeating the last available frame.

### 7.4 Metadata Returned

Includes: model name, device, source/inference resolution, per-sample predictions, and `correlation_timeline` from `build_correlation_timeline()`.

---

## 8. Action Model Backends

**File:** `action_model.py`

### 8.1 `TorchvisionActionModel` (R3D-18)

1. Stack 16 BGR frames → RGB.
2. Per-frame bilinear resize to **112×112** (Kinetics R3D-18 input).
3. Normalize with Kinetics mean/std.
4. Tensor shape: `(1, 3, 16, 112, 112)` — batch, channels, time, height, width.
5. Forward pass → softmax → top class + confidence.
6. `top_label` from `R3D_18_Weights.DEFAULT.meta["categories"]` (Kinetics-400 labels).

### 8.2 `MotionEnergyActionModel` (Fallback)

No neural network. For consecutive frame pairs in the clip:

- **Motion energy:** mean absolute grayscale difference.
- **Edge density:** mean Canny edge response.

Bins:

| energy > 8 | edge > 12 | class_id | label |
|------------|-----------|----------|-------|
| no | no | 0 | static_low |
| no | yes | 1 | static_high |
| yes | no | 2 | motion_low |
| yes | yes | 3 | motion_high |

`confidence = min(1.0, 0.5 + energy/40)`.

### 8.3 `create_action_model()`

```python
if prefer_torch and torch available:
    try: return TorchvisionActionModel(...)
    except: pass
return MotionEnergyActionModel()
```

---

## 9. Video Outputs

**Shared code:** `../common/video_io.py`, `../common/visualize.py`

### 9.1 Kept Indices JSON

`{stem}_kept_indices.json` — list of 0-based frame indices to keep, plus selector metadata.

### 9.2 Annotated Video (`write_annotated_video`)

- **Same frame count** as input.
- Frames **not** in `kept_indices`: red border, dimmed, `REMOVE` label, frame number.
- Frames in `kept_indices`: unchanged (optional green border if configured).
- Banner: `Frames to remove: X/Y (Z%)`.
- Written with OpenCV `mp4v`, optionally re-encoded to H.264 via `ffmpeg`.

### 9.3 Compressed Video (`write_kept_video`)

- **Only kept frames**, same FPS as source → shorter duration.
- Optional resize to `output_width` × `output_height` (default 640×480 from config).
- Optional H.264 re-encode.

### 9.4 Output Path Convention

For input `D:/videos/clip.mp4`:

```
D:/videos/action_aware_output/
  clip_annotated.mp4
  clip_compressed.mp4
  clip_face_annotated.mp4      # if face recognition enabled
  clip_report.json
  clip_benchmark.json
  clip_kept_indices.json
  clip_correlation.png         # if --plot-correlation
  clip_frames_metadata.json    # if Kafka/chunk export
```

---

## 10. Kafka Chunk Export

**Files:** `chunk_exporter.py`, `kafka_producer.py`

### 10.1 Purpose

After compression, split the filtered MP4 into **fixed-duration chunks** (default 5 seconds), save them, attach per-frame metadata, and optionally publish one Kafka message per chunk.

Chunking works **independently of Kafka** (`--chunks-dir` saves UUID-named `.mp4` files even with omitting `--kafka true` (Kafka is off by default)).

### 10.2 Flow (`split_and_publish_chunks`)

```
1. Read filtered video metadata (fps, width, height)
2. frames_per_chunk = round(fps * chunk_duration_sec)
3. Optionally copy full filtered clip to --save-full-clip path
4. If Kafka enabled: connect_kafka() with retries
5. Read filtered video frame by frame into buffer
6. When buffer reaches frames_per_chunk (or EOF):
   a. chunk_id = UUID
   b. Write chunk MP4 to:
      - assets path (if Kafka on): {assets_base}/{site_id}/{camera_id}/{chunk_id}.mp4
      - chunks_dir (if set): {chunks_dir}/{chunk_id}.mp4
   c. Build per-frame metadata (see below)
   d. Write sidecar: {chunk}.frames.json
   e. Build Kafka message via build_chunk_message()
   f. Add event_metadata (full or summary frames)
   g. publish_chunk() if Kafka on
7. Write run-level {stem}_frames_metadata.json (local + assets copy if Kafka on)
```

### 10.3 Per-Frame Metadata Record

Each kept frame in a chunk gets:

```json
{
  "frame_id": "<uuid>",
  "source_frame_number": 42,
  "filtered_index": 7,
  "position_in_chunk": 2,
  "source_time_sec": 1.68,
  "timestamp_ms": 1710000001680,
  "chunk_id": "<uuid>",
  "chunk_index": 0
}
```

- `source_frame_number` comes from `kept_indices[filtered_index]` (original video frame).
- `timestamp_ms` = `base_timestamp_ms` + `source_time_sec * 1000`.

### 10.4 Kafka Message Shape (`build_chunk_message`)

```json
{
  "event_id": "<uuid>",
  "camera_id": "cam-001",
  "site_id": "site-001",
  "chunk_id": "<uuid>",
  "start_timestamp": 1710000000000,
  "end_timestamp": 1710000005000,
  "metadata": {
    "chunk_format": "mp4",
    "path": "<assets_base>/site-001/cam-001/<chunk_id>.mp4",
    "sp_enabled": "true",
    "critic_enabled": "true",
    "alert_level": { "sp": "true", "critic": "true" }
  },
  "run_id": "test-run-12345",
  "event_metadata": { "...": "frame-level metadata or summary" }
}
```

### 10.5 Kafka Producer Behavior

- Uses `confluent_kafka.Producer`.
- Logs to console and `output/kafka_debug.log`.
- On broker failure: spools message to `output/kafka_pending.jsonl` (unless `kafka.required=true`, then export aborts).
- Configurable: brokers, topic, SASL/SSL, `embed_frame_metadata` (full vs summary in message).

---

## 11. Face Recognition (Optional)

Activated by the `--detect` flag. Implemented entirely in vanilla Python + OpenCV —
no NVIDIA DeepStream, no GStreamer, no `pyds`.

### 11.1 Pipeline Integration

When `--detect` is passed:

1. Runs on the filtered clip (output of `--filter`) or directly on the source clip.
2. Calls `faces.annotate.annotate_video(clip, output_path, cfg)`.
3. Output: `{stem}_detection.mp4` — person bounding boxes labelled with recognised names.

### 11.2 OpenCV Detection Pipeline (`faces/annotate.py` + `faces/ package`)

```
cv2.VideoCapture → per-frame loop
  → YOLOv8n ONNX  (person detection, class 0)
  → [for each person ROI]
      → YuNet ONNX  (face detection inside ROI)
      → align_face  (Procrustes warp, 5 keypoints → 112×112)
      → preprocess_face  (soft ellipse mask, bg → gray 127.5)
      → ArcFace R50 ONNX  (512-d unit embedding)
      → Milvus COSINE search  (site_id scoped)
      → IoUTracker  (majority-vote name smoothing)
  → cv2.VideoWriter  (annotated frames)
```

**Per-person labelling** (every `frame_skip` frames):

- **Green box:** tagged match — `Alice (87%)`
- **Orange box:** untagged match — `Unknown [b876bdba]`
- **Dim box:** person detected, no face found yet

**Key design choices:**
- YOLOv8n detects full-body person ROI first → YuNet searches for face only inside that ROI (avoids false positives from background faces).
- Soft ellipse mask on the 112×112 face crop suppresses background before embedding.
- `IoUTracker` smooths noisy identity assignments across frames.

### 11.3 Face Models (`faces/ package`)

| Model | File | Role |
|-------|------|------|
| YOLOv8n | `yolov8n.onnx` | Person detection (COCO class 0), 640×640 input |
| YuNet | `face_detection_yunet_2023mar.onnx` | Face detection inside person ROI, all ages |
| ArcFace R50 | `w600k_r50.onnx` | 112×112 aligned crop → 512-d unit vector |

**Alignment:** Procrustes warp using 5 keypoints to a fixed template (`align_face`).

**Background suppression:** Soft ellipse mask + Gaussian blur fill → gray 127.5 outside the face oval, sharpens embedding quality.

**Milvus:** Collection `face_registry` with COSINE index; fields `id`, `person_id`, `name`, `site_id`, `notes`.

### 11.4 Face Database Workflow (Standalone Scripts)

These are **not** called by `main.py`; run manually to build the Milvus database:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `fr_ingest.py video.mp4` | Detect faces, embed, deduplicate, insert new UUIDs into Milvus |
| 2 | `fr_merge.py` | Centroid clustering + merge pass → assign shared `person_id` |
| 3 | `fr_tag.py tag --uuid ... --name "Alice"` | Attach identity metadata |
| 3a | `fr_inspect.py both video.mp4` | Save crops + contact sheet for untagged UUIDs |

**Ingest dedup:** Gate 1 = in-memory cosine (`dedup_thresh`); Gate 2 = Milvus search.

**Ingest quality:** Auto-calibrated min face size and blur threshold from video percentiles.

---

## 12. Benchmarking & Validation

### 12.1 Per-Run Benchmark (`common/benchmark.py`)

Every `main.py` / `benchmark.py` run attaches a `benchmark` block to the report:

- Phase timings: `selection`, `annotate`, `compress`, `kafka_chunks`, `face_recognition`
- Throughput: ms/frame, processing FPS, realtime factor
- Memory: peak RSS, peak GPU allocated
- Edge compatibility vs fixed device budgets (30 fps, 25 fps CCTV, Jetson Orin/NX/Nano, Pi 5)
- Compute cost estimate (CPU core-seconds, GPU seconds, optional cloud $)
- Deployment tier recommendation

### 12.2 `benchmark.py` CLI

```bash
python benchmark.py video.mp4 -c config.json
python benchmark.py video.mp4 --compare-torch   # R3D-18 vs motion fallback
python benchmark.py video.mp4 --no-torch
```

Runs selection (and outputs per `config.json`); prints summary JSON.

### 12.3 `validate.py`

Generates synthetic video (`common/synthetic.make_action_aware_test_video`):

- 90 frames, 3 segments (static → motion → static), change points at frames 30 and 60.

Tests:

| Test | Criterion |
|------|-----------|
| Determinism | Two runs → identical `kept_indices` |
| Change-point recall | `recall >= 0.5` (tolerance ±4 frames) |
| Event retention | `>= 0.95` of GT change points have a kept frame within ±5 |
| Compression | `kept_frames < total_frames` |

Runs with motion fallback always; also with R3D-18 if torch installed.

---

## 13. Configuration Reference

**File:** `config.json`

### 13.1 Core Selection

| Key | Default | Meaning |
|-----|---------|---------|
| `input_video` | `""` | Input path (or pass on CLI) |
| `clip_len` | `16` | Temporal clip length |
| `sample_stride` | `8` | Model inference every N frames |
| `conf_delta` | `0.15` | Confidence jump threshold |
| `max_gap` | `30` | Max gap between kept frames |
| `neighbor_pad` | `2` | Frames kept around each trigger |
| `prefer_torch` | `true` | Use R3D-18 when possible |
| `device` | `"cpu"` | `"auto"`, `"cpu"`, or `"cuda"` |
| `inference_max_side` | `480` | Max longest side for inference |
| `inference_scale` | `1.0` | Additional scale factor |

### 13.2 Outputs

| Key | Default | Meaning |
|-----|---------|---------|
| `annotate` | `true` | Write annotated review video |
| `no_compressed` | `false` | Skip compressed output when annotating |
| `reencode_h264` | `true` | ffmpeg H.264 pass after OpenCV write |
| `output_resolution.width/height` | `640` / `480` | Compressed video size |
| `plot_correlation` | `false` | Save correlation PNG |
| `visualization.*` | see config | Annotated video colors/labels |

### 13.3 Kafka

| Key | Default | Meaning |
|-----|---------|---------|
| `kafka.enabled` | `true` | Publish chunks (use omitting `--kafka true` (Kafka is off by default) to disable) |
| `kafka.brokers` | `localhost:9092` | Bootstrap servers |
| `kafka.topic` | `semantic-chunks-data` | Topic name |
| `kafka.assets_base` | `<assets_base>` | Chunk storage root |
| `kafka.embed_frame_metadata` | `true` | Full frames in message vs summary |
| `camera_id`, `site_id` | required if Kafka on | Identifiers in messages |
| `chunk_duration_sec` | `5` | Chunk length in seconds |
| `run_id` | `test-run-12345` | Run identifier |
| `chunks_dir` | `""` | Local folder for UUID chunks |
| `kafka.enabled` | `false` | Disable Kafka in config |

### 13.4 Face Recognition

| Key | Meaning |
|-----|---------|
| `face_recognition.enabled` | `false` by default |
| `detector_model`, `embedding_model` | ONNX paths |
| `milvus.host/port/collection` | Vector DB |
| `similarity_threshold` | `0.45` — Milvus cosine match threshold |
| `det_threshold` | `0.5` — face detection confidence |
| `person_conf` | `0.4` — YOLOv8n person confidence |
| `frame_skip` | Run FR every N frames |
| `track_iou_thresh`, `track_max_age`, `track_history` | IoUTracker params |

---

## 14. CLI Reference

### 14.1 Basic Run

```bash
cd 07-action-aware
pip install -r requirements.txt

# Set input_video in config.json, then:
python main.py -c config.json

# Or pass video directly:
python main.py /path/to/video.mp4 --annotate
```

### 14.2 Common Flags

| Flag | Effect |
|------|--------|
| `--cpu` / `--device cuda` | Force compute device |
| `--no-torch` | Motion-energy fallback |
| `--inference-max-side 320` | Smaller inference frames |
| `--sample-stride 12` | Fewer model calls |
| `--no-annotate` | Skip annotated video |
| `--no-compressed` | Skip compressed video |
| omitting `--kafka true` (Kafka is off by default) | Disable Kafka (chunks_dir still works) |
| `--chunks-dir /path/to/chunks` | Save local chunk MP4s |
| `--save-full-clip <assets_base>/.../full/N2.mp4` | Copy whole filtered clip |
| `--detect` | Enable face recognition annotation (YOLOv8n + YuNet + ArcFace) |
| `--plot-correlation` | Save correlation PNG |

### 14.3 Face DB Setup (Before Recognition)

```bash
python fr_ingest.py video.mp4 -c config.json
python fr_merge.py -c config.json
python fr_tag.py tag --uuid <uuid> --name "Alice" -c config.json
```

---

## 15. Output Files

### 15.1 Report JSON (`{stem}_report.json`)

Key fields:

```json
{
  "video": "...",
  "method": "action-aware",
  "model": "TorchvisionActionModel",
  "total_frames": 900,
  "kept_frames": 120,
  "reduction_ratio": 7.5,
  "kept_indices": [0, 8, 16, ...],
  "predictions": [...],
  "correlation_timeline": [...],
  "compressed_video": "...",
  "kafka": { "total_chunks": 12, "published_chunks": 12, ... },
  "benchmark": { ... }
}
```

### 15.2 Kafka Debug Artifacts

| File | Content |
|------|---------|
| `output/kafka_debug.log` | Producer + librdkafka logs |
| `output/kafka_pending.jsonl` | Spooled messages when broker down |

---

## 16. How to Recreate From Scratch

### Phase 1 — Minimal Frame Selector

1. **Read video** with OpenCV (`iter_frames` pattern in `common/video_io.py`).
2. **Downscale** each frame for inference (`preprocess.py`).
3. **Load all resized frames into RAM** (current implementation; plan memory for long 4K clips).
4. **Implement clip extraction** — 16 frames centered on index, edge-padded.
5. **Integrate R3D-18:**
   - `torchvision.models.video.r3d_18(weights=DEFAULT)`
   - BGR→RGB, resize 112×112, Kinetics normalize, shape `(1,3,16,112,112)`.
6. **Sample every `sample_stride` frames**, compare consecutive predictions.
7. **Apply keep rules:** first/last, neighbor pad, max-gap anchors.
8. **Write compressed video** — second pass over source, write only `kept_indices`.

### Phase 2 — Review & Metrics

1. **Annotated video** — same frame count, mark removed frames (`common/visualize.py`).
2. **JSON report** — indices, stats, predictions.
3. **Correlation timeline** — cosine on logits (`correlation_plot.py`).
4. **Benchmark session** — time phases, sample RAM/GPU (`common/benchmark.py`).

### Phase 3 — Motion Fallback

1. Implement `MotionEnergyActionModel` — frame differencing + Canny thresholds.
2. `create_action_model()` factory with `prefer_torch` flag.

### Phase 4 — Kafka Export

1. Split compressed MP4 into time-based chunks.
2. Map filtered index → original `source_frame_number` via `kept_indices`.
3. Build chunk + frame metadata JSON.
4. Publish with `confluent_kafka` Producer; spool on failure.

### Phase 5 — Face Recognition (Optional)

1. Place ONNX models under `FREmbeddings/Model/`:
   - `yolov8n.onnx` (person detection)
   - `face_detection_yunet_2023mar.onnx` (face detection)
   - `w600k_r50.onnx` (ArcFace embeddings)
2. Stand up Milvus with `face_registry` collection (512-d COSINE).
3. Run `fr_discover scan` on enrollment videos, then `fr_discover register` to name each person.
4. Pass `--detect --site <site-id>` to the pipeline.

### Phase 6 — Validation

1. Generate synthetic 90-frame test video with known change points (`common/synthetic.py`).
2. Run `validate.py` — check determinism, retention, recall.

### Minimal Directory to Copy

```
07-action-aware/
  main.py, runner.py, selector.py, action_model.py, preprocess.py
  correlation_plot.py, chunk_exporter.py, kafka_producer.py
  config.json, requirements.txt
../common/
  types.py, video_io.py, visualize.py, selection.py, benchmark.py
  metrics.py, synthetic.py
```

Add face recognition / Kafka components only if you need those features.

---

## Quick Reference — Selection Pseudocode

```text
frames = load_and_downscale(video)
keep = {0, len(frames)-1}
preds = [model(clip_at(i)) for i in range(0, n, sample_stride)] + [last]

for each consecutive pair (prev, curr) in preds:
    if curr.class != prev.class OR |curr.conf - prev.conf| > conf_delta:
        keep frames around curr.index ± neighbor_pad

for each gap in sorted(keep) wider than max_gap:
    keep midpoint

write video containing only frames in keep
```

---

*This guide reflects the code as of the current `07-action-aware` implementation. For research background (not implemented models), see `RESEARCH.md`. For a short usage summary, see `README.md`.*
