# Action-Aware Video Pipeline

A smart processor for surveillance / CCTV footage. It watches a video (or a live
stream), **throws away the boring frames** where nothing changes, optionally
**draws boxes and names** on the people it sees, **chops** the result into short
clips, and **publishes** those clips plus rich per-frame metadata to **Kafka** so
other systems can react.

The core idea: turn a firehose of mostly-boring footage into a small set of
meaningful, richly-labelled clips — without ever silently losing a clip or its
metadata.

---

## Table of contents

- [How it works (the 4 stages)](#how-it-works-the-4-stages)
- [Install](#install)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Configuration (config.json)](#configuration-configjson)
- [Outputs](#outputs)
- [Frame-level metadata & the Kafka message](#frame-level-metadata--the-kafka-message)
- [Kafka behaviour & resilience](#kafka-behaviour--resilience)
- [Stage 1 — Filtering (the brain)](#stage-1--filtering-the-brain)
- [Stage 2 — Detection & face recognition](#stage-2--detection--face-recognition)
- [Live streams (RTSP)](#live-streams-rtsp)
- [Parallel filtering (--workers)](#parallel-filtering---workers)
- [Supporting tools](#supporting-tools)
- [Troubleshooting](#troubleshooting)

---

## How it works (the 4 stages)

The pipeline is an assembly line. Each station is switched on with `true`/`false`,
and the output of one stage feeds the next.

```
Original video / stream
  --filter true   →  Filtered clip   (keep only the interesting frames)
  --detect true   →  Annotated clip  (person boxes + stable name labels)
  --chunk  true   →  N-second chunks (+ frame-level metadata)
  --kafka  true   →  Publish each chunk to Kafka   (requires --chunk true)
```

- **Stage 1 – Filter:** an action-recognition model watches the video and keeps
  only the moments where something changes (see [the brain](#stage-1--filtering-the-brain)).
- **Stage 2 – Detect:** finds people, tracks them across frames, and draws a
  stable label on each.
- **Stage 3 – Chunk:** cuts the clip into short pieces (default 5 s) and writes
  detailed per-frame notes.
- **Kafka:** sends one JSON message per chunk to a message bus.

`main.py` is the entry point (reads/validates flags), and `runner.py` is the
orchestrator that runs the enabled stages in order and writes a final report.

---

## Install

Requires Python 3.10+ and `ffmpeg` on the PATH.

```bash
pip install -r requirements.txt
# Kafka publishing needs:
pip install confluent-kafka
```

Notes:
- **R3D-18** neural-net weights download once and are cached under `models/`.
- The **face-recognition** models (ONNX) live under `FREmbeddings/Model/` and can
  auto-download/build on first use.
- Face recognition also needs a running **Milvus** vector database (see config).

---

## Quick start

```bash
# Filter only
python3 main.py N2.mp4 --filter true --run abc-123

# Filter + detect
python3 main.py N2.mp4 --filter true --detect true --site site-001 --run abc-123

# Full pipeline (filter → detect → chunk → publish to Kafka)
python3 main.py N2.mp4 --filter true --detect true --chunk true --kafka true \
  --site site-001 --camera cam-001 --run abc-123

# Chunk only, save chunks locally, no Kafka
python3 main.py N2.mp4 --chunk true --chunks-dir ./out \
  --site site-001 --camera cam-001 --run abc-123

# Live RTSP stream → filter + chunk + publish (detect is skipped live)
python3 main.py "rtsp://user:pass@camera/stream" --chunk true --kafka true \
  --site site-001 --camera cam-001 --run abc-123
```

### Required identity per stage

| Flag        | Required when                          |
|-------------|----------------------------------------|
| `--run`     | `--filter true`                        |
| `--site`    | `--detect true`                        |
| `--camera`  | `--kafka true`                         |

At least one stage must be selected. `--kafka true` requires `--chunk true`.

---

## Command-line reference

Run `python3 main.py --help` for the full list. Grouped summary:

### Input / config
| Flag | Meaning |
|------|---------|
| `video` | Input file path **or** `rtsp://` / `http(s)://` stream URL (auto-detected) |
| `-c, --config PATH` | Config JSON (default `config.json`) |
| `--out-dir DIR` | Output folder (default `<video_dir>/action_aware_output/`) |
| `-v, --verbose` | Show DEBUG logs |

### Stages (`true`/`false`)
| Flag | Meaning |
|------|---------|
| `--filter` | Stage 1 — action-aware filtering |
| `--detect` | Stage 2 — person detection + face recognition |
| `--chunk` | Stage 3 — split into N-second chunks |
| `--kafka` | Publish chunks to Kafka (requires `--chunk true`) |

### Identity
| Flag | Meaning |
|------|---------|
| `--site SITE_ID` | Site identifier, e.g. `site-001` |
| `--camera CAMERA_ID` | Camera identifier, e.g. `cam-001` |
| `--run RUN_ID` | Run UUID for this clip, e.g. `abc-123` |

### Filter options
| Flag | Meaning (default) |
|------|-------------------|
| `--width` / `--height` | Filtered clip size px (640×480) |
| `--fps` | Output FPS (same as input) |
| `--clip-len` | Frames per analysis window (16) |
| `--stride` | Analyse every Nth frame (config `sample_stride`, 8) |
| `--conf-delta` | Sensitivity to confidence jumps (0.15) |
| `--max-gap` | Max frames between kept frames before forcing a keep (30) |
| `--device auto\|cpu\|cuda` | Inference device (auto) |
| `--workers N` | Parallel filter processes (1). CPU-only; capped to 1 on GPU |
| `--no-torch` | Use the lightweight motion-energy model instead of R3D-18 |
| `--ensemble` | Run R3D-18 **and** motion energy; keep if **either** triggers |
| `--audio-spikes true` | Also keep frames near loud / sudden audio events |
| `--audio-rms-z` / `--audio-delta-z` | Audio sensitivity (2.5 / 2.0) |
| `--max-side` / `--scale` | Downscale frames before inference (speed) |

### Chunk options
| Flag | Meaning (default) |
|------|-------------------|
| `--duration` | Chunk length in seconds (5) |
| `--chunk-width` / `--chunk-height` | Chunk size px (same as input) |
| `--chunks-dir DIR` | Save UUID-named `.mp4` chunks to this folder (works with or without Kafka) |
| `--save-clip PATH` | Copy the entire filtered clip to this exact path |
| `--base-ts MS` | Epoch ms for the first chunk start (default: now) |

### Kafka message overrides (need `--kafka true`)
| Flag | Meaning |
|------|---------|
| `--sp-enabled true\|false` | `sp_enabled` flag in the message |
| `--critic-enabled true\|false` | `critic_enabled` flag in the message |
| `--sp VALUE` / `--critic VALUE` | `alert_level.sp` / `alert_level.critic` values |

### Output overrides
| Flag | Meaning |
|------|---------|
| `--filtered-clip PATH` | Override Stage-1 output path |
| `--detection-clip PATH` | Override Stage-2 output path |
| `--output-clip PATH` | Copy the final stage output here (web-friendly H.264) |
| `--benchmark true` | Collect + save benchmark metrics |
| `--benchmark-out PATH` | Custom benchmark JSON path |
| `--plot-correlation` | Save the correlation-vs-time graph |

---

## Configuration (config.json)

Every setting has a default in `config.json`; any CLI flag overrides it for a
single run. Key blocks:

**Top level (filter / identity / chunk)**

| Key | Default | Meaning |
|-----|---------|---------|
| `input_video` | `""` | Default input if none passed |
| `device` | `auto` | `auto` uses CUDA if present, else CPU |
| `prefer_torch` | `true` | Use R3D-18 (false = motion-energy) |
| `inference_max_side` | `480` | Downscale longest side before inference |
| `inference_scale` | `1.0` | Extra scale factor before inference |
| `clip_len` | `16` | Frames per analysis window |
| `sample_stride` | `8` | Analyse every Nth frame |
| `conf_delta` | `0.15` | Confidence-jump sensitivity |
| `max_gap` | `30` | Max frames between kept frames |
| `neighbor_pad` | `2` | Frames kept on each side of an event |
| `output_resolution` | `640×480` | Filtered clip size |
| `run_id` / `camera_id` / `site_id` | — | Default identity |
| `chunk_duration_sec` | `5` | Chunk length |
| `chunks_dir` | `""` | Default chunk folder |

**`kafka` block**

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Publishing ON by default |
| `required` | `false` | If true, abort when broker unreachable; if false, spool to disk |
| `brokers` | `10.178.120.135:9092` | Broker address(es) |
| `topic` | `semantic-chunks-data` | Kafka topic |
| `assets_base` | `/jvadata/vst/assets` | Where chunks are written for consumers |
| `sp_enabled` / `critic_enabled` | `"true"` | Alert flags in the message |
| `embed_frame_metadata` | `true` | Embed full per-frame list in each message (false = compact summary) |
| `debug` | `broker,topic,msg,protocol` | librdkafka internal logging (set `""` to silence) |

**`face_recognition` block** — `enabled` (false), model paths under
`FREmbeddings/Model/`, `person_conf` (0.4), `detector_type` (`yunet`),
`similarity_threshold` (0.45), tracking params, and **Milvus** `host`/`port`/
`collection` (`face_registry`).

**`benchmark`** (cloud GPU hourly cost) and **`visualization`** (overlay colours/labels).

---

## Outputs

Written to `--out-dir` (or `<video_dir>/action_aware_output/`):

| File | Contents |
|------|----------|
| `<stem>_filtered.mp4` (or `/jvadata/vst/assets/<site>/<camera>/<run>.mp4`) | Filtered clip |
| `<stem>_annotated.mp4` | Stage-2 annotated clip |
| `<stem>_kept_indices.json` | Which original frames were kept |
| `<stem>_filter_metadata.json` | Model, device, reduction ratio, predictions, correlation timeline |
| `<stem>_frames_metadata.json` | **Run-level** list of every frame + which chunk it belongs to |
| `<chunk_id>.mp4` | Individual chunks (in `--chunks-dir` and/or the assets path) |
| `<chunk_id>_frames.json` | **Per-chunk** sidecar with that chunk's frames |
| `<stem>_benchmark.json` | Benchmark metrics (with `--benchmark true`) |
| `<stem>_report.json` | Full run report |
| `output/kafka_debug.log` | Verbose Kafka log |
| `output/kafka_pending.jsonl` | Messages spooled when the broker was down |

---

## Frame-level metadata & the Kafka message

For **every** filtered frame the pipeline records:

| Field | Meaning |
|-------|---------|
| `frame_id` | Unique UUID for the frame |
| `source_frame_number` | Its index in the **original** video |
| `filtered_index` | Its position in the filtered clip (0…K-1) |
| `position_in_chunk` | Its position inside its chunk |
| `source_time_sec` | Time in the original video (seconds) |
| `timestamp_ms` | Real-world epoch time (ms) |
| `chunk_id` / `chunk_index` | Which chunk it belongs to |

This is stored in three places: the **per-chunk sidecar**, the **run-level JSON**,
and **inside each Kafka message** under `event_metadata`.

### Kafka message shape (topic `semantic-chunks-data`)

```json
{
  "event_id": "...",
  "camera_id": "cam-001",
  "site_id": "site-001",
  "chunk_id": "...",
  "start_timestamp": 1000000,
  "end_timestamp": 1005000,
  "run_id": "abc-123",
  "metadata": {
    "chunk_format": "mp4",
    "path": "/jvadata/vst/assets/site-001/cam-001/<chunk_id>.mp4",
    "sp_enabled": "true",
    "critic_enabled": "true",
    "alert_level": { "sp": "true", "critic": "true" }
  },
  "event_metadata": {
    "chunk_id": "...", "chunk_index": 0,
    "run_id": "abc-123", "site_id": "site-001", "camera_id": "cam-001",
    "start_timestamp": 1000000, "end_timestamp": 1005000,
    "source_fps": 30.0, "frame_count": 20,
    "frames_sidecar": "/jvadata/.../<chunk_id>_frames.json",
    "frames_metadata_file": "/jvadata/.../<run>_frames_metadata.json",
    "frames": [
      {
        "frame_id": "...", "source_frame_number": 0, "filtered_index": 0,
        "position_in_chunk": 0, "source_time_sec": 0.0, "timestamp_ms": 1000000,
        "chunk_id": "...", "chunk_index": 0
      }
    ]
  }
}
```

- `event_metadata.frames` carries the **full** per-frame list by default
  (`embed_frame_metadata: true`).
- Set it to `false` (or run `--frame-metadata summary`) to send a **compact**
  summary instead: `first_frame` / `last_frame` + file references, keeping
  messages small. The full data still lives in the sidecar / run-level files.

---

## Kafka behaviour & resilience

- Publishing is **ON by default**. Turn it off with `--kafka false` (or
  `skip_kafka: true` in config).
- Every step is logged to the console **and** `output/kafka_debug.log`, including
  low-level broker chatter (controlled by `kafka.debug`).
- **No data loss if the broker is down:** when `required: false`, chunks are still
  saved to disk and the messages are appended to `output/kafka_pending.jsonl` so
  they can be replayed. When `required: true`, the run aborts instead.
- Watch the log for `CONNECTED …` followed by `SENT chunk_id=…` lines to confirm
  real delivery. If you see `confluent-kafka NOT installed`, run
  `pip install confluent-kafka`.

---

## Stage 1 — Filtering (the brain)

The pipeline samples the video (every `sample_stride` frames) and shows each
snippet to an action-recognition model. A frame is **kept** when:

- the predicted **action label changes** vs. the previous sample, **or**
- the top **confidence jumps** by more than `conf_delta`.

It also always keeps the first/last frame, keeps a few frames around each event
(`neighbor_pad`), and — via `max_gap` — forces an occasional keep during long
quiet stretches so the result stays watchable.

Models (in `action_model.py`):
- **R3D-18** (default): a 3D CNN pre-trained on Kinetics-400 (400 human actions).
- **Motion-energy fallback** (`--no-torch`): measures pixel/edge change; fast,
  no PyTorch needed.
- **Ensemble** (`--ensemble`): runs both and keeps a frame if either triggers.

**Audio** (`--audio-spikes true`) adds frames around loud events and sudden
sound-level changes. Output: the filtered clip plus `kept_indices`, filter
metadata, and a correlation timeline (how stable the scene is over time).

---

## Stage 2 — Detection & face recognition

**In-pipeline detection** (`fr_annotate.py`): uses **YOLO** to box people, tracks
each across frames (IoU + colour re-identification so someone who leaves and
returns keeps their box), and draws a **stable label** per track. Audio is
passed through untouched.

**Face-recognition toolkit** (`fr_*` + `face_recognizer.py`): a separate system to
identify *who* appears. It detects a face, aligns it, turns it into a 512-number
"fingerprint" (ArcFace), and looks it up in **Milvus** (a vector database that
finds the closest stored face). Named faces get **green** boxes; known-but-unnamed
get **orange**. The database is **scoped per site**.

Typical enrolment workflow:

1. `fr_ingest` — scan a video, extract good-quality face fingerprints, dedupe, store.
2. `fr_merge` — cluster fingerprints so each real person becomes one entry.
3. `fr_inspect` / `fr_discover` — build a contact-sheet of face thumbnails to review.
4. `fr_tag` — attach name / role / department (single or bulk CSV).
5. `fr_onboard` — friendly all-in-one: add / list / delete / verify a person.

(`fr_core.py` and `fr_milvus.py` hold the detection/embedding engines and DB helpers.)

---

## Live streams (RTSP)

Pass an `rtsp://` / `http(s)://` URL and the pipeline switches to a real-time loop
(`rtsp_stream.py`). Detection is skipped (too slow live); filtering and chunking
run together. It uses **event clusters**: when action fires, it opens a window
around that moment, extends it while activity continues, then flushes a single
smooth N-second clip centred on the event, writes a sidecar, and publishes to
Kafka. It auto-reconnects on drops and flushes any open cluster on Ctrl-C, then
writes `<run_id>_rtsp_summary.json`.

---

## Parallel filtering (--workers)

`--workers N` splits a long video into N time segments (fast, no re-encode),
filters each in a **separate process**, then concatenates the results. Use it for
long clips on multi-core **CPU** machines. On GPU (`auto`/`cuda`) workers are
automatically capped to 1, because each process would load the model into VRAM
separately and exhaust the GPU.

---

## Supporting tools

| File | Purpose |
|------|---------|
| `benchmark.py` | Measure ms/frame, throughput, real-time factor, RAM/VRAM, edge-device feasibility. `--compare-torch` runs R3D-18 vs motion side-by-side |
| `validate.py` | Correctness self-test on a synthetic video (change detection, event retention, determinism, compression) |
| `loop_test.py` | Soak test: re-run `main.py` repeatedly for a time budget |
| `test_rtsp_mock.py` | Feed a local MP4 into the RTSP processor to test live mode without a camera |
| `correlation_plot.py` | Build / plot the scene-stability (correlation) timeline |
| `preprocess.py` | Downscale frames before inference (speed only) |

---

## Troubleshooting

- **No messages in Kafka.** Check `output/kafka_debug.log`. Ensure
  `confluent-kafka` is installed and the broker in config is reachable. Remember
  publishing needs `--chunk true` alongside `--kafka true`.
- **`camera_id and site_id are required`.** Provide `--site` / `--camera` (or set
  them in config) when using `--detect` / `--kafka`.
- **GPU out of memory with `--workers`.** Multi-worker is CPU-only; on GPU it is
  capped to 1 by design.
- **Messages too large.** Set `kafka.embed_frame_metadata: false` (or
  `--frame-metadata summary`) to send a compact summary; full data stays in the
  sidecar / run-level JSON files.
- **Broker down.** With `kafka.required: false`, chunks are saved and messages are
  spooled to `output/kafka_pending.jsonl` for replay.
