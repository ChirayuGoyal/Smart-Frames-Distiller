# Smart Frames Distiller

Action-aware video pipeline for surveillance and CCTV footage, usable as a CLI
or a FastAPI server. It keeps frames where something meaningful happens,
optionally labels people via face recognition, splits the result into short
chunks, and can publish those chunks (with per-frame metadata) to Kafka.

> Distill long, mostly-static video into a small set of meaningful clips —
> without silently dropping content or metadata.

## Pipeline

Each stage is opt-in (`true` / `false`). Output of one stage feeds the next.

```
Input video / RTSP stream
  --filter true   →  Filtered clip   (action-aware frame selection)
  --detect true   →  Annotated clip  (person boxes + recognized face names)
  --chunk  true   →  N-second chunks (+ frame-level metadata sidecars)
  --kafka  true   →  Publish chunks to Kafka  (requires --chunk true)
```

| Stage | What it does |
|-------|----------------|
| **Filter** | R3D-18 (or motion-energy fallback) scores sliding windows; keeps frames around action / confidence changes. Optional ensemble + audio-spike detection. |
| **Detect** | SCRFD face detection + ArcFace embeddings + Milvus lookup (`faces/` package); optional YOLOv8n person boxes. Degrades to detection-only when Milvus is unreachable. |
| **Chunk** | Splits into fixed-duration MP4 chunks with JSON sidecars; `source_frame_number` maps every chunk frame back to the original video. |
| **Kafka** | One JSON event per chunk (`confluent-kafka`); spools to `kafka_pending.jsonl` when the broker is down. |

Entry points: `main.py` (CLI) → `runner.py` (orchestrator), or `server/`
(FastAPI).

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on `PATH` (optional but recommended — enables
  H.264 re-encode, audio muxing, and parallel filtering; everything degrades
  gracefully without it)
- Optional: CUDA GPU (torch + onnxruntime-gpu)
- Optional: [Milvus](https://milvus.io/) for face recognition
- Optional: Kafka broker for `--kafka true`

```bash
pip install -r requirements.txt
pip install -e .          # required — packages resolve via the editable install
```

GPU PyTorch (example, CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Face models are **not** committed. Place them under:

```
FREmbeddings/Model/
  yolov8n.onnx                # person boxes (auto-exported via ultralytics if absent)
  det_10g.onnx                # SCRFD face detection
  w600k_r50.onnx              # ArcFace embeddings
```

## CLI quick start

```bash
# Filter only
python main.py video.mp4 --filter true --run abc-123

# Filter + detect (face recognition needs Milvus + models; else person/unknown boxes)
python main.py video.mp4 --filter true --detect true --site site-001 --run abc-123

# Full pipeline
python main.py video.mp4 --filter true --detect true --chunk true --kafka true \
  --site site-001 --camera cam-001 --run abc-123

# Chunk only, local files, no Kafka (Kafka is off unless --kafka true)
python main.py video.mp4 --chunk true --chunks-dir ./out \
  --site site-001 --camera cam-001 --run abc-123

# Live RTSP (detect is skipped in streaming mode)
python main.py "rtsp://user:pass@camera/stream" --chunk true --kafka true \
  --site site-001 --camera cam-001 --run abc-123
```

Defaults live in `config.json` (see `config.example.json`); CLI flags override
config only when actually passed.

| Flag | Required when |
|------|----------------|
| `--run` | `--filter true` |
| `--site` | `--detect true` |
| `--camera` | `--kafka true` |

Useful flags: `--device auto|cpu|cuda`, `--workers N` (parallel filter),
`--ensemble`, `--audio-spikes true`, `--duration N` (chunk seconds),
`--out-dir`, `--benchmark true`, `--fps` (default: same as input). Full list:
`python main.py --help`.

## API server

```bash
uvicorn server.app:create_app --factory
```

Configuration via environment (`SFD_` prefix, `__` nesting) — infra endpoints
come **only** from these settings, never from `config.json`:

```bash
SFD_WORK_DIR=./sfd_data
SFD_KAFKA__BROKERS=broker:9092
SFD_KAFKA__ENABLED=true
SFD_MILVUS__HOST=milvus-host
SFD_ALLOWED_INPUT_DIRS='["D:/videos"]'   # server-side paths DENIED unless set
SFD_MAX_CONCURRENT_JOBS=1
SFD_GPU_SLOTS=1
```

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/jobs` | Submit pipeline job — multipart upload (`file` + `options` JSON) or JSON `{server_path, options}` |
| `GET /v1/jobs[?state=]` · `GET /v1/jobs/{id}` | List / inspect jobs (progress, state) |
| `GET /v1/jobs/{id}/report` · `/artifacts` · `/artifacts/{name}` | Report + output downloads |
| `POST /v1/jobs/{id}/cancel` · `DELETE /v1/jobs/{id}` | Cooperative cancel / purge |
| `POST /v1/streams` · `GET /v1/streams[/{id}]` · `DELETE /v1/streams/{id}` | Start / list / stop RTSP stream workers |
| `POST /v1/faces/onboard` · `/search` | Enroll / search a face (upload or server path) |
| `GET/POST/PATCH/DELETE /v1/faces/identities[/{uid}]` | Identity management |
| `POST /v1/faces/ingest` · `/merge` | Background face ingest / identity merge jobs |
| `GET /v1/health` · `/v1/health/ready` | Liveness / per-dependency readiness |

Jobs run on background worker threads with `job.json` state persistence,
cooperative cancellation, and per-job artifact directories under
`{work_dir}/jobs/{job_id}/output/`.

## Face recognition toolkit

Library: `faces/` (engine, tracker, store, annotate, service). CLI wrappers:

```bash
python fr_onboard.py add --site site-001 --name "Alice" person.mp4
python fr_ingest.py video.mp4 -c config.json      # bulk-ingest embeddings
python fr_merge.py -c config.json                 # cluster into person_ids
python fr_tag.py tag --uuid <uuid> --name "Alice" -c config.json
```

Milvus collection (`face_registry`): `id, person_id, name, role, department,
notes, site_id, camera_id, embedding(512)` — COSINE / IVF_FLAT, site-scoped.

## Outputs

Default folder: `<video_dir>/action_aware_output/` (or `--out-dir`).

| Artifact | Description |
|----------|-------------|
| `*_filtered.mp4` / `*_detection.mp4` | Filtered / annotated clips |
| `*_kept_indices.json` | Indices of kept source frames |
| `*_filter_metadata.json` | Model, reduction ratio, predictions |
| `*_frames_metadata.json` | Run-level frame → chunk map |
| `<chunk_id>.mp4` + `<chunk_id>_frames.json` | Chunks and sidecars |
| `*_report.json` / `*_benchmark.json` | Run summary / timing |
| `kafka_pending.jsonl` | Spool when broker unreachable (`required=false`) |

Chunk destinations when Kafka is on use `kafka.assets_base` (default
`assets/`, relative) — override per deployment.

## Project layout

| Path | Role |
|------|------|
| `main.py` / `runner.py` | CLI and stage orchestration |
| `selector.py` / `action_model.py` / `preprocess.py` | Action-aware filtering |
| `audio_filter.py` / `parallel_filter.py` | Audio spikes / multi-worker filter |
| `common/` | Video I/O, benchmark, metrics, synthetic test video |
| `faces/` | Unified face stack: SCRFD+ArcFace engine, tracker, Milvus store, annotate, service |
| `fr_*.py` | Thin CLI wrappers over `faces/service.py` |
| `chunk_exporter.py` / `kafka_producer.py` | Chunking + Kafka publisher |
| `rtsp_stream.py` | Live stream path (stop-event aware) |
| `server/` | FastAPI app, job manager, stream manager, routers |
| `benchmark.py` / `validate.py` | Benchmarks & synthetic validation |
| `ARCHITECTURE.md` | Architecture notes, gotchas, do-not-regress list |

## Testing

```bash
python -m pytest test_faces_unit.py test_server.py -q   # no Milvus/Kafka/ffmpeg needed
python validate.py                                      # synthetic-video validation
```

Markers `integration` / `gpu` / `ffmpeg` gate tests needing external deps.

## Docs

- [DOCUMENTATION.md](DOCUMENTATION.md) — stages, models, Kafka schema, RTSP
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) — algorithms and recreation notes
- [TIER1_Documentation.md](TIER1_Documentation.md) — ops-focused walkthrough
- [RESEARCH.md](RESEARCH.md) — related research notes

## License

[MIT](LICENSE) © 2026 Chirayu Goyal
