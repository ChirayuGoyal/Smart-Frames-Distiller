# Smart Frames Distiller

Action-aware video pipeline for surveillance and CCTV footage. It keeps frames where something meaningful happens, optionally labels people, splits the result into short clips, and can publish those clips (with per-frame metadata) to Kafka.

> Distill long, mostly-static video into a small set of meaningful clips — without silently dropping content or metadata.

## Pipeline

Each stage is opt-in (`true` / `false`). Output of one stage feeds the next.

```
Input video / RTSP stream
  --filter true   →  Filtered clip   (action-aware frame selection)
  --detect true   →  Annotated clip  (person boxes + face names)
  --chunk  true   →  N-second chunks (+ frame-level metadata)
  --kafka  true   →  Publish chunks to Kafka  (requires --chunk true)
```

| Stage | What it does |
|-------|----------------|
| **Filter** | R3D-18 (or motion-energy fallback) scores sliding windows; keeps frames around action / confidence changes |
| **Detect** | YOLOv8n person detection + YuNet/SCRFD faces + ArcFace embeddings + Milvus lookup |
| **Chunk** | Splits into fixed-duration MP4 chunks with JSON sidecars |
| **Kafka** | Publishes one JSON event per chunk (`confluent-kafka`) |

Entry point: `main.py` → orchestrator: `runner.py`.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on `PATH`
- Optional: CUDA GPU for faster R3D-18 / ONNX inference
- Optional: [Milvus](https://milvus.io/) for face recognition
- Optional: Kafka broker for `--kafka true`

```bash
pip install -r requirements.txt
```

GPU PyTorch (example, CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Not committed to git** (see `.gitignore`): input/output videos (`*.mp4`, …), ONNX weights, and `FREmbeddings/`. Place face models under:

```
FREmbeddings/Model/
  yolov8n.onnx
  face_detection_yunet_2023mar.onnx
  w600k_r50.onnx
  det_10g.onnx          # optional SCRFD alternative
```

## Quick start

```bash
# Filter only
python main.py video.mp4 --filter true --run abc-123

# Filter + detect
python main.py video.mp4 --filter true --detect true --site site-001 --run abc-123

# Full pipeline
python main.py video.mp4 --filter true --detect true --chunk true --kafka true \
  --site site-001 --camera cam-001 --run abc-123

# Chunk only (local files, no Kafka)
python main.py video.mp4 --chunk true --chunks-dir ./out \
  --site site-001 --camera cam-001 --run abc-123

# Live RTSP (detect is skipped in streaming mode)
python main.py "rtsp://user:pass@camera/stream" --chunk true --kafka true \
  --site site-001 --camera cam-001 --run abc-123
```

Defaults also live in `config.json` (`input_video`, Kafka, face recognition, etc.). CLI flags override config for that run.

### Identity requirements

| Flag | Required when |
|------|----------------|
| `--run` | `--filter true` |
| `--site` | `--detect true` |
| `--camera` | `--kafka true` |

At least one stage must be enabled. `--kafka true` requires `--chunk true`.

## Common options

```bash
python main.py --help
```

| Area | Useful flags |
|------|----------------|
| **Input** | `video` (file or `rtsp://`), `-c config.json`, `--out-dir`, `-v` |
| **Filter** | `--device auto\|cpu\|cuda`, `--stride`, `--clip-len`, `--conf-delta`, `--workers N`, `--no-torch`, `--ensemble`, `--audio-spikes true` |
| **Chunk** | `--duration`, `--chunks-dir`, `--chunk-width` / `--chunk-height`, `--save-clip` |
| **Kafka** | `--sp-enabled`, `--critic-enabled`, `--sp`, `--critic` |
| **Extras** | `--benchmark true`, `--plot-correlation`, `--output-clip` |

### Filter tip — ensemble + audio

```bash
python main.py video.mp4 --filter true --ensemble --audio-spikes true \
  --site site-001 --camera cam-001 --run abc-123
```

## Outputs

Default folder: `<video_dir>/action_aware_output/` (or `--out-dir`).

| Artifact | Description |
|----------|-------------|
| `*_filtered.mp4` | Kept frames only |
| `*_annotated.mp4` | Person / face overlays (`--detect`) |
| `*_kept_indices.json` | Indices of kept source frames |
| `*_filter_metadata.json` | Model, reduction ratio, predictions |
| `*_frames_metadata.json` | Run-level frame → chunk map |
| `<chunk_id>.mp4` + `*_frames.json` | Chunks and sidecars |
| `*_report.json` / `*_benchmark.json` | Run summary / timing |
| `output/kafka_pending.jsonl` | Spool when broker is unreachable |

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI |
| `runner.py` | Stage orchestration |
| `selector.py` / `action_model.py` / `preprocess.py` | Action-aware filtering |
| `audio_filter.py` / `parallel_filter.py` | Audio spikes / multi-worker filter |
| `fr_*.py` / `face_recognizer.py` | Face recognition toolkit |
| `chunk_exporter.py` / `kafka_producer.py` | Chunking + Kafka |
| `rtsp_stream.py` | Live stream path |
| `benchmark.py` / `validate.py` | Benchmarks & synthetic checks |
| `config.json` | Defaults |
| `DOCUMENTATION.md` | Deep technical reference |
| `IMPLEMENTATION_GUIDE.md` | Theory + rebuild guide |
| `TIER1_Documentation.md` | Extended ops / CLI / Kafka notes |

### Face recognition helpers

```bash
python fr_onboard.py add --site site-001 --name "Alice" person.mp4
python fr_discover.py scan video.mp4
python fr_inspect.py -c config.json
python fr_tag.py tag --uuid <uuid> --name "Alice" -c config.json
```

## Configuration

Edit `config.json` for defaults: filter hyperparameters, `kafka.*`, `face_recognition.*` (model paths, Milvus host/collection), visualization, and benchmark notes.

Kafka is configured under `kafka` (`brokers`, `topic`, `assets_base`, `required`, …). If `required` is `false` and the broker is down, messages are written to `output/kafka_pending.jsonl` instead of aborting the run.

## Docs

- [DOCUMENTATION.md](DOCUMENTATION.md) — stages, models, Kafka schema, RTSP
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) — algorithms and recreation notes
- [TIER1_Documentation.md](TIER1_Documentation.md) — ops-focused walkthrough
- [RESEARCH.md](RESEARCH.md) — related research notes

## License

[MIT](LICENSE) © 2026 Chirayu Goyal
