# Architecture — Smart Frames Distiller

Action-aware video distillation pipeline (CCTV/surveillance) + FastAPI server.
Filters long, mostly-static video down to meaningful clips, optionally labels
people via face recognition, splits into N-second chunks, and publishes chunk
events to Kafka.

Upstream: https://github.com/ChirayuGoyal/Smart-Frames-Distiller (1 commit).
Everything beyond that commit is local work: missing `common` package
reimplemented, full bug-fix pass, face stacks unified, FastAPI server added.
History: `PLAN.md` (approved plan), `HANDOFF.md` (mid-project handoff),
`findings.md` (two review passes, 29 fixed defects).

## Commands

```bash
pip install -e .                       # editable install (required — imports rely on it)
python main.py video.mp4 --filter true --run r1                       # filter only
python main.py video.mp4 --filter true --chunk true --run r1 \
  --site s1 --camera c1 --out-dir out/                                # filter+chunk
python validate.py                     # synthetic-video validation (motion + torch)
python -m pytest test_faces_unit.py test_server.py -q                 # test suites
uvicorn server.app:create_app --factory                               # API server
```

Env for the server: `SFD_` prefix, `__` nesting (`SFD_KAFKA__BROKERS=host:9092`,
`SFD_MILVUS__HOST=...`, `SFD_ALLOWED_INPUT_DIRS='["D:/videos"]'`,
`SFD_WORK_DIR=...`). See `server/settings.py`. Infra endpoints come ONLY from
settings/env — `config.json` holds per-run pipeline defaults.

## Architecture

Pipeline stages (opt-in, output of one feeds the next), orchestrated by
`runner.run_action_aware(RunOptions) -> report dict`:

1. **Filter** — `parallel_filter.do_filter` → `selector.ActionAwareSelector`:
   R3D-18 (torchvision, `action_model.py`) scores sliding 16-frame windows;
   keeps frames near class/confidence changes. CPU fallback:
   `MotionEnergyActionModel` (thresholds: motion energy > 8.0, edge > 12.0).
   Optional ensemble + audio spikes (`audio_filter.py`). `n_workers>1` splits
   the video via frame-accurate ffmpeg re-encode segments (exact kept-index
   offsets depend on this — do NOT revert to `-c copy`).
2. **Detect** — `faces/annotate.annotate_video`: SCRFD faces + ArcFace
   embeddings + Milvus lookup + IoU tracker; optional YOLOv8n person boxes.
3. **Chunk** — `chunk_exporter.split_and_publish_chunks`: N-second MP4 chunks +
   JSON frame-metadata sidecars. `source_frame_number` mapping comes from the
   filter stage's `kept_indices` (runner passes them through — keep it).
4. **Kafka** — `kafka_producer.KafkaPublisher` (thread-safe, spools to
   `kafka_pending.jsonl` when broker down and `required=false`).

RTSP live path: `rtsp_stream.RTSPStreamProcessor` — ring buffer +
event-cluster chunking, `stop_event` for cooperative shutdown, bounded
reconnects with exponential backoff, `stats()` for live counters.

### Packages

- `common/` — video I/O primitives (`video_io.py`: VideoMeta, iter_frames,
  write_kept_video, try_reencode_h264, mux_audio_to_video), benchmark session +
  report, selection types, metrics, synthetic test video generator. This
  package was reimplemented from call sites (upstream never committed it).
- `faces/` — unified face stack ("Stack B": insightface-standard preprocessing).
  `engine.py` (SCRFD + NMS, ArcFace `/128.0`, align_face, ONNX session cache,
  `resolve_execution_provider` maps auto|cuda|cpu), `tracker.py` (FaceTracker:
  IoU-sorted greedy match, no pre-seeded history, `tick()` ages on skip
  frames), `store.py` (`FaceStore` Protocol, `MilvusFaceStore` with per-host
  alias + typed exceptions, `InMemoryFaceStore` for tests), `persons.py`
  (YOLOv8n), `annotate.py` (FaceRecognizer + annotate_video — degrades to
  detection-only when Milvus unavailable unless `face_recognition.required`),
  `service.py` (ingest/merge/onboard/tag/list/delete/search — no CLI coupling).
- `server/` — FastAPI. `app.py` (factory + lifespan), `settings.py`
  (pydantic-settings), `jobs.py` (JobManager: queue.Queue + worker threads,
  job.json persistence, cooperative cancel via `RunOptions.cancel_event`,
  uploads auto-cleaned at terminal state), `stream_manager.py`,
  `model_cache.py` (config-KEYED caches — never collapse these to singletons),
  `errors.py`, `schemas.py`, `routers/{health,jobs,streams,faces}.py`.
  Endpoints under `/v1/`: jobs (upload OR server_path), streams CRUD, faces
  identities/onboard/search/ingest/merge, health + readiness.
- Root modules — CLI (`main.py`), orchestrator (`runner.py`), plus
  `fr_discover/fr_ingest/fr_inspect/fr_merge/fr_onboard/fr_tag` = thin CLI
  wrappers over `faces/service.py`.

## Milvus schema (the only one)

Collection `face_registry`: `id, person_id, name, role, department, notes,
site_id, camera_id, embedding(512)`, COSINE / IVF_FLAT. Site-scoped: every
read/write filters by `site_id`. Old "Stack A" data (normalized `/127.5`) is
incompatible — drop and re-ingest if such a collection exists.

## Config semantics

- `RunOptions.output_fps=None` = keep source fps (default).
- CLI flags `--workers` / `--audio-spikes` default None → only override
  config.json when actually passed. `--benchmark-out` implies benchmarking.
- Face cfg key normalization: accept both `similarity_thresh` and
  `similarity_threshold`; `device: auto|cuda|cpu` resolves ORT providers
  (`execution_provider` remains an explicit override).
- `tag_identity` fields: None = keep stored value, "" = clear.
- Server: `allowed_input_dirs` empty = server paths DENIED (default-deny);
  uploads sanitized to basename and capped by `max_upload_bytes`.

## Environment facts (this dev machine)

- Windows 11, Python 3.13, torch 2.6.0+cu124 (CUDA works).
- **ffmpeg NOT on PATH** — all ffmpeg helpers detect and degrade (return
  False / keep mp4v output). Parallel split, H.264 re-encode, audio mux can't
  run here; pytest markers `ffmpeg`, `integration`, `gpu` exist in pyproject.
- Face ONNX models NOT on disk (`FREmbeddings/Model/` absent) — engine
  construction raises unless models placed there; integration tests gated.
- R3D-18 weights cached at `models/r3d18_weights.pt` (torch.load with
  `weights_only=True` — keep it).

## Testing

- `test_faces_unit.py` + `test_server.py` (root, not tests/): 8 tests, no
  Milvus/Kafka/ffmpeg/models needed (InMemoryFaceStore, TestClient).
- `validate.py` — deterministic synthetic-video check (retention ≥ 0.95,
  recall ≥ 0.5). `common/synthetic.py` regime parameters (24px step, 120×200
  box) are TUNED to pass — don't "improve" without re-running validation; the
  16-frame clip window inherently lags detection ~8 frames, and
  bigger/faster boxes cause wraparound false positives.
- E2E invariant worth re-checking after pipeline edits: run filter+chunk on
  the synthetic video, assert sidecar `source_frame_number` list equals
  `*_kept_indices.json` and chunk `start_timestamp`s are monotone.

## Gotchas / do-not-regress

- `request.form()` yields **starlette** `UploadFile`; `isinstance` against
  `fastapi.UploadFile` is False → import `UploadFile` from
  `starlette.datastructures` in routers.
- `RunOptions.filter_cancel_check` must stay cancel-ONLY (no progress_cb
  call) — it fires every frame; progress writes per frame hammer job.json.
- `parallel_filter` worker `opts_dict` crosses a process boundary — never put
  callables (`progress_cb`, `cancel_event`, `action_model`) in it.
- `chunk_exporter` imports the shared `log` from `kafka_producer`
  ("kafka_pipeline" logger) — intentional. No logging handlers are installed
  at import time anywhere; CLI opts in via `enable_kafka_debug_log()`.
- Chunk `start_ts` is anchored to the chunk's real first-frame cursor
  (`g0/fps`), not `index*chunk_ms`.
- `model_cache` caches are keyed by config tuples (model params, site_id) —
  a singleton here silently reuses the wrong model/site.
- Kafka message shape lives in `build_chunk_message` — downstream consumers
  depend on `event_metadata` (+ embedded frames when
  `embed_frame_metadata=true`). `list_jobs` strips reports; full report via
  `GET /v1/jobs/{id}`.
- Deleted as dead code (do not resurrect): `fr_core.py`, `fr_annotate.py`
  (random-name annotator), `face_recognizer.py`, `fr_milvus.py`.
- All git work is uncommitted on top of the single upstream commit.
