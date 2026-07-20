"""
RTSP streaming pipeline — event-centric filter and chunk in one real-time loop.

Chunk creation logic (chunk duration = N seconds, half_n = fps × N/2 frames):

  Single event at frame F:
      chunk = [F - half_n, F + half_n]  →  N-second clip with F in the middle.

  Multiple events within a cluster:
      Each new trigger extends the window: cluster_end = trigger + half_n.
      Once half_n frames pass with no new trigger the cluster is closed and a
      single chunk is flushed covering [cluster_start, cluster_end].

ALL frames are buffered (including non-interesting ones) so each chunk is a
continuous, watchable video clip centred on the action burst.

Limitations vs. file mode:
  - Detection stage (YOLOv8 + ArcFace) is not supported — too slow for real-time.
  - Parallel segment workers are not applicable (stream is sequential).
  - Audio spike detection is not applicable (no file to extract from).
  - Auto-reconnects on stream drop; partial cluster flushed on Ctrl-C.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.benchmark import (
    BenchmarkSession,
    build_benchmark_report,
    save_benchmark_report,
)
from action_model import create_action_model, ActionPrediction
from preprocess import resize_for_inference
from kafka_producer import (
    KafkaPublisher,
    build_chunk_message,
    chunk_asset_path,
    kafka_settings,
)

log = logging.getLogger(__name__)


class StreamConnectionError(RuntimeError):
    """RTSP source could not be (re)opened within the reconnect budget."""


def _reencode_chunk(src: Path, dst: Path, fps: float | None = None) -> bool:
    """Re-encode OpenCV mp4v output to H.264 MP4 with faststart moov atom.

    fps: when set, applies -vf fps=N to resample the output frame rate.
    """
    tmp = src.with_suffix(".enc.mp4")
    vf_args  = ["-vf", f"fps={fps}", "-r", str(fps)] if fps is not None else []
    aync_arg = ["-async", "1"]                         if fps is not None else []
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i",        str(src),
                "-c:v",      "libx264",
                "-preset",   "fast",
                "-crf",      "23",
                "-pix_fmt",  "yuv420p",
                "-map",      "0:v:0",
                "-map",      "0:a?",
                "-c:a",      "aac",
                "-b:a",      "128k",
                *vf_args,
                *aync_arg,
                "-movflags", "+faststart",
                str(tmp),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        tmp.replace(dst)
        if src != dst:
            src.unlink(missing_ok=True)
        return True
    except Exception as exc:
        log.warning("chunk re-encode failed %s → %s: %s", src.name, dst.name, exc)
        tmp.unlink(missing_ok=True)
        return False


class RTSPStreamProcessor:
    """
    Filter + chunk an RTSP stream using event-centric, N-second chunks.

    The key difference from file mode:
      - ALL frames are stored in a ring buffer (not just interesting ones).
      - A "cluster" tracks the span of frames of interest found so far.
      - Once half_n frames of silence pass (no new frame of interest), the
        cluster is flushed as a single chunk that covers the full event window.
      - The resulting chunk is a normal, continuous video — no frames are
        missing between the action moments.
    """

    def __init__(
        self,
        rtsp_url: str,
        opts,                                   # RunOptions
        *,
        kafka_overrides: dict[str, Any] | None = None,
        stop_event: Any = None,                 # threading.Event | None
        publisher: KafkaPublisher | None = None,
        max_reconnects: int = 10,
        reconnect_backoff_base: float = 2.0,
    ) -> None:
        self.url             = rtsp_url
        if isinstance(opts, dict) or opts is None:
            from runner import options_from_config
            from pathlib import Path
            self.opts = options_from_config(opts or {}, video=Path(rtsp_url))
        else:
            self.opts = opts
        opts = self.opts
        self.kafka_overrides = kafka_overrides or {}
        self.stop_event      = stop_event
        self.max_reconnects  = max_reconnects
        self.reconnect_backoff_base = reconnect_backoff_base
        self._stopped_reason: str = ""

        # Day/night detection and model creation happen in run() after the first
        # frames are read from the already-open cap — opening a second RTSP
        # connection for detection is unreliable and wastes a connection slot.
        self._night_mode: bool = False      # resolved in run()
        self.model      = None              # set in run() if day mode
        self._ir_detector: Any = None      # set in run() if night mode

        # ROI mask — loaded once at startup; None = full frame
        self._roi_mask: np.ndarray | None = None
        _roi_path = getattr(opts, "roi_path", None)
        if _roi_path:
            # Defer mask creation to run() where we know the frame dimensions
            self._roi_path: str | None = _roi_path
        else:
            self._roi_path = None

        # Inference ring — downscaled frames for the action model (day mode only)
        self._infer_ring: deque[np.ndarray] = deque(maxlen=getattr(opts, "clip_len", 16))

        # Frame ring — ALL frames with their frame index, used to build chunks.
        # Capacity is computed in run() once fps is known.
        self._frame_buf:   deque[tuple[int, np.ndarray]] = deque()
        self._buf_capacity: int = 0

        # Active event cluster (in frame indices, inclusive)
        self._cluster_start: int | None = None
        self._cluster_end:   int | None = None

        self._last_pred:  ActionPrediction | None = None
        self._quiet_pred: ActionPrediction | None = None   # baseline during silence
        self._chunk_idx:  int = 0
        self._frame_idx:  int = 0
        self._base_ts:    int = int(time.time() * 1000)
        self._cfg         = kafka_settings(self.kafka_overrides)
        self._publisher   = publisher or KafkaPublisher(self._cfg)
        self._chunks_meta: list[dict] = []   # accumulated per-chunk metadata
        self._bench: BenchmarkSession | None = (
            BenchmarkSession() if getattr(opts, "benchmark_enabled", False) else None
        )

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _build_clip(self, clip_len: int) -> list[np.ndarray]:
        clip = list(self._infer_ring)
        while len(clip) < clip_len:
            clip.insert(0, clip[0])
        return clip[:clip_len]

    def _is_trigger(self, curr: ActionPrediction) -> bool:
        if self._last_pred is None:
            return True
        if curr.class_id != self._last_pred.class_id:
            return True
        # Compare against BOTH the last prediction (catches transitions) and the
        # quiet-baseline prediction (catches sustained action at elevated confidence
        # that would plateau and stop triggering if only delta-from-last were used).
        delta_last  = abs(curr.confidence - self._last_pred.confidence)
        delta_quiet = (
            abs(curr.confidence - self._quiet_pred.confidence)
            if self._quiet_pred is not None else delta_last
        )
        return max(delta_last, delta_quiet) > self.opts.conf_delta

    def _update_cluster(self, trigger_frame: int, half_n: int) -> None:
        """
        Extend (or create) the active cluster to cover
        [trigger_frame - half_n, trigger_frame + half_n].
        """
        start = max(0, trigger_frame - half_n)
        end   = trigger_frame + half_n
        if self._cluster_start is None:
            self._cluster_start = start
            self._cluster_end   = end
            log.debug("cluster open   start=%d  end=%d", start, end)
        else:
            prev_end             = self._cluster_end
            self._cluster_start  = min(self._cluster_start, start)
            self._cluster_end    = max(self._cluster_end, end)
            if self._cluster_end != prev_end:
                log.debug(
                    "cluster extend  start=%d  end=%d  (+%d)",
                    self._cluster_start, self._cluster_end,
                    self._cluster_end - prev_end,
                )

    # ── chunk writer ──────────────────────────────────────────────────────────────

    def _write_chunk(
        self,
        frames:        list[np.ndarray],
        fps:           float,
        src_w:         int,
        src_h:         int,
        cluster_start: int,
        cluster_end:   int,
    ) -> None:
        if not frames:
            return

        opts     = self.opts
        cfg      = self._cfg
        chunk_id = str(uuid.uuid4())

        out_w = getattr(opts, "chunk_width",  None) or src_w
        out_h = getattr(opts, "chunk_height", None) or src_h

        # Destination priority: chunks_dir → output_dir → kafka asset path
        if getattr(opts, "chunks_dir", None):
            dst = Path(opts.chunks_dir) / f"{chunk_id}.mp4"
        elif getattr(opts, "output_dir", None):
            dst = Path(opts.output_dir) / f"{chunk_id}.mp4"
        else:
            dst = Path(
                chunk_asset_path(opts.site_id, opts.camera_id, chunk_id, cfg["assets_base"])
            )

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # OpenCV can only write mp4v; use a temp .mp4 then re-encode to target container
            cv_dst = dst.with_suffix(".tmp.mp4")
            writer  = cv2.VideoWriter(
                str(cv_dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h),
            )
            try:
                if not writer.isOpened():
                    log.error("chunk %d: cannot open writer → %s", self._chunk_idx, cv_dst)
                    return
                for f in frames:
                    if out_w != src_w or out_h != src_h:
                        f = cv2.resize(f, (out_w, out_h))
                    writer.write(f)
            finally:
                writer.release()
            out_fps = getattr(opts, "output_fps", None) or fps
            if self._bench:
                self._bench.start("chunk_write")
            _reencode_chunk(cv_dst, dst, fps=out_fps)
            if self._bench:
                self._bench.end("chunk_write")
        except Exception as exc:
            log.error("chunk %d: write error: %s", self._chunk_idx, exc)
            return

        duration_ms = int(len(frames) / fps * 1000) if fps > 0 else 0
        # Use stream-relative start time for Kafka timestamps
        start_ts   = self._base_ts + int(cluster_start / fps * 1000)
        end_ts     = start_ts + duration_ms
        asset_path = chunk_asset_path(
            opts.site_id, opts.camera_id, chunk_id, cfg["assets_base"]
        )

        # ── Sidecar JSON metadata for this chunk ─────────────────────────────
        chunk_meta: dict = {
            "chunk_id":            chunk_id,
            "chunk_index":         self._chunk_idx,
            "run_id":              opts.run_id,
            "camera_id":           getattr(opts, "camera_id", ""),
            "site_id":             getattr(opts, "site_id", ""),
            "cluster_start_frame": cluster_start,
            "cluster_end_frame":   cluster_end,
            "frame_count":         len(frames),
            "duration_sec":        round(len(frames) / fps, 3) if fps > 0 else 0,
            "start_timestamp_ms":  start_ts,
            "end_timestamp_ms":    end_ts,
            "video_path":          str(dst),
            "last_prediction": {
                "class_id":   self._last_pred.class_id,
                "confidence": round(float(self._last_pred.confidence), 4),
                "label":      self._last_pred.top_label,
            } if self._last_pred else None,
        }
        try:
            json_dst = dst.with_suffix(".json")
            json_dst.write_text(json.dumps(chunk_meta, indent=2))
        except Exception as exc:
            log.warning("chunk %d: could not write metadata JSON: %s", self._chunk_idx, exc)
        self._chunks_meta.append(chunk_meta)

        if cfg.get("enabled"):
            msg = build_chunk_message(
                event_id=str(uuid.uuid4()),
                run_id=opts.run_id,
                chunk_id=chunk_id,
                camera_id=opts.camera_id,
                site_id=opts.site_id,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                chunk_path=asset_path,
                sp_enabled=cfg.get("sp_enabled", "true"),
                critic_enabled=cfg.get("critic_enabled", "true"),
                sp=cfg.get("sp"),
                critic=cfg.get("critic"),
                app_id=int(getattr(opts, "app_id", 0)),
            )
            self._publisher.publish(msg)

        log.info(
            "chunk %d  frames=%d  dur=%.1fs  window=[%d,%d]  → %s",
            self._chunk_idx, len(frames), len(frames) / fps if fps else 0,
            cluster_start, cluster_end, dst,
        )
        self._chunk_idx += 1

    # ── cluster flush ─────────────────────────────────────────────────────────────

    def _flush_cluster(
        self,
        fps:    float,
        width:  int,
        height: int,
        *,
        until:  int | None = None,   # override cluster_end for partial exit flush
    ) -> None:
        """
        Extract frames [cluster_start, cluster_end] from the ring buffer and
        write a chunk.  Resets the cluster state afterwards.
        """
        if self._cluster_start is None:
            return

        c_start = self._cluster_start
        c_end   = until if until is not None else self._cluster_end

        frames = [
            f for (fi, f) in self._frame_buf
            if c_start <= fi <= c_end
        ]

        if not frames:
            log.warning(
                "cluster [%d, %d] has no frames in buffer "
                "(ring buffer too small for this chunk_duration?)",
                c_start, c_end,
            )
        else:
            self._write_chunk(frames, fps, width, height, c_start, c_end)

        self._cluster_start = None
        self._cluster_end   = None

    # ── per-frame processing ──────────────────────────────────────────────────────

    def _step(
        self,
        frame: np.ndarray,
        fps: float,
        width: int,
        height: int,
        half_n: int,
        frames_per_chunk: int,
        clip_len: int,
        sample_stride: int,
        inf_scale: float,
        inf_max_side: int | None,
        buf_capacity: int,
    ) -> None:
        """Buffer one frame, run inference, and manage the active cluster."""
        # Buffer ALL frames so cluster extraction always has enough context
        self._frame_buf.append((self._frame_idx, frame.copy()))
        while len(self._frame_buf) > buf_capacity:
            self._frame_buf.popleft()

        # Apply ROI mask to inference input only; original goes to the chunk
        inf_frame = frame
        if self._roi_mask is not None:
            inf_frame = cv2.bitwise_and(frame, frame, mask=self._roi_mask)

        # Inference — IR every frame, action model at stride
        if self._night_mode:
            if self._bench:
                self._bench.start("inference")
            ir_result = self._ir_detector.process_frame(inf_frame)
            if self._bench:
                self._bench.end("inference")
            if ir_result.is_motion:
                self._update_cluster(self._frame_idx, half_n)
        else:
            resized = resize_for_inference(inf_frame, inf_scale, inf_max_side)
            self._infer_ring.append(resized)
            if self._frame_idx % sample_stride == 0 and self._infer_ring:
                clip = self._build_clip(clip_len)
                if self._bench:
                    self._bench.start("inference")
                pred = self.model.predict_batch([(self._frame_idx, clip)])[0]
                if self._bench:
                    self._bench.end("inference")
                if self._is_trigger(pred):
                    self._update_cluster(self._frame_idx, half_n)
                else:
                    if self._cluster_start is None:
                        self._quiet_pred = pred
                self._last_pred = pred

        # Cluster flush
        if self._cluster_start is not None:
            cluster_age = self._frame_idx - self._cluster_start
            if cluster_age >= frames_per_chunk:
                log.debug("cluster max-dur flush  start=%d  age=%d fr",
                          self._cluster_start, cluster_age)
                self._flush_cluster(fps, width, height)
            elif self._frame_idx > self._cluster_end:
                self._flush_cluster(fps, width, height)

        self._frame_idx += 1

    # ── main loop ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Live counters for external status surfaces (thread-safe reads of ints)."""
        return {
            "url":            self.url,
            "total_frames":   self._frame_idx,
            "total_chunks":   self._chunk_idx,
            "cluster_open":   self._cluster_start is not None,
            "stopped_reason": self._stopped_reason,
        }

    def _should_stop(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def run(self) -> dict[str, Any]:
        """
        Read stream until stop_event is set (or KeyboardInterrupt in CLI use).
        Returns summary: total_frames, total_chunks, chunks_meta, stopped_reason.
        """
        log.info("stream start  url=%s", self.url)

        if self._cfg.get("enabled"):
            self._publisher.connect(wait=True)

        cap = self._open()
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {self.url}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        opts = self.opts

        # ── Sample first frames for day/night detection ───────────────────────
        # Read from the already-open cap — avoids a second RTSP connection that
        # is unreliable and silently defaults to day mode when it fails.
        sat_thresh  = getattr(opts, "day_night_sat_threshold", 30.0)
        n_sample    = getattr(opts, "day_night_sample_frames", 30)
        forced_mode = getattr(opts, "view_mode", "auto")

        sample_frames: list[np.ndarray] = []
        for _ in range(n_sample):
            ok, frame = cap.read()
            if ok:
                sample_frames.append(frame.copy())

        if forced_mode in ("day", "night"):
            _view_mode: str = forced_mode
            log.info("view mode: forced=%s", forced_mode)
        else:
            from day_night_detector import detect_from_frames
            _view_mode = detect_from_frames(sample_frames, saturation_threshold=sat_thresh)

        self._night_mode = (_view_mode == "night")

        # ── Create model / detector ───────────────────────────────────────────
        clip_len      = getattr(opts, "clip_len",          16)
        sample_stride = getattr(opts, "sample_stride",      4)
        inf_scale     = getattr(opts, "inference_scale",    1.0)
        inf_max_side  = getattr(opts, "inference_max_side", None)

        if not self._night_mode:
            self.model = create_action_model(
                clip_len=clip_len,
                prefer_torch=getattr(opts, "prefer_torch",  True),
                device=getattr(opts, "device",              "auto"),
                ensemble=getattr(opts, "ensemble",          False),
                model_path=getattr(opts, "model_path",      None),
                cache_dir=getattr(opts, "model_cache_dir",  None),
            )
            log.info("day mode — action model loaded")
        else:
            from detectors.ir_motion_detector import IRMotionConfig, IRMotionDetector
            _ir_cfg_d: dict[str, Any] = getattr(opts, "ir_mode_cfg", {})
            _ir_cfg = IRMotionConfig(
                method=str(_ir_cfg_d.get("method", "ensemble")),
                sensitivity=str(_ir_cfg_d.get("sensitivity", "medium")),
            )
            _ir_cfg.apply_preset()
            self._ir_detector = IRMotionDetector(_ir_cfg)
            log.info("night mode — IR detector: method=%s  sensitivity=%s",
                     _ir_cfg.method, _ir_cfg.sensitivity)

        # ── Cluster / buffer parameters ───────────────────────────────────────
        half_n           = int(round(fps * opts.chunk_duration_sec / 2))
        frames_per_chunk = int(round(fps * opts.chunk_duration_sec))
        buf_capacity     = int(fps * opts.chunk_duration_sec * 2.5) + 100
        self._buf_capacity = buf_capacity

        # ── ROI mask ──────────────────────────────────────────────────────────
        if self._roi_path and self._roi_mask is None:
            from roi_loader import load_roi
            self._roi_mask = load_roi(self._roi_path, width, height).mask
            log.info("ROI mask loaded from %s", self._roi_path)

        log.info(
            "stream   fps=%.1f  %dx%d  stride=%d  clip_len=%d  "
            "chunk_dur=%.0fs  half_n=%d fr  ring_buf=%d fr  mode=%s  roi=%s",
            fps, width, height, sample_stride, clip_len,
            opts.chunk_duration_sec, half_n, buf_capacity,
            _view_mode, "yes" if self._roi_mask is not None else "no",
        )

        # ── Replay sample frames through the pipeline ─────────────────────────
        # These were read before any model existed; process them now so no
        # motion/action in the opening seconds is missed.
        for frame in sample_frames:
            self._step(frame, fps, width, height, half_n, frames_per_chunk,
                       clip_len, sample_stride, inf_scale, inf_max_side, buf_capacity)

        reconnects = 0
        try:
            while True:
                if self._should_stop():
                    self._stopped_reason = "stop_event"
                    log.info("stream stopped  (stop_event set)")
                    break
                ok, frame = cap.read()
                if not ok:
                    reconnects += 1
                    if reconnects > self.max_reconnects:
                        self._stopped_reason = "reconnects_exhausted"
                        raise StreamConnectionError(
                            f"stream lost and {self.max_reconnects} reconnect "
                            f"attempts failed: {self.url}"
                        )
                    delay = min(
                        60.0, self.reconnect_backoff_base * (2 ** (reconnects - 1))
                    )
                    log.warning(
                        "stream drop — reconnect %d/%d in %.1fs",
                        reconnects, self.max_reconnects, delay,
                    )
                    cap.release()
                    # Sleep in short slices so stop_event stays responsive.
                    t_end = time.monotonic() + delay
                    while time.monotonic() < t_end:
                        if self._should_stop():
                            break
                        time.sleep(min(0.25, max(0.0, t_end - time.monotonic())))
                    if self._should_stop():
                        continue
                    cap = self._open()
                    continue
                reconnects = 0  # healthy read resets the budget

                self._step(frame, fps, width, height, half_n, frames_per_chunk,
                           clip_len, sample_stride, inf_scale, inf_max_side, buf_capacity)

        except KeyboardInterrupt:
            self._stopped_reason = "keyboard_interrupt"
            log.info("stream stopped  (Ctrl-C)")
        finally:
            if self._cluster_start is not None:
                log.info("exit flush  cluster=[%d, %d]",
                         self._cluster_start, self._cluster_end)
                self._flush_cluster(fps, width, height, until=self._frame_idx - 1)
            cap.release()

            # ── Summary JSON ───────────────────────────────────────────────────
            _out = getattr(self.opts, "output_dir", None) or getattr(self.opts, "chunks_dir", None)
            if _out:
                summary = {
                    "run_id":       self.opts.run_id,
                    "url":          self.url,
                    "total_frames": self._frame_idx,
                    "total_chunks": self._chunk_idx,
                    "view_mode":    _view_mode,
                    "chunks":       self._chunks_meta,
                }
                summary_path = Path(str(_out)) / f"{self.opts.run_id}_rtsp_summary.json"
                try:
                    summary_path.write_text(json.dumps(summary, indent=2))
                    log.info("summary → %s", summary_path)
                except Exception as exc:
                    log.warning("could not write summary JSON: %s", exc)

            # ── Benchmark report ───────────────────────────────────────────────
            if self._bench and getattr(self.opts, "benchmark_enabled", False):
                from common.video_io import VideoMeta
                fake_meta = VideoMeta(
                    path=self.url,
                    width=width, height=height,
                    fps=fps,
                    frame_count=self._frame_idx,
                    duration_sec=self._frame_idx / fps if fps > 0 else 0.0,
                )
                bench_out = getattr(self.opts, "benchmark_path", None)
                if bench_out is None and _out:
                    bench_out = Path(str(_out)) / f"{self.opts.run_id}_benchmark.json"
                bench_report = build_benchmark_report(
                    self._bench,
                    video_meta=fake_meta,
                    method="rtsp-stream",
                    selector_model=(
                        type(self.model).__name__ if self.model else "IRMotionDetector"
                    ),
                    benchmark_cfg=getattr(self.opts, "benchmark_cfg", {}),
                )
                if bench_out:
                    save_benchmark_report(bench_report, bench_out)
                    log.info("benchmark → %s", bench_out)
                _b  = bench_report
                tp  = _b.get("throughput", {}).get("pipeline", {})
                mem = _b.get("memory", {})
                dep = _b.get("deployment_recommendation", {})
                log.info("── RTSP Benchmark ───────────────────────────────────────────────")
                log.info("  Total frames       : %d", self._frame_idx)
                log.info("  Total chunks       : %d", self._chunk_idx)
                for ph in ("inference", "chunk_write"):
                    ms = _b.get("timing_ms", {}).get(ph)
                    if ms is not None:
                        log.info("  %-20s : %.2f s", ph, ms / 1000)
                log.info("  Bottleneck         : %s", _b.get("bottleneck_phase", "?"))
                log.info("  Processing fps     : %.1f", tp.get("processing_fps", 0))
                log.info("  ms / frame         : %.1f", tp.get("ms_per_frame", 0))
                log.info("  Real-time factor   : %.2f×  (%s)",
                         tp.get("realtime_factor", 0), tp.get("interpretation", ""))
                log.info("  Peak RAM           : %.0f MB", mem.get("peak_rss_mb", 0))
                log.info("  Peak GPU VRAM      : %.0f MB", mem.get("peak_gpu_allocated_mb", 0))
                log.info("  Deployment tier    : %s", dep.get("label", "?"))
                log.info("─────────────────────────────────────────────────────────────────")

        log.info("stream done  frames=%d  chunks=%d", self._frame_idx, self._chunk_idx)
        return {
            "mode":           "rtsp",
            "url":            self.url,
            "total_frames":   self._frame_idx,
            "total_chunks":   self._chunk_idx,
            "chunks_meta":    self._chunks_meta,
            "stopped_reason": self._stopped_reason or "eof",
            "view_mode":      _view_mode,
        }
