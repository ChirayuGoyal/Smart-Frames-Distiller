"""
server/stream_manager.py — Manage background RTSP stream processing threads.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from rtsp_stream import RTSPStreamProcessor
from server.errors import StreamLimitReachedError, StreamNotFoundError
from server.model_cache import get_kafka_publisher
from server.schemas import StreamResponse, StreamStats
from server.settings import Settings

log = logging.getLogger(__name__)


class StreamInfo:
    def __init__(
        self,
        stream_id: str,
        rtsp_url: str,
        site_id: str,
        camera_id: str,
        processor: RTSPStreamProcessor,
        thread: threading.Thread,
        stop_event: threading.Event,
    ) -> None:
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.site_id = site_id
        self.camera_id = camera_id
        self.processor = processor
        self.thread = thread
        self.stop_event = stop_event
        self.state = "connecting"
        self.error: str | None = None

    def to_response(self) -> StreamResponse:
        pstats = self.processor.stats()
        stats = StreamStats(
            frames_read=pstats.get("total_frames", 0),
            chunks_sent=pstats.get("total_chunks", 0),
            fps_actual=0.0,
        )
        # Check thread health
        if not self.thread.is_alive():
            if self.state in ("connecting", "running", "reconnecting"):
                self.state = "stopped" if self.stop_event.is_set() else "failed"
        return StreamResponse(
            stream_id=self.stream_id,
            rtsp_url=self.rtsp_url,
            site_id=self.site_id,
            camera_id=self.camera_id,
            state=self.state,
            stats=stats,
        )


class StreamManager:
    """Manages active RTSP processing threads capped by `max_streams`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._streams: dict[str, StreamInfo] = {}
        self._lock = threading.Lock()

    def start_stream(
        self, rtsp_url: str, site_id: str, camera_id: str, options: dict[str, Any] | None = None
    ) -> StreamResponse:
        with self._lock:
            # Prune dead stopped/failed threads from map count if needed or check active count
            active = sum(1 for s in self._streams.values() if s.thread.is_alive())
            if active >= self.settings.max_streams:
                raise StreamLimitReachedError(self.settings.max_streams)

            stream_id = str(uuid.uuid4())
            stop_event = threading.Event()

            # Prepare config dict for RTSPStreamProcessor
            cfg = options.copy() if options else {}
            cfg["site_id"] = site_id
            cfg["camera_id"] = camera_id
            if "enabled" not in cfg:
                cfg["enabled"] = True

            from runner import options_from_config
            from pathlib import Path
            run_opts = options_from_config(cfg, video=Path(rtsp_url))
            publisher = get_kafka_publisher(self.settings)
            # kafka_overrides must mirror server Settings — otherwise the
            # processor's internal cfg (enabled flag, assets_base for chunk
            # destinations) falls back to config.json and drifts from the
            # injected publisher.
            proc = RTSPStreamProcessor(
                rtsp_url,
                run_opts,
                kafka_overrides=self.settings.kafka_overrides(),
                publisher=publisher,
                stop_event=stop_event,
            )

            t = threading.Thread(
                target=self._run_wrapper,
                args=(stream_id, proc),
                name=f"RTSPStream-{stream_id}",
                daemon=True,
            )
            info = StreamInfo(stream_id, rtsp_url, site_id, camera_id, proc, t, stop_event)
            self._streams[stream_id] = info
            t.start()
            return info.to_response()

    def _run_wrapper(self, stream_id: str, proc: RTSPStreamProcessor) -> None:
        with self._lock:
            info = self._streams.get(stream_id)
            if info:
                info.state = "running"
        try:
            proc.run()
            with self._lock:
                info = self._streams.get(stream_id)
                if info and info.state != "stopped":
                    info.state = "stopped"
        except Exception as exc:
            log.exception("Stream %s failed during run", stream_id)
            with self._lock:
                info = self._streams.get(stream_id)
                if info:
                    info.state = "failed"
                    info.error = str(exc)

    def get_stream(self, stream_id: str) -> StreamResponse:
        with self._lock:
            info = self._streams.get(stream_id)
            if not info:
                raise StreamNotFoundError(stream_id)
            return info.to_response()

    def list_streams(self) -> list[StreamResponse]:
        with self._lock:
            return [info.to_response() for info in self._streams.values()]

    def stop_stream(self, stream_id: str, timeout: float = 3.0) -> StreamResponse:
        with self._lock:
            info = self._streams.get(stream_id)
            if not info:
                raise StreamNotFoundError(stream_id)
            info.stop_event.set()
            info.state = "stopping"

        info.thread.join(timeout=timeout)
        with self._lock:
            if not info.thread.is_alive():
                info.state = "stopped"
        return info.to_response()

    def shutdown(self, timeout: float = 3.0) -> None:
        with self._lock:
            for info in self._streams.values():
                info.stop_event.set()
                info.state = "stopped"
            threads = [info.thread for info in self._streams.values()]
        for t in threads:
            t.join(timeout=timeout)
