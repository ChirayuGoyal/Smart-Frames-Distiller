"""Kafka publisher for semantic video chunks (confluent_kafka).

Library-safe: no logging handlers, files, or directories are created at
import time. Debug logging to output/kafka_debug.log is opt-in via
enable_kafka_debug_log(). Concurrency-safe: producer state lives in
KafkaPublisher instances (internally locked), not module globals.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

_DEFAULT_BROKERS = "localhost:9092"
_DEFAULT_TOPIC = "semantic-chunks-data"
_DEFAULT_CLIENT_ID = "action-aware-chunk-producer"
_DEFAULT_ASSETS_BASE = "assets"
_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_DEFAULT_SPOOL_PATH = Path(__file__).resolve().parent / "output" / "kafka_pending.jsonl"

log = logging.getLogger("kafka_pipeline")


def enable_kafka_debug_log(log_dir: str | Path | None = None) -> None:
    """Opt-in verbose Kafka logging to console + <log_dir>/kafka_debug.log.

    Called by the CLI; servers configure logging themselves.
    """
    if getattr(log, "_kafka_configured", False):
        return
    log.setLevel(logging.DEBUG)
    log.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s [kafka] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out_dir = Path(log_dir) if log_dir else _DEFAULT_SPOOL_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "kafka_debug.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.addHandler(sh)
    log._kafka_configured = True  # type: ignore[attr-defined]


def _mask(value: str) -> str:
    if not value:
        return ""
    return value[:2] + "***" if len(value) > 2 else "***"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_kafka_config() -> dict[str, Any]:
    if _CONFIG_PATH.is_file():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f).get("kafka", {})
    return {}


def _parse_bool(val: Any, default: bool = True) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() not in ("0", "false", "no", "")
    return default


def _normalize_broker_entry(entry: str) -> str:
    entry = entry.strip().rstrip("/")
    if not entry:
        return entry
    for prefix in ("http://", "https://"):
        if entry.lower().startswith(prefix):
            entry = entry[len(prefix) :].lstrip("/")
    if ":" not in entry:
        return f"{entry}:9092"
    return entry


def _brokers_from_config(k: dict[str, Any]) -> str:
    raw = str(k.get("brokers") or k.get("bootstrap_servers") or _DEFAULT_BROKERS)
    return ",".join(_normalize_broker_entry(p) for p in raw.split(",") if p.strip())


def kafka_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    k = {**load_kafka_config(), **(overrides or {})}
    return {
        "enabled": _parse_bool(k.get("enabled", True)),
        "required": _parse_bool(k.get("required", False)),
        "brokers": _brokers_from_config(k),
        "topic": k.get("topic", _DEFAULT_TOPIC),
        "client_id": k.get("client_id", _DEFAULT_CLIENT_ID),
        "security_protocol": k.get("security_protocol", "PLAINTEXT"),
        "sasl_mechanism": k.get("sasl_mechanism", "PLAIN"),
        "ssl_ca_location": str(k.get("ssl_ca_location", "") or "").strip(),
        "sasl_username": str(k.get("sasl_username", "") or "").strip(),
        "sasl_password": str(k.get("sasl_password", "") or "").strip(),
        "acks": k.get("acks", "all"),
        "compression_type": k.get("compression_type", "none"),
        "batch_size": int(k.get("batch_size", 16384)),
        "linger_ms": int(k.get("linger_ms", 0)),
        "max_in_flight": int(k.get("max_in_flight", 1000000)),
        "connect_retries": int(k.get("connect_retries", 5)),
        "connect_retry_seconds": float(k.get("connect_retry_seconds", 2)),
        "sp_enabled": str(k.get("sp_enabled", "true")),
        "critic_enabled": str(k.get("critic_enabled", "true")),
        "assets_base": str(k.get("assets_base", _DEFAULT_ASSETS_BASE)).rstrip("/"),
        # librdkafka debug contexts, e.g. "broker,topic,msg". Empty = off.
        "debug": str(k.get("debug", "") or "").strip(),
        # embed full per-frame metadata in each message (False = compact summary).
        "embed_frame_metadata": _parse_bool(k.get("embed_frame_metadata", True)),
        # sp / critic alert-level values sent verbatim in alert_level; if None,
        # each inherits from its *_enabled sibling.
        "sp":     str(k.get("sp",     "") or "").strip() or None,
        "critic": str(k.get("critic", "") or "").strip() or None,
        # spool file for undeliverable payloads (required=false mode)
        "spool_path": str(k.get("spool_path", "") or "").strip() or None,
    }


def build_chunk_message(
    *,
    run_id: str,
    chunk_id: str,
    camera_id: str,
    site_id: str,
    start_timestamp: int,
    end_timestamp: int,
    chunk_path: str,
    event_id: str | None = None,
    sp_enabled: str = "true",
    critic_enabled: str = "true",
    sp: str | None = None,
    critic: str | None = None,
    app_id: int = 0,
) -> dict[str, Any]:
    sp_flag    = str(sp_enabled).lower()
    critic_flag = str(critic_enabled).lower()
    # alert_level uses the explicit override if given, otherwise mirrors the flag
    alert_sp     = sp     if sp     is not None else sp_flag
    alert_critic = critic if critic is not None else critic_flag
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "app_id": app_id,
        "camera_id": camera_id,
        "site_id": site_id,
        "chunk_id": chunk_id,
        "start_timestamp": int(start_timestamp),
        "end_timestamp": int(end_timestamp),
        "metadata": {
            "chunk_format": "mp4",
            "path": chunk_path,
            "sp_enabled":     sp_flag,
            "critic_enabled": critic_flag,
            "alert_level": {
                "sp":     alert_sp,
                "critic": alert_critic,
            },
        },
        "asset_id": run_id,
    }


def chunk_asset_path(
    site_id: str,
    camera_id: str,
    chunk_id: str,
    assets_base: str | None = None,
    *,
    ext: str = "mp4",
) -> str:
    base = (assets_base or kafka_settings()["assets_base"]).rstrip("/")
    return f"{base}/{site_id}/{camera_id}/{chunk_id}.{ext}"


def _build_confluent_config(cfg: dict[str, Any]) -> dict[str, Any]:
    conf: dict[str, Any] = {
        "bootstrap.servers": cfg["brokers"],
        "client.id": cfg["client_id"],
        "acks": str(cfg["acks"]),
        "compression.type": cfg["compression_type"],
        "batch.size": cfg["batch_size"],
        "linger.ms": cfg["linger_ms"],
        "max.in.flight.requests.per.connection": cfg["max_in_flight"],
        "socket.timeout.ms": 10000,
        "message.timeout.ms": 30000,
        "error_cb": _error_cb,
    }
    if cfg.get("debug"):
        conf["debug"] = cfg["debug"]
    protocol = str(cfg["security_protocol"]).upper()
    conf["security.protocol"] = protocol
    if protocol in ("SSL", "SASL_SSL") and cfg.get("ssl_ca_location"):
        conf["ssl.ca.location"] = cfg["ssl_ca_location"]
    if protocol in ("SASL_PLAINTEXT", "SASL_SSL"):
        conf["sasl.mechanism"] = cfg["sasl_mechanism"]
        conf["sasl.username"] = cfg["sasl_username"]
        conf["sasl.password"] = cfg["sasl_password"]
    return conf


def _error_cb(err) -> None:
    # librdkafka asynchronous errors (broker down, auth, DNS, transport).
    log.error("librdkafka error_cb: %s", err)


def _log_effective_config(cfg: dict[str, Any]) -> None:
    log.info(
        "config: brokers=%s topic=%s client_id=%s protocol=%s sasl_mech=%s "
        "sasl_user=%s acks=%s compression=%s msg_timeout_ms=30000 debug=%s",
        cfg["brokers"], cfg["topic"], cfg["client_id"], cfg["security_protocol"],
        cfg["sasl_mechanism"], _mask(cfg["sasl_username"]), cfg["acks"],
        cfg["compression_type"], cfg["debug"] or "(off)",
    )


class KafkaProducerWrapper:
    """Thin wrapper over confluent_kafka.Producer. Not thread-safe on its own —
    KafkaPublisher serializes access."""

    def __init__(self, cfg: dict[str, Any]):
        from confluent_kafka import Producer

        self.cfg = cfg
        self.topic = cfg["topic"]
        self._last_error: Optional[str] = None
        self.delivered_records = 0
        conf = _build_confluent_config(cfg)
        log.debug("creating Producer with librdkafka conf keys: %s", sorted(conf.keys()))
        try:
            self.producer = Producer(conf, logger=log)
        except TypeError:
            # Older confluent_kafka without logger kwarg
            self.producer = Producer(conf)

    def delivery_callback(self, err, msg) -> None:
        if err:
            self._last_error = str(err)
            log.error("delivery FAILED: %s", err)
        else:
            self._last_error = None
            self.delivered_records += 1
            try:
                log.info(
                    "delivery OK: topic=%s partition=%s offset=%s key=%s bytes=%s",
                    msg.topic(), msg.partition(), msg.offset(), msg.key(),
                    len(msg.value()) if msg.value() else 0,
                )
            except Exception:
                log.info("delivery OK (message metadata unavailable)")

    def produce_json(self, data: dict[str, Any], *, topic: Optional[str] = None) -> bool:
        self._last_error = None
        t = topic or self.topic
        body = json.dumps(data, default=str).encode("utf-8")
        log.debug("produce -> topic=%s bytes=%d chunk_id=%s", t, len(body), data.get("chunk_id"))
        try:
            self.producer.produce(t, body, callback=self.delivery_callback)
            self.producer.poll(0)
        except BufferError:
            log.warning("local produce queue full — polling 1s and retrying")
            self.producer.poll(1)
            self.producer.produce(t, body, callback=self.delivery_callback)
            self.producer.poll(0)
        except Exception as exc:
            self._last_error = str(exc)
            log.error("produce() raised: %s", exc)
            return False
        return True

    def flush(self, timeout: float = 10) -> bool:
        try:
            remaining = self.producer.flush(timeout)
            if remaining > 0:
                self._last_error = f"{remaining} message(s) not delivered"
                log.error("flush incomplete: %s still in queue (broker not acking)", remaining)
                return False
            return self._last_error is None
        except Exception as exc:
            self._last_error = str(exc)
            log.error("flush() raised: %s", exc)
            return False

    def probe(self, timeout: float = 10) -> bool:
        try:
            md = self.producer.list_topics(timeout=timeout)
            self._last_error = None
            brokers = ",".join(f"{b.host}:{b.port}" for b in md.brokers.values())
            has_topic = self.topic in md.topics
            log.info("probe OK: cluster brokers=[%s] topic '%s' present=%s",
                     brokers, self.topic, has_topic)
            if not has_topic:
                log.warning(
                    "topic '%s' not found in metadata — broker may auto-create on first "
                    "produce, or you are pointed at the wrong cluster", self.topic,
                )
            return True
        except Exception as exc:
            self._last_error = str(exc)
            log.error("probe FAILED (broker unreachable / metadata timeout): %s", exc)
            return False


class KafkaPublisher:
    """Thread-safe chunk publisher with spool-on-failure.

    One instance per pipeline/server (or per job when overrides differ).
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        overrides: dict[str, Any] | None = None,
        spool_path: str | Path | None = None,
    ):
        self.cfg = cfg or kafka_settings(overrides)
        self.spool_path = Path(
            spool_path or self.cfg.get("spool_path") or _DEFAULT_SPOOL_PATH
        )
        self._lock = threading.Lock()
        self._wrapper: Optional[KafkaProducerWrapper] = None
        self.last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg["enabled"])

    # -- lifecycle ---------------------------------------------------------
    def connect(self, *, wait: bool = True) -> bool:
        """Establish a probed producer. False when broker unreachable
        (caller decides based on cfg['required'])."""
        if not self.enabled:
            log.info("Kafka disabled (enabled=false) — skipping connection")
            return True
        _log_effective_config(self.cfg)
        try:
            import confluent_kafka  # noqa: F401
        except ImportError:
            self.last_error = "confluent-kafka not installed"
            log.error("confluent-kafka NOT installed — run: pip install confluent-kafka")
            return False

        attempts = self.cfg["connect_retries"] if wait else 1
        interval = self.cfg["connect_retry_seconds"]
        for n in range(1, attempts + 1):
            log.info("connection attempt %d/%d to %s", n, attempts, self.cfg["brokers"])
            if self._ensure_wrapper(force_new=(n > 1)) is not None:
                log.info("CONNECTED brokers=%s topic=%s", self.cfg["brokers"], self.cfg["topic"])
                return True
            if n < attempts:
                log.warning("broker unreachable, retry %d/%d in %ss...", n, attempts, interval)
                time.sleep(interval)

        msg = f"broker unreachable: {self.cfg['brokers']}"
        self.last_error = msg
        if self.cfg["required"]:
            log.error("%s (required=true)", msg)
        else:
            log.warning("%s (required=false — continuing, chunks will spool)", msg)
        return False

    def close(self) -> None:
        with self._lock:
            self._reset_locked()

    # -- publishing ---------------------------------------------------------
    def publish(self, payload: dict[str, Any], *, topic: Optional[str] = None) -> bool:
        """Produce + flush one payload. Spools and returns False on failure."""
        if not self.enabled:
            log.info("publish: Kafka disabled — chunk_id=%s NOT sent", payload.get("chunk_id"))
            return True

        t = topic or self.cfg["topic"]
        with self._lock:
            wrapper = self._ensure_wrapper_locked() or self._ensure_wrapper_locked(
                force_new=True
            )
            if wrapper is None:
                log.error(
                    "publish: no producer (broker down) — spooling chunk_id=%s",
                    payload.get("chunk_id"),
                )
                self._spool(payload)
                return False
            try:
                ok = self._send_one_locked(wrapper, payload, t)
            except Exception as exc:
                log.error("publish raised (%s) — resetting producer and spooling", exc)
                self.last_error = str(exc)
                self._reset_locked()
                self._spool(payload)
                return False
            if not ok:
                self._spool(payload)
            return ok

    # -- internals -----------------------------------------------------------
    def _ensure_wrapper(self, *, force_new: bool = False) -> Optional[KafkaProducerWrapper]:
        with self._lock:
            return self._ensure_wrapper_locked(force_new=force_new)

    def _ensure_wrapper_locked(
        self, *, force_new: bool = False
    ) -> Optional[KafkaProducerWrapper]:
        if force_new:
            self._reset_locked()
        if self._wrapper is not None:
            return self._wrapper
        try:
            wrapper = KafkaProducerWrapper(self.cfg)
            if wrapper.probe(timeout=10):
                self._wrapper = wrapper
                return self._wrapper
            self.last_error = wrapper._last_error
        except Exception as exc:
            self.last_error = str(exc)
            log.error("Producer init failed: %s", exc)
        self._reset_locked()
        return None

    def _reset_locked(self) -> None:
        if self._wrapper is not None:
            try:
                self._wrapper.flush(2)
            except Exception:
                pass
        self._wrapper = None

    def _send_one_locked(
        self, wrapper: KafkaProducerWrapper, payload: dict[str, Any], topic: str
    ) -> bool:
        if not wrapper.produce_json(payload, topic=topic):
            self.last_error = wrapper._last_error
            log.error("produce failed: %s", wrapper._last_error)
            return False
        if not wrapper.flush(10):
            self.last_error = wrapper._last_error
            log.error("delivery failed: %s", wrapper._last_error)
            return False
        log.info("SENT chunk_id=%s event_id=%s topic=%s",
                 payload.get("chunk_id"), payload.get("event_id"), topic)
        return True

    def _spool(self, payload: dict[str, Any]) -> None:
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.spool_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        log.warning("payload spooled to %s (NOT delivered to Kafka)", self.spool_path)


# --------------------------------------------------------------------------
# Legacy module-level API (CLI compatibility) — backed by one lazily-created
# default publisher guarded by a lock.
# --------------------------------------------------------------------------
_default_publisher: Optional[KafkaPublisher] = None
_default_lock = threading.Lock()


def _get_default_publisher(cfg: dict[str, Any] | None = None) -> KafkaPublisher:
    global _default_publisher
    with _default_lock:
        if _default_publisher is None or (
            cfg is not None and cfg != _default_publisher.cfg
        ):
            _default_publisher = KafkaPublisher(cfg)
        return _default_publisher


def connect_kafka(*, wait: bool = True, overrides: dict[str, Any] | None = None) -> bool:
    cfg = kafka_settings(overrides)
    log.info("connect_kafka: enabled=%s required=%s wait=%s", cfg["enabled"], cfg["required"], wait)
    return _get_default_publisher(cfg).connect(wait=wait)


def publish_chunk(
    payload: dict[str, Any],
    *,
    topic: Optional[str] = None,
    cfg: dict[str, Any] | None = None,
) -> bool:
    return _get_default_publisher(cfg or kafka_settings()).publish(payload, topic=topic)
