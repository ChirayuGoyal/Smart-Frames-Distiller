"""Server configuration via environment variables (SFD_ prefix).

Infrastructure endpoints (Kafka brokers, Milvus host, asset base, work dir)
come ONLY from these settings — never from config.json, which holds
per-run pipeline option defaults.

Examples:
    SFD_WORK_DIR=/srv/sfd
    SFD_KAFKA__BROKERS=broker1:9092
    SFD_MILVUS__HOST=milvus.internal
    SFD_ALLOWED_INPUT_DIRS='["D:/videos", "//nas/cctv"]'
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseModel):
    enabled: bool = True
    required: bool = False
    brokers: str = "localhost:9092"
    topic: str = "semantic-chunks-data"
    client_id: str = "action-aware-chunk-producer"
    sp_enabled: str = "true"
    critic_enabled: str = "true"
    debug: str = ""
    embed_frame_metadata: bool = True
    connect_retries: int = 3
    connect_retry_seconds: float = 2.0


class MilvusSettings(BaseModel):
    host: str = "localhost"
    port: int = 19530
    collection: str = "face_registry"


class ModelSettings(BaseModel):
    device: str = "auto"  # auto | cpu | cuda
    r3d_model_path: str = ""
    r3d_cache_dir: str = "models"
    person_model: str = "FREmbeddings/Model/yolov8n.onnx"
    detector_model: str = "FREmbeddings/Model/det_10g.onnx"
    embedding_model: str = "FREmbeddings/Model/w600k_r50.onnx"
    ort_intra_op_threads: int = 0  # 0 = ORT default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SFD_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    work_dir: Path = Path("./sfd_data")
    assets_base: str = "assets"  # replaces hardcoded /jvadata/vst/assets
    allowed_input_dirs: list[Path] = Field(default_factory=list)
    max_upload_bytes: int = 4 * 1024**3  # 4 GiB
    max_concurrent_jobs: int = 1
    max_filter_workers: int = 4
    max_streams: int = 4
    gpu_slots: int = 1
    warm_models: bool = False
    config_json: Path | None = None  # optional pipeline-defaults file

    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)

    @property
    def jobs_dir(self) -> Path:
        return self.work_dir / "jobs"

    @property
    def uploads_tmp_dir(self) -> Path:
        return self.work_dir / "tmp"

    @property
    def kafka_spool_path(self) -> Path:
        return self.work_dir / "kafka_pending.jsonl"

    def kafka_overrides(self) -> dict:
        """Kafka config dict in the shape kafka_producer.kafka_settings expects."""
        k = self.kafka
        return {
            "enabled": k.enabled,
            "required": k.required,
            "brokers": k.brokers,
            "topic": k.topic,
            "client_id": k.client_id,
            "assets_base": self.assets_base,
            "sp_enabled": k.sp_enabled,
            "critic_enabled": k.critic_enabled,
            "debug": k.debug,
            "embed_frame_metadata": k.embed_frame_metadata,
            "connect_retries": k.connect_retries,
            "connect_retry_seconds": k.connect_retry_seconds,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
