# src/gradient/model_manager/schemas.py
"""Pydantic models for the model catalog, local inventory and downloads."""

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ModelType = Literal["diffusers", "checkpoints", "vae", "lora", "controlnet", "embeddings", "llm"]
JobStatus = Literal["queued", "resolving", "downloading", "completed", "failed", "cancelled"]


class GenerationDefaults(BaseModel):
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    scheduler: str | None = None


class CatalogEntry(BaseModel):
    """One downloadable model in config/model_catalog.yaml."""

    id: str
    name: str
    type: ModelType
    family: str | None = None
    source: Literal["huggingface", "url"] = "huggingface"
    repo_id: str | None = None
    revision: str | None = None
    url: str | None = None
    files: list[str] | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    variant: str | None = None
    torch_dtype: str | None = None
    gated: bool = False
    license: str | None = None
    size_gb: float | None = None
    min_vram_gb: float | None = None
    pipeline: str | None = None
    base_model_family: str | None = None
    defaults: GenerationDefaults | None = None
    tags: list[str] = Field(default_factory=list)
    description: str | None = None
    homepage: str | None = None
    verified: bool = True
    notes: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> "CatalogEntry":
        if self.source == "huggingface" and not self.repo_id:
            raise ValueError(f"catalog entry {self.id!r}: huggingface source needs repo_id")
        if self.source == "url" and not self.url:
            raise ValueError(f"catalog entry {self.id!r}: url source needs url")
        return self


class CatalogEntryView(CatalogEntry):
    """Catalog entry plus local install state (API response)."""

    installed: bool = False
    installed_path: str | None = None


class LocalModel(BaseModel):
    name: str  # identifier used in generation requests (relative to the type folder)
    type: ModelType
    format: Literal["diffusers", "single_file", "peft", "gguf", "directory"]
    path: str
    size_bytes: int
    family: str | None = None
    pipeline: str | None = None
    source: dict[str, Any] | None = None  # repo_id / url / catalog_id ... from the manifest
    defaults: GenerationDefaults | None = None


class DownloadRequest(BaseModel):
    """Start a download from the catalog, a Hugging Face repo, or a direct URL."""

    catalog_id: str | None = Field(default=None, description="ID from the built-in catalog")
    repo_id: str | None = Field(
        default=None,
        description="Hugging Face repo, e.g. 'stabilityai/sdxl-turbo' or a huggingface.co URL",
    )
    url: str | None = Field(
        default=None, description="Direct https URL to a single model file (e.g. Civitai)"
    )
    model_type: ModelType | None = Field(
        default=None, description="Where to store it; inferred when possible"
    )
    files: list[str] | None = Field(
        default=None, description="Specific files from the repo (single-file download)"
    )
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    revision: str | None = None
    target_name: str | None = Field(
        default=None, description="Folder/file name to store under (optional)"
    )
    token: str | None = Field(
        default=None, description="Access token for this download only (not stored)"
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "DownloadRequest":
        given = [x for x in (self.catalog_id, self.repo_id, self.url) if x]
        if len(given) != 1:
            raise ValueError("Provide exactly one of catalog_id, repo_id or url")
        return self


class DownloadJob(BaseModel):
    id: str
    status: JobStatus = "queued"
    name: str
    model_type: ModelType | None = None
    source: str
    target_path: str | None = None
    total_bytes: int | None = None
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    error: str | None = None
    error_code: str | None = (
        None  # gated | auth | not_found | disk_space | network | invalid | internal
    )
    catalog_id: str | None = None
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    @property
    def done(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    @property
    def progress(self) -> float | None:
        if not self.total_bytes:
            return None
        return min(1.0, self.downloaded_bytes / self.total_bytes)

    def model_dump_api(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["progress"] = self.progress
        return data
