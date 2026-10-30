# gradient-project-root/src/gradient/core/resource_manager.py
"""Tracks loaded models and evicts least-recently-used ones.

Modules that load heavy objects (diffusion pipelines, LLMs) register them here
together with an async `unload` callback. When a new model needs room, the
manager calls the callbacks of the least recently used entries.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from gradient.core.device import free_vram_bytes, total_vram_bytes

logger = logging.getLogger(__name__)

UnloadFn = Callable[[], Awaitable[None]]


@dataclass
class TrackedResource:
    resource_id: str
    component_type: str
    vram_mb: int
    unload: UnloadFn | None = None
    last_used: float = field(default_factory=time.time)


class ResourceManager:
    def __init__(self) -> None:
        total = total_vram_bytes()
        self.max_vram_mb: int = int(total / 1024**2) if total else 0
        self.resources: dict[str, TrackedResource] = {}
        logger.info(
            "ResourceManager initialized. VRAM: %s",
            f"{self.max_vram_mb} MB" if self.max_vram_mb else "n/a (no CUDA device)",
        )

    # --- Introspection -------------------------------------------------
    @property
    def allocated_vram_mb(self) -> int:
        return sum(r.vram_mb for r in self.resources.values())

    def get_available_vram_mb(self) -> int:
        free = free_vram_bytes()
        if free is not None:
            return int(free / 1024**2)
        return max(0, self.max_vram_mb - self.allocated_vram_mb)

    def snapshot(self) -> list[dict]:
        return [
            {
                "id": r.resource_id,
                "type": r.component_type,
                "vram_mb": r.vram_mb,
                "last_used": r.last_used,
            }
            for r in sorted(self.resources.values(), key=lambda r: r.last_used, reverse=True)
        ]

    def count(self, component_type_prefix: str) -> int:
        return sum(
            1 for r in self.resources.values() if r.component_type.startswith(component_type_prefix)
        )

    # --- Registration --------------------------------------------------
    def register(
        self, resource_id: str, component_type: str, vram_mb: int, unload: UnloadFn | None = None
    ) -> None:
        self.resources[resource_id] = TrackedResource(
            resource_id, component_type, int(vram_mb), unload
        )
        logger.info(
            "Tracking %s (%s, ~%d MB). Tracked total: %d MB",
            resource_id,
            component_type,
            vram_mb,
            self.allocated_vram_mb,
        )

    def release(self, resource_id: str) -> None:
        """Stop tracking a resource (the owner has already unloaded it)."""
        if self.resources.pop(resource_id, None) is not None:
            logger.info("Released %s. Tracked total: %d MB", resource_id, self.allocated_vram_mb)

    def update_usage(self, resource_id: str) -> None:
        if resource_id in self.resources:
            self.resources[resource_id].last_used = time.time()

    # --- Eviction ------------------------------------------------------
    async def evict_lru(
        self, component_type_prefix: str | None = None, keep: set[str] | None = None
    ) -> bool:
        """Unload the least recently used resource. Returns False if nothing could be evicted."""
        keep = keep or set()
        candidates = [
            r
            for r in self.resources.values()
            if r.resource_id not in keep
            and r.unload is not None
            and (
                component_type_prefix is None or r.component_type.startswith(component_type_prefix)
            )
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda r: r.last_used)
        logger.info("Evicting LRU resource %s (%s)", victim.resource_id, victim.component_type)
        try:
            await victim.unload()  # type: ignore[misc]
        finally:
            self.release(victim.resource_id)
        return True

    async def ensure_capacity(
        self,
        component_type_prefix: str,
        max_items: int,
        required_vram_mb: int = 0,
        keep: set[str] | None = None,
    ) -> None:
        """Evict LRU entries until at most `max_items - 1` of this type remain and VRAM fits."""
        while self.count(component_type_prefix) >= max(1, max_items):
            if not await self.evict_lru(component_type_prefix, keep):
                break
        if required_vram_mb and self.max_vram_mb:
            while self.get_available_vram_mb() < required_vram_mb:
                if not await self.evict_lru(None, keep):
                    break

    async def unload_all(self) -> None:
        for resource_id in list(self.resources):
            res = self.resources.get(resource_id)
            if res and res.unload:
                try:
                    await res.unload()
                except Exception as e:  # pragma: no cover - best effort on shutdown
                    logger.warning("Failed to unload %s: %s", resource_id, e)
            self.release(resource_id)


resource_manager = ResourceManager()
