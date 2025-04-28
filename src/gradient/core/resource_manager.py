# gradient-project-root/src/gradient/core/resource_manager.py
import logging
import time
import torch # Import torch
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages allocatable resources like GPU VRAM and potentially loaded models/components.
    Focuses on VRAM for now.
    """
    def __init__(self):
        self.max_vram_mb: int = self._get_total_vram_mb()
        self.allocated_vram_mb: int = 0
        # Track allocated resources: { resource_id: (vram_mb, timestamp, component_type) }
        self.allocated_resources: Dict[str, Tuple[int, float, str]] = {}
        logger.info(f"ResourceManager initialized. Detected VRAM: {self.max_vram_mb} MB")

    def _get_total_vram_mb(self) -> int:
        """Gets total GPU VRAM in MB if CUDA is available."""
        if torch.cuda.is_available():
            try:
                total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
                total_vram_mb = total_vram_bytes // (1024 * 1024)
                logger.debug(f"Detected {total_vram_mb} MB VRAM via torch.cuda.")
                return total_vram_mb
            except Exception as e:
                logger.error(f"Error getting VRAM info from torch.cuda: {e}", exc_info=True)
                return 8 * 1024 # Fallback dummy value
        else:
            logger.warning("CUDA not available. VRAM management will use dummy values.")
            return 8 * 1024 # Fallback dummy value if no CUDA

    def get_available_vram_mb(self) -> int:
        """Returns currently available VRAM."""
        # Could add torch.cuda.memory_allocated() for more precise tracking if needed
        return self.max_vram_mb - self.allocated_vram_mb

    def can_allocate(self, required_vram_mb: int) -> bool:
        """Checks if a specific amount of VRAM can be allocated."""
        can = self.get_available_vram_mb() >= required_vram_mb
        logger.debug(f"VRAM check: Required={required_vram_mb}, Available={self.get_available_vram_mb()}. Can allocate: {can}")
        return can

    def _find_lru_resource(self, min_vram_to_free: int = 0, component_type: Optional[str] = None) -> Optional[str]:
        """Finds the least recently used resource ID to free up VRAM."""
        if not self.allocated_resources:
            return None

        candidates = self.allocated_resources.items()
        if component_type:
            candidates = [item for item in candidates if item[1][2] == component_type]

        if not candidates:
            return None

        # Sort by timestamp (LRU)
        sorted_candidates = sorted(candidates, key=lambda item: item[1][1]) # Sort by timestamp (index 1)

        # Find the first one that frees enough VRAM, or just the LRU if min_vram_to_free is 0
        for resource_id, (vram, _, _) in sorted_candidates:
            if vram >= min_vram_to_free:
                return resource_id
        # If no single resource is large enough, maybe just return the absolute LRU? For now, only if min=0.
        if min_vram_to_free == 0 and sorted_candidates:
             return sorted_candidates[0][0]

        return None # No suitable candidate found

    async def ensure_available_vram(self, required_vram_mb: int):
        """
        Ensures enough VRAM is available, attempting to unload LRU resources if necessary.
        Raises MemoryError if VRAM cannot be freed.
        """
        if self.can_allocate(required_vram_mb):
            return # Enough space already

        logger.warning(f"Need to free up VRAM. Required: {required_vram_mb} MB, Available: {self.get_available_vram_mb()} MB")

        # Attempt to free resources until enough space is available
        while not self.can_allocate(required_vram_mb):
            needed = required_vram_mb - self.get_available_vram_mb()
            lru_id = self._find_lru_resource(min_vram_to_free=1) # Find any LRU resource first

            if lru_id:
                logger.info(f"Attempting to unload LRU resource: {lru_id} to free VRAM.")
                # --- TODO: Add actual unloading logic here ---
                # This requires coordination with the modules managing these resources (e.g., diffusion loader)
                # Example placeholder: Simulate unloading
                vram_freed, _, comp_type = self.allocated_resources.get(lru_id, (0, 0.0, 'unknown'))
                if vram_freed > 0:
                    # Need a way to signal the actual component manager to unload
                    # For now, just update our tracking
                    await self.release(lru_id, vram_freed)
                    logger.info(f"Simulated unload of {lru_id} (Type: {comp_type}), Freed: {vram_freed} MB.")
                else:
                    logger.error(f"LRU resource {lru_id} reported 0 VRAM. Inconsistency in tracking.")
                    del self.allocated_resources[lru_id] # Remove inconsistent entry
                    # Break to avoid infinite loop if tracking is broken
                    raise MemoryError(f"VRAM tracking inconsistent. Cannot guarantee freeing space.")

                # --- End TODO ---
            else:
                logger.error(f"Cannot free enough VRAM. No unloadable resources found or available.")
                raise MemoryError(f"Insufficient VRAM: Required={required_vram_mb}, Available={self.get_available_vram_mb()}, Max={self.max_vram_mb}")

    async def allocate(self, resource_id: str, vram_mb: int, component_type: str):
        """Allocates VRAM for a resource, potentially unloading others."""
        if resource_id in self.allocated_resources:
            logger.warning(f"Resource {resource_id} is already tracked. Updating timestamp.")
            self.allocated_resources[resource_id] = (vram_mb, time.time(), component_type)
            return

        await self.ensure_available_vram(vram_mb) # Make sure space is available

        self.allocated_vram_mb += vram_mb
        self.allocated_resources[resource_id] = (vram_mb, time.time(), component_type)
        logger.info(f"VRAM allocated for {resource_id} ({component_type}): {vram_mb} MB. Total allocated: {self.allocated_vram_mb} MB.")

    async def release(self, resource_id: str):
        """Releases VRAM allocated for a specific resource."""
        if resource_id in self.allocated_resources:
            vram_amount, _, _ = self.allocated_resources.pop(resource_id)
            self.allocated_vram_mb -= vram_amount
            self.allocated_vram_mb = max(0, self.allocated_vram_mb) # Prevent going below zero
            logger.info(f"VRAM released for {resource_id}: {vram_amount} MB. Total allocated: {self.allocated_vram_mb} MB.")
        else:
            logger.warning(f"Attempted to release VRAM for untracked resource: {resource_id}")

    def update_usage(self, resource_id: str):
         """Updates the last used timestamp for a resource."""
         if resource_id in self.allocated_resources:
              vram, _, comp_type = self.allocated_resources[resource_id]
              self.allocated_resources[resource_id] = (vram, time.time(), comp_type)
              logger.debug(f"Updated usage timestamp for {resource_id}")


# Global instance (or managed via dependency injection later)
resource_manager = ResourceManager()