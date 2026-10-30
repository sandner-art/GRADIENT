# src/gradient/core/families.py
"""Model family names derived from diffusers pipeline class names (no heavy imports)."""

_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("StableDiffusionXL", "sdxl"),
    ("StableDiffusion3", "sd3"),
    ("Flux2", "flux2"),
    ("Flux", "flux1"),
    ("QwenImage", "qwen-image"),
    ("ZImage", "z-image"),
    ("Chroma", "chroma"),
    ("Sana", "sana"),
    ("PixArt", "pixart"),
    ("HiDream", "hidream"),
    ("Kolors", "kolors"),
    ("LatentConsistencyModel", "sd15"),
    ("StableDiffusion", "sd15"),
)


def family_from_class(class_name: str | None) -> str | None:
    if not class_name:
        return None
    for needle, family in _FAMILY_RULES:
        if needle in class_name:
            return family
    return class_name.removesuffix("Pipeline").lower() or None
