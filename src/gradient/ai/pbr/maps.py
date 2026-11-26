# src/gradient/ai/pbr/maps.py
"""PBR map sets from one image: normal, height, AO, albedo, curvature, glow and, with a
property network, roughness, metallic and the engine packs.

The files follow the Texture Map Generator's names (`<name>_normal.png`, `_height`, `_glow`,
`_ao`, `_albedo` and a `_maps.json` sidecar, schema 2) and export_pbr.py's packs, so a set
imports into Unity as the generator's own does. Sets live in <outputs>/pbr_maps/<set id>/.

Methods:
  hybrid     the network's normals plus the picture's fine detail, one Poisson solve (the
             generator's default); needs a pbr_net checkpoint in <models>/pbr/
  network    the network's normals alone
  classical  no network: de-lit multi-scale slopes, AO and curvature from the height
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import threading
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gradient.core.config import get_model_search_dirs, get_model_subdir_name, get_outputs_dir
from gradient.core.version import app_version

from . import classic as C
from .net import UNet, load_net, predict_all

logger = logging.getLogger(__name__)

NET_SUFFIXES = (".pt", ".pth")
MAX_SIDE = 4096
MIN_SIDE = 80  # the network's de-lighting blur pads 64 px
SET_ID = re.compile(r"^[0-9a-f]{32}$")

# kind -> (label, colour data?) in display order
MAP_KINDS: dict[str, tuple[str, bool]] = {
    "albedo": ("Albedo", True),
    "normal": ("Normal", False),
    "height": ("Height", False),
    "ao": ("Ambient occlusion", False),
    "curvature": ("Curvature", False),
    "glow": ("Glow", True),
    "roughness": ("Roughness", False),
    "metallic": ("Metallic", False),
    "metallic_smoothness": ("Metallic + smoothness (Unity Built-in, URP)", False),
    "mask_map": ("Mask map (Unity HDRP)", False),
    "orm": ("ORM (glTF, Unreal)", False),
}

_nets: dict[str, UNet] = {}
_nets_lock = threading.Lock()


# --------------------------------------------------------------------------- networks
def pbr_model_dirs() -> list[Path]:
    sub = get_model_subdir_name("pbr")
    return [base / sub for base in get_model_search_dirs()]


def installed_nets() -> list[dict]:
    """pbr_net checkpoints in <models>/pbr/ (first search dir wins on a name clash)."""
    found: dict[str, dict] = {}
    for d in pbr_model_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if (
                f.suffix.lower() in NET_SUFFIXES
                and not f.name.startswith(".")
                and f.stem not in found
            ):
                found[f.stem] = {
                    "name": f.stem,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 2**20, 1),
                }
    return list(found.values())


def get_net(name: str) -> UNet:
    with _nets_lock:
        if name not in _nets:
            entry = next((n for n in installed_nets() if n["name"] == name), None)
            if entry is None:
                raise ValueError(
                    f"No PBR network '{name}' in {', '.join(map(str, pbr_model_dirs()))}."
                )
            net = load_net(entry["path"])
            if torch.cuda.is_available():
                net = net.cuda()
            _nets[name] = net
            try:
                from gradient.ai.lab.features import track_vision_model

                track_vision_model(f"pbr:{name}", net, lambda n=name: unload_nets([n]))
            except Exception:  # memory tracking is a convenience
                pass
        return _nets[name]


def loaded_net_info() -> list[dict]:
    from gradient.ai.lab.features import model_mb

    return [{"name": f"pbr:{n}", "vram_mb": model_mb(m)} for n, m in _nets.items()]


def unload_nets(names: list[str] | None = None) -> list[str]:
    """Drop PBR networks (all, or `names`) and free their memory. Blocking."""
    from gradient.ai.lab.features import free_memory
    from gradient.core.resource_manager import resource_manager

    with _nets_lock:
        dropped = [n for n in (names if names is not None else list(_nets)) if n in _nets]
        for n in dropped:
            _nets.pop(n, None)
            resource_manager.release(f"vision:pbr:{n}")
    if dropped:
        free_memory()
    return [f"pbr:{n}" for n in dropped]


def default_net() -> str | None:
    """The newest-looking installed network (pbr_net_v3 over v2), or None."""
    names = [n["name"] for n in installed_nets()]
    return sorted(names, reverse=True)[0] if names else None


# --------------------------------------------------------------------------- encoding
def to8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def to16(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 65535 + 0.5).astype(np.uint16)


def normal_rgb(n: np.ndarray) -> np.ndarray:
    return to8(n * 0.5 + 0.5)


def wrap_from_tiling(tiling: str | None) -> tuple[bool, bool]:
    t = (tiling or "none").lower()
    return ("x" in t, "y" in t)


# --------------------------------------------------------------------------- the maps
def compute_maps(
    img: np.ndarray,
    *,
    net: UNet | None,
    wrap: tuple[bool, bool] = (False, False),
    detail: float = 0.35,
    method: str = "hybrid",
) -> dict:
    """img (H,W,3) float sRGB 0..1 -> float maps: normal (unit, OpenGL), slopes p q, height (px),
    albedo, ao, curvature (-1..1), glow (0..1), and with a property network rough, metal, light."""
    out: dict = {}
    if net is not None:
        o = predict_all(net, img, wrap)
        if method == "network":
            p, q = C.normal_to_slopes(o["normal"])
            p, q = C.robust_clip(p, q)
            z = C.frankot_chellappa(p, q, wrap)
            p, q = C.height_slopes(z, wrap)
        else:
            p, q, z = C.hybrid_slopes(
                img, *C.normal_to_slopes(o["normal"]), detail=detail, wrap=wrap
            )
        out.update(albedo=C.albedo_with_color(img, o["albedo"]), glow=o["emissive"])
        if net.props:
            out.update(ao=o["ao"], curvature=o["curv"], rough=o["rough"], metal=o["metal"])
            out["light"] = [float(v) for v in o["light"]]
            out["shadow"] = o["shadow"]
    else:
        p, q = C.classic_slopes(img, wrap)
        z = C.frankot_chellappa(p, q, wrap)
        p, q = C.height_slopes(z, wrap)
        out["glow"] = C.glow_mask(img, wrap)
    if not all(wrap):
        z = C.remove_tilt(z, wrap)
    if "ao" not in out:
        out["ao"] = C.ao_from_height(z, wrap)
    if "curvature" not in out:
        out["curvature"] = C.curvature(z, wrap)
    if "albedo" not in out:
        out["albedo"] = C.albedo_retinex(img, wrap, ao=out["ao"])
    out.update(p=p, q=q, height=z)
    return out


def _sidecar(
    name: str,
    source: str,
    W: int,
    H: int,
    strength: float,
    height_px: float,
    generator: str,
    packs: dict,
    light: list[float] | None,
    wrap,
    method: str,
) -> dict:
    data: dict = {
        "schema": 2,
        "generator": generator,
        "source": source,
        "width": W,
        "height": H,
        "strength": round(strength, 5),
    }
    if packs:
        data["packs"] = packs
    if light is not None:
        x, y, z = light
        data["light"] = {
            "azimuth_deg": round(math.degrees(math.atan2(y, x)), 3),
            "elevation_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, z)))), 3),
            "direction": [round(v, 5) for v in light],
        }
    data["gradient"] = {
        "method": method,
        "seamless": {"x": wrap[0], "y": wrap[1]},
        "height_range_px": round(height_px, 4),
        "name": name,
    }
    return data


def write_set(
    folder: Path, stem: str, M: dict, picture: np.ndarray, alpha: np.ndarray | None, strength: float
) -> tuple[dict[str, str], float, dict]:
    """Write the map files; returns ({kind: file name}, height range in px, packs)."""
    folder.mkdir(parents=True, exist_ok=True)
    H, W = picture.shape[:2]
    files: dict[str, str] = {}

    def save(kind: str, arr: np.ndarray) -> None:
        name = f"{stem}_{kind}.png"
        Image.fromarray(arr).save(folder / name, compress_level=6)  # uint16 -> 16-bit grey
        files[kind] = name

    alb = to8(M["albedo"])
    save("albedo", np.dstack([alb, to8(alpha)]) if alpha is not None else alb)
    save("normal", normal_rgb(C.slopes_to_normal(M["p"], M["q"], strength)))
    z = M["height"].astype(np.float64)
    z = z - z.min()
    zr = float(z.max()) or 1.0
    save("height", to16(z / zr))
    save("ao", to8(M["ao"]))
    save("curvature", to8(M["curvature"] * 0.5 + 0.5))
    save("glow", to8(picture * np.clip(M["glow"], 0, 1)[..., None]))  # lights in their own colour
    packs: dict = {}
    if "rough" in M:
        rough, metal, ao = (np.clip(M[k], 0, 1) for k in ("rough", "metal", "ao"))
        smooth = 1 - rough
        zero = np.zeros((H, W), np.uint8)
        save("roughness", to8(rough))
        save("metallic", to8(metal))
        save("metallic_smoothness", np.dstack([to8(metal)] * 3 + [to8(smooth)]))
        save("mask_map", np.dstack([to8(metal), to8(ao), zero, to8(smooth)]))
        save("orm", np.dstack([to8(ao), to8(rough), to8(metal)]))
        packs = {
            "_metallic_smoothness": ["Metal", "Metal", "Metal", "Smoothness"],
            "_mask_map": ["Metal", "AO", "DetailMask", "Smoothness"],
            "_orm": ["AO", "Roughness", "Metal", "None"],
        }
    return files, zr, packs


def safe_stem(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return s[:80] or "texture"


def make_maps(
    image: Image.Image,
    *,
    name: str = "texture",
    model: str | None = "auto",
    method: str = "hybrid",
    wrap: tuple[bool, bool] = (False, False),
    detail: float = 0.35,
    strength: float = 1.0,
    source: str = "",
) -> dict:
    """Compute and write one map set. Returns the set's description (see MapsResponse)."""
    started = time.time()
    notes: list[str] = []
    W, H = image.size
    if max(W, H) > MAX_SIDE:
        raise ValueError(f"The image is {W}×{H}; the largest side for maps is {MAX_SIDE} px.")
    if min(W, H) < MIN_SIDE:
        raise ValueError(f"The image is {W}×{H}; maps need at least {MIN_SIDE} px per side.")
    alpha = None
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = np.asarray(image.convert("RGBA"), np.float32) / 255.0
        if (rgba[..., 3] < 1).any():
            alpha = rgba[..., 3]
    img = np.asarray(image.convert("RGB"), np.float32) / 255.0

    net_name = None
    net = None
    if method != "classical":
        net_name = default_net() if model in (None, "", "auto") else model
        if net_name is None:
            notes.append(
                "No PBR network installed: used the classical method. Put pbr_net_v3.pt in "
                f"{pbr_model_dirs()[0]} for learned normals, albedo, AO and material maps."
            )
            method = "classical"
        else:
            net = get_net(net_name)
            if not net.props:
                notes.append(f"{net_name} has no property heads: no roughness or metallic maps.")
    if any(wrap) and net is not None and (H % 16 or W % 16):
        notes.append("Seamless is exact when both sides are multiples of 16 px.")

    M = compute_maps(img, net=net, wrap=wrap, detail=detail, method=method)
    set_id = uuid.uuid4().hex
    folder = get_outputs_dir() / "pbr_maps" / set_id
    stem = safe_stem(name)
    files, zr, packs = write_set(folder, stem, M, img, alpha, strength)
    generator = f"GRADIENT {app_version()} " + (f"{net_name} {method}" if net_name else "classical")
    side = _sidecar(
        stem, source, W, H, strength, zr, generator, packs, M.get("light"), wrap, method
    )
    (folder / f"{stem}_maps.json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    files["sidecar"] = f"{stem}_maps.json"
    return {
        "set_id": set_id,
        "name": stem,
        "width": W,
        "height": H,
        "method": method,
        "model": net_name,
        "seamless": {"x": wrap[0], "y": wrap[1]},
        "maps": [
            {"kind": k, "label": MAP_KINDS[k][0], "file": files[k], "color": MAP_KINDS[k][1]}
            for k in MAP_KINDS
            if k in files
        ],
        "sidecar": files["sidecar"],
        "light": side.get("light"),
        "height_range_px": round(zr, 4),
        "notes": notes,
        "seconds": round(time.time() - started, 2),
    }


# --------------------------------------------------------------------------- serving
def set_folder(set_id: str) -> Path:
    if not SET_ID.match(set_id):
        raise KeyError(set_id)
    folder = get_outputs_dir(create=False) / "pbr_maps" / set_id
    if not folder.is_dir():
        raise KeyError(set_id)
    return folder


def set_file(set_id: str, file: str) -> Path:
    """A file of a set; only names that are in the folder (no paths)."""
    folder = set_folder(set_id)
    names = {p.name for p in folder.iterdir() if p.is_file()}
    if file not in names:
        raise KeyError(file)
    return folder / file


def set_zip(set_id: str) -> tuple[bytes, str]:
    folder = set_folder(set_id)
    files = sorted(p for p in folder.iterdir() if p.is_file())
    stem = next(
        (p.name.removesuffix("_maps.json") for p in files if p.name.endswith("_maps.json")), set_id
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # PNGs are compressed already
        for p in files:
            z.write(p, f"{stem}/{p.name}")
    return buf.getvalue(), f"{stem}_maps.zip"
