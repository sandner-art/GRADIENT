# src/gradient/ai/pbr/net.py
"""The multi-head PBR map network (pbr_net) of the Texture Map Generator, for inference.

Ported layer for layer from pbr_experiments/model_torch.py (same author), so its checkpoints
(`pbr_net_v3.pt`: {"config", "model" | "ema"}) load unchanged:

  image (B,3,H,W) sRGB 0..1
    -> preprocess(): sRGB*2-1, normalised log luminance, [de-lit log luminance], [chroma-edge weight]
    -> U-Net: 5 levels, two 3x3 conv + [GroupNorm] + GELU(tanh) per block, avg-pool down,
       bilinear up, skip concat
    -> heads: normal xy, albedo, emissive; with props: shadow, ao, roughness, metal, flow,
       thickness, part edges, curvature and the key-light direction.

Added here: wrap-around per axis (`wrap=(x, y)`), so a seamless texture gets seamless maps.
Every padding (blurs, convolutions, upsampling) then wraps instead of reflecting or zero-padding.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = {
    "xs": (16, 32, 64, 112, 160),  # the shipped model (1.2 M parameters)
    "s": (24, 48, 96, 160, 224),
    "m": (32, 64, 128, 224, 320),
    "l": (48, 96, 192, 320, 448),
}
CUES = {"none": (False, False), "lum": (True, False), "chroma": (False, True), "both": (True, True)}
PROP_CHANNELS = (
    ("shadow", 1),
    ("ao", 1),
    ("rough", 1),
    ("metal", 1),
    ("flow", 2),
    ("thick", 1),
    ("edges", 1),
    ("curv", 1),
)
N_PROP = sum(c for _, c in PROP_CHANNELS)

Wrap = tuple[bool, bool]  # (x: left-right, y: top-bottom)


def n_inputs(cues: str) -> int:
    lum, chroma = CUES[cues]
    return 4 + int(lum) + int(chroma)


def pad2d(x: torch.Tensor, px: int, wrap: Wrap, other: str) -> torch.Tensor:
    """Pad H and W by px: circular on the wrapped axes, `other` on the rest."""
    if px <= 0:
        return x
    x = F.pad(x, (px, px, 0, 0), mode="circular" if wrap[0] else other)
    return F.pad(x, (0, 0, px, px), mode="circular" if wrap[1] else other)


def _gauss(sigma: float, device) -> tuple[torch.Tensor, int]:
    r = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return torch.tensor(k / k.sum(), dtype=torch.float32, device=device), r


def blur(x: torch.Tensor, sigma: float, wrap: Wrap = (False, False)) -> torch.Tensor:
    """Separable Gaussian, truncate 4, reflect padding (circular on wrapped axes)."""
    C = x.shape[1]
    k, r = _gauss(sigma, x.device)
    x = F.pad(x, (r, r, 0, 0), mode="circular" if wrap[0] else "reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.pad(x, (0, 0, r, r), mode="circular" if wrap[1] else "reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def srgb_to_lin(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def preprocess(img: torch.Tensor, cues: str = "both", wrap: Wrap = (False, False)) -> torch.Tensor:
    """img (B,3,H,W) sRGB 0..1 -> (B, n_inputs(cues), H, W), in float32."""
    lum, chroma = CUES[cues]
    with torch.autocast(device_type=img.device.type, enabled=False):
        x = img.float().clamp(0, 1)
        lin = srgb_to_lin(x)
        Y = 0.2126 * lin[:, 0:1] + 0.7152 * lin[:, 1:2] + 0.0722 * lin[:, 2:3]
        lg = torch.log(Y + 1e-3)
        d = lg - lg.mean((2, 3), keepdim=True)
        std = torch.sqrt((d * d).mean((2, 3), keepdim=True))
        ch = [x * 2 - 1, d / (std + 1e-6)]
        if lum:
            ch.append((lg - blur(lg, 16.0, wrap)) * 1.5)
        if chroma:
            le = lin + 1e-3
            c = (le / le.sum(1, keepdim=True))[:, :2]
            cb = pad2d(blur(c, 1.0, wrap), 1, wrap, "reflect")
            sm = torch.tensor([1.0, 2.0, 1.0], device=img.device)
            dv = torch.tensor([-1.0, 0.0, 1.0], device=img.device)
            kc = (torch.outer(sm, dv) / 8.0).view(1, 1, 3, 3).repeat(2, 1, 1, 1)
            kr = (torch.outer(dv, sm) / 8.0).view(1, 1, 3, 3).repeat(2, 1, 1, 1)
            gc, gr = F.conv2d(cb, kc, groups=2), F.conv2d(cb, kr, groups=2)
            g = (gc * gc + gr * gr).sum(1, keepdim=True)
            ch.append(1.0 / (1.0 + g / 0.02**2) * 2 - 1)
        return torch.cat(ch, 1)


class ConvAct(nn.Module):
    def __init__(self, cin: int, cout: int, norm: bool):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(max(1, cout // 8), cout, eps=1e-5) if norm else None
        nn.init.normal_(self.conv.weight, 0.0, math.sqrt(2.0 / (9 * cin)))
        nn.init.zeros_(self.conv.bias)
        self.wrap: Wrap = (False, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(self.wrap):  # zero padding on the open axes, as in training
            x = F.conv2d(pad2d(x, 1, self.wrap, "constant"), self.conv.weight, self.conv.bias)
        else:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return F.gelu(x, approximate="tanh")


class UNet(nn.Module):
    def __init__(self, cues="both", width="xs", norm=True, cout=6, props=False):
        super().__init__()
        ch = WIDTHS[width] if isinstance(width, str) else tuple(width)
        self.cues, self.width, self.ch, self.props = cues, width, ch, props
        if props:
            cout = 6 + N_PROP
        c_prev = n_inputs(cues)
        self.enc = nn.ModuleList()
        for c in ch:
            self.enc.append(nn.Sequential(ConvAct(c_prev, c, norm), ConvAct(c, c, norm)))
            c_prev = c
        self.dec = nn.ModuleList()
        for i in range(len(ch) - 2, -1, -1):
            self.dec.append(
                nn.Sequential(ConvAct(ch[i + 1] + ch[i], ch[i], norm), ConvAct(ch[i], ch[i], norm))
            )
        self.head = nn.Conv2d(ch[0], cout, 1)
        self.light = nn.Linear(ch[-1], 3) if props else None
        self.wrap: Wrap = (False, False)

    def set_wrap(self, wrap: Wrap) -> None:
        self.wrap = (bool(wrap[0]), bool(wrap[1]))
        for m in self.modules():
            if isinstance(m, ConvAct):
                m.wrap = self.wrap

    def _up(self, x: torch.Tensor) -> torch.Tensor:
        if not any(self.wrap):
            return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # replicate padding reproduces the clamped border of align_corners=False exactly
        y = F.interpolate(
            pad2d(x, 1, self.wrap, "replicate"),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return y[:, :, 2:-2, 2:-2]

    def _trunk(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.avg_pool2d(x, 2)
        bottleneck = x
        for blk, s in zip(self.dec, skips[::-1], strict=True):
            x = blk(torch.cat([self._up(x), s], 1))
        return self.head(x).float(), bottleneck

    def forward_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """All outputs as raw values (logits where a sigmoid applies)."""
        y, b = self._trunk(x)
        out = {
            "nxy": torch.tanh(y[:, 0:2]) * 0.999,
            "albedo": torch.sigmoid(y[:, 2:5]),
            "emissive": y[:, 5:6],
        }
        if self.props:
            c = 6
            for name, k in PROP_CHANNELS:
                out[name] = y[:, c : c + k]
                c += k
            g = self.light(b.float().mean((2, 3)))
            L = torch.cat([g[:, :2], torch.sigmoid(g[:, 2:3])], 1)
            out["light"] = L / L.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return out


def full_normal(nxy: torch.Tensor) -> torch.Tensor:
    nz = torch.sqrt(torch.clamp(1 - (nxy**2).sum(1, keepdim=True), 1e-4, 1))
    n = torch.cat([nxy, nz], 1)
    return n / n.norm(dim=1, keepdim=True)


def load_net(path: str | Path) -> UNet:
    """A pbr_net checkpoint: {"config": {cues, width, norm, props}, "model", "ema"?}. Loaded with
    weights_only=True, so a checkpoint can't run code."""
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "config" not in ck or "model" not in ck:
        raise ValueError(f"{Path(path).name} is not a pbr_net checkpoint (no config/model).")
    c = ck["config"]
    width = c["width"] if isinstance(c["width"], str) else tuple(c["width"])
    m = UNet(cues=c["cues"], width=width, norm=c["norm"], props=c.get("props", False))
    m.load_state_dict(ck.get("ema") or ck["model"])
    return m.eval()


def save_net(model: UNet, path: str | Path) -> None:
    """Write a checkpoint in the same layout (used by the tests)."""
    width = model.width if isinstance(model.width, str) else list(model.ch)
    config = {"cues": model.cues, "width": width, "norm": True, "props": model.props}
    torch.save({"config": config, "model": model.state_dict()}, str(path))


def _pad_np(img: np.ndarray, Hp: int, Wp: int, wrap: Wrap) -> np.ndarray:
    H, W = img.shape[:2]
    img = np.pad(img, ((0, 0), (0, Wp - W), (0, 0)), mode="wrap" if wrap[0] else "reflect")
    return np.pad(img, ((0, Hp - H), (0, 0), (0, 0)), mode="wrap" if wrap[1] else "reflect")


@torch.no_grad()
def predict_all(model: UNet, img: np.ndarray, wrap: Wrap = (False, False), device=None) -> dict:
    """img (H,W,3) float sRGB -> every output as numpy, activations applied: normal (H,W,3),
    albedo (H,W,3), emissive; with props also shadow, ao, rough, metal, flow (H,W,2), thick
    (radius px), edges, curv (-1..1) and light (3,). Pads to a multiple of 16."""
    device = device or next(model.parameters()).device
    H, W = img.shape[:2]
    Hp, Wp = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    x = _pad_np(img, Hp, Wp, wrap)
    t = torch.tensor(x.transpose(2, 0, 1)[None].copy(), dtype=torch.float32, device=device)
    model.set_wrap(wrap)
    try:
        o = model.forward_all(preprocess(t, model.cues, wrap))
    finally:
        model.set_wrap((False, False))

    def cut(a):
        return a[0].float().cpu().numpy()[..., :H, :W]

    r = {
        "normal": full_normal(o["nxy"])[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W],
        "albedo": cut(o["albedo"]).transpose(1, 2, 0),
        "emissive": cut(torch.sigmoid(o["emissive"]))[0],
    }
    if model.props:
        r.update(
            shadow=cut(torch.sigmoid(o["shadow"]))[0],
            ao=cut(torch.sigmoid(o["ao"]))[0],
            rough=cut(torch.sigmoid(o["rough"]))[0],
            metal=cut(torch.sigmoid(o["metal"]))[0],
            flow=cut(torch.tanh(o["flow"])).transpose(1, 2, 0),
            thick=cut(torch.sigmoid(o["thick"]))[0] * 8,
            edges=cut(torch.sigmoid(o["edges"]))[0],
            curv=cut(torch.tanh(o["curv"]))[0],
            light=o["light"][0].float().cpu().numpy(),
        )
    return r
