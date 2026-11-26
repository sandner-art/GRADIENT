# src/gradient/ai/pbr/classic.py
"""Classical image -> slopes / height / AO / albedo / glow pieces, and the Hybrid method.

Ported from pbr_experiments/methods.py (same author). Slopes (p, q) are of a pseudo-height in
pixel units, p = dh/dcol, q = dh/drow; slopes_to_normal() builds an OpenGL tangent-space
normal (green up). Every function takes `wrap=(x, y)`: wrapped axes are treated as periodic, so
the maps of a seamless texture tile too.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
Wrap = tuple[bool, bool]
OPEN: Wrap = (False, False)


def modes(wrap: Wrap, other: str = "reflect") -> tuple[str, str]:
    """scipy.ndimage modes per axis (rows, cols)."""
    return ("wrap" if wrap[1] else other, "wrap" if wrap[0] else other)


def srgb_to_lin(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def slopes_to_normal(p: np.ndarray, q: np.ndarray, k: float = 1.0) -> np.ndarray:
    n = np.stack([-k * p, k * q, np.ones_like(p)], -1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def normal_to_slopes(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = np.clip(n[..., 2], 1e-3, None)
    return -n[..., 0] / nz, n[..., 1] / nz


def grad(a: np.ndarray, wrap: Wrap = OPEN) -> tuple[np.ndarray, np.ndarray]:
    m = modes(wrap)
    return ndi.sobel(a, 1, mode=m) / 8.0, ndi.sobel(a, 0, mode=m) / 8.0


def gauss(a: np.ndarray, sigma: float, wrap: Wrap = OPEN) -> np.ndarray:
    return ndi.gaussian_filter(a, sigma, mode=modes(wrap))


def delight_log(img: np.ndarray, wrap: Wrap = OPEN, sigma_frac: float = 1 / 16):
    """Log luminance with low-frequency illumination (pools, vignette) removed."""
    Y = srgb_to_lin(img[..., :3]) @ LUMA
    lg = np.log(Y + 1e-3)
    illum = gauss(lg, max(img.shape[:2]) * sigma_frac, wrap)
    return lg - illum, illum


def chroma_weight(img: np.ndarray, wrap: Wrap = OPEN, tau: float = 0.02, sigma: float = 1.0):
    """~0 where chromaticity changes sharply (paint edges), ~1 elsewhere."""
    lin = srgb_to_lin(img[..., :3]) + 1e-3
    c = lin / lin.sum(-1, keepdims=True)
    g = 0
    for ch in range(2):
        a, b = grad(gauss(c[..., ch], sigma, wrap), wrap)
        g = g + a * a + b * b
    return 1.0 / (1.0 + g / tau**2)


def glow_mask(img: np.ndarray, wrap: Wrap = OPEN, sat_min: float = 0.25, lum_k: float = 3.0):
    """Small, locally very bright and/or saturated blobs -> emissive mask."""
    lin = srgb_to_lin(img[..., :3])
    Y = lin @ LUMA
    ratio = Y / (gauss(Y, 6, wrap) + 1e-4)
    mx, mn = lin.max(-1), lin.min(-1)
    sat = (mx - mn) / (mx + 1e-4)
    peak = (ratio > lum_k) & (Y > np.percentile(Y, 97))
    m = peak.astype(np.float32) * np.clip(0.5 + sat, 0, 1)
    m = np.maximum(m, gauss(m, 1.0, wrap) * 1.5)
    return np.clip(m, 0, 1)


def robust_clip(p, q, pct: float = 99.7, mult: float = 1.5):
    """Cap outlier slopes (saturated pixels, LED edges) before integration."""
    m = np.sqrt(p * p + q * q)
    cap = np.percentile(m, pct) * mult + 1e-9
    s = np.minimum(1.0, cap / (m + 1e-12))
    return p * s, q * s


def frankot_chellappa(p: np.ndarray, q: np.ndarray, wrap: Wrap = OPEN) -> np.ndarray:
    """Least-squares integrable height from slopes (FFT). Open axes are mirrored first, so their
    borders get no seam; wrapped axes are integrated as periodic, so the height tiles."""
    H, W = p.shape
    P, Q = p, q
    if not wrap[0]:
        P, Q = np.concatenate([P, -P[:, ::-1]], 1), np.concatenate([Q, Q[:, ::-1]], 1)
    if not wrap[1]:
        P, Q = np.concatenate([P, P[::-1]], 0), np.concatenate([Q, -Q[::-1]], 0)
    h, w = P.shape
    wy = np.fft.fftfreq(h)[:, None] * 2 * np.pi
    wx = np.fft.fftfreq(w)[None, :] * 2 * np.pi
    sx, sy = np.sin(wx), np.sin(wy)  # the central difference's response
    denom = sx**2 + sy**2
    dead = denom < 1e-8  # DC and Nyquist modes are unconstrained
    Z = (-1j * sx * np.fft.fft2(P) - 1j * sy * np.fft.fft2(Q)) / np.where(dead, 1, denom)
    Z[dead] = 0
    z = np.real(np.fft.ifft2(Z))[:H, :W]
    return (z - z.mean()).astype(np.float32)


def height_slopes(z: np.ndarray, wrap: Wrap = OPEN) -> tuple[np.ndarray, np.ndarray]:
    gc = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) * 0.5
    gr = (np.roll(z, -1, 0) - np.roll(z, 1, 0)) * 0.5
    if not wrap[0]:
        gc[:, 0] = z[:, 1] - z[:, 0]
        gc[:, -1] = z[:, -1] - z[:, -2]
    if not wrap[1]:
        gr[0] = z[1] - z[0]
        gr[-1] = z[-1] - z[-2]
    return gc, gr


def remove_tilt(z: np.ndarray, wrap: Wrap = OPEN) -> np.ndarray:
    """Subtract the best-fit plane along the open axes (a wrapped axis can't tilt)."""
    H, W = z.shape
    cols = [np.ones(z.size)]
    if not wrap[0]:
        cols.append(np.tile(np.linspace(-1, 1, W), H))
    if not wrap[1]:
        cols.append(np.repeat(np.linspace(-1, 1, H), W))
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, z.ravel().astype(np.float64), rcond=None)
    return (z - (A @ coef).reshape(H, W)).astype(np.float32)


def classic_slopes(img: np.ndarray, wrap: Wrap = OPEN, scales=(0.7, 1.5, 3, 6, 12)):
    """No network: slopes of the de-lit log luminance over several scales, quiet across paint
    edges and lights (the classical part of the generator's Enhanced method, unfitted)."""
    lg, _ = delight_log(img, wrap)
    p = np.zeros_like(lg)
    q = np.zeros_like(lg)
    for s in scales:
        a, b = grad(gauss(lg, s, wrap), wrap)
        p += a
        q += b
    calm = 1 - np.clip(ndi.maximum_filter(glow_mask(img, wrap), 7, mode=modes(wrap)), 0, 1)
    calm = calm * chroma_weight(img, wrap)
    return robust_clip(p / len(scales) * calm, q / len(scales) * calm)


def hybrid_slopes(img, lp, lq, detail: float = 0.35, sigma: float = 2.0, wrap: Wrap = OPEN):
    """Learned low/mid frequencies + the picture's fine detail, one Poisson solve.
    img and (lp, lq) at the same resolution. Returns integrable slopes and the height."""
    lg, _ = delight_log(img, wrap)
    dp, dq = grad(lg - gauss(lg, sigma, wrap), wrap)
    dp, dq = robust_clip(dp, dq, 99.0, 1.0)
    calm = 1 - np.clip(ndi.maximum_filter(glow_mask(img, wrap), 7, mode=modes(wrap)), 0, 1)
    calm = calm * chroma_weight(img, wrap)
    lp, lq = robust_clip(lp, lq)
    mag = np.mean(np.sqrt(lp**2 + lq**2))
    dmag = np.mean(np.sqrt(dp**2 + dq**2)) + 1e-9
    hp = lp + detail * mag / dmag * dp * calm
    hq = lq + detail * mag / dmag * dq * calm
    z = frankot_chellappa(*robust_clip(hp, hq), wrap=wrap)
    p, q = height_slopes(z, wrap)
    return p, q, z


def ao_from_height(z: np.ndarray, wrap: Wrap = OPEN, radii=(1, 2, 4, 8, 16, 32), dirs: int = 8):
    """Horizon-based ambient occlusion from the height (in pixel units)."""
    r_max = int(max(radii)) + 2
    zp = np.pad(z, ((0, 0), (r_max, r_max)), mode="wrap" if wrap[0] else "edge")
    zp = np.pad(zp, ((r_max, r_max), (0, 0)), mode="wrap" if wrap[1] else "edge")
    acc = np.zeros(zp.shape, np.float32)
    for k in range(dirs):
        a = 2 * np.pi * k / dirs
        hor = np.zeros(zp.shape, np.float32)
        for r in radii:
            dc, dr = np.cos(a) * r, -np.sin(a) * r
            zs = ndi.shift(zp, (-dr, -dc), order=1, mode="nearest")
            hor = np.maximum(hor, np.arctan2(zs - zp, r))
        acc += np.sin(hor)
    ao = np.clip(1 - acc / dirs, 0, 1)
    return ao[r_max:-r_max, r_max:-r_max].astype(np.float32)


def curvature(z: np.ndarray, wrap: Wrap = OPEN, sigma: float = 1.5) -> np.ndarray:
    """-1..1: + creases (cavities), - ridges (edges), like the network's curvature head."""
    lap = ndi.gaussian_laplace(z, sigma, mode=modes(wrap))
    s = np.percentile(np.abs(lap), 99) + 1e-9
    return np.tanh(lap / s * 2).astype(np.float32)


def albedo_retinex(img: np.ndarray, wrap: Wrap = OPEN, ao=None, sigma_frac: float = 1 / 16):
    """Divide out low-frequency illumination (and optionally AO)."""
    lin = srgb_to_lin(img[..., :3])
    Y = lin @ LUMA + 1e-3
    illum = np.exp(gauss(np.log(Y), max(img.shape[:2]) * sigma_frac, wrap))
    a = lin / illum[..., None]
    if ao is not None:
        a = a / np.clip(ao, 0.3, 1)[..., None]
    a = a / (np.percentile(a @ LUMA, 99) + 1e-6) * 0.8
    return lin_to_srgb(a).astype(np.float32)


def albedo_with_color(img: np.ndarray, net_albedo: np.ndarray, median: float = 0.12) -> np.ndarray:
    """The network's de-lit brightness with the picture's colour. pbr_net v3's albedo head removes
    the light well but comes out nearly grey, so its luminance is kept and the chromaticity is
    taken from the picture. Normalised like the generator: median luminance 0.12 linear."""
    lin = srgb_to_lin(img[..., :3])
    Y = lin @ LUMA
    Yn = srgb_to_lin(net_albedo) @ LUMA
    chroma = (lin + 1e-3) / (Y[..., None] + 1e-3)
    w = (Y / (Y + 0.005))[..., None]  # near black the picture's colour is noise: fade to grey
    a = (1 + (chroma - 1) * w) * Yn[..., None]
    a = a * (median / (np.median(a @ LUMA) + 1e-6))
    return lin_to_srgb(a).astype(np.float32)
