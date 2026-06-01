"""Preset closed parametric curves for the dance app.

Each preset returns an (N, 2) ndarray centered at the origin and normalized so
that its closed-curve path length is approximately ``2*pi``. That length is the
natural circumference of a unit-radius rolling sphere, which keeps the
trajectoid mismatch-angle small and the auto-scale stable.

Presets come in two flavors:

* **Fixed**       — no parameters (e.g. ``circle``, ``heart``).
* **Parametric**  — one or more ``Param`` knobs (e.g. ``polygon`` with N sides,
                    ``rose`` with k petals). The UI exposes them as sliders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from trajectoids_adapter import path_length, resample_uniform, smooth_path


_TARGET_LENGTH = 2.0 * np.pi
_N_SAMPLES = 320


# ---------------------------------------------------------------------------
# Data structures describing parametric preset families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """A single tunable parameter for a parametric preset."""
    name: str           # internal key, used in Dancer.curve_params dict
    label: str          # UI label
    min: float
    max: float
    default: float
    kind: type = float  # float or int
    step: float = 0.05


@dataclass(frozen=True)
class PresetSpec:
    key: str
    label: str
    params: Tuple[Param, ...]  # empty = no parameters
    generator: Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Finalization helpers
# ---------------------------------------------------------------------------


def _normalize_length(points: np.ndarray) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    length = path_length(closed)
    if length < 1e-9:
        return points
    return points * (_TARGET_LENGTH / length)


def _finalize(points: np.ndarray) -> np.ndarray:
    pts = points - np.mean(points, axis=0, keepdims=True)
    pts = smooth_path(pts, passes=2, closed=True)
    pts = resample_uniform(pts, n_points=_N_SAMPLES, closed=True)
    pts = _normalize_length(pts)
    return pts.astype(float)


def _t(n: int = 256) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


# ---------------------------------------------------------------------------
# Generators — fixed shapes
# ---------------------------------------------------------------------------


def gen_circle() -> np.ndarray:
    t = _t()
    return _finalize(np.column_stack([np.cos(t), np.sin(t)]))


def gen_figure_eight() -> np.ndarray:
    t = _t()
    a = 1.2
    return _finalize(np.column_stack([a * np.sin(t), a * np.sin(t) * np.cos(t)]))


def gen_infinity() -> np.ndarray:
    # Bernoulli lemniscate, smoother than figure_eight.
    t = _t()
    denom = 1.0 + np.sin(t) ** 2
    return _finalize(np.column_stack([
        np.cos(t) / denom,
        np.sin(t) * np.cos(t) / denom,
    ]))


def gen_heart() -> np.ndarray:
    t = _t()
    x = 16.0 * np.sin(t) ** 3
    y = (13.0 * np.cos(t)
         - 5.0 * np.cos(2 * t)
         - 2.0 * np.cos(3 * t)
         - np.cos(4 * t))
    raw = np.column_stack([x, y])
    extent = np.max(np.linalg.norm(raw, axis=1))
    raw = raw / max(extent, 1e-9)
    return _finalize(raw)


def gen_cardioid() -> np.ndarray:
    # r = 1 - cos(t)  →  classic single-cusp heart shape, smoother than gen_heart.
    t = _t()
    r = 1.0 - np.cos(t)
    return _finalize(np.column_stack([r * np.cos(t), r * np.sin(t)]))


def gen_peanut() -> np.ndarray:
    t = _t()
    radius = 1.0 + 0.4 * np.cos(2 * t)
    return _finalize(np.column_stack([radius * np.cos(t), radius * np.sin(t)]))


# ---------------------------------------------------------------------------
# Generators — parametric shape families
# ---------------------------------------------------------------------------


def gen_ellipse(aspect: float = 1.5) -> np.ndarray:
    aspect = max(float(aspect), 0.01)
    t = _t()
    return _finalize(np.column_stack([np.cos(t), np.sin(t) / aspect]))


def _polygon_perimeter(verts: np.ndarray, samples_total: int) -> np.ndarray:
    """Sample uniformly along a polygon's edges."""
    n = verts.shape[0]
    per_edge = max(4, samples_total // n)
    out = []
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        for j in range(per_edge):
            t = j / per_edge
            out.append(a + (b - a) * t)
    return np.asarray(out)


def gen_polygon(n_sides: float = 6) -> np.ndarray:
    n = max(3, int(round(n_sides)))
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _finalize(_polygon_perimeter(verts, _N_SAMPLES))


def gen_star(n_points: float = 5, inner_ratio: float = 0.45) -> np.ndarray:
    n_points = max(4, int(round(n_points)))
    inner = float(np.clip(inner_ratio, 0.05, 0.95))
    total = 2 * n_points
    verts = []
    for i in range(total):
        r = 1.0 if i % 2 == 0 else inner
        ang = 2 * np.pi * i / total
        verts.append([r * np.cos(ang), r * np.sin(ang)])
    return _finalize(_polygon_perimeter(np.asarray(verts), _N_SAMPLES))


def gen_star_polygon(n_points: float = 5, step: float = 2) -> np.ndarray:
    # {n/k} star polygon: place n vertices on a circle, then connect every
    # k-th one. n_points=5, step=2 is the classic pentagram (K5 minus the
    # outer pentagon). The self-intersecting "drawn in one stroke" star.
    n = max(5, int(round(n_points)))
    s = int(round(step))
    s = max(2, min(s, n - 2))
    verts = []
    idx = 0
    while True:
        ang = 2 * np.pi * idx / n
        verts.append([np.cos(ang), np.sin(ang)])
        idx = (idx + s) % n
        if idx == 0:
            break
    return _finalize(_polygon_perimeter(np.asarray(verts), _N_SAMPLES))


def gen_rose(k: float = 3) -> np.ndarray:
    k = max(2, int(round(k)))
    # r = |cos(k t)| with a small offset so the curve never passes through the
    # origin (which makes the trajectoid mesh ill-conditioned).
    t = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    r = np.abs(np.cos(k * t)) + 0.15
    return _finalize(np.column_stack([r * np.cos(t), r * np.sin(t)]))


def gen_lissajous(a: float = 1, b: float = 2) -> np.ndarray:
    a = max(1, int(round(a)))
    b = max(1, int(round(b)))
    t = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    return _finalize(np.column_stack([
        np.sin(a * t),
        np.sin(b * t + np.pi / 2),
    ]))


def gen_clover(n_leaves: float = 3) -> np.ndarray:
    k = max(2, int(round(n_leaves)))
    t = _t()
    radius = 1.0 + 0.5 * np.cos(k * t)
    return _finalize(np.column_stack([radius * np.cos(t), radius * np.sin(t)]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


PRESET_SPECS: Tuple[PresetSpec, ...] = (
    PresetSpec("circle",       "Circle",                (),                                                        gen_circle),
    PresetSpec("ellipse",      "Ellipse",
               (Param("aspect", "Aspect (y/x)", 0.3, 3.0, 1.5, float, 0.1),),
               gen_ellipse),
    PresetSpec("polygon",      "Polygon (N-gon)",
               (Param("n_sides", "Sides", 3, 12, 6, int, 1),),
               gen_polygon),
    PresetSpec("star",         "Star",
               (Param("n_points",    "Points",      4,   12,   5,   int,   1),
                Param("inner_ratio", "Inner ratio", 0.2, 0.8,  0.45, float, 0.05)),
               gen_star),
    PresetSpec("star_polygon", "Star polygon {n/k}",
               (Param("n_points", "Points", 5, 12, 5, int, 1),
                Param("step",     "Step",   2,  5, 2, int, 1)),
               gen_star_polygon),
    PresetSpec("rose",         "Rose curve",
               (Param("k", "Petal count", 2, 8, 3, int, 1),),
               gen_rose),
    PresetSpec("lissajous",    "Lissajous",
               (Param("a", "a", 1, 6, 1, int, 1),
                Param("b", "b", 1, 6, 2, int, 1)),
               gen_lissajous),
    PresetSpec("figure_eight", "Figure 8",              (),                                                        gen_figure_eight),
    PresetSpec("infinity",     "Infinity (lemniscate)", (),                                                        gen_infinity),
    PresetSpec("heart",        "Heart",                 (),                                                        gen_heart),
    PresetSpec("cardioid",     "Cardioid",              (),                                                        gen_cardioid),
    PresetSpec("peanut",       "Peanut",                (),                                                        gen_peanut),
    PresetSpec("clover",       "Clover",
               (Param("n_leaves", "Leaves", 2, 8, 3, int, 1),),
               gen_clover),
)


PRESETS_BY_KEY: Dict[str, PresetSpec] = {s.key: s for s in PRESET_SPECS}

# Back-compat: a few existing callers (and old .tdance files) still use the
# old name-only map. Kept as a {key: label} dict so ``PRESET_LABELS.items()``
# iteration patterns keep working.
PRESET_LABELS: Dict[str, str] = {s.key: s.label for s in PRESET_SPECS}


# Legacy preset keys saved in older .tdance files. Map old key →
# (new_key, default_param_kwargs). Used by scene_io to migrate at load time.
LEGACY_KEY_MIGRATION: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "star_5":   ("star",   {"n_points": 5, "inner_ratio": 0.45}),
    "clover_3": ("clover", {"n_leaves": 3}),
}


def get_preset_spec(name: str) -> Optional[PresetSpec]:
    """Return the spec for ``name``, transparently migrating legacy keys."""
    if name in LEGACY_KEY_MIGRATION:
        new_key, _ = LEGACY_KEY_MIGRATION[name]
        return PRESETS_BY_KEY.get(new_key)
    return PRESETS_BY_KEY.get(name)


def get_preset(name: str, **params: Any) -> np.ndarray:
    """Generate a preset curve. Extra kwargs override the preset's default params.

    Legacy keys (``star_5``, ``clover_3``) auto-migrate.
    """
    if name in LEGACY_KEY_MIGRATION:
        new_key, defaults = LEGACY_KEY_MIGRATION[name]
        merged = {**defaults, **params}
        return get_preset(new_key, **merged)
    spec = PRESETS_BY_KEY.get(name)
    if spec is None:
        raise KeyError(f"Unknown preset: {name}")
    # Fill in defaults for any params the caller omitted.
    kwargs = {p.name: p.default for p in spec.params}
    for k, v in params.items():
        if k in kwargs:
            kwargs[k] = v
    return spec.generator(**kwargs)
