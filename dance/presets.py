"""Preset closed parametric curves for the dance app.

Each preset returns an (N, 2) ndarray centered at the origin and normalized so
that its closed-curve path length is approximately ``2*pi``. That length is the
natural circumference of a unit-radius rolling sphere, which keeps the
trajectoid mismatch-angle small and the auto-scale stable.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from trajectoids_adapter import path_length, resample_uniform, smooth_path


_TARGET_LENGTH = 2.0 * np.pi
_N_SAMPLES = 320


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


def _t() -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)


def circle() -> np.ndarray:
    t = _t()
    return _finalize(np.column_stack([np.cos(t), np.sin(t)]))


def figure_eight() -> np.ndarray:
    t = _t()
    a = 1.2
    return _finalize(np.column_stack([a * np.sin(t), a * np.sin(t) * np.cos(t)]))


def heart() -> np.ndarray:
    t = _t()
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    raw = np.column_stack([x, y])
    extent = np.max(np.linalg.norm(raw, axis=1))
    raw = raw / max(extent, 1e-9)
    return _finalize(raw)


def star_5() -> np.ndarray:
    t = _t()
    R = 1.0
    r_amp = 0.45
    radius = R + r_amp * np.cos(5 * t)
    return _finalize(np.column_stack([radius * np.cos(t), radius * np.sin(t)]))


def peanut() -> np.ndarray:
    t = _t()
    radius = 1.0 + 0.4 * np.cos(2 * t)
    return _finalize(np.column_stack([radius * np.cos(t), radius * np.sin(t)]))


def clover_3() -> np.ndarray:
    t = _t()
    radius = 1.0 + 0.5 * np.cos(3 * t)
    return _finalize(np.column_stack([radius * np.cos(t), radius * np.sin(t)]))


PRESETS: Dict[str, Callable[[], np.ndarray]] = {
    "circle": circle,
    "figure_eight": figure_eight,
    "heart": heart,
    "star_5": star_5,
    "peanut": peanut,
    "clover_3": clover_3,
}


PRESET_LABELS: Dict[str, str] = {
    "circle": "Circle",
    "figure_eight": "Figure 8",
    "heart": "Heart",
    "star_5": "5-Point Star",
    "peanut": "Peanut",
    "clover_3": "3-Leaf Clover",
}


def get_preset(name: str) -> np.ndarray:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return PRESETS[name]()
