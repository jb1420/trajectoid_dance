"""Data model and generation pipeline for one trajectoid "dancer".

A Dancer owns its source curve, motion parameters, and (after generation) its
mesh and roll-simulation results. The DanceScene aggregates dancers and the
shared playback timeline length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import uuid

import numpy as np

from trajectoids_adapter import (
    GenerationResult,
    RollSimulationResult,
    build_roll_simulation,
    generate_trajectoid_mesh,
    path_length,
    resample_uniform,
    smooth_path,
    validate_path,
)


# 8-color palette (Tableau 10 minus gray + light variants), cycled for new dancers.
COLOR_PALETTE = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
    "#bcf60c",  # lime
]


@dataclass
class Dancer:
    dancer_id: str
    name: str
    curve_source: str  # e.g. "preset:circle" or "freehand"
    curve_xy: np.ndarray  # (N, 2) raw input curve
    color_hex: str
    start_offset_xy: tuple[float, float] = (0.0, 0.0)
    phase_offset: float = 0.0       # [0, 1)
    speed_multiplier: float = 1.0   # 0.25..4.0
    n_cycles: int = 1
    closed: bool = True

    # Computed (None until generated successfully):
    gen_result: Optional[GenerationResult] = None
    sim_result: Optional[RollSimulationResult] = None
    cycle_arc_length: float = 0.0

    @staticmethod
    def new(curve_source: str, curve_xy: np.ndarray, name: str, color_hex: str) -> "Dancer":
        return Dancer(
            dancer_id=uuid.uuid4().hex,
            name=name,
            curve_source=curve_source,
            curve_xy=np.asarray(curve_xy, dtype=float).copy(),
            color_hex=color_hex,
        )


@dataclass
class DanceScene:
    dancers: List[Dancer] = field(default_factory=list)
    global_ticks: int = 480
    duration_seconds: float = 12.0
    loop: bool = False

    def add(self, dancer: Dancer) -> None:
        self.dancers.append(dancer)

    def remove(self, dancer_id: str) -> None:
        self.dancers = [d for d in self.dancers if d.dancer_id != dancer_id]

    def find(self, dancer_id: str) -> Optional[Dancer]:
        for d in self.dancers:
            if d.dancer_id == dancer_id:
                return d
        return None

    def next_color(self) -> str:
        return COLOR_PALETTE[len(self.dancers) % len(COLOR_PALETTE)]

    def next_name(self) -> str:
        existing = {d.name for d in self.dancers}
        i = 1
        while True:
            candidate = f"Dancer {i}"
            if candidate not in existing:
                return candidate
            i += 1


def prepare_curve(raw_xy: np.ndarray, source: str) -> np.ndarray:
    """Apply Y-flip (screen→math), smooth, resample, and recentre."""
    pts = np.asarray(raw_xy, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Curve must be an Nx2 array.")
    if source.startswith("freehand"):
        # Freehand input is in Qt screen coords (y-down); flip to math (y-up).
        pts[:, 1] *= -1.0
    # Recentre so the trajectoid is built around the origin.
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    pts = smooth_path(pts, passes=1, closed=True)
    pts = resample_uniform(pts, n_points=320, closed=True)
    return pts


def _resample_translations(translations: np.ndarray, target_len: int) -> np.ndarray:
    src_n = translations.shape[0]
    if src_n == target_len:
        return translations.astype(float).copy()
    src_t = np.linspace(0.0, 1.0, src_n)
    tgt_t = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, translations.shape[1]), dtype=float)
    for c in range(translations.shape[1]):
        out[:, c] = np.interp(tgt_t, src_t, translations[:, c])
    return out


def _resample_rotations_nearest(rotations: np.ndarray, target_len: int) -> np.ndarray:
    src_n = rotations.shape[0]
    if src_n == target_len:
        return rotations.astype(float).copy()
    src_idx = np.round(np.linspace(0.0, src_n - 1, target_len)).astype(int)
    src_idx = np.clip(src_idx, 0, src_n - 1)
    return rotations[src_idx].astype(float).copy()


@dataclass
class NormalizedSim:
    translations: np.ndarray  # (T, 3)
    rotations: np.ndarray     # (T, 3, 3)
    trajectory_xy: np.ndarray # (T, 2)


def normalize_sim(sim: RollSimulationResult, target_len: int) -> NormalizedSim:
    trans = _resample_translations(sim.translations_xyz, target_len)
    rots = _resample_rotations_nearest(sim.rotations, target_len)
    traj = _resample_translations(sim.trajectory_xy, target_len)
    return NormalizedSim(translations=trans, rotations=rots, trajectory_xy=traj)


def generate_dancer(
    dancer: Dancer,
    *,
    resolution: int = 96,
    core_radius: float = 1.0,
) -> Dancer:
    """Run the full curve→mesh→sim pipeline for one dancer.

    Mutates and returns the dancer with gen_result / sim_result / cycle_arc_length filled in.
    Raises ValueError on validation failure (with a human-readable message).
    """
    pts = prepare_curve(dancer.curve_xy, dancer.curve_source)

    validation = validate_path(pts, require_closed=True)
    if validation.errors:
        message = "\n".join(validation.errors + validation.suggestions)
        raise ValueError(message)

    gen = generate_trajectoid_mesh(
        pts,
        require_closed=True,
        smooth_passes=1,
        resample_points=320,
        resolution=resolution,
        core_radius=core_radius,
    )

    cycle_pts = gen.resampled_points
    closed_cycle = np.vstack([cycle_pts, cycle_pts[0]])
    cycle_arc = float(path_length(closed_cycle))
    n_cycles = max(1, int(dancer.n_cycles))
    target_angle = (cycle_arc / max(core_radius, 1e-9)) * n_cycles
    n_frames = max(240, 200 * n_cycles)

    sim = build_roll_simulation(
        cycle_pts,
        target_roll_angle_rad=target_angle,
        closed=True,
        n_frames=n_frames,
        core_radius=core_radius,
    )

    dancer.gen_result = gen
    dancer.sim_result = sim
    dancer.cycle_arc_length = cycle_arc
    return dancer
