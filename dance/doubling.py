from __future__ import annotations

"""Two-period (doubling) trajectoid generation for closed curves.

Mathematical basis
------------------
Rolling a unit sphere once around a closed planar loop produces a net rotation
``R(s) in SO(3)`` that depends on the loop's overall scale ``s``.  For a body
that rolls along the loop forever the orientation must return to its start after
a whole number of laps.

* A *single* lap requires ``R = I`` -- three scalar conditions on the one free
  parameter ``s`` -- which is generically impossible, so the single-period
  construction in :mod:`trajectoids_adapter` is only approximate (it tolerates a
  residual mismatch up to 65 degrees).
* Rolling the loop *twice* gives net rotation ``R**2``.  ``R**2 = I`` holds
  whenever ``R`` is a rotation by exactly pi about *any* axis (the axis is
  free).  The set of pi-rotations has codimension 1 in SO(3), so the
  one-parameter curve ``R(s)`` generically crosses it: there is always a scale
  ``s*`` at which ``R(s*)`` is a pi-rotation.  We find it by scanning the
  rotation angle ``theta(s)`` for its first peak at pi.

At ``s*`` the spherical contact trace over two laps closes back onto the south
pole (``R**2`` fixes it) and the body returns to its exact starting orientation
every two laps -- a mathematically exact two-period trajectoid.

Only the scale search and the two-lap path assembly are new here; rotation
accumulation, sphere trace, the implicit field and marching cubes are reused
from :mod:`trajectoids_adapter`.

Theory: Sobolev et al., "Solid-body trajectoids shaped to roll along desired
pathways", Nature 620, 310-315 (2023).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from trajectoids_adapter import (
    EPS,
    GenerationResult,
    _compute_normals,
    _field_to_mesh,
    _implicit_field,
    _rotation_angle,
    path_length,
    resample_uniform,
    rotations_to_origin,
    smooth_path,
    trace_on_sphere,
    validate_path,
)


def _get_minimize_scalar():
    try:
        from scipy.optimize import minimize_scalar as _minimize_scalar

        return _minimize_scalar
    except Exception:  # pragma: no cover - optional fallback if scipy is unavailable
        return None


@dataclass
class PiScaleResult:
    """Outcome of the search for the pi-rotation scale of a closed loop."""

    scale: float                # s* : scale at which one lap is a pi-rotation
    one_loop_angle: float       # theta(s*) ~ pi
    two_period_mismatch: float  # _rotation_angle(R(s*) @ R(s*)) ~ 0
    scan_scales: np.ndarray     # scanned scales (diagnostics)
    scan_angles: np.ndarray     # theta at each scanned scale (diagnostics)


def _one_loop_path(loop_xy: np.ndarray) -> np.ndarray:
    """Close an (N, 2) loop into an (N+1, 2) one-lap polyline (N segments).

    A trailing duplicate of the first point is appended only if the loop is not
    already explicitly closed.
    """
    pts = np.asarray(loop_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("loop must be an Nx2 array.")
    if pts.shape[0] < 2:
        raise ValueError("loop has too few points.")
    if np.linalg.norm(pts[0] - pts[-1]) > EPS:
        return np.vstack([pts, pts[:1]])
    return pts.copy()


def _doubled_closed_path(loop_xy: np.ndarray) -> np.ndarray:
    """Two full laps of a closed loop as a (2N+1, 2) polyline (2N segments).

    Any explicit closing-point duplicate is dropped first so that both laps have
    identical segment sequences, then the loop is repeated twice and finally
    closed back onto its first point.  The closing segment of each lap
    (``loop[N-1] -> loop[0]``) is therefore included.
    """
    pts = np.asarray(loop_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("loop must be an Nx2 array.")
    if pts.shape[0] < 2:
        raise ValueError("loop has too few points.")
    if np.linalg.norm(pts[0] - pts[-1]) <= EPS:
        pts = pts[:-1]
    return np.vstack([pts, pts, pts[:1]])


def _one_loop_angle(one_loop: np.ndarray, scale: float) -> float:
    """Rotation angle theta(s) in [0, pi] of one lap at the given scale.

    The angle is independent of ``core_radius`` (the accumulated rotation only
    depends on the planar path), matching the r=1 convention used throughout
    :mod:`trajectoids_adapter`.
    """
    net = rotations_to_origin(one_loop * float(scale))[-1]
    return _rotation_angle(net)


def _refine_peak(one_loop: np.ndarray, s_lo: float, s_hi: float) -> Tuple[float, float]:
    """Locate the scale of maximum theta within [s_lo, s_hi].

    theta has a tent-shaped peak (cusp) at exactly pi when the unwrapped rotation
    angle crosses pi, so maximizing the capped [0, pi] angle converges onto that
    crossing.  Uses scipy's bounded minimizer when available, else a two-level
    grid refine.
    """
    minimize_scalar = _get_minimize_scalar()
    if minimize_scalar is not None:
        result = minimize_scalar(
            lambda s: -_one_loop_angle(one_loop, s),
            bounds=(s_lo, s_hi),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 80},
        )
        s_best = float(result.x)
        return s_best, _one_loop_angle(one_loop, s_best)

    best_s, best_a = s_lo, _one_loop_angle(one_loop, s_lo)
    lo, hi = s_lo, s_hi
    for _ in range(2):
        for s in np.linspace(lo, hi, 41):
            a = _one_loop_angle(one_loop, float(s))
            if a > best_a:
                best_a, best_s = a, float(s)
        span = (hi - lo) / 41.0
        lo, hi = best_s - span, best_s + span
    return best_s, best_a


def _local_max_indices(values: np.ndarray) -> List[int]:
    """Indices of interior local maxima of a 1-D array (plateau-tolerant)."""
    out: List[int] = []
    n = values.shape[0]
    for i in range(1, n - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            out.append(i)
    return out


def find_pi_rotation_scale(
    loop_xy: np.ndarray,
    core_radius: float = 1.0,  # noqa: ARG001 - accepted for API symmetry; theta is r-independent
    n_scan: int = 120,
    lo_factor: float = 0.15,
    hi_factor: float = 3.0,
    coarse_tol: float = 0.12,
    fine_tol: float = 0.03,
    search_points: int = 160,
) -> PiScaleResult:
    """Find the smallest scale s* where one lap of ``loop_xy`` is a pi-rotation.

    Strategy: scan theta(s) over ``[lo_factor, hi_factor] * base`` (with
    ``base = 2*pi / one_lap_length``); the first local maximum that reaches pi
    (within ``coarse_tol``) is refined and accepted if its refined angle is
    within ``fine_tol`` of pi.  If no candidate qualifies the scale range is
    extended once before giving up.

    Raises ``ValueError`` if no pi-rotation scale can be located.
    """
    # Light-weight loop for the scan; the final mesh still uses full resolution.
    scan_loop = resample_uniform(np.asarray(loop_xy, dtype=float), n_points=search_points, closed=True)
    one = _one_loop_path(scan_loop)
    base = 2.0 * np.pi / max(path_length(one), EPS)

    def scan(lo_f: float, hi_f: float, count: int):
        scales = np.linspace(lo_f * base, hi_f * base, count)
        angles = np.array([_one_loop_angle(one, float(s)) for s in scales])
        return scales, angles

    attempts = [(lo_factor, hi_factor, n_scan), (lo_factor, hi_factor * 1.7, int(n_scan * 1.5))]
    scan_scales, scan_angles = scan(*attempts[0])

    for lo_f, hi_f, count in attempts:
        scan_scales, scan_angles = scan(lo_f, hi_f, count)
        target = np.pi - coarse_tol
        candidates = [i for i in _local_max_indices(scan_angles) if scan_angles[i] >= target]
        # Also consider a rising-to-the-end tail as a candidate bracket.
        if scan_angles[-1] >= target and (len(scan_angles) < 2 or scan_angles[-1] >= scan_angles[-2]):
            candidates.append(scan_angles.shape[0] - 1)

        for i in sorted(set(candidates)):
            lo = scan_scales[max(0, i - 1)]
            hi = scan_scales[min(scan_scales.shape[0] - 1, i + 1)]
            if hi <= lo:
                continue
            s_star, theta_star = _refine_peak(one, float(lo), float(hi))
            if theta_star >= np.pi - fine_tol:
                net = rotations_to_origin(one * s_star)[-1]
                two_period_mismatch = _rotation_angle(net @ net)
                return PiScaleResult(
                    scale=float(s_star),
                    one_loop_angle=float(theta_star),
                    two_period_mismatch=float(two_period_mismatch),
                    scan_scales=scan_scales,
                    scan_angles=scan_angles,
                )

    raise ValueError(
        "Could not find a two-period (pi-rotation) scale for this curve.\n"
        f"Max one-lap rotation angle reached was {np.degrees(float(np.max(scan_angles))):.1f} deg "
        "(needs to reach 180 deg). Try smooth + resample, simplify high-curvature "
        "corners, or redraw a more balanced curve."
    )


def generate_two_period_trajectoid_mesh(
    input_path: np.ndarray,
    *,
    smooth_passes: int = 1,
    resample_points: int = 320,
    resolution: int = 96,
    core_radius: float = 1.0,
    outer_radius: float = 1.25,
    max_planes: int = 180,
) -> GenerationResult:
    """Build an exact two-period trajectoid mesh from a closed curve.

    Mirrors :func:`trajectoids_adapter.generate_trajectoid_mesh` but scales the
    loop to the pi-rotation scale and cuts the sphere with the contact planes of
    *two* laps, so the body returns to its exact starting orientation every two
    laps.
    """
    points = np.asarray(input_path, dtype=float)
    validation = validate_path(points, require_closed=True)
    if validation.errors:
        message = "\n".join(validation.errors + validation.suggestions)
        raise ValueError(message)

    # Same preprocessing as the single-period path: recentre, smooth, resample.
    points = points - np.mean(points, axis=0, keepdims=True)
    points = smooth_path(points, passes=max(0, smooth_passes), closed=True)
    points = resample_uniform(points, n_points=resample_points, closed=True)

    pi_scale = find_pi_rotation_scale(points, core_radius=core_radius)
    scale = pi_scale.scale

    doubled = _doubled_closed_path(points)
    normals = _compute_normals(doubled, scale=scale, core_radius=core_radius, max_planes=max_planes)
    lin, field = _implicit_field(
        normals=normals,
        outer_radius=outer_radius,
        core_radius=core_radius,
        resolution=resolution,
    )
    vertices, faces = _field_to_mesh(lin, field)

    trace = trace_on_sphere(doubled, scale=scale, core_radius=core_radius)
    endpoint_gap = float(np.linalg.norm(trace[-1] - trace[0]))

    return GenerationResult(
        vertices=vertices,
        faces=faces,
        scale=scale,
        # Net rotation after two laps (R^2) -- the meaningful "does it close" metric.
        mismatch_angle=pi_scale.two_period_mismatch,
        endpoint_gap=endpoint_gap,
        # Single scaled lap: the visible path the roll simulation rolls along.
        resampled_points=points * scale,
        normals=normals,
        # Contact locus over a full two-lap period.
        surface_contact_curve=trace,
    )
