r"""3D-printable trajectoid generator (steel-ball-in-the-middle version).

A trajectoid only rolls along its design path if its **centre of mass sits at
the geometric centre** of the bounding sphere (the rolling theory assumes a
uniform sphere). A bare FDM-printed shell fails this: the contact planes carve
plastic away asymmetrically, so the plastic-only centroid is off-centre. The
fix used in the original paper (Sobolev et al., Nature 620, 2023) and in
``trajectoid/ver2_PyBullet/config.py`` is to embed a dense **steel ball** at the
centre — the ball's mass dominates and pins the CoM to the centre.

This script therefore builds the trajectoid as a **hollow shell with a central
spherical cavity** sized for a real steel ball:

    solid region  =  { ‖p‖ ≤ R_outer }  ∩  { nₖ·p ≥ -R_core ∀k }  \  { ‖p‖ < R_cavity }
                     └ outer sphere ┘    └ contact-plane cuts  ┘     └ ball pocket ┘

The closed-curve maths reuses the maintained period-2 (doubling) pipeline so the
body returns to its exact starting orientation every two laps.

Print modes
-----------
* ``split``   (default) — two watertight halves split through the centre. Print
                both flat-face-down, drop the steel ball into the bowl, glue the
                halves together. Easiest to actually assemble.
* ``inplace`` — one watertight shell with a fully enclosed cavity. Pause the
                print at mid-height, drop the ball in, resume. No glue.
* ``solid``   — the plain trajectoid (no cavity), for reference.

Run from the ``dance/`` directory, e.g.::

    python make_printable_trajectoid.py --preset trefoil --ball-mm 19.05
    python make_printable_trajectoid.py --preset heart --mode inplace --ball-mm 12.7
    python make_printable_trajectoid.py --path-file mycurve.npy --mode split

Output goes to ``output/printable/<name>/``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Console may be a non-UTF-8 codepage (e.g. cp949 on Korean Windows); the report
# uses ≈, ×, °, π … so force UTF-8 output where the runtime allows it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent))

# Maintained closed-curve (period-2) pipeline + low-level field/mesh helpers.
from doubling import find_pi_rotation_scale, _doubled_closed_path
from trajectoids_adapter import (
    validate_path,
    smooth_path,
    resample_uniform,
    estimate_scale,
    path_length,
    _compute_normals,
    _implicit_field,
    _field_to_mesh,
)

# ---------------------------------------------------------------------------
# Physical constants (mirrors ver2_PyBullet/config.py)
# ---------------------------------------------------------------------------
PLA_DENSITY = 1240.0     # kg/m^3 — printed shell
STEEL_DENSITY = 7874.0   # kg/m^3 — ball bearing

# A few common steel-ball diameters [mm] for the closest-size hint.
COMMON_BALL_MM = {
    "1/4\"": 6.35, "5/16\"": 7.9375, "3/8\"": 9.525, "1/2\"": 12.7,
    "5/8\"": 15.875, "3/4\"": 19.05, "7/8\"": 22.225, "1\"": 25.4,
    "10 mm": 10.0, "16 mm": 16.0, "20 mm": 20.0, "25 mm": 25.0,
}


# ---------------------------------------------------------------------------
# Curve input
# ---------------------------------------------------------------------------
def _normalize_open_length(points: np.ndarray, target: float) -> np.ndarray:
    """Centre an open polyline and scale it to a given arc length."""
    pts = points - np.mean(points, axis=0, keepdims=True)
    length = path_length(pts)
    if length < 1e-9:
        return pts
    return pts * (target / length)


def periodic_sinusoid(periods: int = 1, amp: float = 0.35, n: int = 400) -> np.ndarray:
    """One or more periods of a sine wave — the canonical open periodic path."""
    t = np.linspace(0.0, 1.0, n)
    y = amp * np.sin(2.0 * np.pi * periods * t)
    pts = np.column_stack([t, y])
    return _normalize_open_length(pts, target=2.0 * np.pi * periods)


def periodic_zigzag(periods: int = 1, amp: float = 0.3, n: int = 400) -> np.ndarray:
    """Triangle (zig-zag) wave; corners are softened later by smoothing."""
    t = np.linspace(0.0, 1.0, n)
    y = amp * (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * periods * t))
    pts = np.column_stack([t, y])
    return _normalize_open_length(pts, target=2.0 * np.pi * periods)


PERIODIC_PATHS = {"sinusoid": periodic_sinusoid, "zigzag": periodic_zigzag}


def load_curve(preset, path_file, periodic, periods, amp, open_file):
    """Return ``(curve (N,2), name, closed)`` from one of three sources."""
    if periodic:
        if periodic not in PERIODIC_PATHS:
            raise ValueError(f"Unknown periodic path '{periodic}'. "
                             f"Choose from {sorted(PERIODIC_PATHS)}.")
        curve = PERIODIC_PATHS[periodic](periods=periods, amp=amp)
        return curve, f"{periodic}_p{periods}", False

    if path_file:
        p = Path(path_file)
        if p.suffix.lower() == ".npy":
            curve = np.load(p)
        else:  # csv / txt: two columns
            curve = np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None)
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(f"{path_file} must contain an Nx2 array of points.")
        return curve, p.stem, not open_file

    from presets import get_preset, get_preset_spec

    name = preset or "trefoil"
    if get_preset_spec(name) is None:
        raise ValueError(f"Unknown preset '{name}'.")
    return get_preset(name), name, True


# ---------------------------------------------------------------------------
# Geometry: hollow trajectoid field with central ball cavity
# ---------------------------------------------------------------------------
class PrintGeometry:
    """All derived dimensions, in millimetres, plus the algorithm-unit field.

    ``size_scale`` uniformly enlarges the whole object (e.g. 3.0 = print it at
    triple size). Algorithm-unit ratios are scale-independent; only ``scale_mm``
    and the displayed millimetre dimensions carry the factor.
    """

    def __init__(self, ball_mm, wall_mm, clearance_mm, shell_ratio, size_scale=1.0):
        self.size_scale = float(size_scale)
        self.shell_ratio = float(shell_ratio)

        ball_r = float(ball_mm) / 2.0
        cav = ball_r + float(clearance_mm)          # base (1x) pocket radius
        core = cav + float(wall_mm)                 # base (1x) inscribed core radius

        # Algorithm units fix the inscribed (core) sphere at radius 1.0, so the
        # cavity radius as a fraction of it is independent of the print size.
        self.cavity_r_algo = cav / core             # < 1.0, inside every cut plane

        s = self.size_scale
        self.ball_mm = float(ball_mm) * s
        self.wall_mm = float(wall_mm) * s
        self.clearance_mm = float(clearance_mm) * s
        self.cavity_r_mm = cav * s
        self.core_r_mm = core * s
        self.outer_r_mm = core * self.shell_ratio * s
        self.scale_mm = core * s                    # algo 1.0 -> mm

    # -- pretty hint for the nearest off-the-shelf ball --------------------
    def nearest_ball_label(self) -> str:
        label, mm = min(COMMON_BALL_MM.items(), key=lambda kv: abs(kv[1] - self.ball_mm))
        if abs(mm - self.ball_mm) < 0.05:
            return f"{label} ({mm:g} mm)"
        return f"nearest standard: {label} ({mm:g} mm)"


@dataclass
class ScaleInfo:
    scale: float            # algorithm-unit curve scale
    mismatch_rad: float     # net-rotation mismatch over one printed period (≈0 = exact)
    period_label: str       # human description of the closure scheme


def build_normals(curve, *, closed, smooth_passes, resample_points, max_planes):
    """Preprocess the curve → contact-plane unit normals (algorithm units).

    * **closed** loops use the exact period-2 (doubling) scheme — the body
      returns to its starting orientation every two laps.
    * **open periodic** paths use the single-period scheme: one period of the
      path already rolls the body back to its starting orientation (mismatch≈0),
      so the trajectoid tiles forward along the repeated path.
    """
    validation = validate_path(curve, require_closed=closed)
    if validation.errors:
        raise ValueError("\n".join(validation.errors + validation.suggestions))

    points = curve - np.mean(curve, axis=0, keepdims=True)
    points = smooth_path(points, passes=max(0, smooth_passes), closed=closed)
    points = resample_uniform(points, n_points=resample_points, closed=closed)

    if closed:
        pi = find_pi_rotation_scale(points, core_radius=1.0)
        doubled = _doubled_closed_path(points)
        normals = _compute_normals(doubled, scale=pi.scale, core_radius=1.0, max_planes=max_planes)
        info = ScaleInfo(pi.scale, pi.two_period_mismatch, "period-2 (closed loop)")
    else:
        scale, angle, _gap = estimate_scale(points, core_radius=1.0)
        if angle > np.deg2rad(65.0):
            raise ValueError(
                f"Open-path orientation mismatch too high ({np.degrees(angle):.1f}°). "
                "Use a gentler amplitude (--amp) or fewer periods (--periods)."
            )
        normals = _compute_normals(points, scale=scale, core_radius=1.0, max_planes=max_planes)
        info = ScaleInfo(scale, angle, "single-period (open periodic)")
    return normals, info, points


def hollow_fields(normals, geom: PrintGeometry, resolution: int):
    """Build the solid-trajectoid field and derive hollow / split variants.

    Returns ``(lin, fields)`` where ``fields`` maps a name to a level-0 scalar
    grid (``<= 0`` is solid). The expensive plane evaluation runs once; the
    cavity and the z-cut are cheap ``max`` combinations on top of it.
    """
    lin, solid = _implicit_field(
        normals=normals,
        outer_radius=geom.shell_ratio,   # outer sphere radius in algo units
        core_radius=1.0,                 # cut planes tangent to the unit core sphere
        resolution=int(resolution),
    )

    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)

    cavity_term = geom.cavity_r_algo - r           # > 0 inside the ball pocket → carved out
    hollow = np.maximum(solid, cavity_term)        # shell between cavity and outer/cuts

    fields = {"solid": solid, "hollow": hollow}
    # Split the hollow shell with the z = 0 plane into two watertight halves.
    # max(field, -z) keeps z >= 0; max(field, z) keeps z <= 0. Marching cubes
    # caps the flat z = 0 cross-section automatically, so each half is closed.
    fields["top"] = np.maximum(hollow, -Z)
    fields["bottom"] = np.maximum(hollow, Z)
    return lin, fields


# ---------------------------------------------------------------------------
# Mesh assembly + reporting
# ---------------------------------------------------------------------------
def field_to_trimesh(lin, field, geom: PrintGeometry):
    """Marching cubes → millimetre-scaled, outward-oriented trimesh."""
    import trimesh

    verts_algo, faces = _field_to_mesh(lin, field)
    verts_mm = verts_algo.astype(float) * geom.scale_mm
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _g(volume_mm3, density_kg_m3):
    """Volume [mm^3] × density [kg/m^3] → mass [g]."""
    return volume_mm3 * 1e-9 * density_kg_m3 * 1e3


def report(geom: PrintGeometry, info: "ScaleInfo", mesh, halves, *, mode: str):
    """Print human-readable dimensions, masses and the all-important CoM check."""
    solid = mode == "solid"
    vol = float(mesh.volume)
    m_plastic = _g(vol, PLA_DENSITY)
    centroid = np.asarray(mesh.center_mass, dtype=float)
    bb = mesh.bounding_box.extents

    print("\n" + "=" * 60)
    print("  Printable trajectoid — geometry & physics")
    print("=" * 60)
    print(f"  Print size scale     : {geom.size_scale:g}×")
    print("  Dimensions [mm]")
    if not solid:
        print(f"    steel ball Ø      : {geom.ball_mm:7.3f}   ({geom.nearest_ball_label()})")
        print(f"    cavity Ø (pocket) : {2 * geom.cavity_r_mm:7.3f}   (ball + {geom.clearance_mm:g} clearance)")
        print(f"    inscribed core Ø  : {2 * geom.core_r_mm:7.3f}")
        print(f"    min wall (@ cut)  : {geom.wall_mm:7.3f}")
    print(f"    outer sphere Ø    : {2 * geom.outer_r_mm:7.3f}")
    print(f"    bounding box      : {bb[0]:.2f} × {bb[1]:.2f} × {bb[2]:.2f}")

    print("  Mass / centre of mass")
    if solid:
        com_off = float(np.linalg.norm(centroid))
        print(f"    solid (PLA)       : {m_plastic:7.2f} g   (volume {vol / 1000:.2f} cm³)")
        print(f"    centroid offset   : {com_off:7.3f} mm  "
              f"({100 * com_off / geom.outer_r_mm:.2f}% of outer radius)")
        print("    note: a SOLID trajectoid is not a uniform sphere, so its centre")
        print("          of mass is off-centre and it will not roll true on its own.")
        print("          (That offset is exactly why the ball version exists.)")
    else:
        ball_vol = 4.0 / 3.0 * np.pi * (geom.ball_mm / 2.0) ** 3
        m_ball = _g(ball_vol, STEEL_DENSITY)
        com = (m_plastic * centroid) / (m_plastic + m_ball)  # ball is centred at origin
        com_off = float(np.linalg.norm(com))
        print(f"    shell (PLA)       : {m_plastic:7.2f} g   (volume {vol / 1000:.2f} cm³)")
        print(f"    ball  (steel)     : {m_ball:7.2f} g")
        print(f"    ball : shell ratio: {m_ball / max(m_plastic, 1e-9):7.2f} ×")
        print(f"    CoM offset        : {com_off:7.3f} mm  "
              f"({100 * com_off / geom.outer_r_mm:.2f}% of outer radius)")
        if com_off > 0.05 * geom.outer_r_mm:
            print("    ⚠ CoM offset > 5% of radius — rolling accuracy may suffer.")
            print("      Use a larger/denser ball or a thinner wall (--wall-mm).")

    print("  Maths")
    print(f"    curve scale         : {info.scale:.6f}  ({info.period_label})")
    print(f"    period mismatch     : {np.degrees(info.mismatch_rad):.4f}°  (≈0 = exact)")
    print(f"    watertight          : {mesh.is_watertight}")
    for name, half in halves:
        print(f"    {name:<18}: watertight={half.is_watertight}, "
              f"volume={half.volume / 1000:.2f} cm³")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Optional preview (cross-section showing the ball pocket + 3D silhouette)
# ---------------------------------------------------------------------------
def save_preview(curve, hollow_mesh, geom: PrintGeometry, out_png: Path, *, solid: bool = False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as exc:  # pragma: no cover
        print(f"  (preview skipped: {exc})")
        return

    fig = plt.figure(figsize=(15, 5))

    # 1) input curve
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(curve[:, 0], curve[:, 1], "b-", lw=1.5)
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.set_title("Input path")

    # 2) vertical cross-section (y = 0) — shows wall thickness + the steel ball
    ax2 = fig.add_subplot(1, 3, 2)
    try:
        section = hollow_mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
        planar, _ = section.to_2D() if hasattr(section, "to_2D") else section.to_planar()
        for entity in planar.discrete:
            ax2.plot(entity[:, 0], entity[:, 1], "k-", lw=1.0)
    except Exception:
        pass
    if not solid:
        ball = plt.Circle((0, 0), geom.ball_mm / 2.0, color="#b0b0b0", zorder=3)
        ax2.add_patch(ball)
    ax2.plot(0, 0, "r+", ms=10)
    lim = geom.outer_r_mm * 1.1
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_title("Cross-section [mm]" if solid else "Cross-section (steel ball in grey) [mm]")

    # 3) 3D silhouette of the outer shell (drop the inner cavity faces so the
    #    form reads as a solid; downsample for a light-weight render)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    faces = hollow_mesh.faces
    centroid_r = np.linalg.norm(hollow_mesh.vertices[faces].mean(axis=1), axis=1)
    outer = faces[centroid_r > geom.cavity_r_mm + 0.5 * geom.wall_mm]
    if len(outer) == 0:
        outer = faces
    if len(outer) > 8000:
        outer = outer[np.linspace(0, len(outer) - 1, 8000).astype(int)]
    tris = hollow_mesh.vertices[outer]
    coll = Poly3DCollection(tris, alpha=0.9, facecolor="#f0a030", edgecolor="#a8701c", linewidths=0.05)
    ax3.add_collection3d(coll)
    v = hollow_mesh.vertices
    for setlim, lo, hi in (
        (ax3.set_xlim, v[:, 0].min(), v[:, 0].max()),
        (ax3.set_ylim, v[:, 1].min(), v[:, 1].max()),
        (ax3.set_zlim, v[:, 2].min(), v[:, 2].max()),
    ):
        setlim(lo, hi)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_title("Trajectoid shell")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  preview : {out_png}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate(args):
    curve, name, closed = load_curve(
        args.preset, args.path_file, args.periodic, args.periods, args.amp, args.open
    )
    geom = PrintGeometry(
        args.ball_mm, args.wall_mm, args.clearance_mm, args.shell_ratio, size_scale=args.scale
    )

    size_tag = "" if abs(args.scale - 1.0) < 1e-9 else f"_x{args.scale:g}"
    stem = f"{name}{size_tag}"

    print("=" * 60)
    print(f"  Trajectoid (printable)  —  '{name}', mode={args.mode}, "
          f"{'closed loop' if closed else 'open periodic'}, scale={args.scale:g}×")
    print("=" * 60)
    print(f"  resolution={args.resolution}, max_planes={args.max_planes}")

    normals, info, _ = build_normals(
        curve,
        closed=closed,
        smooth_passes=args.smooth_passes,
        resample_points=args.resample_points,
        max_planes=args.max_planes,
    )
    print("  building implicit field …")
    lin, fields = hollow_fields(normals, geom, args.resolution)

    out_dir = Path(__file__).parent / "output" / "printable" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[tuple[str, Path]] = []
    halves_for_report: list[tuple[str, object]] = []

    def _export(mesh, suffix):
        path = out_dir / f"{stem}_{suffix}.stl"
        mesh.export(path)
        exported.append((suffix, path))
        return path

    if args.mode == "solid":
        # "Filled" trajectoid — no cavity, no room for a ball.
        ref_mesh = field_to_trimesh(lin, fields["solid"], geom)
        _export(ref_mesh, "solid")

    elif args.mode == "inplace":
        # Single enclosed shell — pause the print mid-way and drop the ball in.
        ref_mesh = field_to_trimesh(lin, fields["hollow"], geom)
        _export(ref_mesh, "shell")

    else:  # split (default)
        import trimesh

        ref_mesh = field_to_trimesh(lin, fields["hollow"], geom)
        top = field_to_trimesh(lin, fields["top"], geom)
        bottom = field_to_trimesh(lin, fields["bottom"], geom)
        # Print-ready orientation: both halves flat cut-face on the bed.
        # The top half already rests on its z = 0 face; flip the bottom half
        # 180° about X so its flat face is down and the dome points up.
        bottom_print = bottom.copy()
        bottom_print.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        bottom_print.apply_translation([0, 0, -bottom_print.bounds[0][2]])

        _export(top, "half_top")
        _export(bottom_print, "half_bottom")
        _export(ref_mesh, "assembled_ref")
        halves_for_report = [("half_top", top), ("half_bottom", bottom)]

    report(geom, info, ref_mesh, halves_for_report, mode=args.mode)

    if not args.no_preview:
        save_preview(curve, ref_mesh, geom, out_dir / f"{stem}_preview.png",
                     solid=(args.mode == "solid"))

    print("\n  Files written:")
    for suffix, path in exported:
        print(f"    [{suffix}] {path}")

    if args.mode == "split":
        print("\n  Assembly:")
        print("    1. Print half_top and half_bottom (flat face on the bed).")
        print(f"    2. Drop a Ø{geom.ball_mm:g} mm steel ball into the bowl.")
        print("    3. Glue the two halves together at the flat faces.")
    elif args.mode == "inplace":
        print("\n  Assembly:")
        print(f"    Pause the print at the cavity mid-height (z ≈ 0), drop a "
              f"Ø{geom.ball_mm:g} mm steel ball in, resume.")
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a 3D-printable trajectoid STL with a central steel-ball pocket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("curve source")
    src.add_argument("--preset", default="trefoil",
                     help="preset closed curve (circle, heart, trefoil, figure_eight, …)")
    src.add_argument("--periodic", default=None, choices=sorted(PERIODIC_PATHS),
                     help="open periodic path instead of a closed preset (sinusoid, zigzag)")
    src.add_argument("--periods", type=int, default=1,
                     help="number of periods for --periodic")
    src.add_argument("--amp", type=float, default=0.35,
                     help="relative amplitude for --periodic")
    src.add_argument("--path-file", default=None,
                     help="custom curve: .npy / .csv / .txt with an Nx2 array")
    src.add_argument("--open", action="store_true",
                     help="treat --path-file as an open periodic path (default: closed)")

    geo = p.add_argument_group("physical geometry [mm]")
    geo.add_argument("--scale", type=float, default=1.0,
                     help="uniform print-size multiplier (e.g. 3 = triple size)")
    geo.add_argument("--ball-mm", type=float, default=25.4,
                     help="steel ball diameter (e.g. 12.7=1/2\", 19.05=3/4\", 25.4=1\")")
    geo.add_argument("--wall-mm", type=float, default=1.6,
                     help="minimum plastic wall thickness over the ball (at the deepest cut)")
    geo.add_argument("--clearance-mm", type=float, default=0.2,
                     help="radial clearance added to the ball pocket for a loose fit")
    geo.add_argument("--shell-ratio", type=float, default=1.25,
                     help="outer sphere radius / inscribed core radius (keep near 1.25)")

    msh = p.add_argument_group("meshing")
    msh.add_argument("--mode", choices=["split", "inplace", "solid"], default="split",
                     help="split=two halves, inplace=single enclosed shell, solid=no cavity")
    msh.add_argument("--resolution", type=int, default=140,
                     help="marching-cubes grid resolution (higher = finer, slower)")
    msh.add_argument("--max-planes", type=int, default=240,
                     help="number of contact-plane cuts (higher = sharper detail)")
    msh.add_argument("--smooth-passes", type=int, default=1)
    msh.add_argument("--resample-points", type=int, default=320)
    msh.add_argument("--no-preview", action="store_true", help="skip the preview PNG")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    generate(args)


if __name__ == "__main__":
    main()
