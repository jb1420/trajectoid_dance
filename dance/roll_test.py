"""Standalone rolling-only debug page.

This window isolates the rolling kinematics from the rest of the app so the
motion can be sanity-checked visually. It supports two body modes:

* **sphere** — a unit sphere (the analytic ground truth: a sphere rolling
  along a planar curve must stay tangent to z=0 with center height = r).
* **trajectoid** — a generated trajectoid mesh from a preset curve, using the
  same generation pipeline as the main app.

Tools:
* preset paths (line, circle, square, figure-8) and adjustable scale
* manual time scrub + play/pause
* visible contact-point marker, body-frame axes triad, full path on the floor
* live HUD: t, frame index, center xyz, contact point xyz, distance to floor

Run::

    python roll_test.py
"""
from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np

from viewer import HAS_GPU_VIEWER  # also runs cache-dir bootstrap

from PySide6 import QtCore, QtGui, QtWidgets

if HAS_GPU_VIEWER:
    import pyqtgraph.opengl as gl

from trajectoids_adapter import (
    GenerationResult,
    build_roll_simulation,
    generate_trajectoid_mesh,
    resample_uniform,
    smooth_path,
)


CORE_RADIUS = 1.0
N_FRAMES = 600


# ---------------------------------------------------------------------------
# Path presets — already in "math" coords, centered at origin where applicable.
# ---------------------------------------------------------------------------

def path_line(length: float = 4.0 * np.pi, n: int = 200) -> np.ndarray:
    xs = np.linspace(0.0, length, n)
    ys = np.zeros_like(xs)
    return np.column_stack([xs, ys])


def path_circle(radius: float = 1.5, n: int = 320) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def path_square(side: float = 3.0, n_per_edge: int = 60) -> np.ndarray:
    s = side
    edges = [
        np.column_stack([np.linspace(-s/2, s/2, n_per_edge), np.full(n_per_edge, -s/2)]),
        np.column_stack([np.full(n_per_edge, s/2), np.linspace(-s/2, s/2, n_per_edge)]),
        np.column_stack([np.linspace(s/2, -s/2, n_per_edge), np.full(n_per_edge, s/2)]),
        np.column_stack([np.full(n_per_edge, -s/2), np.linspace(s/2, -s/2, n_per_edge)]),
    ]
    pts = np.vstack(edges)
    pts = smooth_path(pts, passes=2, closed=True)
    return resample_uniform(pts, n_points=320, closed=True)


def path_figure_eight(scale: float = 1.5, n: int = 320) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([scale * np.sin(t), scale * np.sin(t) * np.cos(t)])


PATH_PRESETS = {
    "line": ("Line (open)", False, path_line),
    "circle": ("Circle (closed)", True, path_circle),
    "square": ("Square (closed)", True, path_square),
    "figure_eight": ("Figure 8 (closed)", True, path_figure_eight),
}


# ---------------------------------------------------------------------------
# Body builders
# ---------------------------------------------------------------------------

def build_sphere_mesh(radius: float = CORE_RADIUS, lat_steps: int = 24, lon_steps: int = 36) -> Tuple[np.ndarray, np.ndarray]:
    """Build a UV sphere centered at origin in the body frame."""
    lats = np.linspace(0.0, np.pi, lat_steps)
    lons = np.linspace(0.0, 2.0 * np.pi, lon_steps, endpoint=False)
    verts = []
    for lat in lats:
        for lon in lons:
            x = radius * np.sin(lat) * np.cos(lon)
            y = radius * np.sin(lat) * np.sin(lon)
            z = radius * np.cos(lat)
            verts.append([x, y, z])
    verts = np.asarray(verts, dtype=np.float32)

    faces = []
    for i in range(lat_steps - 1):
        for j in range(lon_steps):
            j1 = (j + 1) % lon_steps
            a = i * lon_steps + j
            b = i * lon_steps + j1
            c = (i + 1) * lon_steps + j
            d = (i + 1) * lon_steps + j1
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces = np.asarray(faces, dtype=np.int32)
    return verts, faces


def build_trajectoid_from_preset(curve_xy: np.ndarray, closed: bool) -> Optional[GenerationResult]:
    if not closed:
        return None
    pts = np.asarray(curve_xy, dtype=float)
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    pts = smooth_path(pts, passes=1, closed=True)
    pts = resample_uniform(pts, n_points=320, closed=True)
    return generate_trajectoid_mesh(
        pts,
        require_closed=True,
        smooth_passes=0,
        resample_points=320,
        resolution=80,
        core_radius=CORE_RADIUS,
    )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class RollTestWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trajectoid Roll Debug")
        self.resize(1200, 780)

        if not HAS_GPU_VIEWER:
            QtWidgets.QMessageBox.critical(
                self, "GPU viewer required",
                "pyqtgraph.opengl is not available. Install pyqtgraph + PyOpenGL to run this debug page.",
            )
            raise SystemExit(1)

        # Simulation state
        self._sim_translations: Optional[np.ndarray] = None  # (T, 3)
        self._sim_rotations: Optional[np.ndarray] = None     # (T, 3, 3)
        self._traj_xy: Optional[np.ndarray] = None           # (T, 2)
        self._body_verts: Optional[np.ndarray] = None        # (V, 3) base frame
        self._body_faces: Optional[np.ndarray] = None
        self._frame_count = 0

        # GL items we own and rebuild on each setup
        self._mesh_item = None
        self._path_item = None
        self._contact_item = None
        self._axis_items: list = []
        self._grid_items: list = []

        # ---- Layout ----
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # Left: controls
        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(320)
        cl = QtWidgets.QVBoxLayout(controls)
        cl.setContentsMargins(4, 4, 4, 4)

        cl.addWidget(QtWidgets.QLabel("<b>Body</b>"))
        self._body_combo = QtWidgets.QComboBox()
        self._body_combo.addItem("Sphere (ground truth)", userData="sphere")
        self._body_combo.addItem("Trajectoid (from path)", userData="trajectoid")
        cl.addWidget(self._body_combo)

        cl.addWidget(QtWidgets.QLabel("<b>Path</b>"))
        self._path_combo = QtWidgets.QComboBox()
        for key, (label, _closed, _fn) in PATH_PRESETS.items():
            self._path_combo.addItem(label, userData=key)
        cl.addWidget(self._path_combo)

        form = QtWidgets.QFormLayout()
        self._scale_spin = QtWidgets.QDoubleSpinBox()
        self._scale_spin.setRange(0.1, 10.0)
        self._scale_spin.setSingleStep(0.1)
        self._scale_spin.setValue(1.0)
        form.addRow("Path scale:", self._scale_spin)
        self._target_revs_spin = QtWidgets.QDoubleSpinBox()
        self._target_revs_spin.setRange(0.25, 8.0)
        self._target_revs_spin.setSingleStep(0.25)
        self._target_revs_spin.setValue(1.0)
        form.addRow("Cycles to roll:", self._target_revs_spin)
        cl.addLayout(form)

        self._build_btn = QtWidgets.QPushButton("Build Simulation")
        self._build_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._build_btn.clicked.connect(self._on_build)
        cl.addWidget(self._build_btn)

        cl.addWidget(QtWidgets.QLabel("<b>Playback</b>"))
        scrub_row = QtWidgets.QHBoxLayout()
        self._scrub = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._scrub.setRange(0, 1000)
        self._scrub.valueChanged.connect(self._on_scrub)
        scrub_row.addWidget(self._scrub)
        cl.addLayout(scrub_row)

        play_row = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._on_play_toggle)
        self._play_btn.setEnabled(False)
        play_row.addWidget(self._play_btn)
        self._reset_btn = QtWidgets.QPushButton("⟲ Reset")
        self._reset_btn.clicked.connect(lambda: self._scrub.setValue(0))
        play_row.addWidget(self._reset_btn)
        cl.addLayout(play_row)

        speed_form = QtWidgets.QFormLayout()
        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(1.0, 60.0)
        self._duration_spin.setValue(8.0)
        self._duration_spin.setSuffix(" s")
        speed_form.addRow("Duration:", self._duration_spin)
        cl.addLayout(speed_form)

        cl.addWidget(QtWidgets.QLabel("<b>Visibility</b>"))
        self._show_path = QtWidgets.QCheckBox("Path on floor")
        self._show_path.setChecked(True)
        self._show_path.toggled.connect(lambda v: self._path_item and self._path_item.setVisible(v))
        cl.addWidget(self._show_path)
        self._show_contact = QtWidgets.QCheckBox("Contact point")
        self._show_contact.setChecked(True)
        self._show_contact.toggled.connect(lambda v: self._contact_item and self._contact_item.setVisible(v))
        cl.addWidget(self._show_contact)
        self._show_axes = QtWidgets.QCheckBox("Body axes")
        self._show_axes.setChecked(True)
        self._show_axes.toggled.connect(self._set_axes_visible)
        cl.addWidget(self._show_axes)
        self._wire_check = QtWidgets.QCheckBox("Wireframe")
        self._wire_check.toggled.connect(self._set_wireframe)
        cl.addWidget(self._wire_check)

        # HUD
        cl.addWidget(QtWidgets.QLabel("<b>Live</b>"))
        self._hud = QtWidgets.QPlainTextEdit()
        self._hud.setReadOnly(True)
        self._hud.setMaximumHeight(180)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self._hud.setFont(font)
        cl.addWidget(self._hud)

        cl.addStretch(1)

        # Right: GL view
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor(QtGui.QColor("#f4f6f9"))
        self._view.setCameraPosition(distance=10.0, elevation=28.0, azimuth=-55.0)

        root.addWidget(controls)
        root.addWidget(self._view, stretch=1)
        self.setCentralWidget(central)

        # Playback timer
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._t01 = 0.0

        self._draw_grid(radius=4.0)

        self.statusBar().showMessage("Pick a body + path, then Build Simulation.")

    # -- grid / axes helpers ------------------------------------------------

    def _draw_grid(self, radius: float, center: Tuple[float, float] = (0.0, 0.0)) -> None:
        for it in self._grid_items:
            try:
                self._view.removeItem(it)
            except Exception:
                pass
        self._grid_items = []
        ticks = np.linspace(-radius, radius, 13, dtype=np.float32)
        cx, cy = center
        for t in ticks:
            x_line = np.array(
                [[cx - radius, cy + float(t), 0.0], [cx + radius, cy + float(t), 0.0]],
                dtype=np.float32,
            )
            y_line = np.array(
                [[cx + float(t), cy - radius, 0.0], [cx + float(t), cy + radius, 0.0]],
                dtype=np.float32,
            )
            for pos in (x_line, y_line):
                it = gl.GLLinePlotItem(
                    pos=pos,
                    color=(0.76, 0.81, 0.86, 0.95),
                    width=1.2,
                    antialias=True,
                    mode="line_strip",
                )
                it.setGLOptions("opaque")
                self._view.addItem(it)
                self._grid_items.append(it)

        # World axes at origin: X red, Y green, Z blue
        L = 1.0
        for vec, color in (
            (np.array([[0, 0, 0], [L, 0, 0]]), (1.0, 0.2, 0.2, 1.0)),
            (np.array([[0, 0, 0], [0, L, 0]]), (0.2, 0.8, 0.2, 1.0)),
            (np.array([[0, 0, 0], [0, 0, L]]), (0.2, 0.4, 1.0, 1.0)),
        ):
            it = gl.GLLinePlotItem(pos=vec.astype(np.float32), color=color, width=2.0, antialias=True)
            self._view.addItem(it)
            self._grid_items.append(it)

    def _set_axes_visible(self, v: bool) -> None:
        for it in self._axis_items:
            it.setVisible(v)

    def _set_wireframe(self, v: bool) -> None:
        if self._mesh_item is None:
            return
        self._mesh_item.opts["drawEdges"] = bool(v)
        self._mesh_item.update()

    # -- build simulation ---------------------------------------------------

    def _on_build(self) -> None:
        body_kind = self._body_combo.currentData()
        path_key = self._path_combo.currentData()
        path_label, closed, fn = PATH_PRESETS[path_key]
        scale = float(self._scale_spin.value())
        target_cycles = float(self._target_revs_spin.value())

        try:
            curve = fn() * scale
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Path error", str(e))
            return

        # Body
        try:
            if body_kind == "sphere":
                verts, faces = build_sphere_mesh(radius=CORE_RADIUS)
            else:
                if not closed:
                    QtWidgets.QMessageBox.warning(
                        self, "Trajectoid build",
                        "Trajectoid generation requires a closed path. "
                        "Pick a closed preset (Circle / Square / Figure 8) or use the sphere body.",
                    )
                    return
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                self.statusBar().showMessage("Generating trajectoid mesh…")
                gen = build_trajectoid_from_preset(curve, closed)
                QtWidgets.QApplication.restoreOverrideCursor()
                if gen is None:
                    return
                verts = gen.vertices.astype(np.float32)
                faces = gen.faces.astype(np.int32)
                # Use the resampled, scaled curve actually used during generation,
                # so the body's contact geometry matches the rolling path exactly.
                curve = gen.resampled_points
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Body build failed", f"{e}")
            return

        # Roll simulation
        seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        if closed:
            seg = np.append(seg, np.linalg.norm(curve[0] - curve[-1]))
        cycle_len = float(np.sum(seg))
        target_arc = (cycle_len / CORE_RADIUS) * target_cycles * CORE_RADIUS  # = cycle_len * cycles
        try:
            sim = build_roll_simulation(
                curve,
                target_roll_angle_rad=target_arc / CORE_RADIUS,
                closed=closed,
                n_frames=N_FRAMES,
                core_radius=CORE_RADIUS,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Roll sim failed", f"{e}")
            return

        self._sim_translations = sim.translations_xyz
        self._sim_rotations = sim.rotations
        self._traj_xy = sim.trajectory_xy
        self._body_verts = verts
        self._body_faces = faces
        self._frame_count = int(sim.translations_xyz.shape[0])

        self._rebuild_scene()
        self._t01 = 0.0
        self._scrub.blockSignals(True)
        self._scrub.setValue(0)
        self._scrub.blockSignals(False)
        self._draw_frame(0.0)
        self._play_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Built sim: body={body_kind}, path={path_label}, frames={self._frame_count}, "
            f"cycle_len={cycle_len:.3f}, achieved_roll={sim.achieved_roll_angle_rad:.3f} rad"
        )

    def _rebuild_scene(self) -> None:
        # Tear down previous items
        for it in (self._mesh_item, self._path_item, self._contact_item):
            if it is not None:
                try:
                    self._view.removeItem(it)
                except Exception:
                    pass
        for it in self._axis_items:
            try:
                self._view.removeItem(it)
            except Exception:
                pass
        self._mesh_item = None
        self._path_item = None
        self._contact_item = None
        self._axis_items = []

        if self._traj_xy is None or self._body_verts is None or self._body_faces is None:
            return

        # Path on floor (full traveled curve)
        traj = self._traj_xy
        pts = np.column_stack([traj[:, 0], traj[:, 1], np.full(traj.shape[0], 0.005)]).astype(np.float32)
        self._path_item = gl.GLLinePlotItem(
            pos=pts, color=(0.85, 0.30, 0.30, 0.95), width=2.6, antialias=True, mode="line_strip"
        )
        self._path_item.setGLOptions("translucent")
        self._view.addItem(self._path_item)
        self._path_item.setVisible(self._show_path.isChecked())

        # Body mesh
        rot0 = self._sim_rotations[0]
        trans0 = self._sim_translations[0]
        v0 = (self._body_verts @ rot0.T) + trans0
        self._mesh_item = gl.GLMeshItem(
            vertexes=v0.astype(np.float32),
            faces=self._body_faces,
            smooth=True,
            drawEdges=self._wire_check.isChecked(),
            drawFaces=True,
            edgeColor=(0.15, 0.18, 0.22, 0.85),
            color=(0.30, 0.55, 0.95, 1.0),
            shader="shaded",
            computeNormals=True,
        )
        self._mesh_item.setGLOptions("opaque")
        self._view.addItem(self._mesh_item)

        # Contact-point marker — small magenta sphere at floor below body center
        contact_v, contact_f = build_sphere_mesh(radius=0.05, lat_steps=8, lon_steps=12)
        self._contact_item = gl.GLMeshItem(
            vertexes=contact_v + np.array([trans0[0], trans0[1], 0.0], dtype=np.float32),
            faces=contact_f,
            color=(1.0, 0.1, 0.6, 1.0),
            smooth=True,
            shader="shaded",
        )
        self._contact_item.setGLOptions("opaque")
        self._view.addItem(self._contact_item)
        self._contact_item.setVisible(self._show_contact.isChecked())

        # Body-frame axes (drawn each frame)
        for color in ((1.0, 0.2, 0.2, 1.0), (0.2, 0.8, 0.2, 1.0), (0.2, 0.4, 1.0, 1.0)):
            it = gl.GLLinePlotItem(pos=np.zeros((2, 3), dtype=np.float32), color=color, width=2.4, antialias=True)
            it.setVisible(self._show_axes.isChecked())
            self._view.addItem(it)
            self._axis_items.append(it)

        # Recenter camera
        mins = np.min(traj, axis=0)
        maxs = np.max(traj, axis=0)
        center = 0.5 * (mins + maxs)
        radius = float(max(np.max(maxs - mins) * 0.6, 2.0))
        self._draw_grid(radius=max(radius * 1.4, 4.0), center=(float(center[0]), float(center[1])))
        self._view.setCameraPosition(
            pos=QtGui.QVector3D(float(center[0]), float(center[1]), 0.0),
            distance=max(radius * 3.0, 6.0),
            elevation=30.0,
            azimuth=-55.0,
        )

    # -- playback -----------------------------------------------------------

    def _on_play_toggle(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._play_btn.setText("▶ Play")
        else:
            if self._frame_count <= 0:
                return
            if self._t01 >= 1.0:
                self._t01 = 0.0
                self._scrub.setValue(0)
            self._timer.start()
            self._play_btn.setText("❚❚ Pause")

    def _on_tick(self) -> None:
        dur = max(0.5, float(self._duration_spin.value()))
        self._t01 += (self._timer.interval() / 1000.0) / dur
        if self._t01 >= 1.0:
            self._t01 = 1.0
            self._timer.stop()
            self._play_btn.setText("▶ Play")
        self._scrub.blockSignals(True)
        self._scrub.setValue(int(round(self._t01 * 1000)))
        self._scrub.blockSignals(False)
        self._draw_frame(self._t01)

    def _on_scrub(self, value: int) -> None:
        self._t01 = value / 1000.0
        self._draw_frame(self._t01)

    def _draw_frame(self, t01: float) -> None:
        if self._frame_count <= 0 or self._mesh_item is None:
            return
        f = int(round(t01 * (self._frame_count - 1)))
        f = max(0, min(self._frame_count - 1, f))
        rot = self._sim_rotations[f]
        trans = self._sim_translations[f]
        verts = (self._body_verts @ rot.T) + trans
        self._mesh_item.setMeshData(
            vertexes=verts.astype(np.float32),
            faces=self._body_faces,
            color=(0.30, 0.55, 0.95, 1.0),
            smooth=True,
            drawEdges=self._wire_check.isChecked(),
            edgeColor=(0.15, 0.18, 0.22, 0.85),
        )

        # Contact point: world point directly below center (since body must touch z=0).
        if self._contact_item is not None:
            base_v, base_f = build_sphere_mesh(radius=0.05, lat_steps=8, lon_steps=12)
            self._contact_item.setMeshData(
                vertexes=(base_v + np.array([trans[0], trans[1], 0.0], dtype=np.float32)).astype(np.float32),
                faces=base_f,
                color=(1.0, 0.1, 0.6, 1.0),
                smooth=True,
            )

        # Body-frame axes (rotated, anchored at center)
        L = 0.9
        anchor = trans
        body_x = anchor + rot @ np.array([L, 0, 0])
        body_y = anchor + rot @ np.array([0, L, 0])
        body_z = anchor + rot @ np.array([0, 0, L])
        for it, end in zip(self._axis_items, (body_x, body_y, body_z)):
            line = np.array([anchor, end], dtype=np.float32)
            it.setData(pos=line)

        # HUD
        z_below = float(np.min(verts[:, 2]))
        text = (
            f"t        : {t01:.3f}\n"
            f"frame    : {f} / {self._frame_count - 1}\n"
            f"center   : ({trans[0]:+.3f}, {trans[1]:+.3f}, {trans[2]:+.3f})\n"
            f"min vert z: {z_below:+.4f}   (should be ≈ 0 for proper rolling)\n"
            f"|center-z - r|: {abs(trans[2] - CORE_RADIUS):.4e}\n"
        )
        self._hud.setPlainText(text)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RollTestWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
