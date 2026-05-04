"""Multi-trajectoid 3D viewer.

Holds N independent dancer meshes on a shared ground plane and animates them
together off a single global timer. Picks the GPU (pyqtgraph.opengl) backend
when available and falls back to a Matplotlib viewer otherwise.

Frame transform formula matches trajectoid-main/app.py:1468-1470 :
    vertices_world = (vertices_base @ rot.T) + translation
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Match the trajectoid-main bootstrap so cache dirs exist under user home.
import tempfile
from pathlib import Path


def _ensure_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


for var, suffix in (("XDG_CACHE_HOME", ".cache"), ("MPLCONFIGDIR", ".cache/dance-mpl")):
    if var not in os.environ:
        target = Path.home() / suffix
        if not _ensure_dir(target):
            target = Path(tempfile.gettempdir()) / suffix.replace("/", "-")
            target.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(target)

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except Exception:
    pg = None
    gl = None
    HAS_GPU_VIEWER = False
else:
    pg.setConfigOptions(antialias=True)
    HAS_GPU_VIEWER = True

from dancer import Dancer, normalize_sim


GROUND_GRID_COLOR = (0.76, 0.81, 0.86, 0.95)
GROUND_GRID_WIDTH = 1.4
GROUND_GRID_TICKS = 13
TRAJECTORY_ALPHA = 0.85


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return 0.5, 0.5, 0.5, float(alpha)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return r, g, b, float(alpha)


def _make_animation_mesh(
    vertices: np.ndarray, faces: np.ndarray, max_faces: int = 1800
) -> Tuple[np.ndarray, np.ndarray]:
    if faces.shape[0] <= max_faces:
        return vertices.astype(np.float32), faces.astype(np.int32)
    stride = max(1, int(np.ceil(faces.shape[0] / float(max_faces))))
    faces_sub = faces[::stride]
    unique_indices, inverse = np.unique(faces_sub.reshape(-1), return_inverse=True)
    vertices_sub = vertices[unique_indices]
    return vertices_sub.astype(np.float32), inverse.reshape(-1, 3).astype(np.int32)


class _DancerState:
    __slots__ = (
        "vertices", "faces", "lod_vertices", "lod_faces",
        "translations", "rotations", "trajectory_xy",
        "start_offset", "color_rgba", "phase", "speed", "frame_count",
        "mesh_face_item", "trajectory_item",
    )

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        lod_vertices: np.ndarray,
        lod_faces: np.ndarray,
        translations: np.ndarray,
        rotations: np.ndarray,
        trajectory_xy: np.ndarray,
        start_offset: np.ndarray,
        color_rgba: Tuple[float, float, float, float],
        phase: float,
        speed: float,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.lod_vertices = lod_vertices
        self.lod_faces = lod_faces
        self.translations = translations
        self.rotations = rotations
        self.trajectory_xy = trajectory_xy
        self.start_offset = start_offset
        self.color_rgba = color_rgba
        self.phase = float(phase)
        self.speed = float(speed)
        self.frame_count = int(translations.shape[0])
        self.mesh_face_item = None
        self.trajectory_item = None


# ---------------------------------------------------------------------------
# GPU viewer
# ---------------------------------------------------------------------------

class _MultiTrajectoidGLViewer(QtWidgets.QWidget):
    playFinished = QtCore.Signal()
    backend_name = "PyQtGraph OpenGL (GPU)"

    GLOBAL_TICKS = 480
    TIMER_INTERVAL_MS = 33

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._dancer_state: Dict[str, _DancerState] = {}
        self._grid_items: List = []
        self._wireframe = False
        self._show_trajectories = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = gl.GLViewWidget(self)
        self._view.setBackgroundColor(QtGui.QColor("#f4f6f9"))
        layout.addWidget(self._view, stretch=1)

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self.TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._global_t = 0.0
        self._duration_s = 12.0
        self._loop = False
        self._is_playing = False

        self._update_grid(radius=4.0)
        self._reset_camera_default()

    # -- camera helpers ------------------------------------------------------

    def _reset_camera_default(self) -> None:
        self._view.setCameraPosition(
            pos=QtGui.QVector3D(0.0, 0.0, 0.0),
            distance=12.0,
            elevation=35.0,
            azimuth=-60.0,
        )

    def _fit_camera_to_scene(self) -> None:
        if not self._dancer_state:
            self._reset_camera_default()
            return
        all_xy: List[np.ndarray] = []
        for st in self._dancer_state.values():
            xy = st.trajectory_xy + st.start_offset[:2]
            all_xy.append(xy)
        stacked = np.vstack(all_xy)
        mins = np.min(stacked, axis=0)
        maxs = np.max(stacked, axis=0)
        center_xy = 0.5 * (mins + maxs)
        radius = float(max(np.max(maxs - mins) * 0.6, 2.0))
        self._update_grid(radius=max(radius * 1.4, 4.0), center_xy=center_xy)
        distance = max(radius * 3.2, 6.0)
        self._view.setCameraPosition(
            pos=QtGui.QVector3D(float(center_xy[0]), float(center_xy[1]), 0.0),
            distance=distance,
            elevation=45.0,
            azimuth=-55.0,
        )

    def _update_grid(self, radius: float, center_xy: Optional[np.ndarray] = None) -> None:
        for item in self._grid_items:
            try:
                self._view.removeItem(item)
            except Exception:
                pass
        self._grid_items = []
        cx, cy = (0.0, 0.0) if center_xy is None else (float(center_xy[0]), float(center_xy[1]))
        ticks = np.linspace(-radius, radius, GROUND_GRID_TICKS, dtype=np.float32)
        z0 = np.float32(0.0)
        for t in ticks:
            line_x = np.array(
                [[cx - radius, cy + float(t), z0], [cx + radius, cy + float(t), z0]],
                dtype=np.float32,
            )
            line_y = np.array(
                [[cx + float(t), cy - radius, z0], [cx + float(t), cy + radius, z0]],
                dtype=np.float32,
            )
            for pos in (line_x, line_y):
                item = gl.GLLinePlotItem(
                    pos=pos,
                    color=GROUND_GRID_COLOR,
                    width=GROUND_GRID_WIDTH,
                    antialias=True,
                    mode="line_strip",
                )
                item.setGLOptions("opaque")
                self._view.addItem(item)
                self._grid_items.append(item)

    # -- dancer management ---------------------------------------------------

    def add_or_update_dancer(self, dancer: Dancer) -> None:
        if dancer.gen_result is None or dancer.sim_result is None:
            # Nothing to render yet; remove any prior state.
            self.remove_dancer(dancer.dancer_id)
            return
        # Tear down any existing items for this dancer first.
        self._tear_down_dancer(dancer.dancer_id)

        gen = dancer.gen_result
        sim = dancer.sim_result
        normalized = normalize_sim(sim, self.GLOBAL_TICKS)

        lod_v, lod_f = _make_animation_mesh(gen.vertices, gen.faces, max_faces=1500)
        color_rgba = _hex_to_rgba(dancer.color_hex, alpha=1.0)
        start_offset = np.array(
            [float(dancer.start_offset_xy[0]), float(dancer.start_offset_xy[1]), 0.0],
            dtype=float,
        )

        st = _DancerState(
            vertices=gen.vertices.astype(np.float32),
            faces=gen.faces.astype(np.int32),
            lod_vertices=lod_v,
            lod_faces=lod_f,
            translations=normalized.translations,
            rotations=normalized.rotations,
            trajectory_xy=normalized.trajectory_xy,
            start_offset=start_offset,
            color_rgba=color_rgba,
            phase=dancer.phase_offset,
            speed=dancer.speed_multiplier,
        )

        # Initial mesh at frame 0.
        rot0 = st.rotations[0]
        trans0 = st.translations[0] + st.start_offset
        verts0 = (st.lod_vertices @ rot0.T) + trans0
        st.mesh_face_item = gl.GLMeshItem(
            vertexes=verts0.astype(np.float32),
            faces=st.lod_faces,
            smooth=True,
            drawEdges=self._wireframe,
            drawFaces=True,
            edgeColor=(0.15, 0.18, 0.22, 0.85),
            color=color_rgba,
            shader="shaded",
            computeNormals=True,
        )
        st.mesh_face_item.setGLOptions("opaque")
        self._view.addItem(st.mesh_face_item)

        # Trajectory line on the ground plane (full curve, slightly above z=0).
        if self._show_trajectories:
            traj_pts = np.column_stack(
                [
                    st.trajectory_xy[:, 0] + start_offset[0],
                    st.trajectory_xy[:, 1] + start_offset[1],
                    np.full(st.trajectory_xy.shape[0], 0.005, dtype=float),
                ]
            ).astype(np.float32)
            st.trajectory_item = gl.GLLinePlotItem(
                pos=traj_pts,
                color=(color_rgba[0], color_rgba[1], color_rgba[2], TRAJECTORY_ALPHA),
                width=2.2,
                antialias=True,
                mode="line_strip",
            )
            st.trajectory_item.setGLOptions("translucent")
            self._view.addItem(st.trajectory_item)

        self._dancer_state[dancer.dancer_id] = st
        if not self._is_playing:
            self._fit_camera_to_scene()

    def remove_dancer(self, dancer_id: str) -> None:
        self._tear_down_dancer(dancer_id)
        if not self._is_playing:
            self._fit_camera_to_scene()

    def _tear_down_dancer(self, dancer_id: str) -> None:
        st = self._dancer_state.pop(dancer_id, None)
        if st is None:
            return
        for item in (st.mesh_face_item, st.trajectory_item):
            if item is None:
                continue
            try:
                self._view.removeItem(item)
            except Exception:
                pass

    def clear_dancers(self) -> None:
        for did in list(self._dancer_state.keys()):
            self._tear_down_dancer(did)
        if not self._is_playing:
            self._fit_camera_to_scene()

    # -- playback ------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def start_play(self, duration_seconds: float, loop: bool = False) -> bool:
        if not self._dancer_state:
            return False
        self._duration_s = max(0.5, float(duration_seconds))
        self._loop = bool(loop)
        self._global_t = 0.0
        self._fit_camera_to_scene()
        self._draw_frame(0.0)
        self._is_playing = True
        self._timer.start()
        return True

    def stop_play(self) -> None:
        if not self._is_playing:
            return
        self._timer.stop()
        self._is_playing = False
        self.playFinished.emit()

    def reset_view(self) -> None:
        self._fit_camera_to_scene()

    def set_wireframe(self, enabled: bool) -> None:
        self._wireframe = bool(enabled)
        for st in self._dancer_state.values():
            if st.mesh_face_item is not None:
                st.mesh_face_item.opts["drawEdges"] = self._wireframe
                st.mesh_face_item.update()

    def _on_tick(self) -> None:
        dt = self.TIMER_INTERVAL_MS / 1000.0
        self._global_t += dt / max(self._duration_s, 1e-3)
        if self._global_t >= 1.0:
            if self._loop:
                self._global_t = self._global_t % 1.0
            else:
                self._global_t = 1.0
                self._draw_frame(self._global_t)
                self._timer.stop()
                self._is_playing = False
                self.playFinished.emit()
                return
        self._draw_frame(self._global_t)

    def _draw_frame(self, global_t: float) -> None:
        for st in self._dancer_state.values():
            local_t = ((global_t * st.speed) + st.phase) % 1.0
            f = int(local_t * (st.frame_count - 1))
            f = max(0, min(st.frame_count - 1, f))
            rot = st.rotations[f]
            trans = st.translations[f] + st.start_offset
            verts = (st.lod_vertices @ rot.T) + trans
            if st.mesh_face_item is not None:
                st.mesh_face_item.setMeshData(
                    vertexes=verts.astype(np.float32),
                    faces=st.lod_faces,
                    color=st.color_rgba,
                    smooth=True,
                    drawEdges=self._wireframe,
                    edgeColor=(0.15, 0.18, 0.22, 0.85),
                )


# ---------------------------------------------------------------------------
# Matplotlib fallback (functional but slower)
# ---------------------------------------------------------------------------

class _MultiTrajectoidMplViewer(QtWidgets.QWidget):
    playFinished = QtCore.Signal()
    backend_name = "Matplotlib (CPU fallback)"

    GLOBAL_TICKS = 240
    TIMER_INTERVAL_MS = 50

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._dancer_state: Dict[str, _DancerState] = {}
        self._wireframe = False
        self._show_trajectories = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(6.0, 5.0), facecolor="#f4f6f9")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111, projection="3d")
        self._ax.set_facecolor("#f4f6f9")
        layout.addWidget(self._canvas, stretch=1)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._global_t = 0.0
        self._duration_s = 12.0
        self._loop = False
        self._is_playing = False
        self._render_static()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def add_or_update_dancer(self, dancer: Dancer) -> None:
        if dancer.gen_result is None or dancer.sim_result is None:
            self.remove_dancer(dancer.dancer_id)
            return
        gen = dancer.gen_result
        sim = dancer.sim_result
        normalized = normalize_sim(sim, self.GLOBAL_TICKS)
        lod_v, lod_f = _make_animation_mesh(gen.vertices, gen.faces, max_faces=520)
        color_rgba = _hex_to_rgba(dancer.color_hex, alpha=1.0)
        start_offset = np.array(
            [float(dancer.start_offset_xy[0]), float(dancer.start_offset_xy[1]), 0.0],
            dtype=float,
        )
        st = _DancerState(
            vertices=gen.vertices.astype(np.float32),
            faces=gen.faces.astype(np.int32),
            lod_vertices=lod_v,
            lod_faces=lod_f,
            translations=normalized.translations,
            rotations=normalized.rotations,
            trajectory_xy=normalized.trajectory_xy,
            start_offset=start_offset,
            color_rgba=color_rgba,
            phase=dancer.phase_offset,
            speed=dancer.speed_multiplier,
        )
        self._dancer_state[dancer.dancer_id] = st
        if not self._is_playing:
            self._render_static()

    def remove_dancer(self, dancer_id: str) -> None:
        self._dancer_state.pop(dancer_id, None)
        if not self._is_playing:
            self._render_static()

    def clear_dancers(self) -> None:
        self._dancer_state.clear()
        if not self._is_playing:
            self._render_static()

    def start_play(self, duration_seconds: float, loop: bool = False) -> bool:
        if not self._dancer_state:
            return False
        self._duration_s = max(0.5, float(duration_seconds))
        self._loop = bool(loop)
        self._global_t = 0.0
        self._is_playing = True
        self._timer.start()
        return True

    def stop_play(self) -> None:
        if not self._is_playing:
            return
        self._timer.stop()
        self._is_playing = False
        self.playFinished.emit()

    def reset_view(self) -> None:
        self._render_static()

    def set_wireframe(self, enabled: bool) -> None:
        self._wireframe = bool(enabled)
        if not self._is_playing:
            self._render_static()

    def _on_tick(self) -> None:
        dt = self.TIMER_INTERVAL_MS / 1000.0
        self._global_t += dt / max(self._duration_s, 1e-3)
        if self._global_t >= 1.0:
            if self._loop:
                self._global_t = self._global_t % 1.0
            else:
                self._global_t = 1.0
                self._render_animated()
                self._timer.stop()
                self._is_playing = False
                self.playFinished.emit()
                return
        self._render_animated()

    def _scene_bounds(self) -> Tuple[np.ndarray, float]:
        if not self._dancer_state:
            return np.zeros(3), 4.0
        all_xy: List[np.ndarray] = []
        for st in self._dancer_state.values():
            all_xy.append(st.trajectory_xy + st.start_offset[:2])
        stacked = np.vstack(all_xy)
        mins = np.min(stacked, axis=0)
        maxs = np.max(stacked, axis=0)
        center = np.array([0.5 * (mins[0] + maxs[0]), 0.5 * (mins[1] + maxs[1]), 0.5], dtype=float)
        radius = float(max(np.max(maxs - mins) * 0.7, 2.0))
        return center, radius

    def _set_axes(self) -> None:
        center, radius = self._scene_bounds()
        self._ax.set_xlim(center[0] - radius, center[0] + radius)
        self._ax.set_ylim(center[1] - radius, center[1] + radius)
        self._ax.set_zlim(0.0, 2.0 * radius * 0.5)
        self._ax.view_init(elev=35, azim=-55)
        self._ax.set_box_aspect((1, 1, 0.4))
        self._ax.grid(True)

    def _render_static(self) -> None:
        self._ax.clear()
        self._set_axes()
        for st in self._dancer_state.values():
            rot0 = st.rotations[0]
            trans0 = st.translations[0] + st.start_offset
            verts = (st.lod_vertices @ rot0.T) + trans0
            self._draw_mesh(verts, st.lod_faces, st.color_rgba)
            if self._show_trajectories:
                xs = st.trajectory_xy[:, 0] + st.start_offset[0]
                ys = st.trajectory_xy[:, 1] + st.start_offset[1]
                zs = np.full_like(xs, 0.005)
                self._ax.plot(xs, ys, zs, color=st.color_rgba[:3], linewidth=1.6, alpha=0.85)
        self._canvas.draw_idle()

    def _render_animated(self) -> None:
        self._ax.clear()
        self._set_axes()
        for st in self._dancer_state.values():
            local_t = ((self._global_t * st.speed) + st.phase) % 1.0
            f = int(local_t * (st.frame_count - 1))
            f = max(0, min(st.frame_count - 1, f))
            rot = st.rotations[f]
            trans = st.translations[f] + st.start_offset
            verts = (st.lod_vertices @ rot.T) + trans
            self._draw_mesh(verts, st.lod_faces, st.color_rgba)
            if self._show_trajectories:
                xs = st.trajectory_xy[:, 0] + st.start_offset[0]
                ys = st.trajectory_xy[:, 1] + st.start_offset[1]
                zs = np.full_like(xs, 0.005)
                self._ax.plot(xs, ys, zs, color=st.color_rgba[:3], linewidth=1.4, alpha=0.7)
        self._canvas.draw_idle()

    def _draw_mesh(self, verts: np.ndarray, faces: np.ndarray, color_rgba) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        tri = verts[faces]
        coll = Poly3DCollection(
            tri,
            facecolors=(color_rgba[0], color_rgba[1], color_rgba[2], 0.85),
            edgecolors=(0.15, 0.18, 0.22, 0.6) if self._wireframe else "none",
            linewidths=0.4 if self._wireframe else 0.0,
        )
        self._ax.add_collection3d(coll)


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def make_viewer(parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QWidget:
    """Return the best available multi-trajectoid viewer for this environment.

    Honors DANCE_VIEWER_BACKEND=matplotlib for forced fallback.
    """
    backend_pref = os.environ.get("DANCE_VIEWER_BACKEND", "").strip().lower()
    if HAS_GPU_VIEWER and backend_pref != "matplotlib":
        return _MultiTrajectoidGLViewer(parent)
    return _MultiTrajectoidMplViewer(parent)
