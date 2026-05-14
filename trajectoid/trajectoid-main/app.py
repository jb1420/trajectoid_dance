from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# Keep Matplotlib and Fontconfig caches in writable locations to avoid startup stalls.
def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


if "XDG_CACHE_HOME" not in os.environ:
    preferred_cache_home = Path.home() / ".cache"
    if _is_writable_dir(preferred_cache_home):
        os.environ["XDG_CACHE_HOME"] = str(preferred_cache_home)
    else:
        fallback_cache_home = Path(tempfile.gettempdir()) / "cache"
        fallback_cache_home.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(fallback_cache_home)

if "MPLCONFIGDIR" not in os.environ:
    preferred = Path.home() / ".cache" / "trajectoids-mpl"
    if _is_writable_dir(preferred):
        os.environ["MPLCONFIGDIR"] = str(preferred)
    else:
        fallback = Path(tempfile.gettempdir()) / "trajectoids-mpl"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback)

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    import pyqtgraph.opengl.shaders as gl_shaders
except Exception:  # pragma: no cover - optional GPU dependency
    pg = None
    gl = None
    gl_shaders = None
    _HAS_GPU_VIEWER = False
else:
    pg.setConfigOptions(antialias=True)
    _HAS_GPU_VIEWER = True

from curve_editor import CurveEditorWidget, Tool


_METAL_SURFACE_HEX = "#b8c2cd"
_METAL_SURFACE_FAST_HEX = "#adb8c4"
_METAL_EDGE_HEX = "#2d3744"


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        return 0.0, 0.0, 0.0, float(alpha)
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    return float(r), float(g), float(b), float(alpha)


_METAL_GPU_SURFACE_RGBA = _hex_to_rgba("#b8c2cd", alpha=1.0)
_METAL_GPU_SURFACE_FAST_RGBA = _hex_to_rgba("#adb8c4", alpha=1.0)
_METAL_GPU_EDGE_RGBA = _hex_to_rgba("#2d3744", alpha=0.86)
_METAL_GPU_FAST_SHADER = "shaded"
_METAL_LIGHT_SOURCE = LightSource(azdeg=330, altdeg=48)
_INTERACTION_GHOST_ALPHA = 0.56


def _build_stainless_gpu_shader():
    if gl_shaders is None:
        return "shaded"
    try:
        return gl_shaders.ShaderProgram(
            "trajectoidsStainlessSoft",
            [
                gl_shaders.VertexShader(
                    """
                    varying vec3 normal;
                    varying vec3 viewDir;
                    void main() {
                        vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
                        normal = normalize(gl_NormalMatrix * gl_Normal);
                        viewDir = normalize(-viewPos.xyz);
                        gl_FrontColor = gl_Color;
                        gl_BackColor = gl_Color;
                        gl_Position = ftransform();
                    }
                    """
                ),
                gl_shaders.FragmentShader(
                    """
                    varying vec3 normal;
                    varying vec3 viewDir;
                    void main() {
                        vec3 N = normalize(normal);
                        vec3 V = normalize(viewDir);
                        vec3 LKey = normalize(vec3(0.45, -0.55, -0.70));
                        vec3 LFill = normalize(vec3(-0.28, 0.35, -0.20));

                        float key = max(dot(N, LKey), 0.0);
                        float fill = max(dot(N, LFill), 0.0) * 0.50;
                        float diffuse = min(key + fill, 1.20);

                        vec3 HKey = normalize(LKey + V);
                        vec3 HFill = normalize(LFill + V);
                        float specularTight = pow(max(dot(N, HKey), 0.0), 84.0) * 0.58;
                        float specularWide = pow(max(dot(N, HFill), 0.0), 34.0) * 0.12;
                        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.6);
                        float rim = fresnel * 0.28;

                        vec3 base = gl_Color.rgb;
                        vec3 lit = base * (0.30 + 0.74 * diffuse);
                        vec3 metalTint = vec3(0.92, 0.96, 1.00);
                        vec3 color = lit + metalTint * (specularTight + specularWide + rim);
                        gl_FragColor = vec4(min(color, vec3(1.0)), gl_Color.a);
                    }
                    """
                ),
            ],
        )
    except Exception:
        return "shaded"


_METAL_GPU_PRIMARY_SHADER = _build_stainless_gpu_shader()


def _rgba_with_alpha(rgba: tuple[float, float, float, float], alpha: float) -> tuple[float, float, float, float]:
    return float(rgba[0]), float(rgba[1]), float(rgba[2]), float(np.clip(alpha, 0.0, 1.0))


GROUND_GRID_WIDTH = 1.5
GROUND_GRID_COLOR = _hex_to_rgba("#c2cedb", alpha=0.98)
GROUND_GRID_SIZE_SCALE = 0.50
GROUND_GRID_MIN_RADIUS = 1.4
GROUND_GRID_CLEARANCE_FACTOR = 0.0025
GROUND_GRID_CLEARANCE_MIN = 0.0015
_TRAJECTORY_GPU_RGBA = _hex_to_rgba("#d97706", alpha=0.96)


class MatplotlibTrajectoidViewerWidget(QtWidgets.QWidget):
    simulationStarted = QtCore.Signal()
    simulationFinished = QtCore.Signal(bool, str)
    backend_name = "Matplotlib (CPU fallback)"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._vertices: Optional[np.ndarray] = None
        self._faces: Optional[np.ndarray] = None
        self._full_vertices: Optional[np.ndarray] = None
        self._full_faces: Optional[np.ndarray] = None
        self._lod_vertices: Optional[np.ndarray] = None
        self._lod_faces: Optional[np.ndarray] = None
        # Keep Matplotlib rotation responsive by using a lightweight interaction mesh.
        self._lod_max_faces = 520
        self._sim_lod_max_faces = 1400
        self._is_interacting = False
        self._wireframe = False
        self._default_view = (24.0, 36.0)
        self._contact_curve: Optional[np.ndarray] = None

        self._sim_timer = QtCore.QTimer(self)
        self._sim_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._sim_timer.timeout.connect(self._on_simulation_tick)
        self._simulation_running = False
        self._sim_frame_index = 0
        self._sim_total_frames = 0
        self._sim_message = ""
        self._sim_completed_target = True
        self._sim_translations: Optional[np.ndarray] = None
        self._sim_rotations: Optional[np.ndarray] = None
        self._sim_trajectory_xy: Optional[np.ndarray] = None
        self._sim_anim_vertices: Optional[np.ndarray] = None
        self._sim_anim_faces: Optional[np.ndarray] = None
        self._sim_axis_center = np.zeros(3, dtype=float)
        self._sim_axis_radius = 1.0
        self._sim_grid_view_radius = 1.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._figure = Figure(figsize=(6.0, 5.0))
        self._figure.patch.set_facecolor("#f8f9fb")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._ax = self._figure.add_subplot(111, projection="3d")
        self._ax.set_facecolor("#f8f9fb")

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)
        self._connect_interaction_events()
        self._render_mesh(reset_camera=True)

    @property
    def is_simulation_running(self) -> bool:
        return self._simulation_running

    def set_wireframe(self, enabled: bool) -> None:
        self._wireframe = bool(enabled)
        if not self._simulation_running:
            self._render_mesh(reset_camera=False, preserve_view=True)

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self._vertices = np.asarray(vertices, dtype=float)
        self._faces = np.asarray(faces, dtype=np.int32)
        self._full_vertices = self._vertices
        self._full_faces = self._faces
        self._lod_vertices, self._lod_faces = self._make_animation_mesh(
            self._full_vertices,
            self._full_faces,
            max_faces=self._lod_max_faces,
        )
        self._is_interacting = False
        if not self._simulation_running:
            self._render_mesh(reset_camera=True)

    def set_contact_curve(self, points_xyz: Optional[np.ndarray]) -> None:
        if points_xyz is None or len(points_xyz) < 2:
            self._contact_curve = None
        else:
            self._contact_curve = np.asarray(points_xyz, dtype=float)
        if not self._simulation_running:
            self._render_mesh(reset_camera=False, preserve_view=True)

    def reset_view(self) -> None:
        if self._simulation_running:
            return
        self._render_mesh(reset_camera=True)

    def fit_to_screen(self) -> None:
        if self._simulation_running:
            return
        if self._full_vertices is None:
            return
        self._set_equal_limits(self._full_vertices)
        self._canvas.draw_idle()

    def start_simulation(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        translations_xyz: np.ndarray,
        rotations: np.ndarray,
        trajectory_xy: np.ndarray,
        wireframe: bool,
        duration_seconds: float = 8.0,
        message: str = "",
        completed_target: bool = True,
    ) -> bool:
        if self._simulation_running:
            return False
        self._set_interaction_mode(False, redraw=False)

        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=np.int32)
        translations_xyz = np.asarray(translations_xyz, dtype=float)
        rotations = np.asarray(rotations, dtype=float)
        trajectory_xy = np.asarray(trajectory_xy, dtype=float)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must be Nx3.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must be Mx3 triangles.")
        if translations_xyz.ndim != 2 or translations_xyz.shape[1] != 3:
            raise ValueError("translations_xyz must be Fx3.")
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must be Fx3x3.")
        if translations_xyz.shape[0] != rotations.shape[0]:
            raise ValueError("translations and rotations frame counts do not match.")
        if trajectory_xy.ndim != 2 or trajectory_xy.shape[1] != 2:
            raise ValueError("trajectory_xy must be Tx2.")
        if translations_xyz.shape[0] < 2:
            raise ValueError("Simulation must include at least 2 frames.")

        self._vertices = vertices
        self._faces = faces
        self._full_vertices = vertices
        self._full_faces = faces
        self._lod_vertices, self._lod_faces = self._make_animation_mesh(
            vertices,
            faces,
            max_faces=self._lod_max_faces,
        )
        self._wireframe = bool(wireframe)

        self._sim_translations = translations_xyz
        self._sim_rotations = rotations
        self._sim_trajectory_xy = trajectory_xy
        self._sim_anim_vertices, self._sim_anim_faces = self._make_animation_mesh(
            vertices,
            faces,
            max_faces=self._sim_lod_max_faces,
        )
        self._sim_frame_index = 0
        self._sim_total_frames = translations_xyz.shape[0]
        self._sim_message = message
        self._sim_completed_target = bool(completed_target)
        if self._full_vertices is not None and self._full_vertices.size > 0:
            mins = np.min(self._full_vertices, axis=0)
            maxs = np.max(self._full_vertices, axis=0)
            span = float(np.max(maxs - mins))
            self._sim_grid_view_radius = max(span * 0.55, 1e-3)
        else:
            self._sim_grid_view_radius = 1.0

        self._compute_simulation_limits()
        self._simulation_running = True
        self._toolbar.setEnabled(False)
        self._ax.set_navigate(False)
        self.simulationStarted.emit()

        self._draw_simulation_frame(frame_index=0, show_trajectory=True, use_full_mesh=False)

        total_ms = max(100, int(round(float(duration_seconds) * 1000.0)))
        interval_ms = max(1, int(round(total_ms / max(1, self._sim_total_frames - 1))))
        self._sim_timer.start(interval_ms)
        return True

    def stop_simulation(
        self,
        show_trajectory: bool = True,
        interrupted: bool = False,
        custom_message: Optional[str] = None,
    ) -> None:
        if not self._simulation_running:
            return

        self._sim_timer.stop()
        final_index = max(0, self._sim_total_frames - 1)
        if show_trajectory:
            self._draw_simulation_frame(
                frame_index=final_index,
                show_trajectory=True,
                use_full_mesh=True,
            )

        self._simulation_running = False
        self._set_interaction_mode(False, redraw=False)
        self._toolbar.setEnabled(True)
        self._ax.set_navigate(True)

        if interrupted:
            completed = False
            message = custom_message or "Simulation interrupted."
        else:
            completed = self._sim_completed_target
            message = custom_message or self._sim_message
        self.simulationFinished.emit(completed, message)

    def _connect_interaction_events(self) -> None:
        self._canvas.mpl_connect(
            "button_press_event",
            self._on_mouse_press,
        )
        self._canvas.mpl_connect(
            "button_release_event",
            self._on_mouse_release,
        )
        self._canvas.mpl_connect(
            "motion_notify_event",
            self._on_mouse_motion,
        )
        self._canvas.mpl_connect(
            "figure_leave_event",
            self._on_figure_leave,
        )

    def _on_mouse_press(self, event) -> None:
        if self._simulation_running:
            return
        if self._full_vertices is None or self._full_faces is None:
            return
        if event.inaxes is not self._ax:
            return
        # Left/middle/right press can start rotate/pan interactions.
        if getattr(event, "button", None) in (1, 2, 3):
            self._set_interaction_mode(True, redraw=True)

    def _on_mouse_release(self, _event) -> None:
        if self._simulation_running:
            return
        if self._is_interacting:
            self._set_interaction_mode(False, redraw=True)

    def _on_figure_leave(self, _event) -> None:
        if self._simulation_running:
            return
        if self._is_interacting:
            self._set_interaction_mode(False, redraw=True)

    def _on_mouse_motion(self, event) -> None:
        if self._simulation_running or not self._is_interacting:
            return
        buttons = getattr(event, "buttons", None)
        if not buttons:
            self._set_interaction_mode(False, redraw=True)

    def _set_interaction_mode(self, interacting: bool, redraw: bool) -> None:
        interacting = bool(interacting)
        if self._is_interacting == interacting:
            return
        self._is_interacting = interacting
        if redraw and not self._simulation_running:
            self._render_mesh(reset_camera=False, preserve_view=True)

    def _on_simulation_tick(self) -> None:
        if not self._simulation_running:
            return
        self._sim_frame_index += 1
        if self._sim_frame_index >= self._sim_total_frames:
            self.stop_simulation(show_trajectory=True, interrupted=False)
            return
        self._draw_simulation_frame(
            frame_index=self._sim_frame_index,
            show_trajectory=True,
            use_full_mesh=False,
        )

    def _make_animation_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        max_faces: int = 2600,
    ) -> tuple[np.ndarray, np.ndarray]:
        if faces.shape[0] <= max_faces:
            return vertices, faces
        stride = max(1, int(np.ceil(faces.shape[0] / float(max_faces))))
        faces_sub = faces[::stride]
        unique_indices, inverse = np.unique(faces_sub.reshape(-1), return_inverse=True)
        vertices_sub = vertices[unique_indices]
        faces_remapped = inverse.reshape(-1, 3)
        return vertices_sub, faces_remapped.astype(np.int32)

    def _compute_simulation_limits(self) -> None:
        assert self._sim_translations is not None
        assert self._sim_trajectory_xy is not None
        assert self._vertices is not None

        center_local = np.mean(self._vertices, axis=0)
        mesh_radius = float(np.max(np.linalg.norm(self._vertices - center_local, axis=1)))

        x_values = np.concatenate([self._sim_translations[:, 0], self._sim_trajectory_xy[:, 0]])
        y_values = np.concatenate([self._sim_translations[:, 1], self._sim_trajectory_xy[:, 1]])
        z_values = np.array(
            [
                np.min(self._sim_translations[:, 2]) - mesh_radius,
                np.max(self._sim_translations[:, 2]) + mesh_radius,
                0.0,
            ],
            dtype=float,
        )

        mins = np.array([np.min(x_values), np.min(y_values), np.min(z_values)], dtype=float)
        maxs = np.array([np.max(x_values), np.max(y_values), np.max(z_values)], dtype=float)
        center = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))

        # Zoom out for cinematic simulation framing.
        radius = max(1.35 * span * 0.5, mesh_radius * 2.0, 1.5)
        self._sim_axis_center = center
        self._sim_axis_radius = radius

    def _apply_simulation_axes(self) -> None:
        c = self._sim_axis_center
        r = self._sim_axis_radius
        self._ax.set_xlim(c[0] - r, c[0] + r)
        self._ax.set_ylim(c[1] - r, c[1] + r)
        self._ax.set_zlim(max(-0.1 * r, c[2] - 0.55 * r), c[2] + 0.85 * r)
        self._ax.set_box_aspect((1.0, 1.0, 0.75))
        self._ax.view_init(elev=45.0, azim=45.0)
        self._ax.set_proj_type("persp")
        self._ax.set_axis_off()

    def _draw_ground_grid(self) -> None:
        # Keep simulation grid world-size consistent with normal mesh view mode.
        grid_r = max(GROUND_GRID_MIN_RADIUS, float(self._sim_grid_view_radius) * GROUND_GRID_SIZE_SCALE)
        ticks = np.linspace(-grid_r, grid_r, 9)
        for t in ticks:
            self._ax.plot(
                [-grid_r, grid_r],
                [t, t],
                [0.0, 0.0],
                color="#dbe2ea",
                linewidth=0.6,
                alpha=0.8,
                zorder=1,
            )
            self._ax.plot(
                [t, t],
                [-grid_r, grid_r],
                [0.0, 0.0],
                color="#dbe2ea",
                linewidth=0.6,
                alpha=0.8,
                zorder=1,
            )

    def _draw_simulation_frame(
        self,
        frame_index: int,
        show_trajectory: bool,
        use_full_mesh: bool,
    ) -> None:
        assert self._sim_translations is not None
        assert self._sim_rotations is not None
        assert self._sim_trajectory_xy is not None
        assert self._vertices is not None
        assert self._faces is not None
        assert self._sim_anim_vertices is not None
        assert self._sim_anim_faces is not None

        self._ax.clear()
        self._ax.set_facecolor("#f8f9fb")
        self._draw_ground_grid()

        if use_full_mesh:
            vertices_base = self._vertices
            faces = self._faces
        else:
            vertices_base = self._sim_anim_vertices
            faces = self._sim_anim_faces

        rot = self._sim_rotations[frame_index]
        trans = self._sim_translations[frame_index]
        vertices_world = (vertices_base @ rot.T) + trans
        use_fast_style = not use_full_mesh

        if self._wireframe:
            self._ax.plot_trisurf(
                vertices_world[:, 0],
                vertices_world[:, 1],
                vertices_world[:, 2],
                triangles=faces,
                color=(0.0, 0.0, 0.0, 0.0),
                edgecolor=_METAL_EDGE_HEX,
                linewidth=0.28 if use_fast_style else 0.45,
                shade=False,
                antialiased=False,
                alpha=1.0,
            )
        else:
            self._ax.plot_trisurf(
                vertices_world[:, 0],
                vertices_world[:, 1],
                vertices_world[:, 2],
                triangles=faces,
                color=_METAL_SURFACE_FAST_HEX if use_fast_style else _METAL_SURFACE_HEX,
                edgecolor="none",
                linewidth=0.0,
                shade=True,
                lightsource=_METAL_LIGHT_SOURCE,
                antialiased=False,
                alpha=1.0,
            )

        if show_trajectory:
            traj = self._sim_trajectory_xy
            if use_full_mesh:
                traj_display = traj
            else:
                t_count = traj.shape[0]
                if t_count <= 1 or self._sim_total_frames <= 1:
                    traj_display = traj[:1]
                else:
                    frac = frame_index / float(max(1, self._sim_total_frames - 1))
                    upto = int(np.clip(np.ceil(frac * (t_count - 1)) + 1, 1, t_count))
                    traj_display = traj[:upto]
            self._ax.plot(
                traj_display[:, 0],
                traj_display[:, 1],
                np.zeros(traj_display.shape[0], dtype=float),
                color="#d97706",
                linewidth=2.1,
                alpha=0.95,
                zorder=3,
            )

        if self._contact_curve is not None and self._contact_curve.shape[0] >= 2:
            curve_world = (self._contact_curve * 1.005) @ rot.T + trans
            self._ax.plot(
                curve_world[:, 0],
                curve_world[:, 1],
                curve_world[:, 2],
                color="#d97706",
                linewidth=1.6,
                alpha=0.92,
                zorder=4,
            )

        self._apply_simulation_axes()
        self._ax.set_navigate(False)
        self._canvas.draw_idle()

    def _set_equal_limits(self, vertices: np.ndarray) -> None:
        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)
        center = 0.5 * (mins + maxs)
        span = np.max(maxs - mins)
        radius = max(span * 0.55, 1e-3)
        self._ax.set_xlim(center[0] - radius, center[0] + radius)
        self._ax.set_ylim(center[1] - radius, center[1] + radius)
        self._ax.set_zlim(center[2] - radius, center[2] + radius)
        self._ax.set_box_aspect((1.0, 1.0, 1.0))

    def _draw_empty(self) -> None:
        self._ax.text2D(
            0.05,
            0.96,
            "Generate a shape to preview it here.",
            transform=self._ax.transAxes,
            color="#475569",
            fontsize=10,
        )
        self._ax.text2D(
            0.05,
            0.90,
            "Rotate: left drag | Pan: toolbar Pan mode | Zoom: wheel/toolbar",
            transform=self._ax.transAxes,
            color="#64748b",
            fontsize=9,
        )
        self._ax.set_axis_off()

    def _capture_view_state(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float, float]:
        return (
            self._ax.get_xlim(),
            self._ax.get_ylim(),
            self._ax.get_zlim(),
            float(self._ax.elev),
            float(self._ax.azim),
        )

    def _restore_view_state(
        self,
        state: tuple[tuple[float, float], tuple[float, float], tuple[float, float], float, float],
    ) -> None:
        xlim, ylim, zlim, elev, azim = state
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.set_zlim(*zlim)
        self._ax.view_init(elev=elev, azim=azim)

    def _render_mesh(self, reset_camera: bool = False, preserve_view: bool = False) -> None:
        prior_state = None
        if preserve_view:
            try:
                prior_state = self._capture_view_state()
            except Exception:
                prior_state = None

        self._ax.clear()
        self._ax.set_facecolor("#f8f9fb")
        if self._full_vertices is None or self._full_faces is None or self._full_faces.size == 0:
            self._draw_empty()
        else:
            # Keep full geometry while rotating so object shape does not collapse into coarse triangles.
            vertices_to_draw = self._full_vertices
            faces_to_draw = self._full_faces
            render_wireframe = self._wireframe and (not self._is_interacting)
            surface_alpha = _INTERACTION_GHOST_ALPHA if self._is_interacting else 1.0

            if render_wireframe:
                self._ax.plot_trisurf(
                    vertices_to_draw[:, 0],
                    vertices_to_draw[:, 1],
                    vertices_to_draw[:, 2],
                    triangles=faces_to_draw,
                    color=(0.0, 0.0, 0.0, 0.0),
                    edgecolor=_METAL_EDGE_HEX,
                    linewidth=0.35,
                    shade=False,
                    antialiased=False,
                )
            else:
                self._ax.plot_trisurf(
                    vertices_to_draw[:, 0],
                    vertices_to_draw[:, 1],
                    vertices_to_draw[:, 2],
                    triangles=faces_to_draw,
                    color=_METAL_SURFACE_HEX,
                    edgecolor="none",
                    linewidth=0.0,
                    shade=not self._is_interacting,
                    lightsource=_METAL_LIGHT_SOURCE,
                    antialiased=False,
                    alpha=surface_alpha,
                )

            self._ax.grid(False)
            self._ax.set_axis_off()
            if prior_state is not None and not reset_camera:
                self._restore_view_state(prior_state)
            else:
                self._set_equal_limits(self._full_vertices)
                if reset_camera:
                    self._ax.view_init(elev=self._default_view[0], azim=self._default_view[1])
            self._ax.set_proj_type("persp")

            # Avoid extra decorative lines while interacting to minimize redraw cost.
            if not self._is_interacting:
                theta = np.linspace(0, 2 * np.pi, 120)
                radius = float(max(np.ptp(self._full_vertices[:, 0]), np.ptp(self._full_vertices[:, 1]))) * 0.65
                radius = max(radius, 0.6)
                z0 = float(np.min(self._full_vertices[:, 2]) - 0.02 * radius)
                self._ax.plot(
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    np.full_like(theta, z0),
                    color="#cbd5e1",
                    linewidth=0.6,
                    alpha=0.9,
                )

            if self._contact_curve is not None and self._contact_curve.shape[0] >= 2:
                # Slight outward push so the curve stays visible on top of the surface.
                lifted = self._contact_curve * 1.005
                self._ax.plot(
                    lifted[:, 0],
                    lifted[:, 1],
                    lifted[:, 2],
                    color="#d97706",
                    linewidth=2.0,
                    alpha=0.95,
                    zorder=5,
                )
        self._canvas.draw_idle()


if _HAS_GPU_VIEWER:

    class _InteractiveGLViewWidget(gl.GLViewWidget):
        interactionChanged = QtCore.Signal(bool)

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self._interacting = False
            self._input_enabled = True

        def set_input_enabled(self, enabled: bool) -> None:
            self._input_enabled = bool(enabled)
            if not self._input_enabled and self._interacting:
                self._set_interacting(False)

        def _set_interacting(self, interacting: bool) -> None:
            interacting = bool(interacting)
            if self._interacting == interacting:
                return
            self._interacting = interacting
            self.interactionChanged.emit(interacting)

        def mousePressEvent(self, event) -> None:
            if not self._input_enabled:
                event.ignore()
                return
            if event.button() in (
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.MiddleButton,
                QtCore.Qt.MouseButton.RightButton,
            ):
                self._set_interacting(True)
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            if not self._input_enabled:
                event.ignore()
                return
            super().mouseReleaseEvent(event)
            if self._interacting and event.buttons() == QtCore.Qt.MouseButton.NoButton:
                self._set_interacting(False)

        def wheelEvent(self, event) -> None:
            if not self._input_enabled:
                event.ignore()
                return
            super().wheelEvent(event)

        def leaveEvent(self, event) -> None:
            super().leaveEvent(event)
            if self._interacting:
                self._set_interacting(False)


    class GPUTrajectoidViewerWidget(QtWidgets.QWidget):
        simulationStarted = QtCore.Signal()
        simulationFinished = QtCore.Signal(bool, str)
        backend_name = "PyQtGraph OpenGL (GPU)"

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self._vertices: Optional[np.ndarray] = None
            self._faces: Optional[np.ndarray] = None
            self._full_vertices: Optional[np.ndarray] = None
            self._full_faces: Optional[np.ndarray] = None
            self._lod_vertices: Optional[np.ndarray] = None
            self._lod_faces: Optional[np.ndarray] = None
            self._lod_max_faces = 1500
            self._is_interacting = False
            self._wireframe = False
            self._default_view = (24.0, 36.0)
            self._mesh_face_item: Optional[gl.GLMeshItem] = None
            self._mesh_edge_item: Optional[gl.GLMeshItem] = None
            self._mesh_topology: Optional[tuple[int, int, bool, bool]] = None
            self._trajectory_item: Optional[gl.GLLinePlotItem] = None
            self._contact_curve_item: Optional[gl.GLLinePlotItem] = None
            self._contact_curve: Optional[np.ndarray] = None
            self._grid_items: list[gl.GLLinePlotItem] = []
            self._grid_radius = 0.0
            self._grid_z_level = 0.0

            self._sim_timer = QtCore.QTimer(self)
            self._sim_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self._sim_timer.timeout.connect(self._on_simulation_tick)
            self._simulation_running = False
            self._sim_frame_index = 0
            self._sim_total_frames = 0
            self._sim_message = ""
            self._sim_completed_target = True
            self._sim_translations: Optional[np.ndarray] = None
            self._sim_rotations: Optional[np.ndarray] = None
            self._sim_trajectory_xy: Optional[np.ndarray] = None
            self._sim_anim_vertices: Optional[np.ndarray] = None
            self._sim_anim_faces: Optional[np.ndarray] = None
            self._sim_axis_center = np.zeros(3, dtype=float)
            self._sim_axis_radius = 1.0
            self._sim_grid_z_level = -GROUND_GRID_CLEARANCE_MIN
            self._sim_grid_view_radius = 1.0

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            self._view = _InteractiveGLViewWidget(self)
            self._view.setBackgroundColor(QtGui.QColor("#f8f9fb"))
            self._view.interactionChanged.connect(self._on_interaction_changed)
            layout.addWidget(self._view, stretch=1)

            self._update_ground_grid(
                radius=2.0,
                z_level=-GROUND_GRID_CLEARANCE_MIN,
                force=True,
            )
            self._apply_camera(
                center=np.zeros(3, dtype=float),
                radius=1.0,
                elev=self._default_view[0],
                azim=self._default_view[1],
                zoom_factor=3.0,
            )

        @property
        def is_simulation_running(self) -> bool:
            return self._simulation_running

        def set_wireframe(self, enabled: bool) -> None:
            self._wireframe = bool(enabled)
            if not self._simulation_running:
                self._render_mesh(reset_camera=False)

        def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
            self._vertices = np.asarray(vertices, dtype=float)
            self._faces = np.asarray(faces, dtype=np.int32)
            self._full_vertices = self._vertices
            self._full_faces = self._faces
            self._lod_vertices, self._lod_faces = self._make_animation_mesh(
                self._full_vertices,
                self._full_faces,
                max_faces=self._lod_max_faces,
            )
            self._is_interacting = False
            if not self._simulation_running:
                self._render_mesh(reset_camera=True)

        def set_contact_curve(self, points_xyz: Optional[np.ndarray]) -> None:
            if points_xyz is None or len(points_xyz) < 2:
                self._contact_curve = None
            else:
                self._contact_curve = np.asarray(points_xyz, dtype=float)
            if not self._simulation_running:
                self._render_mesh(reset_camera=False)

        def reset_view(self) -> None:
            if self._simulation_running:
                return
            if self._full_vertices is None:
                self._apply_camera(
                    center=np.zeros(3, dtype=float),
                    radius=1.0,
                    elev=self._default_view[0],
                    azim=self._default_view[1],
                    zoom_factor=3.0,
                )
                return
            center, radius = self._mesh_center_radius(self._full_vertices)
            self._apply_camera(
                center=center,
                radius=radius,
                elev=self._default_view[0],
                azim=self._default_view[1],
                zoom_factor=2.6,
            )

        def fit_to_screen(self) -> None:
            if self._simulation_running:
                return
            if self._full_vertices is None:
                return
            center, radius = self._mesh_center_radius(self._full_vertices)
            elev, azim = self._camera_angles()
            self._apply_camera(
                center=center,
                radius=radius,
                elev=elev,
                azim=azim,
                zoom_factor=2.3,
            )

        def start_simulation(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            translations_xyz: np.ndarray,
            rotations: np.ndarray,
            trajectory_xy: np.ndarray,
            wireframe: bool,
            duration_seconds: float = 8.0,
            message: str = "",
            completed_target: bool = True,
        ) -> bool:
            if self._simulation_running:
                return False

            vertices = np.asarray(vertices, dtype=float)
            faces = np.asarray(faces, dtype=np.int32)
            translations_xyz = np.asarray(translations_xyz, dtype=float)
            rotations = np.asarray(rotations, dtype=float)
            trajectory_xy = np.asarray(trajectory_xy, dtype=float)

            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("vertices must be Nx3.")
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("faces must be Mx3 triangles.")
            if translations_xyz.ndim != 2 or translations_xyz.shape[1] != 3:
                raise ValueError("translations_xyz must be Fx3.")
            if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
                raise ValueError("rotations must be Fx3x3.")
            if translations_xyz.shape[0] != rotations.shape[0]:
                raise ValueError("translations and rotations frame counts do not match.")
            if trajectory_xy.ndim != 2 or trajectory_xy.shape[1] != 2:
                raise ValueError("trajectory_xy must be Tx2.")
            if translations_xyz.shape[0] < 2:
                raise ValueError("Simulation must include at least 2 frames.")

            self._vertices = vertices
            self._faces = faces
            self._full_vertices = vertices
            self._full_faces = faces
            self._lod_vertices, self._lod_faces = self._make_animation_mesh(
                vertices,
                faces,
                max_faces=self._lod_max_faces,
            )
            self._wireframe = bool(wireframe)
            self._is_interacting = False

            self._sim_translations = translations_xyz
            self._sim_rotations = rotations
            self._sim_trajectory_xy = trajectory_xy
            self._sim_anim_vertices, self._sim_anim_faces = self._make_animation_mesh(
                vertices,
                faces,
                max_faces=max(self._lod_max_faces, 3000),
            )
            self._sim_frame_index = 0
            self._sim_total_frames = translations_xyz.shape[0]
            self._sim_message = message
            self._sim_completed_target = bool(completed_target)
            _, self._sim_grid_view_radius = self._mesh_center_radius(self._full_vertices)
            self._sim_grid_z_level = self._compute_simulation_grid_z_level(
                self._vertices,
                rotations,
                translations_xyz,
            )

            self._compute_simulation_limits()
            self._simulation_running = True
            self._view.set_input_enabled(False)
            self.simulationStarted.emit()

            self._draw_simulation_frame(frame_index=0, show_trajectory=True, use_full_mesh=False)

            total_ms = max(100, int(round(float(duration_seconds) * 1000.0)))
            interval_ms = max(1, int(round(total_ms / max(1, self._sim_total_frames - 1))))
            self._sim_timer.start(interval_ms)
            return True

        def stop_simulation(
            self,
            show_trajectory: bool = True,
            interrupted: bool = False,
            custom_message: Optional[str] = None,
        ) -> None:
            if not self._simulation_running:
                return

            self._sim_timer.stop()
            final_index = max(0, self._sim_total_frames - 1)
            if show_trajectory:
                self._draw_simulation_frame(
                    frame_index=final_index,
                    show_trajectory=True,
                    use_full_mesh=True,
                )

            self._simulation_running = False
            self._view.set_input_enabled(True)
            self._is_interacting = False

            if not show_trajectory:
                self._render_mesh(reset_camera=False)

            if interrupted:
                completed = False
                message = custom_message or "Simulation interrupted."
            else:
                completed = self._sim_completed_target
                message = custom_message or self._sim_message
            self.simulationFinished.emit(completed, message)

        def _on_interaction_changed(self, interacting: bool) -> None:
            if self._simulation_running:
                return
            interacting = bool(interacting)
            if self._is_interacting == interacting:
                return
            self._is_interacting = interacting
            self._render_mesh(reset_camera=False)

        def _on_simulation_tick(self) -> None:
            if not self._simulation_running:
                return
            self._sim_frame_index += 1
            if self._sim_frame_index >= self._sim_total_frames:
                self.stop_simulation(show_trajectory=True, interrupted=False)
                return
            self._draw_simulation_frame(
                frame_index=self._sim_frame_index,
                show_trajectory=True,
                use_full_mesh=False,
            )

        def _make_animation_mesh(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            max_faces: int = 2600,
        ) -> tuple[np.ndarray, np.ndarray]:
            if faces.shape[0] <= max_faces:
                return vertices, faces
            stride = max(1, int(np.ceil(faces.shape[0] / float(max_faces))))
            faces_sub = faces[::stride]
            unique_indices, inverse = np.unique(faces_sub.reshape(-1), return_inverse=True)
            vertices_sub = vertices[unique_indices]
            faces_remapped = inverse.reshape(-1, 3)
            return vertices_sub, faces_remapped.astype(np.int32)

        def _mesh_center_radius(self, vertices: np.ndarray) -> tuple[np.ndarray, float]:
            mins = np.min(vertices, axis=0)
            maxs = np.max(vertices, axis=0)
            center = 0.5 * (mins + maxs)
            span = float(np.max(maxs - mins))
            radius = max(span * 0.55, 1e-3)
            return center, radius

        def _camera_angles(self) -> tuple[float, float]:
            elev = float(self._view.opts.get("elevation", self._default_view[0]))
            azim = float(self._view.opts.get("azimuth", self._default_view[1]))
            return elev, azim

        def _apply_camera(
            self,
            center: np.ndarray,
            radius: float,
            elev: float,
            azim: float,
            zoom_factor: float,
        ) -> None:
            distance = max(float(radius) * float(zoom_factor), 1.2)
            self._view.setCameraPosition(
                pos=QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2])),
                distance=distance,
                elevation=float(elev),
                azimuth=float(azim),
            )

        def _clear_ground_grid(self) -> None:
            for item in self._grid_items:
                self._remove_item(item)
            self._grid_items = []

        def _update_ground_grid(self, radius: float, z_level: float, force: bool = False) -> None:
            grid_r = max(GROUND_GRID_MIN_RADIUS, float(radius) * GROUND_GRID_SIZE_SCALE)
            z_level = float(z_level)
            if (
                (not force)
                and self._grid_items
                and abs(grid_r - self._grid_radius) < 1e-6
                and abs(z_level - self._grid_z_level) < 1e-6
            ):
                return

            self._clear_ground_grid()
            self._grid_radius = grid_r
            self._grid_z_level = z_level

            ticks = np.linspace(-grid_r, grid_r, 9, dtype=np.float32)
            z0 = np.float32(z_level)
            for t in ticks:
                line_xy = np.array([[-grid_r, float(t), float(z0)], [grid_r, float(t), float(z0)]], dtype=np.float32)
                line_yx = np.array([[float(t), -grid_r, float(z0)], [float(t), grid_r, float(z0)]], dtype=np.float32)
                for pos in (line_xy, line_yx):
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

        def _remove_item(self, item) -> None:
            if item is None:
                return
            try:
                self._view.removeItem(item)
            except Exception:
                pass

        def _compute_grid_z_level(self, vertices: np.ndarray) -> float:
            if vertices.size == 0:
                return -GROUND_GRID_CLEARANCE_MIN
            min_z = float(np.min(vertices[:, 2]))
            span = float(np.max(np.ptp(vertices, axis=0)))
            clearance = max(
                GROUND_GRID_CLEARANCE_MIN,
                GROUND_GRID_CLEARANCE_FACTOR * max(1.0, span),
            )
            return min(0.0, min_z - clearance)

        def _compute_simulation_grid_z_level(
            self,
            vertices: np.ndarray,
            rotations: np.ndarray,
            translations: np.ndarray,
        ) -> float:
            if vertices.size == 0:
                return -GROUND_GRID_CLEARANCE_MIN

            vertices_f32 = np.ascontiguousarray(vertices, dtype=np.float32)
            span = float(np.max(np.ptp(vertices_f32, axis=0)))
            clearance = max(
                GROUND_GRID_CLEARANCE_MIN,
                GROUND_GRID_CLEARANCE_FACTOR * max(1.0, span),
            )

            min_world_z = float("inf")
            for rot, trans in zip(rotations, translations):
                z_values = (vertices_f32 @ rot[2, :]) + trans[2]
                frame_min = float(np.min(z_values))
                if frame_min < min_world_z:
                    min_world_z = frame_min
            if not np.isfinite(min_world_z):
                min_world_z = float(np.min(vertices_f32[:, 2]))
            return min(0.0, min_world_z - clearance)

        def _clear_mesh_items(self) -> None:
            self._remove_item(self._mesh_edge_item)
            self._remove_item(self._mesh_face_item)
            self._mesh_face_item = None
            self._mesh_edge_item = None

        def _build_shaded_face_mesh_item(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            use_fast_style: bool,
            face_alpha: float,
        ) -> gl.GLMeshItem:
            face_base = _METAL_GPU_SURFACE_FAST_RGBA if use_fast_style else _METAL_GPU_SURFACE_RGBA
            face_color = _rgba_with_alpha(face_base, face_alpha)
            if face_alpha < 0.999:
                # Keep drag-preview smooth and legible; avoid sharp specular facets while interacting.
                shader_name = "shaded"
            else:
                shader_name = _METAL_GPU_FAST_SHADER if use_fast_style else _METAL_GPU_PRIMARY_SHADER
            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                smooth=not use_fast_style,
                drawEdges=False,
                drawFaces=True,
                edgeColor=_METAL_GPU_EDGE_RGBA,
                color=face_color,
                shader=shader_name,
                computeNormals=True,
            )
            mesh.setGLOptions("translucent" if face_alpha < 0.999 else "opaque")
            return mesh

        def _build_edge_mesh_item(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
        ) -> gl.GLMeshItem:
            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                smooth=False,
                drawEdges=True,
                drawFaces=False,
                edgeColor=_METAL_GPU_EDGE_RGBA,
                color=(0.0, 0.0, 0.0, 0.0),
                shader="shaded",
                computeNormals=False,
            )
            mesh.setGLOptions("opaque")
            return mesh

        def _update_mesh_item(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            allow_reuse: bool,
            use_fast_style: bool,
            display_wireframe: bool,
            face_alpha: float,
        ) -> None:
            vertices_f32 = np.ascontiguousarray(vertices, dtype=np.float32)
            faces_i32 = np.ascontiguousarray(faces, dtype=np.int32)
            alpha_key = round(float(face_alpha), 3)
            topology = (vertices_f32.shape[0], faces_i32.shape[0], display_wireframe, use_fast_style, alpha_key)
            if allow_reuse and self._mesh_topology == topology:
                if display_wireframe and self._mesh_face_item is not None:
                    self._mesh_face_item.setMeshData(vertexes=vertices_f32, faces=faces_i32)
                    return
                if (not display_wireframe) and self._mesh_face_item is not None:
                    self._mesh_face_item.setMeshData(vertexes=vertices_f32, faces=faces_i32)
                    if self._mesh_edge_item is not None:
                        self._mesh_edge_item.setMeshData(vertexes=vertices_f32, faces=faces_i32)
                    return

            self._clear_mesh_items()
            if display_wireframe:
                self._mesh_face_item = self._build_edge_mesh_item(vertices_f32, faces_i32)
                self._view.addItem(self._mesh_face_item)
            else:
                self._mesh_face_item = self._build_shaded_face_mesh_item(
                    vertices_f32,
                    faces_i32,
                    use_fast_style=use_fast_style,
                    face_alpha=face_alpha,
                )
                self._view.addItem(self._mesh_face_item)
            self._mesh_topology = topology

        def _clear_trajectory(self) -> None:
            if self._trajectory_item is None:
                return
            self._remove_item(self._trajectory_item)
            self._trajectory_item = None

        def _clear_contact_curve_item(self) -> None:
            if self._contact_curve_item is None:
                return
            self._remove_item(self._contact_curve_item)
            self._contact_curve_item = None

        def _set_contact_curve_world(self, points_world: Optional[np.ndarray]) -> None:
            if points_world is None or len(points_world) < 2:
                self._clear_contact_curve_item()
                return
            arr = np.ascontiguousarray(points_world, dtype=np.float32)
            if self._contact_curve_item is None:
                self._contact_curve_item = gl.GLLinePlotItem(
                    pos=arr,
                    color=_TRAJECTORY_GPU_RGBA,
                    width=2.4,
                    antialias=True,
                    mode="line_strip",
                )
                self._contact_curve_item.setGLOptions("translucent")
                self._view.addItem(self._contact_curve_item)
            else:
                self._contact_curve_item.setData(
                    pos=arr,
                    color=_TRAJECTORY_GPU_RGBA,
                    width=2.4,
                    antialias=True,
                    mode="line_strip",
                )

        def _update_contact_curve_item(self) -> None:
            if self._contact_curve is None or self._contact_curve.shape[0] < 2:
                self._clear_contact_curve_item()
                return
            # Push slightly outward from the body center so the curve sits on top
            # of the surface instead of fighting the depth buffer.
            lifted = self._contact_curve * 1.005
            self._set_contact_curve_world(lifted)

        def _update_trajectory(self, points_xy: np.ndarray, z_level: float) -> None:
            points_xyz = np.column_stack(
                [
                    points_xy[:, 0],
                    points_xy[:, 1],
                    np.full(points_xy.shape[0], float(z_level) + 1e-4, dtype=float),
                ]
            ).astype(np.float32, copy=False)
            if self._trajectory_item is None:
                self._trajectory_item = gl.GLLinePlotItem(
                    pos=points_xyz,
                    color=_TRAJECTORY_GPU_RGBA,
                    width=2.2,
                    antialias=True,
                    mode="line_strip",
                )
                self._view.addItem(self._trajectory_item)
            else:
                self._trajectory_item.setData(
                    pos=points_xyz,
                    color=_TRAJECTORY_GPU_RGBA,
                    width=2.2,
                    antialias=True,
                    mode="line_strip",
                )

        def _render_mesh(self, reset_camera: bool = False) -> None:
            self._clear_trajectory()
            if self._full_vertices is None or self._full_faces is None or self._full_faces.size == 0:
                self._clear_mesh_items()
                self._clear_contact_curve_item()
                self._mesh_topology = None
                self._update_ground_grid(
                    radius=2.0,
                    z_level=-GROUND_GRID_CLEARANCE_MIN,
                )
                if reset_camera:
                    self._apply_camera(
                        center=np.zeros(3, dtype=float),
                        radius=1.0,
                        elev=self._default_view[0],
                        azim=self._default_view[1],
                        zoom_factor=3.0,
                    )
                self._view.update()
                return

            # Keep full geometry while rotating so object shape remains stable.
            vertices_to_draw = self._full_vertices
            faces_to_draw = self._full_faces
            self._update_mesh_item(
                vertices_to_draw,
                faces_to_draw,
                allow_reuse=True,
                use_fast_style=False,
                display_wireframe=self._wireframe and (not self._is_interacting),
                face_alpha=_INTERACTION_GHOST_ALPHA if self._is_interacting else 1.0,
            )
            self._update_contact_curve_item()

            center, radius = self._mesh_center_radius(self._full_vertices)
            grid_z = self._compute_grid_z_level(self._full_vertices)
            self._update_ground_grid(radius=radius, z_level=grid_z)
            if reset_camera:
                self._apply_camera(
                    center=center,
                    radius=radius,
                    elev=self._default_view[0],
                    azim=self._default_view[1],
                    zoom_factor=2.6,
                )
            self._view.update()

        def _compute_simulation_limits(self) -> None:
            assert self._sim_translations is not None
            assert self._sim_trajectory_xy is not None
            assert self._vertices is not None

            center_local = np.mean(self._vertices, axis=0)
            mesh_radius = float(np.max(np.linalg.norm(self._vertices - center_local, axis=1)))

            x_values = np.concatenate([self._sim_translations[:, 0], self._sim_trajectory_xy[:, 0]])
            y_values = np.concatenate([self._sim_translations[:, 1], self._sim_trajectory_xy[:, 1]])
            z_values = np.array(
                [
                    np.min(self._sim_translations[:, 2]) - mesh_radius,
                    np.max(self._sim_translations[:, 2]) + mesh_radius,
                    0.0,
                ],
                dtype=float,
            )

            mins = np.array([np.min(x_values), np.min(y_values), np.min(z_values)], dtype=float)
            maxs = np.array([np.max(x_values), np.max(y_values), np.max(z_values)], dtype=float)
            center = 0.5 * (mins + maxs)
            span = float(np.max(maxs - mins))
            radius = max(1.35 * span * 0.5, mesh_radius * 2.0, 1.5)
            self._sim_axis_center = center
            self._sim_axis_radius = radius

        def _simulation_camera_center(self) -> np.ndarray:
            c = self._sim_axis_center
            r = self._sim_axis_radius
            z_min = max(-0.1 * r, c[2] - 0.55 * r)
            z_max = c[2] + 0.85 * r
            return np.array([c[0], c[1], 0.5 * (z_min + z_max)], dtype=float)

        def _apply_simulation_camera(self) -> None:
            self._apply_camera(
                center=self._simulation_camera_center(),
                radius=self._sim_axis_radius,
                elev=45.0,
                azim=45.0,
                zoom_factor=2.45,
            )

        def _draw_simulation_frame(
            self,
            frame_index: int,
            show_trajectory: bool,
            use_full_mesh: bool,
        ) -> None:
            assert self._sim_translations is not None
            assert self._sim_rotations is not None
            assert self._sim_trajectory_xy is not None
            assert self._vertices is not None
            assert self._faces is not None
            assert self._sim_anim_vertices is not None
            assert self._sim_anim_faces is not None

            if use_full_mesh:
                vertices_base = self._vertices
                faces = self._faces
            else:
                vertices_base = self._sim_anim_vertices
                faces = self._sim_anim_faces

            rot = self._sim_rotations[frame_index]
            trans = self._sim_translations[frame_index]
            vertices_world = (vertices_base @ rot.T) + trans
            self._update_ground_grid(
                radius=self._sim_grid_view_radius,
                z_level=self._sim_grid_z_level,
            )
            self._update_mesh_item(
                vertices_world,
                faces,
                allow_reuse=True,
                use_fast_style=not use_full_mesh,
                display_wireframe=self._wireframe,
                face_alpha=1.0,
            )

            if self._contact_curve is not None and self._contact_curve.shape[0] >= 2:
                curve_world = (self._contact_curve * 1.005) @ rot.T + trans
                self._set_contact_curve_world(curve_world)
            else:
                self._clear_contact_curve_item()

            if show_trajectory:
                traj = self._sim_trajectory_xy
                if use_full_mesh:
                    traj_display = traj
                else:
                    t_count = traj.shape[0]
                    if t_count <= 1 or self._sim_total_frames <= 1:
                        traj_display = traj[:1]
                    else:
                        frac = frame_index / float(max(1, self._sim_total_frames - 1))
                        upto = int(np.clip(np.ceil(frac * (t_count - 1)) + 1, 1, t_count))
                        traj_display = traj[:upto]
                self._update_trajectory(traj_display, z_level=self._sim_grid_z_level)
            else:
                self._clear_trajectory()

            self._apply_simulation_camera()
            self._view.update()


_viewer_backend_pref = os.environ.get("TRAJECTOIDS_VIEWER_BACKEND", "auto").strip().lower()
if _viewer_backend_pref == "matplotlib":
    TrajectoidViewerWidget = MatplotlibTrajectoidViewerWidget
else:
    # Keep runtime on the original OpenGL viewer path.
    # Any non-matplotlib preference resolves to PyQtGraph GPU when available.
    if _HAS_GPU_VIEWER:
        TrajectoidViewerWidget = GPUTrajectoidViewerWidget
    else:
        TrajectoidViewerWidget = MatplotlibTrajectoidViewerWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trajectoids Curve-to-3D Designer")
        self.resize(1520, 920)

        self._last_generation_result = None
        self._last_closed_mode = True
        self._last_simulation_result = None
        self._has_simulated_once = False
        self._last_export_dir = Path.home()
        self._viewer_init_warning: Optional[str] = None
        self._ui_scale = 1.0
        self._ui_scale_min = 0.65
        self._ui_scale_max = 2.4
        app_font = QtWidgets.QApplication.font()
        base_font_size = float(app_font.pointSizeF())
        if base_font_size <= 0:
            base_font_size = float(app_font.pointSize())
        if base_font_size <= 0:
            base_font_size = 12.0
        self._base_font_point_size = base_font_size

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        self.editor = CurveEditorWidget()
        try:
            self.viewer = TrajectoidViewerWidget()
        except Exception as exc:  # pragma: no cover - runtime OpenGL stack dependent
            self.viewer = MatplotlibTrajectoidViewerWidget()
            self._viewer_init_warning = f"GPU viewer unavailable, using CPU fallback: {exc}"

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setChildrenCollapsible(True)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([640, 900])
        root_layout.addWidget(splitter)

        self.viewer.simulationStarted.connect(self._on_simulation_started)
        self.viewer.simulationFinished.connect(self._on_simulation_finished)

        self.editor.curveChanged.connect(self._update_curve_stats)
        self._update_curve_stats()
        self._build_sim_lock_widget_list()
        self._setup_ui_scaling_controls()
        self._append_status(f"3D viewer backend: {self.viewer.backend_name}")
        if self._viewer_init_warning:
            self._append_status(self._viewer_init_warning)
        self._append_status(
            "GUI zoom enabled: Ctrl/Cmd + mouse wheel, Ctrl/Cmd +/- (reset with Ctrl/Cmd+0)."
        )

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        tools_row = QtWidgets.QHBoxLayout()
        tools_row.setSpacing(4)
        self._tool_buttons = {}
        for name, label in [
            (Tool.FREEHAND, "Freehand"),
            (Tool.BEZIER, "Bezier"),
            (Tool.POLYLINE, "Polyline"),
            (Tool.ERASER, "Eraser"),
            (Tool.SELECT, "Select"),
        ]:
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, n=name: self._set_tool(n))
            self._tool_buttons[name] = button
            tools_row.addWidget(button)
        self._tool_buttons[Tool.FREEHAND].setChecked(True)

        layout.addLayout(tools_row)
        layout.addWidget(self.editor, stretch=1)

        controls = QtWidgets.QVBoxLayout()
        controls.setSpacing(6)

        edit_row = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.redo_button = QtWidgets.QPushButton("Redo")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.undo_button.clicked.connect(self.editor.undo)
        self.redo_button.clicked.connect(self.editor.redo)
        self.clear_button.clicked.connect(self.editor.clear_curve)
        edit_row.addWidget(self.undo_button)
        edit_row.addWidget(self.redo_button)
        edit_row.addWidget(self.clear_button)
        controls.addLayout(edit_row)

        smooth_row = QtWidgets.QHBoxLayout()
        self.smooth_on_draw_checkbox = QtWidgets.QCheckBox("Smooth while drawing")
        self.smooth_on_draw_checkbox.toggled.connect(self.editor.set_smooth_on_draw)
        self.closed_checkbox = QtWidgets.QCheckBox("Treat as closed/periodic")
        self.closed_checkbox.setChecked(True)
        self.closed_checkbox.toggled.connect(self.editor.set_closed_hint)
        smooth_row.addWidget(self.smooth_on_draw_checkbox)
        smooth_row.addWidget(self.closed_checkbox)
        controls.addLayout(smooth_row)

        process_row = QtWidgets.QHBoxLayout()
        self.smooth_button = QtWidgets.QPushButton("Smooth")
        self.smooth_button.clicked.connect(lambda: self.editor.apply_smooth(passes=1))
        self.resample_points_spin = QtWidgets.QSpinBox()
        self.resample_points_spin.setRange(60, 1200)
        self.resample_points_spin.setValue(320)
        self.resample_points_spin.setPrefix("Resample ")
        self.resample_points_spin.setSuffix(" pts")
        self.resample_button = QtWidgets.QPushButton("Apply")
        self.resample_button.clicked.connect(
            lambda: self.editor.apply_resample(self.resample_points_spin.value())
        )
        process_row.addWidget(self.smooth_button)
        process_row.addWidget(self.resample_points_spin)
        process_row.addWidget(self.resample_button)
        controls.addLayout(process_row)

        transform_row = QtWidgets.QHBoxLayout()
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.05, 8.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(1.10)
        self.scale_button = QtWidgets.QPushButton("Scale")
        self.scale_button.clicked.connect(lambda: self.editor.apply_scale(self.scale_spin.value()))

        self.rotate_spin = QtWidgets.QDoubleSpinBox()
        self.rotate_spin.setRange(-360.0, 360.0)
        self.rotate_spin.setSingleStep(5.0)
        self.rotate_spin.setValue(15.0)
        self.rotate_button = QtWidgets.QPushButton("Rotate")
        self.rotate_button.clicked.connect(lambda: self.editor.apply_rotate(self.rotate_spin.value()))

        transform_row.addWidget(self.scale_spin)
        transform_row.addWidget(self.scale_button)
        transform_row.addWidget(self.rotate_spin)
        transform_row.addWidget(self.rotate_button)
        controls.addLayout(transform_row)

        translate_row = QtWidgets.QHBoxLayout()
        self.dx_spin = QtWidgets.QDoubleSpinBox()
        self.dx_spin.setRange(-2000.0, 2000.0)
        self.dx_spin.setSingleStep(5.0)
        self.dx_spin.setValue(10.0)
        self.dx_spin.setPrefix("dx ")
        self.dy_spin = QtWidgets.QDoubleSpinBox()
        self.dy_spin.setRange(-2000.0, 2000.0)
        self.dy_spin.setSingleStep(5.0)
        self.dy_spin.setValue(0.0)
        self.dy_spin.setPrefix("dy ")
        self.translate_button = QtWidgets.QPushButton("Translate")
        self.translate_button.clicked.connect(
            lambda: self.editor.apply_translate(self.dx_spin.value(), self.dy_spin.value())
        )
        translate_row.addWidget(self.dx_spin)
        translate_row.addWidget(self.dy_spin)
        translate_row.addWidget(self.translate_button)
        controls.addLayout(translate_row)

        stats_row = QtWidgets.QHBoxLayout()
        self.length_label = QtWidgets.QLabel("Length: 0.00 px")
        self.curvature_button = QtWidgets.QPushButton("Curvature Plot")
        self.curvature_button.clicked.connect(self._show_curvature_plot)
        stats_row.addWidget(self.length_label, stretch=1)
        stats_row.addWidget(self.curvature_button)
        controls.addLayout(stats_row)

        generation_row = QtWidgets.QHBoxLayout()
        self.grid_res_spin = QtWidgets.QSpinBox()
        self.grid_res_spin.setRange(40, 180)
        self.grid_res_spin.setValue(96)
        self.grid_res_spin.setPrefix("Grid ")
        self.grid_res_spin.setSuffix("^3")

        self.outer_radius_spin = QtWidgets.QDoubleSpinBox()
        self.outer_radius_spin.setRange(1.01, 5.0)
        self.outer_radius_spin.setSingleStep(0.02)
        self.outer_radius_spin.setValue(1.25)
        self.outer_radius_spin.setPrefix("R ")
        generation_row.addWidget(self.grid_res_spin)
        generation_row.addWidget(self.outer_radius_spin)
        controls.addLayout(generation_row)

        self.generate_button = QtWidgets.QPushButton("Generate 3D Shape")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.clicked.connect(self._generate_shape)
        controls.addWidget(self.generate_button)

        sim_row = QtWidgets.QHBoxLayout()
        self.simulate_button = QtWidgets.QPushButton("Simulate")
        self.simulate_button.setMinimumHeight(36)
        self.simulate_button.setVisible(False)
        self.simulate_button.setEnabled(False)
        self.simulate_button.clicked.connect(self._simulate_last_shape)
        sim_row.addWidget(self.simulate_button, stretch=1)

        self.sim_cycles_spin = QtWidgets.QSpinBox()
        self.sim_cycles_spin.setRange(1, 50)
        self.sim_cycles_spin.setValue(1)
        self.sim_cycles_spin.setPrefix("Cycles ")
        self.sim_cycles_spin.setToolTip(
            "Number of full trajectory cycles to roll during the simulation."
        )
        sim_row.addWidget(self.sim_cycles_spin)
        controls.addLayout(sim_row)

        self.export_stl_button = QtWidgets.QPushButton("Export STL")
        self.export_stl_button.setMinimumHeight(36)
        self.export_stl_button.setEnabled(False)
        self.export_stl_button.clicked.connect(self._export_last_shape_stl)
        controls.addWidget(self.export_stl_button)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumBlockCount(120)
        self.status_box.setMinimumHeight(110)
        controls.addWidget(self.status_box)

        layout.addLayout(controls)
        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        self.wireframe_checkbox = QtWidgets.QCheckBox("Wireframe")
        self.wireframe_checkbox.toggled.connect(self.viewer.set_wireframe)
        self.reset_view_button = QtWidgets.QPushButton("Reset View")
        self.fit_view_button = QtWidgets.QPushButton("Fit-to-Screen")
        self.reset_view_button.clicked.connect(self.viewer.reset_view)
        self.fit_view_button.clicked.connect(self.viewer.fit_to_screen)
        controls.addWidget(self.wireframe_checkbox)
        controls.addWidget(self.reset_view_button)
        controls.addWidget(self.fit_view_button)
        controls.addStretch(1)

        layout.addLayout(controls)
        layout.addWidget(self.viewer, stretch=1)
        return panel

    def _build_sim_lock_widget_list(self) -> None:
        self._sim_lock_widgets = [
            self.editor,
            *self._tool_buttons.values(),
            self.undo_button,
            self.redo_button,
            self.clear_button,
            self.smooth_on_draw_checkbox,
            self.closed_checkbox,
            self.smooth_button,
            self.resample_points_spin,
            self.resample_button,
            self.scale_spin,
            self.scale_button,
            self.rotate_spin,
            self.rotate_button,
            self.dx_spin,
            self.dy_spin,
            self.translate_button,
            self.curvature_button,
            self.grid_res_spin,
            self.outer_radius_spin,
            self.generate_button,
            self.sim_cycles_spin,
            self.export_stl_button,
            self.wireframe_checkbox,
            self.reset_view_button,
            self.fit_view_button,
        ]

    def _setup_ui_scaling_controls(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self._zoom_in_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.ZoomIn, self)
        self._zoom_out_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.ZoomOut, self)
        self._zoom_in_shortcut.activated.connect(lambda: self._scale_ui_by(1.10))
        self._zoom_out_shortcut.activated.connect(lambda: self._scale_ui_by(1.0 / 1.10))

        self._zoom_reset_shortcuts = []
        for sequence in ("Ctrl+0", "Meta+0"):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.activated.connect(self._reset_ui_scale)
            self._zoom_reset_shortcuts.append(shortcut)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Wheel:
            modifiers = event.modifiers() if hasattr(event, "modifiers") else QtCore.Qt.KeyboardModifier.NoModifier
            zoom_modifiers = (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
            if modifiers & zoom_modifiers:
                delta = 0
                if hasattr(event, "angleDelta"):
                    delta = int(event.angleDelta().y())
                if delta == 0 and hasattr(event, "pixelDelta"):
                    delta = int(event.pixelDelta().y())
                if delta != 0:
                    self._scale_ui_by(1.08 if delta > 0 else 1.0 / 1.08)
                    if hasattr(event, "accept"):
                        event.accept()
                    return True
        return super().eventFilter(watched, event)

    def _scale_ui_by(self, factor: float) -> None:
        self._set_ui_scale(self._ui_scale * float(factor))

    def _reset_ui_scale(self) -> None:
        self._set_ui_scale(1.0)

    def _set_ui_scale(self, requested_scale: float) -> None:
        scale = float(np.clip(float(requested_scale), self._ui_scale_min, self._ui_scale_max))
        if abs(scale - self._ui_scale) < 1e-6:
            return
        self._ui_scale = scale
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        font = QtGui.QFont(app.font())
        font.setPointSizeF(self._base_font_point_size * self._ui_scale)
        app.setFont(font)
        self.statusBar().showMessage(f"UI scale: {self._ui_scale * 100.0:.0f}%", 1400)

    def _set_simulation_ui_locked(self, locked: bool) -> None:
        for widget in self._sim_lock_widgets:
            widget.setEnabled(not locked)
        has_generation = self._last_generation_result is not None
        if locked:
            self.simulate_button.setEnabled(False)
            self.export_stl_button.setEnabled(False)
        else:
            self.simulate_button.setEnabled(has_generation)
            self.export_stl_button.setEnabled(has_generation)

    def _set_tool(self, tool: str) -> None:
        self.editor.set_tool(tool)
        for name, button in self._tool_buttons.items():
            button.blockSignals(True)
            button.setChecked(name == tool)
            button.blockSignals(False)

    def _append_status(self, text: str) -> None:
        self.status_box.appendPlainText(text)

    def _update_curve_stats(self) -> None:
        length = self.editor.curve_length()
        control_points = self.editor.points().shape[0]
        self.length_label.setText(f"Length: {length:.2f} px | control pts: {control_points}")

    def _show_curvature_plot(self) -> None:
        import matplotlib.pyplot as plt
        from trajectoids_adapter import curvature_profile

        curve = self.editor.sampled_points(n_samples=700)
        if curve.shape[0] < 3:
            QtWidgets.QMessageBox.information(
                self,
                "Curvature",
                "Draw or resample a curve first.",
            )
            return
        curve_math = curve.copy()
        curve_math[:, 1] *= -1.0
        s, kappa = curvature_profile(curve_math)
        if s.size == 0:
            QtWidgets.QMessageBox.information(self, "Curvature", "Need at least 3 points.")
            return
        fig = plt.figure("Curvature Profile", figsize=(8, 3))
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(s, kappa, color="#1f5fa8", linewidth=1.5)
        ax.set_xlabel("Arc length")
        ax.set_ylabel("Curvature")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()

    def _generate_shape(self) -> None:
        from trajectoids_adapter import generate_trajectoid_mesh

        if self.viewer.is_simulation_running:
            self.viewer.stop_simulation(show_trajectory=False, interrupted=True)

        curve = self.editor.sampled_points(n_samples=max(600, self.resample_points_spin.value() * 2))
        if curve.shape[0] < 8:
            QtWidgets.QMessageBox.warning(
                self,
                "Generation failed",
                "Curve has too few points. Draw more points or apply resampling.",
            )
            return

        # Convert from screen coordinates (Y down) to mathematical coordinates (Y up).
        curve_math = curve.copy()
        curve_math[:, 1] *= -1.0

        self._append_status("Generating mesh...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            result = generate_trajectoid_mesh(
                curve_math,
                require_closed=self.closed_checkbox.isChecked(),
                smooth_passes=1 if self.smooth_on_draw_checkbox.isChecked() else 0,
                resample_points=self.resample_points_spin.value(),
                resolution=self.grid_res_spin.value(),
                core_radius=1.0,
                outer_radius=self.outer_radius_spin.value(),
            )
        except ValueError as exc:
            message = str(exc).strip() or "Unknown validation failure."
            QtWidgets.QMessageBox.warning(self, "Generation failed", message)
            self._append_status(f"Generation failed: {message}")
            return
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            message = (
                f"Unexpected error while generating mesh:\n{exc}\n\n"
                "Try smoothing, resampling, reducing grid size, or simplifying the curve."
            )
            QtWidgets.QMessageBox.critical(self, "Generation error", message)
            self._append_status(f"Generation error: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._last_generation_result = result
        self._last_closed_mode = bool(self.closed_checkbox.isChecked())

        self.viewer.set_mesh(result.vertices, result.faces)
        if hasattr(self.viewer, "set_contact_curve"):
            self.viewer.set_contact_curve(result.surface_contact_curve)
        self.viewer.set_wireframe(self.wireframe_checkbox.isChecked())

        self.simulate_button.setVisible(True)
        self.simulate_button.setEnabled(True)
        if self._has_simulated_once:
            self.simulate_button.setText("Replay Simulation")
        else:
            self.simulate_button.setText("Simulate")
        self.export_stl_button.setEnabled(True)

        self._append_status(
            "Generated mesh: "
            f"{result.vertices.shape[0]} vertices, "
            f"{result.faces.shape[0]} triangles, "
            f"scale={result.scale:.5f}, "
            f"mismatch={np.rad2deg(result.mismatch_angle):.2f} deg, "
            f"endpoint gap={result.endpoint_gap:.5f}"
        )
        self._append_status("Ready to simulate. Click Simulate to animate one full trajectory.")

    def _export_last_shape_stl(self) -> None:
        from trajectoids_adapter import export_binary_stl

        if self._last_generation_result is None:
            QtWidgets.QMessageBox.information(
                self,
                "No mesh to export",
                "Generate a 3D shape before exporting STL.",
            )
            return

        base_dir = self._last_export_dir if self._last_export_dir.exists() else Path.home()
        default_path = base_dir / "trajectoid.stl"
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export STL",
            str(default_path),
            "STL Files (*.stl)",
        )
        if not selected_path:
            return

        output_path = Path(selected_path).expanduser()
        if output_path.suffix.lower() != ".stl":
            output_path = output_path.with_suffix(".stl")

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            saved_path = export_binary_stl(
                vertices=self._last_generation_result.vertices,
                faces=self._last_generation_result.faces,
                output_path=output_path,
                solid_name="trajectoid",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "STL export failed", f"Could not export STL:\n{exc}")
            self._append_status(f"STL export failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._last_export_dir = saved_path.parent
        self._append_status(
            f"Exported binary STL: {saved_path} "
            f"({self._last_generation_result.faces.shape[0]} triangles)."
        )

    def _simulate_last_shape(self) -> None:
        from trajectoids_adapter import build_roll_simulation, path_length

        if self._last_generation_result is None:
            return
        if self.viewer.is_simulation_running:
            return

        path_xy = self._last_generation_result.resampled_points
        if self._last_closed_mode and np.linalg.norm(path_xy[0] - path_xy[-1]) > 1e-9:
            path_for_length = np.vstack([path_xy, path_xy[0]])
        else:
            path_for_length = path_xy
        trajectory_length = path_length(path_for_length)
        n_cycles = max(1, int(self.sim_cycles_spin.value()))
        target_roll_angle_rad = (trajectory_length / 1.0) * n_cycles

        self._append_status(
            f"Preparing simulation ({n_cycles} cycle{'s' if n_cycles != 1 else ''})..."
        )
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            sim_result = build_roll_simulation(
                path_xy,
                target_roll_angle_rad=target_roll_angle_rad,
                closed=self._last_closed_mode,
                n_frames=max(240, 200 * n_cycles),
                core_radius=1.0,
            )
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            message = (
                f"Could not build simulation:\n{exc}\n\n"
                "Try smoothing/resampling the curve and generating again."
            )
            QtWidgets.QMessageBox.warning(self, "Simulation error", message)
            self._append_status(f"Simulation setup failed: {exc}")
            return

        self._last_simulation_result = sim_result
        try:
            started = self.viewer.start_simulation(
                vertices=self._last_generation_result.vertices,
                faces=self._last_generation_result.faces,
                translations_xyz=sim_result.translations_xyz,
                rotations=sim_result.rotations,
                trajectory_xy=sim_result.trajectory_xy,
                wireframe=self.wireframe_checkbox.isChecked(),
                duration_seconds=8.0 * n_cycles,
                message=sim_result.message,
                # Total wall-clock = 8s per cycle × N. n_frames scales sub-linearly
                # (max(240, 120·N)) so each frame advances roughly one cycle worth
                # of arc per ~8s, giving the visual "N laps in 8N seconds" feel.
                completed_target=sim_result.completed_target,
            )
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Simulation error", f"Simulation failed: {exc}")
            self._append_status(f"Simulation failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if started:
            self._append_status("Simulation started: camera locked to 45 degrees, zoomed out.")

    def _on_simulation_started(self) -> None:
        self._set_simulation_ui_locked(True)

    def _on_simulation_finished(self, completed_target: bool, message: str) -> None:
        self._set_simulation_ui_locked(False)
        self.simulate_button.setVisible(True)
        self.simulate_button.setEnabled(self._last_generation_result is not None)
        self.simulate_button.setText("Replay Simulation")
        self._has_simulated_once = True

        if self._last_simulation_result is not None:
            achieved_deg = np.rad2deg(self._last_simulation_result.achieved_roll_angle_rad)
            self._append_status(
                f"Simulation complete: achieved rolling angle {achieved_deg:.2f} deg."
            )

        if completed_target:
            if message:
                self._append_status(message)
            self._append_status("Full trajectory displayed.")
        else:
            self._append_status(message or "Simulation finished before full trajectory display.")


def main() -> int:
    try:
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_UseDesktopOpenGL,
            True,
        )
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
