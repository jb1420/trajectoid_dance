"""2D top-down layout canvas for the dance scene.

Displays each dancer's input ``curve_xy`` (translated by ``start_offset_xy``)
on a pan/zoomable world plane. Lets the user

* drag dancers around    → updates ``start_offset_xy`` only (no regen)
* rotate a dancer        → rotates ``curve_xy`` about its centroid
* scale a dancer         → uniform scale of ``curve_xy`` about its centroid

Rotation and scale invalidate the cached ``gen_result`` / ``sim_result`` so the
user must click *Generate Mesh* to apply the changes to the rolling body. The
canvas marks such "dirty" dancers with a dashed outline.

World coordinates are math-style y-up; Qt screen coordinates are y-down. All
conversions go through ``_world_to_screen_xy`` / ``_screen_to_world``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from dancer import Dancer


# Screen pixel sizes for transform-gizmo handles (independent of zoom).
_HANDLE_HALF = 5
_ROTATE_HANDLE_OFFSET = 24
_ROTATE_HANDLE_RADIUS = 6
_HIT_PIXEL_TOLERANCE = 6


@dataclass
class _DragInitial:
    start_offset: tuple[float, float]
    curve_xy: np.ndarray  # copy taken at drag begin


class LayoutCanvasWidget(QtWidgets.QWidget):
    """2D top-down editor for dancer positions and curve transforms."""

    dancerTranslated = QtCore.Signal(str)        # dancer_id
    dancerCurveModified = QtCore.Signal(str)     # dancer_id (curve_xy changed, regen needed)
    selectionChanged = QtCore.Signal(list)       # list[str] of dancer_ids

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._dancers: list[Dancer] = []
        self._selected_ids: set[str] = set()

        # View transform: world point W maps to screen point
        #   sx = w/2 + (W.x - center.x) * scale
        #   sy = h/2 - (W.y - center.y) * scale     (y flipped vs Qt)
        self._view_scale: float = 40.0  # pixels per world unit
        self._view_center: np.ndarray = np.zeros(2, dtype=float)

        # Drag state machine
        self._drag_mode: Optional[str] = None  # "translate" | "rotate" | "scale" | "pan"
        self._drag_dancer_id: Optional[str] = None  # rotate/scale single target
        self._drag_initial: dict[str, _DragInitial] = {}
        self._drag_anchor_world: Optional[np.ndarray] = None
        self._drag_initial_pan_center: Optional[np.ndarray] = None
        self._drag_initial_screen: Optional[QtCore.QPoint] = None
        self._drag_init_dist: float = 0.0
        self._drag_init_angle: float = 0.0

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(220)
        self.setAutoFillBackground(False)

    # -- public API ---------------------------------------------------------

    def set_dancers(self, dancers: list[Dancer]) -> None:
        self._dancers = list(dancers)
        existing_ids = {d.dancer_id for d in self._dancers}
        self._selected_ids &= existing_ids
        self.update()

    def set_selection(self, ids: Iterable[str]) -> None:
        new_ids = set(ids)
        if new_ids == self._selected_ids:
            return
        self._selected_ids = new_ids
        self.update()

    def refresh(self) -> None:
        self.update()

    def fit_to_scene(self) -> None:
        if not self._dancers:
            self._view_center = np.zeros(2)
            self._view_scale = 40.0
            self.update()
            return
        merged_pts = [
            d.curve_xy + np.asarray(d.start_offset_xy, dtype=float)
            for d in self._dancers
            if d.curve_xy.shape[0] >= 2
        ]
        if not merged_pts:
            self._view_center = np.zeros(2)
            self._view_scale = 40.0
            self.update()
            return
        merged = np.vstack(merged_pts)
        lo = merged.min(axis=0)
        hi = merged.max(axis=0)
        span = hi - lo
        self._view_center = 0.5 * (lo + hi)
        avail_w = max(self.width(), 100) * 0.8
        avail_h = max(self.height(), 100) * 0.8
        if span[0] > 0 and span[1] > 0:
            self._view_scale = float(min(avail_w / span[0], avail_h / span[1]))
        elif span[0] > 0:
            self._view_scale = float(avail_w / span[0])
        elif span[1] > 0:
            self._view_scale = float(avail_h / span[1])
        else:
            self._view_scale = 40.0
        self._view_scale = float(np.clip(self._view_scale, 4.0, 400.0))
        self.update()

    # -- coordinate helpers -------------------------------------------------

    def _world_to_screen_xy(self, world_xy: np.ndarray) -> np.ndarray:
        """Vectorized world → screen. ``world_xy`` may be (N,2) or (2,)."""
        w = self.width()
        h = self.height()
        x = w / 2 + (world_xy[..., 0] - self._view_center[0]) * self._view_scale
        y = h / 2 - (world_xy[..., 1] - self._view_center[1]) * self._view_scale
        return np.stack([x, y], axis=-1)

    def _screen_to_world(self, sx: float, sy: float) -> np.ndarray:
        w = self.width()
        h = self.height()
        wx = self._view_center[0] + (sx - w / 2) / self._view_scale
        wy = self._view_center[1] - (sy - h / 2) / self._view_scale
        return np.array([wx, wy])

    # -- paint --------------------------------------------------------------

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#1f2329"))
        self._draw_grid(painter)
        for d in self._dancers:
            self._draw_dancer(painter, d)
        for d in self._dancers:
            if d.dancer_id in self._selected_ids:
                self._draw_handles(painter, d)
        self._draw_status(painter)
        painter.end()

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        w = self.width()
        h = self.height()
        # Determine world-space corners and maximum radius from origin
        corners = [
            self._screen_to_world(0, 0),
            self._screen_to_world(w, 0),
            self._screen_to_world(w, h),
            self._screen_to_world(0, h),
        ]
        max_r = max(float(np.linalg.norm(c)) for c in corners) * 1.05

        # Choose a world-unit spacing that aims for ~40 pixels between concentric
        # circles, then snap to a "nice" step (1,2,5 × 10^k).
        target_px = 40.0
        world_spacing = max(0.01, target_px / max(1.0, self._view_scale))

        def _nice_step(s: float) -> float:
            exp = np.floor(np.log10(s))
            base = s / (10 ** exp)
            if base <= 1.5:
                b = 1.0
            elif base <= 3.5:
                b = 2.0
            elif base <= 7.5:
                b = 5.0
            else:
                b = 10.0
            return float(b * 10 ** exp)

        spacing = _nice_step(world_spacing)
        if spacing <= 0:
            spacing = 1.0

        radii = np.arange(spacing, max_r + spacing, spacing)

        pen_grid = QtGui.QPen(QtGui.QColor("#2c3138"), 1)
        pen_grid.setCosmetic(True)
        painter.setPen(pen_grid)

        origin_screen = self._world_to_screen_xy(np.array([0.0, 0.0]))
        cx, cy = float(origin_screen[0]), float(origin_screen[1])

        # Draw concentric circles (polar grid)
        for r in radii:
            pr = r * self._view_scale
            if pr < 1.0:
                continue
            painter.drawEllipse(QtCore.QPointF(cx, cy), pr, pr)

        # Draw radial lines. Angle density adapts with zoom; more detail when
        # zoomed in.
        angle_step_deg = 30
        if self._view_scale > 120:
            angle_step_deg = 15
        if self._view_scale > 320:
            angle_step_deg = 10
        n_angles = max(4, int(360 // angle_step_deg))
        for i in range(n_angles):
            theta = 2.0 * np.pi * i / float(n_angles)
            ex = np.cos(theta) * max_r
            ey = np.sin(theta) * max_r
            p = self._world_to_screen_xy(np.array([ex, ey]))
            painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(float(p[0]), float(p[1])))

        # Origin crosshair / axes (stronger lines)
        pen_axis = QtGui.QPen(QtGui.QColor("#3a4a55"), 2)
        pen_axis.setCosmetic(True)
        painter.setPen(pen_axis)
        painter.drawLine(QtCore.QPointF(0, cy), QtCore.QPointF(w, cy))
        painter.drawLine(QtCore.QPointF(cx, 0), QtCore.QPointF(cx, h))

    def _draw_dancer(self, painter: QtGui.QPainter, d: Dancer) -> None:
        if d.curve_xy.shape[0] < 2:
            return
        offset = np.asarray(d.start_offset_xy, dtype=float)
        screen_pts = self._world_to_screen_xy(d.curve_xy + offset)
        is_selected = d.dancer_id in self._selected_ids
        needs_regen = d.gen_result is None
        pen = QtGui.QPen(QtGui.QColor(d.color_hex), 2.5 if is_selected else 1.6)
        if needs_regen:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        path = QtGui.QPainterPath()
        path.moveTo(screen_pts[0, 0], screen_pts[0, 1])
        for pt in screen_pts[1:]:
            path.lineTo(pt[0], pt[1])
        if d.closed:
            path.lineTo(screen_pts[0, 0], screen_pts[0, 1])
        painter.drawPath(path)
        # Name label near curve centroid (which equals start_offset_xy since
        # curve_xy is centered at the origin).
        center = self._world_to_screen_xy(offset)
        painter.setPen(QtGui.QColor("#ffffff"))
        suffix = " ⟳" if needs_regen else ""
        painter.drawText(QtCore.QPointF(center[0] + 6, center[1] - 6),
                         f"{d.name}{suffix}")

    def _draw_handles(self, painter: QtGui.QPainter, d: Dancer) -> None:
        if d.curve_xy.shape[0] < 2:
            return
        bbox = self._curve_screen_bbox(d)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        pen = QtGui.QPen(QtGui.QColor("#f5f6f8"), 1, QtCore.Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
        # Corner scale handles
        painter.setPen(QtGui.QPen(QtGui.QColor("#f5f6f8"), 1))
        painter.setBrush(QtGui.QColor("#2b6cb0"))
        for cx, cy in self._scale_handle_screen_positions(bbox):
            painter.drawRect(QtCore.QRectF(cx - _HANDLE_HALF, cy - _HANDLE_HALF,
                                           2 * _HANDLE_HALF, 2 * _HANDLE_HALF))
        # Tether + rotation handle above the top edge
        rx, ry = self._rotate_handle_screen_position(bbox)
        painter.setPen(QtGui.QPen(QtGui.QColor("#f5f6f8"), 1, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(QtCore.QPointF((x1 + x2) / 2, y1),
                         QtCore.QPointF(rx, ry))
        painter.setPen(QtGui.QPen(QtGui.QColor("#f5f6f8"), 1))
        painter.setBrush(QtGui.QColor("#38a169"))
        painter.drawEllipse(QtCore.QPointF(rx, ry),
                            _ROTATE_HANDLE_RADIUS, _ROTATE_HANDLE_RADIUS)

    def _curve_screen_bbox(
        self, d: Dancer
    ) -> Optional[tuple[float, float, float, float]]:
        if d.curve_xy.shape[0] < 2:
            return None
        offset = np.asarray(d.start_offset_xy, dtype=float)
        screen_pts = self._world_to_screen_xy(d.curve_xy + offset)
        x1, y1 = screen_pts.min(axis=0)
        x2, y2 = screen_pts.max(axis=0)
        pad = 4.0
        return (float(x1 - pad), float(y1 - pad),
                float(x2 + pad), float(y2 + pad))

    @staticmethod
    def _scale_handle_screen_positions(
        bbox: tuple[float, float, float, float]
    ) -> list[tuple[float, float]]:
        x1, y1, x2, y2 = bbox
        return [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    @staticmethod
    def _rotate_handle_screen_position(
        bbox: tuple[float, float, float, float]
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y1 - _ROTATE_HANDLE_OFFSET)

    def _draw_status(self, painter: QtGui.QPainter) -> None:
        painter.setPen(QtGui.QColor("#6b7280"))
        n_sel = len(self._selected_ids)
        sel_part = f"  |  selected: {n_sel}" if n_sel else ""
        msg = (f"zoom: {self._view_scale:.1f}px/u  |  "
               f"center: ({self._view_center[0]:+.2f}, {self._view_center[1]:+.2f})"
               f"{sel_part}  |  F: fit  |  wheel: zoom  |  MMB: pan")
        painter.drawText(QtCore.QPointF(8, self.height() - 8), msg)

    # -- hit testing --------------------------------------------------------

    def _hit_test(
        self, pos: QtCore.QPoint
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (dancer_id, kind) where kind ∈ {"curve","rotate","scale"} or (None, None)."""
        # Check handles on currently-selected dancers first (they're on top).
        for d in self._dancers:
            if d.dancer_id not in self._selected_ids:
                continue
            bbox = self._curve_screen_bbox(d)
            if bbox is None:
                continue
            rx, ry = self._rotate_handle_screen_position(bbox)
            if (pos.x() - rx) ** 2 + (pos.y() - ry) ** 2 \
                    <= (_ROTATE_HANDLE_RADIUS + 2) ** 2:
                return d.dancer_id, "rotate"
            for cx, cy in self._scale_handle_screen_positions(bbox):
                if (abs(pos.x() - cx) <= _HANDLE_HALF + 2
                        and abs(pos.y() - cy) <= _HANDLE_HALF + 2):
                    return d.dancer_id, "scale"
        # Curve: nearest within tolerance.
        best_id: Optional[str] = None
        best_dist = float("inf")
        for d in self._dancers:
            dist = self._curve_pixel_distance(d, pos)
            if dist < best_dist:
                best_id = d.dancer_id
                best_dist = dist
        if best_dist <= _HIT_PIXEL_TOLERANCE:
            return best_id, "curve"
        return None, None

    def _curve_pixel_distance(self, d: Dancer, pos: QtCore.QPoint) -> float:
        if d.curve_xy.shape[0] < 2:
            return float("inf")
        offset = np.asarray(d.start_offset_xy, dtype=float)
        screen_pts = self._world_to_screen_xy(d.curve_xy + offset)
        a = screen_pts[:-1]
        b = screen_pts[1:]
        if d.closed:
            a = np.vstack([a, screen_pts[-1:]])
            b = np.vstack([b, screen_pts[:1]])
        ab = b - a
        ap = np.array([pos.x(), pos.y()]) - a
        ab_len2 = (ab * ab).sum(axis=1)
        ab_len2 = np.where(ab_len2 == 0, 1e-12, ab_len2)
        t = np.clip((ap * ab).sum(axis=1) / ab_len2, 0, 1)
        closest = a + ab * t[:, None]
        diff = closest - np.array([pos.x(), pos.y()])
        return float(np.sqrt((diff * diff).sum(axis=1)).min())

    # -- mouse events -------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position().toPoint()
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._begin_pan(pos)
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        dancer_id, kind = self._hit_test(pos)
        ctrl = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if kind == "rotate" and dancer_id is not None:
            self._begin_rotate(dancer_id, pos)
        elif kind == "scale" and dancer_id is not None:
            self._begin_scale(dancer_id, pos)
        elif kind == "curve" and dancer_id is not None:
            self._handle_curve_click(dancer_id, ctrl)
            if dancer_id in self._selected_ids:
                self._begin_translate(pos)
        else:
            if not ctrl and self._selected_ids:
                self._selected_ids = set()
                self.selectionChanged.emit([])
                self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position().toPoint()
        if self._drag_mode == "translate":
            self._update_translate(pos)
        elif self._drag_mode == "rotate":
            self._update_rotate(pos)
        elif self._drag_mode == "scale":
            self._update_scale(pos)
        elif self._drag_mode == "pan":
            self._update_pan(pos)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # Rotation/scale fire on release so listeners can refresh editor info.
        if self._drag_mode in ("rotate", "scale") and self._drag_dancer_id:
            self.dancerCurveModified.emit(self._drag_dancer_id)
        # Translate emits live in _update_translate; nothing extra here.
        self._drag_mode = None
        self._drag_dancer_id = None
        self._drag_initial = {}
        self._drag_anchor_world = None
        self._drag_initial_pan_center = None
        self._drag_initial_screen = None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        pos = event.position()
        world_before = self._screen_to_world(pos.x(), pos.y())
        delta = event.angleDelta().y() / 120.0
        factor = 1.1 ** delta
        self._view_scale = float(np.clip(self._view_scale * factor, 4.0, 800.0))
        world_after = self._screen_to_world(pos.x(), pos.y())
        self._view_center += (world_before - world_after)
        self.update()

    def mouseDoubleClickEvent(self, _event: QtGui.QMouseEvent) -> None:
        self.fit_to_scene()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_F:
            self.fit_to_scene()
        else:
            super().keyPressEvent(event)

    # -- selection / drag begin / update ------------------------------------

    def _handle_curve_click(self, dancer_id: str, ctrl: bool) -> None:
        if ctrl:
            new_ids = set(self._selected_ids)
            if dancer_id in new_ids:
                new_ids.discard(dancer_id)
            else:
                new_ids.add(dancer_id)
            self._selected_ids = new_ids
            self.selectionChanged.emit(list(new_ids))
        elif dancer_id not in self._selected_ids:
            self._selected_ids = {dancer_id}
            self.selectionChanged.emit([dancer_id])
        self.update()

    def _begin_translate(self, pos: QtCore.QPoint) -> None:
        self._drag_mode = "translate"
        self._drag_anchor_world = self._screen_to_world(pos.x(), pos.y())
        self._drag_initial = {
            d.dancer_id: _DragInitial(
                start_offset=tuple(d.start_offset_xy),
                curve_xy=d.curve_xy.copy(),
            )
            for d in self._dancers
            if d.dancer_id in self._selected_ids
        }

    def _update_translate(self, pos: QtCore.QPoint) -> None:
        if self._drag_anchor_world is None:
            return
        current = self._screen_to_world(pos.x(), pos.y())
        delta = current - self._drag_anchor_world
        for d in self._dancers:
            init = self._drag_initial.get(d.dancer_id)
            if init is None:
                continue
            d.start_offset_xy = (
                init.start_offset[0] + float(delta[0]),
                init.start_offset[1] + float(delta[1]),
            )
            # Live mirror to 3D viewer (listener decides cost-tradeoff).
            self.dancerTranslated.emit(d.dancer_id)
        self.update()

    def _begin_rotate(self, dancer_id: str, pos: QtCore.QPoint) -> None:
        d = self._find(dancer_id)
        if d is None:
            return
        self._drag_mode = "rotate"
        self._drag_dancer_id = dancer_id
        self._drag_initial = {
            dancer_id: _DragInitial(
                start_offset=tuple(d.start_offset_xy),
                curve_xy=d.curve_xy.copy(),
            )
        }
        center = np.asarray(d.start_offset_xy, dtype=float)
        mouse_world = self._screen_to_world(pos.x(), pos.y())
        self._drag_init_angle = float(np.arctan2(
            mouse_world[1] - center[1], mouse_world[0] - center[0]
        ))

    def _update_rotate(self, pos: QtCore.QPoint) -> None:
        if self._drag_dancer_id is None:
            return
        d = self._find(self._drag_dancer_id)
        if d is None:
            return
        init = self._drag_initial[d.dancer_id]
        center = np.asarray(init.start_offset, dtype=float)
        mouse_world = self._screen_to_world(pos.x(), pos.y())
        cur_angle = float(np.arctan2(
            mouse_world[1] - center[1], mouse_world[0] - center[0]
        ))
        theta = cur_angle - self._drag_init_angle
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        d.curve_xy = init.curve_xy @ R.T
        d.gen_result = None
        d.sim_result = None
        self.update()

    def _begin_scale(self, dancer_id: str, pos: QtCore.QPoint) -> None:
        d = self._find(dancer_id)
        if d is None:
            return
        self._drag_mode = "scale"
        self._drag_dancer_id = dancer_id
        self._drag_initial = {
            dancer_id: _DragInitial(
                start_offset=tuple(d.start_offset_xy),
                curve_xy=d.curve_xy.copy(),
            )
        }
        center = np.asarray(d.start_offset_xy, dtype=float)
        mouse_world = self._screen_to_world(pos.x(), pos.y())
        self._drag_init_dist = max(
            float(np.linalg.norm(mouse_world - center)), 1e-6
        )

    def _update_scale(self, pos: QtCore.QPoint) -> None:
        if self._drag_dancer_id is None:
            return
        d = self._find(self._drag_dancer_id)
        if d is None:
            return
        init = self._drag_initial[d.dancer_id]
        center = np.asarray(init.start_offset, dtype=float)
        mouse_world = self._screen_to_world(pos.x(), pos.y())
        cur_dist = float(np.linalg.norm(mouse_world - center))
        factor = float(np.clip(cur_dist / self._drag_init_dist, 0.1, 10.0))
        d.curve_xy = init.curve_xy * factor
        d.gen_result = None
        d.sim_result = None
        self.update()

    def _begin_pan(self, pos: QtCore.QPoint) -> None:
        self._drag_mode = "pan"
        self._drag_initial_pan_center = self._view_center.copy()
        self._drag_initial_screen = pos

    def _update_pan(self, pos: QtCore.QPoint) -> None:
        if (self._drag_initial_pan_center is None
                or self._drag_initial_screen is None):
            return
        dx = pos.x() - self._drag_initial_screen.x()
        dy = pos.y() - self._drag_initial_screen.y()
        self._view_center = self._drag_initial_pan_center - np.array(
            [dx / self._view_scale, -dy / self._view_scale]
        )
        self.update()

    # -- misc ---------------------------------------------------------------

    def _find(self, dancer_id: str) -> Optional[Dancer]:
        for d in self._dancers:
            if d.dancer_id == dancer_id:
                return d
        return None
