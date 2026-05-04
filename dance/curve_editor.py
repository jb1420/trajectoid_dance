from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from trajectoids_adapter import path_length, resample_uniform, smooth_path

try:
    from scipy.interpolate import splprep, splev
except Exception:  # pragma: no cover - optional fallback
    splprep = None
    splev = None


class Tool:
    FREEHAND = "freehand"
    BEZIER = "bezier"
    POLYLINE = "polyline"
    ERASER = "eraser"
    SELECT = "select"


class CurveEditorWidget(QtWidgets.QWidget):
    curveChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Keep a usable canvas while allowing the overall GUI to scale down freely.
        self.setMinimumSize(260, 260)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self._tool = Tool.FREEHAND
        self._curve_kind = Tool.FREEHAND
        self._points = np.empty((0, 2), dtype=float)

        self._history: list[Tuple[np.ndarray, str, bool]] = []
        self._redo: list[Tuple[np.ndarray, str, bool]] = []

        self._drawing = False
        self._polyline_active = False
        self._bezier_active = False
        self._drag_index = -1
        self._hover_pos: Optional[np.ndarray] = None
        self._smooth_on_draw = False
        self._closed_hint = True
        self._sample_cache: dict[int, np.ndarray] = {}

    def set_closed_hint(self, value: bool) -> None:
        value = bool(value)
        if self._closed_hint == value:
            return
        self._closed_hint = value
        self._invalidate_sample_cache()
        self.update()
        self.curveChanged.emit()

    def set_smooth_on_draw(self, value: bool) -> None:
        self._smooth_on_draw = bool(value)

    def set_tool(self, tool: str) -> None:
        self._tool = tool

    def points(self) -> np.ndarray:
        return self._points.copy()

    def sampled_points(self, n_samples: int = 500) -> np.ndarray:
        return self._sampled_points_cached(n_samples=n_samples, copy_points=True)

    def curve_length(self) -> float:
        pts = self.sampled_points(n_samples=700)
        return path_length(pts)

    def clear_curve(self) -> None:
        if self._points.shape[0] == 0:
            return
        self._push_history()
        self._points = np.empty((0, 2), dtype=float)
        self._invalidate_sample_cache()
        self._polyline_active = False
        self._bezier_active = False
        self._drag_index = -1
        self.update()
        self.curveChanged.emit()

    def undo(self) -> None:
        if not self._history:
            return
        self._redo.append((self._points.copy(), self._curve_kind, self._closed_hint))
        points, kind, closed = self._history.pop()
        self._points = points
        self._curve_kind = kind
        self._closed_hint = closed
        self._invalidate_sample_cache()
        self._polyline_active = False
        self._bezier_active = False
        self._drag_index = -1
        self.update()
        self.curveChanged.emit()

    def redo(self) -> None:
        if not self._redo:
            return
        self._history.append((self._points.copy(), self._curve_kind, self._closed_hint))
        points, kind, closed = self._redo.pop()
        self._points = points
        self._curve_kind = kind
        self._closed_hint = closed
        self._invalidate_sample_cache()
        self._polyline_active = False
        self._bezier_active = False
        self._drag_index = -1
        self.update()
        self.curveChanged.emit()

    def apply_smooth(self, passes: int = 1) -> None:
        if self._points.shape[0] < 3:
            return
        self._push_history()
        self._points = smooth_path(self._points, passes=passes, closed=self._closed_hint)
        self._invalidate_sample_cache()
        self.update()
        self.curveChanged.emit()

    def apply_resample(self, n_points: int = 240) -> None:
        pts = self.sampled_points(n_samples=max(n_points, 50))
        if pts.shape[0] < 2:
            return
        self._push_history()
        self._points = resample_uniform(pts, n_points=n_points, closed=self._closed_hint)
        self._curve_kind = Tool.POLYLINE
        self._invalidate_sample_cache()
        self.update()
        self.curveChanged.emit()

    def apply_scale(self, factor: float) -> None:
        if self._points.shape[0] < 2:
            return
        self._push_history()
        center = np.mean(self._points, axis=0, keepdims=True)
        self._points = center + (self._points - center) * float(factor)
        self._invalidate_sample_cache()
        self.update()
        self.curveChanged.emit()

    def apply_rotate(self, angle_degrees: float) -> None:
        if self._points.shape[0] < 2:
            return
        self._push_history()
        center = np.mean(self._points, axis=0, keepdims=True)
        theta = np.deg2rad(float(angle_degrees))
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        self._points = center + (self._points - center) @ rot.T
        self._invalidate_sample_cache()
        self.update()
        self.curveChanged.emit()

    def apply_translate(self, dx: float, dy: float) -> None:
        if self._points.shape[0] < 1:
            return
        self._push_history()
        self._points = self._points + np.array([dx, dy], dtype=float)
        self._invalidate_sample_cache()
        self.update()
        self.curveChanged.emit()

    def _sample_bezier_spline(self, n_samples: int = 500) -> np.ndarray:
        pts = self._points
        if pts.shape[0] < 3:
            return pts.copy()
        if splprep is None or splev is None:
            return self._sample_bezier_fallback(pts, n_samples=n_samples)
        k = min(3, pts.shape[0] - 1)
        x = pts[:, 0]
        y = pts[:, 1]
        per = int(self._closed_hint)
        try:
            tck, _ = splprep([x, y], s=0.0, k=k, per=per)
            u_new = np.linspace(0.0, 1.0, max(80, n_samples))
            x_new, y_new = splev(u_new, tck)
            return np.column_stack([x_new, y_new])
        except Exception:
            return self._sample_bezier_fallback(pts, n_samples=n_samples)

    def _sample_bezier_fallback(self, pts: np.ndarray, n_samples: int) -> np.ndarray:
        # Keep editing responsive even when spline fitting fails on degenerate control points.
        n = max(pts.shape[0] * 3, n_samples // 4)
        smoothed = smooth_path(pts, passes=2, closed=self._closed_hint)
        return resample_uniform(smoothed, n_points=n, closed=self._closed_hint)

    def _sampled_points_cached(self, n_samples: int, copy_points: bool) -> np.ndarray:
        if self._points.shape[0] < 2:
            return self._points.copy() if copy_points else self._points
        if self._curve_kind != Tool.BEZIER:
            return self._points.copy() if copy_points else self._points
        key = max(2, int(n_samples))
        cached = self._sample_cache.get(key)
        if cached is None:
            cached = self._sample_bezier_spline(n_samples=key)
            self._sample_cache[key] = cached
        return cached.copy() if copy_points else cached

    def _invalidate_sample_cache(self) -> None:
        self._sample_cache.clear()

    def _push_history(self) -> None:
        self._history.append((self._points.copy(), self._curve_kind, self._closed_hint))
        if len(self._history) > 200:
            self._history.pop(0)
        self._redo.clear()

    def _mouse_xy(self, event: QtGui.QMouseEvent) -> np.ndarray:
        p = event.position()
        return np.array([p.x(), p.y()], dtype=float)

    def _append_point(self, point: np.ndarray) -> None:
        if self._points.shape[0] == 0:
            self._points = np.array([point], dtype=float)
        else:
            self._points = np.vstack([self._points, point])
        self._invalidate_sample_cache()

    def _nearest_point_index(self, point: np.ndarray, radius: float = 10.0) -> Optional[int]:
        if self._points.shape[0] == 0:
            return None
        d = np.linalg.norm(self._points - point, axis=1)
        idx = int(np.argmin(d))
        return idx if d[idx] <= radius else None

    def _erase_at(self, point: np.ndarray, radius: float = 14.0) -> bool:
        idx = self._nearest_point_index(point, radius=radius)
        if idx is None:
            return False
        if self._points.shape[0] <= 2:
            self._points = np.empty((0, 2), dtype=float)
        else:
            self._points = np.delete(self._points, idx, axis=0)
        self._invalidate_sample_cache()
        return True

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        point = self._mouse_xy(event)
        self._hover_pos = point
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if self._polyline_active:
                self._polyline_active = False
            if self._bezier_active:
                self._bezier_active = False
            self.update()
            return

        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        if self._tool == Tool.FREEHAND:
            self._push_history()
            self._curve_kind = Tool.FREEHAND
            self._points = np.array([point], dtype=float)
            self._invalidate_sample_cache()
            self._drawing = True
            self._polyline_active = False
            self._bezier_active = False
            self.curveChanged.emit()
            self.update()
            return

        if self._tool == Tool.POLYLINE:
            if not self._polyline_active:
                self._push_history()
                self._curve_kind = Tool.POLYLINE
                self._points = np.array([point], dtype=float)
                self._invalidate_sample_cache()
                self._polyline_active = True
                self._bezier_active = False
            else:
                self._append_point(point)
            self.curveChanged.emit()
            self.update()
            return

        if self._tool == Tool.BEZIER:
            if not self._bezier_active:
                self._push_history()
                self._curve_kind = Tool.BEZIER
                self._points = np.array([point], dtype=float)
                self._invalidate_sample_cache()
                self._bezier_active = True
                self._polyline_active = False
            else:
                self._append_point(point)
            self.curveChanged.emit()
            self.update()
            return

        if self._tool == Tool.ERASER:
            self._push_history()
            if self._erase_at(point):
                self.curveChanged.emit()
                self.update()
            return

        if self._tool == Tool.SELECT:
            idx = self._nearest_point_index(point, radius=13.0)
            if idx is not None:
                self._push_history()
                self._drag_index = idx
                self.update()
            return

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        point = self._mouse_xy(event)
        self._hover_pos = point

        if self._tool == Tool.FREEHAND and self._drawing:
            if self._points.shape[0] == 0 or np.linalg.norm(point - self._points[-1]) > 1.6:
                self._append_point(point)
                self.curveChanged.emit()
                self.update()
            return

        if self._tool == Tool.SELECT and self._drag_index >= 0:
            self._points[self._drag_index] = point
            self._invalidate_sample_cache()
            self.curveChanged.emit()
            self.update()
            return

        if self._tool == Tool.ERASER and (
            event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            if self._erase_at(point):
                self.curveChanged.emit()
                self.update()
            return

        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if self._smooth_on_draw and self._points.shape[0] > 5:
                self._points = smooth_path(self._points, passes=1, closed=False)
                self._invalidate_sample_cache()
            self.curveChanged.emit()
            self.update()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drag_index >= 0:
            self._drag_index = -1
            self.curveChanged.emit()
            self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        point = self._mouse_xy(event)
        if self._tool == Tool.POLYLINE and self._polyline_active:
            self._append_point(point)
            self._polyline_active = False
            self.curveChanged.emit()
            self.update()
            return
        if self._tool == Tool.BEZIER and self._bezier_active:
            self._append_point(point)
            self._bezier_active = False
            self.curveChanged.emit()
            self.update()
            return

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if self._polyline_active or self._bezier_active:
                self._polyline_active = False
                self._bezier_active = False
                self.update()
                self.curveChanged.emit()
            return
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._polyline_active = False
            self._bezier_active = False
            self._drag_index = -1
            self.update()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#ffffff"))
        self._draw_grid(painter)
        self._draw_curve(painter)
        self._draw_control_points(painter)
        self._draw_polyline_preview(painter)
        painter.end()

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        spacing = 40
        pen = QtGui.QPen(QtGui.QColor("#f2f3f5"), 1)
        painter.setPen(pen)
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)

    def _draw_curve(self, painter: QtGui.QPainter) -> None:
        sampled = self._sampled_points_cached(n_samples=700, copy_points=False)
        if sampled.shape[0] < 2:
            return
        curve_pen = QtGui.QPen(QtGui.QColor("#1f5fa8"), 2.4)
        painter.setPen(curve_pen)
        path = QtGui.QPainterPath(QtCore.QPointF(sampled[0, 0], sampled[0, 1]))
        for i in range(1, sampled.shape[0]):
            path.lineTo(sampled[i, 0], sampled[i, 1])
        if self._closed_hint and not self._polyline_active and not self._bezier_active:
            path.lineTo(sampled[0, 0], sampled[0, 1])
        painter.drawPath(path)

        if self._curve_kind == Tool.BEZIER and self._points.shape[0] > 1:
            painter.setPen(QtGui.QPen(QtGui.QColor("#8d9aa5"), 1.2, QtCore.Qt.PenStyle.DashLine))
            cpath = QtGui.QPainterPath(QtCore.QPointF(self._points[0, 0], self._points[0, 1]))
            for i in range(1, self._points.shape[0]):
                cpath.lineTo(self._points[i, 0], self._points[i, 1])
            painter.drawPath(cpath)

    def _draw_control_points(self, painter: QtGui.QPainter) -> None:
        if self._points.shape[0] == 0:
            return
        for i, p in enumerate(self._points):
            if i == self._drag_index:
                color = QtGui.QColor("#c0342b")
                r = 5.8
            else:
                color = QtGui.QColor("#2f3f4f")
                r = 4.3
            painter.setBrush(color)
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
            painter.drawEllipse(QtCore.QPointF(p[0], p[1]), r, r)

    def _draw_polyline_preview(self, painter: QtGui.QPainter) -> None:
        if self._hover_pos is None:
            return
        if self._polyline_active and self._points.shape[0] > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor("#8694a2"), 1.1, QtCore.Qt.PenStyle.DashLine))
            p = self._points[-1]
            h = self._hover_pos
            painter.drawLine(
                QtCore.QPointF(float(p[0]), float(p[1])),
                QtCore.QPointF(float(h[0]), float(h[1])),
            )
        if self._bezier_active and self._points.shape[0] > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor("#8694a2"), 1.1, QtCore.Qt.PenStyle.DotLine))
            p = self._points[-1]
            h = self._hover_pos
            painter.drawLine(
                QtCore.QPointF(float(p[0]), float(p[1])),
                QtCore.QPointF(float(h[0]), float(h[1])),
            )
