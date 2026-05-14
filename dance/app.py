"""Multi-trajectoid dance choreography app.

Lets the user manage multiple trajectoid "dancers" (each with its own curve,
color, start position, phase offset, and speed) and watch them roll together
on a shared ground plane.

Run::

    cd c:/Users/jb142/Desktop/Code_Drive/DEV/SuYeah/dance
    python app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from scene_io import load_scene, save_scene

import numpy as np

# viewer.py sets up cache env vars; import it before the rest of Qt land.
from viewer import make_viewer, MESH_SHADERS

from PySide6 import QtCore, QtGui, QtWidgets

from curve_editor import CurveEditorWidget, Tool
from dancer import Dancer, DanceScene, generate_dancer
from layout_canvas import LayoutCanvasWidget
from presets import PRESET_LABELS, PRESETS_BY_KEY, Param, PresetSpec, get_preset
from trajectoids_adapter import export_binary_stl


FREEHAND_OPTION = "__freehand__"


def _color_swatch(hex_color: str, size: int = 14) -> QtGui.QIcon:
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtGui.QColor(hex_color))
    p = QtGui.QPainter(pix)
    p.setPen(QtGui.QPen(QtGui.QColor("#1f2329"), 1))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QtGui.QIcon(pix)


# ---------------------------------------------------------------------------
# Dancer editor panel
# ---------------------------------------------------------------------------

MIXED_SOURCE_OPTION = "__mixed__"


class DancerEditorPanel(QtWidgets.QWidget):
    dancerChanged = QtCore.Signal(str)             # dancer_id (motion-only update)
    generateRequested = QtCore.Signal(str)         # dancer_id (regenerate mesh)
    nameChanged = QtCore.Signal(str)               # dancer_id (refresh roster label)
    colorChanged = QtCore.Signal(str)              # dancer_id (refresh roster swatch)

    # Sentinel values used to represent "mixed" state across N selected dancers.
    # Each spinbox's minimum is set one step below its legitimate range and
    # QDoubleSpinBox.setSpecialValueText("—") renders that value as a dash.
    _START_X_SENTINEL = -50.5
    _START_Y_SENTINEL = -50.5
    _PHASE_SENTINEL = -0.05
    _SPEED_SENTINEL = 0.20
    _CYCLES_SENTINEL = 0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._dancers: list[Dancer] = []
        self._suppress_signals = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Header: dancer name + color
        header = QtWidgets.QHBoxLayout()
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("Dancer name")
        self._name_edit.editingFinished.connect(self._on_name_changed)
        header.addWidget(QtWidgets.QLabel("Name:"))
        header.addWidget(self._name_edit, stretch=1)
        self._color_btn = QtWidgets.QPushButton("Color")
        self._color_btn.clicked.connect(self._on_pick_color)
        header.addWidget(self._color_btn)
        layout.addLayout(header)

        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # ---- Curve tab ----
        curve_tab = QtWidgets.QWidget()
        ct_layout = QtWidgets.QVBoxLayout(curve_tab)
        ct_layout.setContentsMargins(4, 4, 4, 4)

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(QtWidgets.QLabel("Source:"))
        self._source_combo = QtWidgets.QComboBox()
        for key, label in PRESET_LABELS.items():
            self._source_combo.addItem(label, userData=f"preset:{key}")
        self._source_combo.addItem("Custom (draw your own)", userData=FREEHAND_OPTION)
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self._source_combo, stretch=1)
        ct_layout.addLayout(source_row)

        # Parameter panel — rebuilt dynamically per preset. Shown only when the
        # current source is a parametric preset (and either single-select or
        # all selected dancers share that source).
        self._param_group = QtWidgets.QGroupBox("Parameters")
        self._param_form = QtWidgets.QFormLayout(self._param_group)
        self._param_form.setContentsMargins(8, 8, 8, 8)
        self._param_widgets: dict[str, QtWidgets.QAbstractSpinBox] = {}
        ct_layout.addWidget(self._param_group)
        self._param_group.setVisible(False)

        # Tool palette — visible only in Freehand source mode. Wraps the
        # already-implemented tools in `curve_editor.Tool`.
        self._tool_group = QtWidgets.QGroupBox("Drawing tool")
        tool_row = QtWidgets.QHBoxLayout(self._tool_group)
        tool_row.setContentsMargins(6, 6, 6, 6)
        self._tool_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._tool_button_group = QtWidgets.QButtonGroup(self)
        self._tool_button_group.setExclusive(True)
        for label, tool_id in (
            ("Bezier",   Tool.BEZIER),
            ("Polyline", Tool.POLYLINE),
            ("Freehand", Tool.FREEHAND),
            ("Eraser",   Tool.ERASER),
            ("Select",   Tool.SELECT),
        ):
            btn = QtWidgets.QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, t=tool_id: self._on_tool_clicked(t))
            self._tool_buttons[tool_id] = btn
            self._tool_button_group.addButton(btn)
            tool_row.addWidget(btn)
        tool_row.addStretch(1)
        self._tool_buttons[Tool.BEZIER].setChecked(True)
        ct_layout.addWidget(self._tool_group)
        self._tool_group.setVisible(False)

        self._curve_editor = CurveEditorWidget()
        # Default to Bezier (control-point spline) — gives cleaner curves than freehand.
        self._curve_editor.set_tool(Tool.BEZIER)
        self._curve_editor.set_closed_hint(True)
        self._curve_editor.curveChanged.connect(self._on_freehand_curve_changed)
        self._curve_editor.setMinimumHeight(180)
        ct_layout.addWidget(self._curve_editor, stretch=1)

        ce_buttons = QtWidgets.QHBoxLayout()
        smooth_btn = QtWidgets.QPushButton("Smooth")
        smooth_btn.clicked.connect(lambda: self._curve_editor.apply_smooth(passes=1))
        resample_btn = QtWidgets.QPushButton("Resample")
        resample_btn.clicked.connect(lambda: self._curve_editor.apply_resample(n_points=240))
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._curve_editor.clear_curve)
        for b in (smooth_btn, resample_btn, clear_btn):
            ce_buttons.addWidget(b)
        ce_buttons.addStretch(1)
        ct_layout.addLayout(ce_buttons)

        self._generate_btn = QtWidgets.QPushButton("Generate Mesh")
        self._generate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._generate_btn.clicked.connect(self._on_generate_clicked)
        ct_layout.addWidget(self._generate_btn)

        self._tabs.addTab(curve_tab, "Curve")

        # ---- Motion tab ----
        motion_tab = QtWidgets.QWidget()
        m_layout = QtWidgets.QFormLayout(motion_tab)
        m_layout.setContentsMargins(8, 8, 8, 8)

        self._start_x_spin = QtWidgets.QDoubleSpinBox()
        self._start_x_spin.setRange(self._START_X_SENTINEL, 50.0)
        self._start_x_spin.setSingleStep(0.5)
        self._start_x_spin.setDecimals(2)
        self._start_x_spin.setSpecialValueText("—")
        self._start_x_spin.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Start X:", self._start_x_spin)

        self._start_y_spin = QtWidgets.QDoubleSpinBox()
        self._start_y_spin.setRange(self._START_Y_SENTINEL, 50.0)
        self._start_y_spin.setSingleStep(0.5)
        self._start_y_spin.setDecimals(2)
        self._start_y_spin.setSpecialValueText("—")
        self._start_y_spin.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Start Y:", self._start_y_spin)

        self._phase_slider = QtWidgets.QDoubleSpinBox()
        self._phase_slider.setRange(self._PHASE_SENTINEL, 0.999)
        self._phase_slider.setSingleStep(0.05)
        self._phase_slider.setDecimals(3)
        self._phase_slider.setSpecialValueText("—")
        self._phase_slider.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Phase offset:", self._phase_slider)

        self._speed_slider = QtWidgets.QDoubleSpinBox()
        self._speed_slider.setRange(self._SPEED_SENTINEL, 4.0)
        self._speed_slider.setSingleStep(0.05)
        self._speed_slider.setDecimals(2)
        self._speed_slider.setSpecialValueText("—")
        self._speed_slider.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Speed:", self._speed_slider)

        self._cycles_spin = QtWidgets.QSpinBox()
        self._cycles_spin.setRange(self._CYCLES_SENTINEL, 8)
        self._cycles_spin.setSpecialValueText("—")
        self._cycles_spin.valueChanged.connect(self._on_cycles_changed)
        m_layout.addRow("Cycles per build:", self._cycles_spin)

        self._tabs.addTab(motion_tab, "Motion")

        # ---- Layout tab (2D top-down canvas) ----
        layout_tab = QtWidgets.QWidget()
        lt_layout = QtWidgets.QVBoxLayout(layout_tab)
        lt_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = LayoutCanvasWidget()
        lt_layout.addWidget(self.canvas, stretch=1)
        layout_toolbar = QtWidgets.QHBoxLayout()
        fit_btn = QtWidgets.QPushButton("Fit view")
        fit_btn.clicked.connect(self.canvas.fit_to_scene)
        layout_toolbar.addWidget(fit_btn)
        layout_toolbar.addStretch(1)
        layout_toolbar.addWidget(QtWidgets.QLabel(
            "drag: move • corner: scale • top handle: rotate"
        ))
        lt_layout.addLayout(layout_toolbar)
        self._tabs.addTab(layout_tab, "Layout")

        # Status hint at bottom
        self._info_label = QtWidgets.QLabel("(no dancer selected)")
        self._info_label.setStyleSheet("color: #6b7280; font-style: italic;")
        layout.addWidget(self._info_label)

        self.setEnabled(False)

    # -- public API ----------------------------------------------------------

    def set_dancer(self, dancer: Optional[Dancer]) -> None:
        """Backwards-compatible single-target wrapper around set_dancers."""
        self.set_dancers([dancer] if dancer is not None else [])

    def set_dancers(self, dancers: list[Dancer]) -> None:
        self._dancers = list(dancers)
        # Mirror selection to the layout canvas. The canvas owns the full
        # roster separately (set via MainWindow on scene mutations); this only
        # syncs which dancers it should highlight as selected.
        self.canvas.set_selection([d.dancer_id for d in self._dancers])
        if not self._dancers:
            # Layout canvas stays interactive even when no dancer is selected
            # (user might want to click on a curve to pick one). Disable only
            # the per-dancer form fields by switching tabs' container off.
            self._name_edit.setEnabled(False)
            self._tabs.widget(0).setEnabled(False)   # Curve tab
            self._tabs.widget(1).setEnabled(False)   # Motion tab
            self._color_btn.setEnabled(False)
            self._update_section_visibility()  # hides param + tool groups
            self._info_label.setText("(no dancer selected)")
            return
        # Re-enable per-dancer form fields when something is selected.
        self._tabs.widget(0).setEnabled(True)
        self._tabs.widget(1).setEnabled(True)
        self._color_btn.setEnabled(True)
        self.setEnabled(True)
        self._suppress_signals = True
        try:
            if len(self._dancers) == 1:
                self._apply_single_state(self._dancers[0])
            else:
                self._apply_bulk_state(self._dancers)
        finally:
            self._suppress_signals = False

    def refresh_info(self) -> None:
        if not self._dancers:
            return
        if len(self._dancers) == 1:
            self._info_label.setText(self._build_info_text(self._dancers[0]))
        else:
            self._info_label.setText(self._build_info_text_bulk(self._dancers))

    # -- state application --------------------------------------------------

    def _apply_single_state(self, d: Dancer) -> None:
        self._name_edit.setEnabled(True)
        self._name_edit.setText(d.name)
        self._update_color_button(d.color_hex)
        # Source combo: ensure no leftover "(mixed)" entry
        self._remove_mixed_source()
        target_data = d.curve_source if d.curve_source.startswith("preset:") else FREEHAND_OPTION
        self._select_source_data(target_data)
        # Curve editor visibility
        is_freehand = d.curve_source.startswith("freehand")
        self._curve_editor.setVisible(is_freehand)
        if is_freehand and d.curve_xy.size > 0:
            # No public setter for restoring points; original behavior is to clear.
            self._curve_editor.clear_curve()
        # Parametric panel + tool palette
        self._rebuild_param_widgets()
        self._update_section_visibility()
        # Motion fields
        self._start_x_spin.setValue(float(d.start_offset_xy[0]))
        self._start_y_spin.setValue(float(d.start_offset_xy[1]))
        self._phase_slider.setValue(float(d.phase_offset))
        self._speed_slider.setValue(float(d.speed_multiplier))
        self._cycles_spin.setValue(int(d.n_cycles))
        # Buttons
        self._generate_btn.setText("Generate Mesh")
        self._info_label.setText(self._build_info_text(d))

    def _apply_bulk_state(self, dancers: list[Dancer]) -> None:
        n = len(dancers)
        # Name not bulk-editable (each dancer keeps its own).
        self._name_edit.setEnabled(False)
        self._name_edit.setText(f"({n} dancers)")
        # Color: show first dancer's color as button background; bulk-set on pick.
        self._update_color_button(dancers[0].color_hex)
        # Source combo
        sources = {d.curve_source for d in dancers}
        if len(sources) == 1:
            self._remove_mixed_source()
            src = next(iter(sources))
            target_data = src if src.startswith("preset:") else FREEHAND_OPTION
            self._select_source_data(target_data)
        else:
            self._add_mixed_source_if_needed()
            self._select_source_data(MIXED_SOURCE_OPTION)
        # Freehand canvas always hidden in bulk (per-dancer drawing isn't meaningful).
        self._curve_editor.setVisible(False)
        # Parametric panel: visible only when all bulk-selected share one
        # parametric preset. Values shown are the first dancer's; editing
        # applies to all selected dancers.
        self._rebuild_param_widgets()
        self._update_section_visibility()
        # Motion fields with mixed-state sentinels
        self._set_float_field(self._start_x_spin,
                              [d.start_offset_xy[0] for d in dancers],
                              self._START_X_SENTINEL)
        self._set_float_field(self._start_y_spin,
                              [d.start_offset_xy[1] for d in dancers],
                              self._START_Y_SENTINEL)
        self._set_float_field(self._phase_slider,
                              [d.phase_offset for d in dancers],
                              self._PHASE_SENTINEL)
        self._set_float_field(self._speed_slider,
                              [d.speed_multiplier for d in dancers],
                              self._SPEED_SENTINEL)
        self._set_int_field(self._cycles_spin,
                            [d.n_cycles for d in dancers],
                            self._CYCLES_SENTINEL)
        # Buttons
        self._generate_btn.setText(f"Generate Mesh ({n})")
        self._info_label.setText(self._build_info_text_bulk(dancers))

    @staticmethod
    def _set_float_field(spin: QtWidgets.QDoubleSpinBox,
                         values: list[float], sentinel: float) -> None:
        if values and all(v == values[0] for v in values):
            spin.setValue(float(values[0]))
        else:
            spin.setValue(sentinel)

    @staticmethod
    def _set_int_field(spin: QtWidgets.QSpinBox,
                       values: list[int], sentinel: int) -> None:
        if values and all(v == values[0] for v in values):
            spin.setValue(int(values[0]))
        else:
            spin.setValue(sentinel)

    def _select_source_data(self, target_data: str) -> None:
        for i in range(self._source_combo.count()):
            if self._source_combo.itemData(i) == target_data:
                self._source_combo.setCurrentIndex(i)
                return

    def _add_mixed_source_if_needed(self) -> None:
        for i in range(self._source_combo.count()):
            if self._source_combo.itemData(i) == MIXED_SOURCE_OPTION:
                return
        self._source_combo.insertItem(0, "(mixed)", userData=MIXED_SOURCE_OPTION)

    def _remove_mixed_source(self) -> None:
        for i in range(self._source_combo.count()):
            if self._source_combo.itemData(i) == MIXED_SOURCE_OPTION:
                self._source_combo.removeItem(i)
                return

    # -- info text helpers ---------------------------------------------------

    def _build_info_text(self, dancer: Dancer) -> str:
        if dancer.gen_result is None:
            return "Mesh: not generated yet."
        scale = dancer.gen_result.scale
        mismatch_deg = float(np.rad2deg(dancer.gen_result.mismatch_angle))
        return (
            f"Mesh: {dancer.gen_result.faces.shape[0]} faces, "
            f"scale={scale:.3f}, mismatch={mismatch_deg:.1f}°"
        )

    def _build_info_text_bulk(self, dancers: list[Dancer]) -> str:
        with_mesh = sum(1 for d in dancers if d.gen_result is not None)
        return f"{len(dancers)} dancers selected — {with_mesh}/{len(dancers)} with mesh."

    def _update_color_button(self, hex_color: str) -> None:
        self._color_btn.setStyleSheet(
            f"background-color: {hex_color}; color: white; font-weight: bold;"
        )

    # -- callbacks (all iterate self._dancers) ------------------------------

    def _on_name_changed(self) -> None:
        if self._suppress_signals or len(self._dancers) != 1:
            return
        d = self._dancers[0]
        new_name = self._name_edit.text().strip() or d.name
        if new_name != d.name:
            d.name = new_name
            self.nameChanged.emit(d.dancer_id)

    def _on_pick_color(self) -> None:
        if not self._dancers:
            return
        initial = QtGui.QColor(self._dancers[0].color_hex)
        color = QtWidgets.QColorDialog.getColor(initial, self, "Pick dancer color")
        if not color.isValid():
            return
        new_hex = color.name()
        self._update_color_button(new_hex)
        for d in self._dancers:
            d.color_hex = new_hex
            self.colorChanged.emit(d.dancer_id)
            # Re-emit dancerChanged so viewer recolors the mesh.
            self.dancerChanged.emit(d.dancer_id)

    def _on_source_changed(self, _index: int) -> None:
        if not self._dancers or self._suppress_signals:
            return
        data = self._source_combo.currentData()
        if data == MIXED_SOURCE_OPTION:
            return  # No-op: user re-selected the "(mixed)" placeholder.

        is_bulk = len(self._dancers) > 1
        if data == FREEHAND_OPTION:
            for d in self._dancers:
                d.curve_source = "freehand"
                d.curve_xy = np.empty((0, 2), dtype=float)
                d.curve_params = {}
                d.gen_result = None
                d.sim_result = None
            # In bulk mode we keep the freehand canvas hidden (per-dancer drawing
            # isn't meaningful for many dancers at once).
            self._curve_editor.setVisible(not is_bulk)
            if not is_bulk:
                self._curve_editor.clear_curve()
        else:
            preset_key = data.split(":", 1)[1]
            spec = PRESETS_BY_KEY.get(preset_key)
            # Fresh defaults — switching source resets parametric knobs.
            default_params: dict = (
                {p.name: p.default for p in spec.params} if spec is not None else {}
            )
            curve = get_preset(preset_key, **default_params)
            for d in self._dancers:
                d.curve_source = data
                d.curve_params = dict(default_params)
                d.curve_xy = curve.copy()
                d.gen_result = None
                d.sim_result = None
            self._curve_editor.setVisible(False)
        # Once we've applied a concrete source, the "(mixed)" entry is stale.
        self._remove_mixed_source()
        # Rebuild parameter widgets and toggle param/tool group visibility.
        self._suppress_signals = True
        try:
            self._rebuild_param_widgets()
        finally:
            self._suppress_signals = False
        self._update_section_visibility()
        self.refresh_info()
        # Curve shape changed → repaint canvas + tell main window so it can
        # update the "Generate All" dirty counter.
        self.canvas.refresh()
        for d in self._dancers:
            self.dancerChanged.emit(d.dancer_id)

    def _on_freehand_curve_changed(self) -> None:
        if self._suppress_signals or len(self._dancers) != 1:
            return
        d = self._dancers[0]
        if not d.curve_source.startswith("freehand"):
            return
        pts = self._curve_editor.sampled_points(n_samples=400)
        if pts.shape[0] >= 2:
            d.curve_xy = pts

    def _on_motion_changed(self) -> None:
        if not self._dancers or self._suppress_signals:
            return
        sx = self._start_x_spin.value()
        sy = self._start_y_spin.value()
        phase = self._phase_slider.value()
        speed = self._speed_slider.value()
        # In bulk mode, fields still at their sentinel value mean "values differ;
        # don't apply" — let those fields preserve each dancer's existing value.
        apply_sx = sx != self._START_X_SENTINEL
        apply_sy = sy != self._START_Y_SENTINEL
        apply_phase = phase != self._PHASE_SENTINEL
        apply_speed = speed != self._SPEED_SENTINEL
        for d in self._dancers:
            new_x = sx if apply_sx else d.start_offset_xy[0]
            new_y = sy if apply_sy else d.start_offset_xy[1]
            d.start_offset_xy = (new_x, new_y)
            if apply_phase:
                d.phase_offset = float(phase)
            if apply_speed:
                d.speed_multiplier = float(speed)
            self.dancerChanged.emit(d.dancer_id)

    def _on_cycles_changed(self, value: int) -> None:
        if not self._dancers or self._suppress_signals:
            return
        if value == self._CYCLES_SENTINEL:
            return  # mixed-state sentinel; don't clobber individual values
        for d in self._dancers:
            d.n_cycles = int(value)
            # Cycle count affects sim length, so the mesh+sim need regeneration.
            d.sim_result = None

    def _on_generate_clicked(self) -> None:
        for d in self._dancers:
            self.generateRequested.emit(d.dancer_id)

    # -- parametric preset helpers ------------------------------------------

    def _current_spec(self) -> PresetSpec | None:
        """PresetSpec shared by all currently-selected dancers, or None.

        Returns None when nothing is selected, sources differ, or the source
        is freehand.
        """
        if not self._dancers:
            return None
        sources = {d.curve_source for d in self._dancers}
        if len(sources) != 1:
            return None
        src = next(iter(sources))
        if not src.startswith("preset:"):
            return None
        return PRESETS_BY_KEY.get(src.split(":", 1)[1])

    def _build_param_widget(self, param: Param) -> QtWidgets.QAbstractSpinBox:
        if param.kind is int:
            w: QtWidgets.QAbstractSpinBox = QtWidgets.QSpinBox()
            w.setRange(int(param.min), int(param.max))
            w.setSingleStep(max(1, int(param.step)))
        else:
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(float(param.min), float(param.max))
            w.setSingleStep(float(param.step))
            w.setDecimals(3 if param.step < 0.1 else 2)
        name = param.name
        w.valueChanged.connect(lambda v, n=name: self._on_param_changed(n, v))
        return w

    def _clear_param_form(self) -> None:
        while self._param_form.rowCount() > 0:
            self._param_form.removeRow(0)
        self._param_widgets.clear()

    def _rebuild_param_widgets(self) -> None:
        """Rebuild the parameter form from the current preset spec.

        Populates each widget with the first selected dancer's stored value
        (or the param's default if missing).
        """
        self._clear_param_form()
        spec = self._current_spec()
        if spec is None or not spec.params:
            return
        first = self._dancers[0] if self._dancers else None
        for p in spec.params:
            widget = self._build_param_widget(p)
            if first is not None:
                value = first.curve_params.get(p.name, p.default)
                # Block valueChanged so populating doesn't fire _on_param_changed.
                widget.blockSignals(True)
                try:
                    if p.kind is int:
                        widget.setValue(int(value))
                    else:
                        widget.setValue(float(value))
                finally:
                    widget.blockSignals(False)
            self._param_form.addRow(p.label + ":", widget)
            self._param_widgets[p.name] = widget

    def _update_section_visibility(self) -> None:
        if not self._dancers:
            self._param_group.setVisible(False)
            self._tool_group.setVisible(False)
            return
        sources = {d.curve_source for d in self._dancers}
        if len(sources) != 1:
            self._param_group.setVisible(False)
            self._tool_group.setVisible(False)
            return
        src = next(iter(sources))
        is_freehand = src.startswith("freehand")
        spec = self._current_spec()
        has_params = spec is not None and bool(spec.params)
        self._param_group.setVisible(has_params)
        self._tool_group.setVisible(is_freehand)

    def _on_tool_clicked(self, tool_id: str) -> None:
        self._curve_editor.set_tool(tool_id)

    def _on_param_changed(self, name: str, value) -> None:
        if not self._dancers or self._suppress_signals:
            return
        # Only act when all selected dancers share a parametric preset source.
        spec = self._current_spec()
        if spec is None or not spec.params:
            return
        source_data = f"preset:{spec.key}"
        for d in self._dancers:
            if d.curve_source != source_data:
                continue
            d.curve_params[name] = float(value) if not isinstance(value, int) else int(value)
            # Re-generate curve from the spec with the new params.
            d.curve_xy = get_preset(spec.key, **d.curve_params)
            d.gen_result = None
            d.sim_result = None
            self.dancerChanged.emit(d.dancer_id)
        self.refresh_info()
        self.canvas.refresh()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trajectoid Dance — multi-rolling-shape playground")
        self.resize(1400, 850)

        self._scene = DanceScene()
        self._current_file: Optional[Path] = None
        self._is_dirty: bool = False
        self._suppress_roster_signal: bool = False
        self._viewer = make_viewer(self)

        # ----- Left panel: Roster + global controls -----
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(QtWidgets.QLabel("<b>Dancers</b>"))
        self._roster = QtWidgets.QListWidget()
        self._roster.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._roster.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._roster, stretch=1)

        roster_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ Add")
        add_btn.clicked.connect(self._on_add_dancer)
        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.clicked.connect(self._on_duplicate_dancer)
        del_btn = QtWidgets.QPushButton("Remove")
        del_btn.clicked.connect(self._on_remove_dancer)
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_dancers)
        for b in (add_btn, dup_btn, del_btn, clear_btn):
            roster_buttons.addWidget(b)
        left_layout.addLayout(roster_buttons)

        left_layout.addWidget(QtWidgets.QLabel("<b>Build</b>"))
        self._generate_all_btn = QtWidgets.QPushButton("Generate All Meshes")
        self._generate_all_btn.setStyleSheet(
            "font-weight: bold; padding: 8px; background-color: #2b6cb0; color: white;"
        )
        self._generate_all_btn.clicked.connect(self._on_generate_all)
        left_layout.addWidget(self._generate_all_btn)

        left_layout.addWidget(QtWidgets.QLabel("<b>Playback</b>"))
        playback_form = QtWidgets.QFormLayout()
        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(2.0, 120.0)
        self._duration_spin.setSingleStep(1.0)
        self._duration_spin.setValue(self._scene.duration_seconds)
        self._duration_spin.setSuffix(" s")
        playback_form.addRow("Duration:", self._duration_spin)
        self._loop_check = QtWidgets.QCheckBox("Loop")
        playback_form.addRow("", self._loop_check)
        self._wireframe_check = QtWidgets.QCheckBox("Wireframe")
        self._wireframe_check.toggled.connect(self._viewer.set_wireframe)
        playback_form.addRow("", self._wireframe_check)

        self._shader_combo = QtWidgets.QComboBox()
        for label, key in MESH_SHADERS.items():
            self._shader_combo.addItem(label, userData=key)
        self._shader_combo.currentIndexChanged.connect(self._on_shader_changed)
        playback_form.addRow("Shader:", self._shader_combo)

        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(10, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setTickInterval(10)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        playback_form.addRow("Opacity:", self._opacity_slider)

        left_layout.addLayout(playback_form)

        playback_buttons = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("▶ Play All")
        self._play_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._play_btn.clicked.connect(self._on_play)
        self._stop_btn = QtWidgets.QPushButton("■ Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        reset_btn = QtWidgets.QPushButton("Reset View")
        reset_btn.clicked.connect(self._viewer.reset_view)
        for b in (self._play_btn, self._stop_btn, reset_btn):
            playback_buttons.addWidget(b)
        left_layout.addLayout(playback_buttons)

        export_btn = QtWidgets.QPushButton("Export STLs")
        export_btn.clicked.connect(self._on_export_stls)
        left_layout.addWidget(export_btn)

        # ----- Middle panel: Editor -----
        self._editor = DancerEditorPanel()
        self._editor.dancerChanged.connect(self._on_dancer_motion_changed)
        self._editor.generateRequested.connect(self._on_generate_requested)
        self._editor.nameChanged.connect(self._on_dancer_name_changed)
        self._editor.colorChanged.connect(self._on_dancer_color_changed)
        self._editor.dancerChanged.connect(lambda _: self._mark_dirty())
        self._editor.nameChanged.connect(lambda _: self._mark_dirty())
        self._editor.colorChanged.connect(lambda _: self._mark_dirty())
        # Source/curve changes can invalidate gen_result → refresh dirty counter.
        self._editor.dancerChanged.connect(lambda _: self._update_generate_all_label())

        # Layout canvas → main-window plumbing
        self._editor.canvas.dancerTranslated.connect(self._on_canvas_translated)
        self._editor.canvas.dancerCurveModified.connect(self._on_canvas_curve_modified)
        self._editor.canvas.selectionChanged.connect(self._on_canvas_selection)

        # ----- Splitter -----
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._editor)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([320, 380, 700])
        self.setCentralWidget(splitter)

        # Status bar
        self._status = self.statusBar()
        self._status.showMessage(f"Ready — viewer: {getattr(self._viewer, 'backend_name', 'unknown')}")

        # Hook viewer playback signal
        if hasattr(self._viewer, "playFinished"):
            self._viewer.playFinished.connect(self._on_play_finished)

        self._build_menu()
        # Initial state: empty scene → button shows "Generate All Meshes" disabled.
        self._update_generate_all_label()

    # -- menu ----------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        act_new = QtGui.QAction("&New", self)
        act_new.setShortcut(QtGui.QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new)

        act_open = QtGui.QAction("&Open…", self)
        act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open)

        act_save = QtGui.QAction("&Save", self)
        act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save)

        act_save_as = QtGui.QAction("Save &As…", self)
        act_save_as.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._on_save_as)

        file_menu.addAction(act_new)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        file_menu.addAction(act_save)
        file_menu.addAction(act_save_as)

    # -- dirty state ---------------------------------------------------------

    def _mark_dirty(self) -> None:
        if not self._is_dirty:
            self._is_dirty = True
            self._update_title()

    def _mark_clean(self) -> None:
        self._is_dirty = False
        self._update_title()

    def _update_title(self) -> None:
        base = "Trajectoid Dance"
        if self._current_file is not None:
            base = f"{self._current_file.name} — {base}"
        if self._is_dirty:
            base = f"*{base}"
        self.setWindowTitle(base)

    # -- file operations -----------------------------------------------------

    def _confirm_discard_changes(self) -> bool:
        if not self._is_dirty:
            return True
        name = self._current_file.name if self._current_file else "Untitled"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Unsaved Changes",
            f'"{name}" has unsaved changes.\nDo you want to discard them?',
            QtWidgets.QMessageBox.StandardButton.Discard |
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()

    def _on_new(self) -> None:
        if not self._confirm_discard_changes():
            return
        self._scene.dancers.clear()
        self._viewer.clear_dancers()
        self._refresh_roster()
        self._current_file = None
        self._mark_clean()
        self._status.showMessage("New scene.")

    def _on_open(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Scene", str(Path.home()),
            "Trajectoid Dance (*.tdance);;All Files (*)"
        )
        if not path:
            return
        self._load_from_path(Path(path))

    def _on_save(self) -> None:
        if self._current_file is None:
            self._on_save_as()
        else:
            self._save_to_path(self._current_file)

    def _on_save_as(self) -> None:
        default = str(self._current_file or Path.home() / "untitled.tdance")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Scene As", default,
            "Trajectoid Dance (*.tdance);;All Files (*)"
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".tdance":
            p = p.with_suffix(".tdance")
        self._save_to_path(p)

    def _save_to_path(self, path: Path) -> None:
        self._scene.duration_seconds = float(self._duration_spin.value())
        self._scene.loop = self._loop_check.isChecked()
        try:
            save_scene(self._scene, path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Failed", str(exc))
            return
        self._current_file = path
        self._mark_clean()
        self._status.showMessage(f"Saved to {path.name}.")

    def _load_from_path(self, path: Path) -> None:
        try:
            scene = load_scene(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Failed", str(exc))
            return
        self._scene = scene
        self._viewer.clear_dancers()
        for d in self._scene.dancers:
            if d.gen_result is not None and d.sim_result is not None:
                self._viewer.add_or_update_dancer(d)
        self._refresh_roster()
        self._duration_spin.setValue(self._scene.duration_seconds)
        self._loop_check.setChecked(self._scene.loop)
        self._current_file = path
        self._mark_clean()
        self._status.showMessage(f"Opened {path.name} — {len(self._scene.dancers)} dancer(s).")

    # -- roster operations ---------------------------------------------------

    def _on_add_dancer(self) -> None:
        name = self._scene.next_name()
        color = self._scene.next_color()
        # Default to circle preset for instant gratification.
        from presets import get_preset

        d = Dancer.new("preset:circle", get_preset("circle"), name, color)
        self._scene.add(d)
        self._refresh_roster(select_id=d.dancer_id)
        self._mark_dirty()

    def _on_duplicate_dancer(self) -> None:
        targets = self._selected_dancers()
        if not targets:
            return
        import uuid

        clones: list[Dancer] = []
        for d in targets:
            clone = Dancer(
                dancer_id=uuid.uuid4().hex,
                name=self._scene.next_name(),
                curve_source=d.curve_source,
                curve_xy=d.curve_xy.copy(),
                color_hex=self._scene.next_color(),
                start_offset_xy=(d.start_offset_xy[0] + 2.0, d.start_offset_xy[1]),
                phase_offset=d.phase_offset,
                speed_multiplier=d.speed_multiplier,
                n_cycles=d.n_cycles,
                closed=d.closed,
            )
            self._scene.add(clone)
            clones.append(clone)
        self._refresh_roster()
        self._select_dancers_in_roster(clones)
        self._mark_dirty()

    def _on_remove_dancer(self) -> None:
        targets = self._selected_dancers()
        if not targets:
            return
        for d in targets:
            self._viewer.remove_dancer(d.dancer_id)
            self._scene.remove(d.dancer_id)
        self._refresh_roster()
        self._mark_dirty()

    def _on_clear_dancers(self) -> None:
        self._scene.dancers.clear()
        self._viewer.clear_dancers()
        self._refresh_roster()
        self._mark_dirty()

    def _refresh_roster(self, select_id: Optional[str] = None) -> None:
        self._roster.blockSignals(True)
        self._roster.clear()
        target_row = -1
        for i, d in enumerate(self._scene.dancers):
            item = QtWidgets.QListWidgetItem(_color_swatch(d.color_hex), d.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, d.dancer_id)
            self._roster.addItem(item)
            if d.dancer_id == select_id:
                target_row = i
        self._roster.blockSignals(False)
        if target_row >= 0:
            self._roster.setCurrentRow(target_row)
        elif self._roster.count() > 0:
            self._roster.setCurrentRow(0)
        else:
            self._editor.set_dancers([])
        # Keep the layout canvas in sync with the full scene roster
        # (per-dancer selection is handled separately via set_selection).
        self._editor.canvas.set_dancers(self._scene.dancers)
        self._update_generate_all_label()

    def _current_dancer(self) -> Optional[Dancer]:
        row = self._roster.currentRow()
        if row < 0 or row >= len(self._scene.dancers):
            return None
        return self._scene.dancers[row]

    def _selected_dancers(self) -> list[Dancer]:
        ids = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self._roster.selectedItems()
        }
        return [d for d in self._scene.dancers if d.dancer_id in ids]

    def _select_dancers_in_roster(self, dancers: list[Dancer]) -> None:
        ids = {d.dancer_id for d in dancers}
        self._roster.clearSelection()
        for i in range(self._roster.count()):
            item = self._roster.item(i)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) in ids:
                item.setSelected(True)

    def _on_selection_changed(self) -> None:
        if self._suppress_roster_signal:
            return
        self._editor.set_dancers(self._selected_dancers())

    # -- layout-canvas signal handlers --------------------------------------

    def _on_canvas_translated(self, dancer_id: str) -> None:
        # Drag-move: only start_offset_xy changed, so reuse the motion handler
        # to refresh the 3D viewer transform. Mark dirty per drag tick.
        self._on_dancer_motion_changed(dancer_id)
        self._mark_dirty()

    def _on_canvas_curve_modified(self, dancer_id: str) -> None:
        # Rotate/scale dirtied curve_xy → the cached mesh+sim are now stale
        # (the canvas already cleared d.gen_result / d.sim_result). Drop the
        # stale mesh from the viewer so the user sees they must regenerate.
        self._viewer.remove_dancer(dancer_id)
        self._editor.refresh_info()
        self._update_generate_all_label()
        self._mark_dirty()

    def _on_canvas_selection(self, ids: list) -> None:
        # User picked dancers from the canvas → sync the roster selection.
        # Guard against the roster firing _on_selection_changed back into us.
        self._suppress_roster_signal = True
        try:
            self._roster.clearSelection()
            id_set = set(ids)
            for i in range(self._roster.count()):
                item = self._roster.item(i)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) in id_set:
                    item.setSelected(True)
        finally:
            self._suppress_roster_signal = False
        # Manually drive the editor since we suppressed the auto-signal.
        self._editor.set_dancers(self._selected_dancers())

    # -- per-dancer signal handlers -----------------------------------------

    def _on_dancer_motion_changed(self, dancer_id: str) -> None:
        d = self._scene.find(dancer_id)
        if d is None:
            return
        # Only push to viewer if mesh exists.
        if d.gen_result is not None and d.sim_result is not None:
            self._viewer.add_or_update_dancer(d)

    def _on_dancer_name_changed(self, dancer_id: str) -> None:
        for i, d in enumerate(self._scene.dancers):
            if d.dancer_id == dancer_id:
                item = self._roster.item(i)
                if item is not None:
                    item.setText(d.name)
                break

    def _on_dancer_color_changed(self, dancer_id: str) -> None:
        for i, d in enumerate(self._scene.dancers):
            if d.dancer_id == dancer_id:
                item = self._roster.item(i)
                if item is not None:
                    item.setIcon(_color_swatch(d.color_hex))
                break

    def _on_generate_requested(self, dancer_id: str) -> None:
        d = self._scene.find(dancer_id)
        if d is None:
            return
        if d.curve_xy.shape[0] < 4:
            QtWidgets.QMessageBox.warning(
                self, "Generate", "This dancer has no usable curve yet. "
                "Pick a preset or draw one."
            )
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._status.showMessage(f"Generating mesh for {d.name}…")
        # Let the status bar repaint before the (potentially multi-second)
        # generate call blocks the event loop. Important when bulk-generating
        # several dancers back to back.
        QtWidgets.QApplication.processEvents()
        try:
            generate_dancer(d)
        except ValueError as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Generation failed", str(e))
            self._status.showMessage(f"Generation failed for {d.name}.")
            return
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", f"Unexpected error:\n{e}")
            self._status.showMessage(f"Error: {e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._viewer.add_or_update_dancer(d)
        self._editor.refresh_info()
        # gen_result is now populated → canvas redraws this dancer's curve
        # with a solid (not dashed) stroke.
        self._editor.canvas.refresh()
        self._update_generate_all_label()
        self._mark_dirty()
        self._status.showMessage(
            f"Generated {d.name}: {d.gen_result.faces.shape[0]} faces."
        )

    # -- playback ------------------------------------------------------------

    def _on_shader_changed(self, _: int) -> None:
        key = self._shader_combo.currentData()
        if key:
            self._viewer.set_shader(key)

    def _on_opacity_changed(self, value: int) -> None:
        self._viewer.set_opacity(value / 100.0)

    def _on_play(self) -> None:
        ready = [d for d in self._scene.dancers if d.gen_result is not None]
        if not ready:
            QtWidgets.QMessageBox.information(
                self, "Play", "No dancer has a generated mesh yet. "
                "Add a dancer and click Generate Mesh first."
            )
            return
        duration = float(self._duration_spin.value())
        loop = self._loop_check.isChecked()
        if self._viewer.start_play(duration, loop=loop):
            self._play_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._status.showMessage(
                f"Playing {len(ready)} dancer(s), duration {duration:.1f}s"
                + (" (loop)" if loop else "")
            )

    def _on_stop(self) -> None:
        self._viewer.stop_play()

    def _on_play_finished(self) -> None:
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Playback finished.")

    # -- batch generate ------------------------------------------------------

    def _dirty_dancers(self) -> list[Dancer]:
        """Dancers whose cached mesh is stale (or never generated)."""
        return [d for d in self._scene.dancers if d.gen_result is None]

    def _update_generate_all_label(self) -> None:
        n_dirty = len(self._dirty_dancers())
        total = len(self._scene.dancers)
        if total == 0:
            self._generate_all_btn.setText("Generate All Meshes")
            self._generate_all_btn.setEnabled(False)
        elif n_dirty == 0:
            self._generate_all_btn.setText("All meshes up to date")
            self._generate_all_btn.setEnabled(False)
        else:
            self._generate_all_btn.setText(f"Generate All Meshes  ({n_dirty} dirty)")
            self._generate_all_btn.setEnabled(True)

    def _on_generate_all(self) -> None:
        dirty = self._dirty_dancers()
        if not dirty:
            return
        total = len(dirty)
        # Disable the button during the batch so the user can't double-click.
        self._generate_all_btn.setEnabled(False)
        self._generate_all_btn.setText(f"Generating 0/{total}…")
        QtWidgets.QApplication.processEvents()
        for i, d in enumerate(dirty, start=1):
            self._generate_all_btn.setText(f"Generating {i}/{total}…")
            QtWidgets.QApplication.processEvents()
            self._on_generate_requested(d.dancer_id)
        # _on_generate_requested already calls _update_generate_all_label, but
        # call it once more to settle the final state cleanly.
        self._update_generate_all_label()

    # -- export --------------------------------------------------------------

    def _on_export_stls(self) -> None:
        ready = [d for d in self._scene.dancers if d.gen_result is not None]
        if not ready:
            QtWidgets.QMessageBox.information(self, "Export", "No generated dancers to export.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output directory",
            str(Path(__file__).resolve().parent / "output"),
        )
        if not out_dir:
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        wrote = []
        for d in ready:
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in d.name).strip() or d.dancer_id[:8]
            target = Path(out_dir) / f"{safe_name}.stl"
            export_binary_stl(d.gen_result.vertices, d.gen_result.faces, target, solid_name=d.name)
            wrote.append(str(target))
        QtWidgets.QMessageBox.information(
            self, "Export complete", "Wrote:\n" + "\n".join(wrote)
        )
        self._status.showMessage(f"Exported {len(wrote)} STL file(s) to {out_dir}.")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
