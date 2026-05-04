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

import numpy as np

# viewer.py sets up cache env vars; import it before the rest of Qt land.
from viewer import make_viewer

from PySide6 import QtCore, QtGui, QtWidgets

from curve_editor import CurveEditorWidget, Tool
from dancer import Dancer, DanceScene, generate_dancer
from presets import PRESET_LABELS, get_preset
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

class DancerEditorPanel(QtWidgets.QWidget):
    dancerChanged = QtCore.Signal(str)             # dancer_id (motion-only update)
    generateRequested = QtCore.Signal(str)         # dancer_id (regenerate mesh)
    nameChanged = QtCore.Signal(str)               # dancer_id (refresh roster label)
    colorChanged = QtCore.Signal(str)              # dancer_id (refresh roster swatch)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._dancer: Optional[Dancer] = None
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
        self._source_combo.addItem("Freehand…", userData=FREEHAND_OPTION)
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self._source_combo, stretch=1)
        ct_layout.addLayout(source_row)

        self._curve_editor = CurveEditorWidget()
        self._curve_editor.set_tool(Tool.FREEHAND)
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
        self._start_x_spin.setRange(-50.0, 50.0)
        self._start_x_spin.setSingleStep(0.5)
        self._start_x_spin.setDecimals(2)
        self._start_x_spin.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Start X:", self._start_x_spin)

        self._start_y_spin = QtWidgets.QDoubleSpinBox()
        self._start_y_spin.setRange(-50.0, 50.0)
        self._start_y_spin.setSingleStep(0.5)
        self._start_y_spin.setDecimals(2)
        self._start_y_spin.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Start Y:", self._start_y_spin)

        self._phase_slider = QtWidgets.QDoubleSpinBox()
        self._phase_slider.setRange(0.0, 0.999)
        self._phase_slider.setSingleStep(0.05)
        self._phase_slider.setDecimals(3)
        self._phase_slider.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Phase offset:", self._phase_slider)

        self._speed_slider = QtWidgets.QDoubleSpinBox()
        self._speed_slider.setRange(0.25, 4.0)
        self._speed_slider.setSingleStep(0.05)
        self._speed_slider.setDecimals(2)
        self._speed_slider.valueChanged.connect(self._on_motion_changed)
        m_layout.addRow("Speed:", self._speed_slider)

        self._cycles_spin = QtWidgets.QSpinBox()
        self._cycles_spin.setRange(1, 8)
        self._cycles_spin.valueChanged.connect(self._on_cycles_changed)
        m_layout.addRow("Cycles per build:", self._cycles_spin)

        self._tabs.addTab(motion_tab, "Motion")

        # Status hint at bottom
        self._info_label = QtWidgets.QLabel("(no dancer selected)")
        self._info_label.setStyleSheet("color: #6b7280; font-style: italic;")
        layout.addWidget(self._info_label)

        self.setEnabled(False)

    # -- public API ----------------------------------------------------------

    def set_dancer(self, dancer: Optional[Dancer]) -> None:
        self._dancer = dancer
        if dancer is None:
            self.setEnabled(False)
            self._info_label.setText("(no dancer selected)")
            return
        self.setEnabled(True)
        self._suppress_signals = True
        try:
            self._name_edit.setText(dancer.name)
            self._update_color_button(dancer.color_hex)
            # Source combo
            target_data = dancer.curve_source if dancer.curve_source.startswith("preset:") else FREEHAND_OPTION
            for i in range(self._source_combo.count()):
                if self._source_combo.itemData(i) == target_data:
                    self._source_combo.setCurrentIndex(i)
                    break
            # Curve editor visibility
            is_freehand = dancer.curve_source.startswith("freehand")
            self._curve_editor.setVisible(is_freehand)
            if is_freehand and dancer.curve_xy.size > 0:
                # Restore freehand drawing by setting points (no public setter; clear + nothing).
                self._curve_editor.clear_curve()
            # Motion fields
            self._start_x_spin.setValue(float(dancer.start_offset_xy[0]))
            self._start_y_spin.setValue(float(dancer.start_offset_xy[1]))
            self._phase_slider.setValue(float(dancer.phase_offset))
            self._speed_slider.setValue(float(dancer.speed_multiplier))
            self._cycles_spin.setValue(int(dancer.n_cycles))
            self._info_label.setText(self._build_info_text(dancer))
        finally:
            self._suppress_signals = False

    def refresh_info(self) -> None:
        if self._dancer is not None:
            self._info_label.setText(self._build_info_text(self._dancer))

    # -- internal callbacks --------------------------------------------------

    def _build_info_text(self, dancer: Dancer) -> str:
        if dancer.gen_result is None:
            return "Mesh: not generated yet."
        scale = dancer.gen_result.scale
        mismatch_deg = float(np.rad2deg(dancer.gen_result.mismatch_angle))
        return (
            f"Mesh: {dancer.gen_result.faces.shape[0]} faces, "
            f"scale={scale:.3f}, mismatch={mismatch_deg:.1f}°"
        )

    def _update_color_button(self, hex_color: str) -> None:
        self._color_btn.setStyleSheet(
            f"background-color: {hex_color}; color: white; font-weight: bold;"
        )

    def _on_name_changed(self) -> None:
        if self._dancer is None or self._suppress_signals:
            return
        new_name = self._name_edit.text().strip() or self._dancer.name
        if new_name != self._dancer.name:
            self._dancer.name = new_name
            self.nameChanged.emit(self._dancer.dancer_id)

    def _on_pick_color(self) -> None:
        if self._dancer is None:
            return
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._dancer.color_hex), self, "Pick dancer color"
        )
        if not color.isValid():
            return
        self._dancer.color_hex = color.name()
        self._update_color_button(self._dancer.color_hex)
        self.colorChanged.emit(self._dancer.dancer_id)
        # Re-emit dancerChanged so viewer recolors the mesh.
        self.dancerChanged.emit(self._dancer.dancer_id)

    def _on_source_changed(self, _index: int) -> None:
        if self._dancer is None or self._suppress_signals:
            return
        data = self._source_combo.currentData()
        if data == FREEHAND_OPTION:
            self._dancer.curve_source = "freehand"
            self._dancer.curve_xy = np.empty((0, 2), dtype=float)
            self._curve_editor.setVisible(True)
            self._curve_editor.clear_curve()
        else:
            preset_key = data.split(":", 1)[1]
            self._dancer.curve_source = data
            self._dancer.curve_xy = get_preset(preset_key)
            self._curve_editor.setVisible(False)
        # Mesh becomes stale when source changes; clear cached results.
        self._dancer.gen_result = None
        self._dancer.sim_result = None
        self.refresh_info()

    def _on_freehand_curve_changed(self) -> None:
        if self._dancer is None or self._suppress_signals:
            return
        if not self._dancer.curve_source.startswith("freehand"):
            return
        pts = self._curve_editor.sampled_points(n_samples=400)
        if pts.shape[0] >= 2:
            self._dancer.curve_xy = pts

    def _on_motion_changed(self) -> None:
        if self._dancer is None or self._suppress_signals:
            return
        self._dancer.start_offset_xy = (self._start_x_spin.value(), self._start_y_spin.value())
        self._dancer.phase_offset = float(self._phase_slider.value())
        self._dancer.speed_multiplier = float(self._speed_slider.value())
        self.dancerChanged.emit(self._dancer.dancer_id)

    def _on_cycles_changed(self, value: int) -> None:
        if self._dancer is None or self._suppress_signals:
            return
        self._dancer.n_cycles = int(value)
        # Cycle count affects sim length, so the mesh+sim need regeneration.
        self._dancer.sim_result = None

    def _on_generate_clicked(self) -> None:
        if self._dancer is None:
            return
        self.generateRequested.emit(self._dancer.dancer_id)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trajectoid Dance — multi-rolling-shape playground")
        self.resize(1400, 850)

        self._scene = DanceScene()
        self._viewer = make_viewer(self)

        # ----- Left panel: Roster + global controls -----
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(QtWidgets.QLabel("<b>Dancers</b>"))
        self._roster = QtWidgets.QListWidget()
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

    # -- roster operations ---------------------------------------------------

    def _on_add_dancer(self) -> None:
        name = self._scene.next_name()
        color = self._scene.next_color()
        # Default to circle preset for instant gratification.
        from presets import get_preset

        d = Dancer.new("preset:circle", get_preset("circle"), name, color)
        self._scene.add(d)
        self._refresh_roster(select_id=d.dancer_id)

    def _on_duplicate_dancer(self) -> None:
        d = self._current_dancer()
        if d is None:
            return
        import uuid

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
        self._refresh_roster(select_id=clone.dancer_id)

    def _on_remove_dancer(self) -> None:
        d = self._current_dancer()
        if d is None:
            return
        self._viewer.remove_dancer(d.dancer_id)
        self._scene.remove(d.dancer_id)
        self._refresh_roster()

    def _on_clear_dancers(self) -> None:
        self._scene.dancers.clear()
        self._viewer.clear_dancers()
        self._refresh_roster()

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
            self._editor.set_dancer(None)

    def _current_dancer(self) -> Optional[Dancer]:
        row = self._roster.currentRow()
        if row < 0 or row >= len(self._scene.dancers):
            return None
        return self._scene.dancers[row]

    def _on_selection_changed(self) -> None:
        self._editor.set_dancer(self._current_dancer())

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
        self._status.showMessage(
            f"Generated {d.name}: {d.gen_result.faces.shape[0]} faces."
        )

    # -- playback ------------------------------------------------------------

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
