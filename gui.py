"""
BatchStack — PySide6 GUI
=========================
Minimalistic, native-style interface for batch FITS image stacking.
Inspired by classic Windows utility design (compact, clean, functional).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import cast

from PySide6.QtCore import QPointF, QUrl, Slot
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QGridLayout,
    QWidget,
)

from stacker import StackWorker


class BatchStackWindow(QMainWindow):
    """Main application window for BatchStack."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BatchStack")
        self.setWindowIcon(self._make_window_icon())
        self.setMinimumSize(640, 480)
        self.resize(660, 500)

        self.worker: StackWorker | None = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 2)
        main_layout.setSpacing(6)

        # ── Left side: all settings ──
        left = QVBoxLayout()
        left.setSpacing(4)

        # ── Folders group ──
        folders_grp = QGroupBox("Folders")
        fg = QGridLayout(folders_grp)
        fg.setContentsMargins(6, 14, 6, 6)
        fg.setHorizontalSpacing(6)
        fg.setVerticalSpacing(3)

        lbl_input = QLabel("Input:")
        fg.addWidget(lbl_input, 0, 0)
        self.txt_input = QLineEdit()
        self.txt_input.setReadOnly(True)
        self.txt_input.setPlaceholderText("Select folder")
        self.txt_input.setMinimumWidth(260)
        fg.addWidget(self.txt_input, 0, 1)
        self.btn_browse_input = QPushButton("Browse")
        self.btn_browse_input.setMinimumWidth(72)
        self.btn_browse_input.clicked.connect(self._browse_input)
        fg.addWidget(self.btn_browse_input, 0, 2)

        lbl_output = QLabel("Output:")
        fg.addWidget(lbl_output, 1, 0)
        self.txt_output = QLineEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setPlaceholderText("Select folder")
        fg.addWidget(self.txt_output, 1, 1)
        self.btn_browse_output = QPushButton("Browse")
        self.btn_browse_output.setMinimumWidth(72)
        self.btn_browse_output.clicked.connect(self._browse_output)
        fg.addWidget(self.btn_browse_output, 1, 2)

        left.addWidget(folders_grp)

        # ── Settings group ──
        settings_grp = QGroupBox("Stacking Settings")
        sg = QGridLayout(settings_grp)
        sg.setContentsMargins(6, 14, 6, 6)
        sg.setHorizontalSpacing(6)
        sg.setVerticalSpacing(3)

        lbl_group = QLabel("Group Size:")
        sg.addWidget(lbl_group, 0, 0)
        self.spin_group = QSpinBox()
        self.spin_group.setRange(2, 999)
        self.spin_group.setValue(3)
        self.spin_group.setMinimumWidth(74)
        self.spin_group.valueChanged.connect(self._update_summary)
        sg.addWidget(self.spin_group, 0, 1)

        lbl_method = QLabel("Method:")
        sg.addWidget(lbl_method, 0, 2)
        self.combo_method = QComboBox()
        self.combo_method.addItems(StackWorker.METHODS)
        self.combo_method.setMinimumWidth(170)
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        sg.addWidget(self.combo_method, 0, 3)

        lbl_sort = QLabel("Sort By:")
        sg.addWidget(lbl_sort, 1, 0)
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Filename", "DATE-OBS (Header)"])
        sg.addWidget(self.combo_sort, 1, 1, 1, 1)

        lbl_dtype = QLabel("Data Type:")
        sg.addWidget(lbl_dtype, 1, 2)
        self.combo_dtype = QComboBox()
        self.combo_dtype.addItems(["float32", "float64"])
        sg.addWidget(self.combo_dtype, 1, 3)

        # K-Sigma row
        self.lbl_iter = QLabel("Iter:")
        sg.addWidget(self.lbl_iter, 2, 0)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 50)
        self.spin_iter.setValue(5)
        self.spin_iter.setMinimumWidth(74)
        self.spin_iter.setToolTip("Maximum K-Sigma clipping iterations.")
        sg.addWidget(self.spin_iter, 2, 1)

        self.lbl_kappa = QLabel("Kappa:")
        sg.addWidget(self.lbl_kappa, 2, 2)
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(0.10, 10.00)
        self.spin_kappa.setDecimals(2)
        self.spin_kappa.setValue(2.00)
        self.spin_kappa.setSingleStep(0.10)
        self.spin_kappa.setMinimumWidth(74)
        self.spin_kappa.setToolTip("K-Sigma rejection threshold. Default is 2.00.")
        sg.addWidget(self.spin_kappa, 2, 3)

        # Hide K-Sigma controls initially.
        self._set_k_sigma_visible(False)

        left.addWidget(settings_grp)

        # ── Output naming group ──
        naming_grp = QGroupBox("Output & Options")
        ng = QGridLayout(naming_grp)
        ng.setContentsMargins(6, 14, 6, 6)
        ng.setHorizontalSpacing(6)
        ng.setVerticalSpacing(3)

        self.radio_seq = QRadioButton("Sequential")
        self.radio_seq.setChecked(True)
        ng.addWidget(self.radio_seq, 0, 0)
        self.radio_range = QRadioButton("Range")
        ng.addWidget(self.radio_range, 0, 1)
        lbl_prefix = QLabel("Prefix:")
        ng.addWidget(lbl_prefix, 0, 2)
        self.txt_prefix = QLineEdit("stack")
        self.txt_prefix.setMinimumWidth(90)
        ng.addWidget(self.txt_prefix, 0, 3)

        self.chk_incomplete = QCheckBox("Include incomplete last group")
        self.chk_incomplete.stateChanged.connect(self._update_summary)
        ng.addWidget(self.chk_incomplete, 1, 0, 1, 4)

        left.addWidget(naming_grp)

        # ── Summary label ──
        self.lbl_summary = QLabel("Select an input folder to begin.")
        self.lbl_summary.setWordWrap(True)
        left.addWidget(self.lbl_summary)

        # ── Progress bar ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m groups")
        self.progress_bar.setFixedHeight(18)
        left.addWidget(self.progress_bar)

        # ── Log ──
        log_grp = QGroupBox("Log")
        lg = QVBoxLayout(log_grp)
        lg.setContentsMargins(4, 14, 4, 4)
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 8))
        self.log_area.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log_area.setMaximumBlockCount(1000)
        self.log_area.setMinimumHeight(78)
        self.log_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lg.addWidget(self.log_area)
        left.addWidget(log_grp, 1)

        main_layout.addLayout(left, 1)

        # ── Right side: buttons ──
        right = QVBoxLayout()
        right.setSpacing(4)

        self.btn_start = QPushButton("Start")
        self.btn_start.setMinimumWidth(82)
        self.btn_start.setDefault(True)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        right.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumWidth(82)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        right.addWidget(self.btn_cancel)

        right.addSpacing(12)

        self.btn_help = QPushButton("Help")
        self.btn_help.setMinimumWidth(82)
        self.btn_help.clicked.connect(self._help)
        right.addWidget(self.btn_help)

        self.btn_about = QPushButton("About")
        self.btn_about.setMinimumWidth(82)
        self.btn_about.clicked.connect(self._about)
        right.addWidget(self.btn_about)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setMinimumWidth(82)
        self.btn_exit.clicked.connect(self.close)
        right.addWidget(self.btn_exit)

        right.addStretch()

        main_layout.addLayout(right)

        # ── Status bar ──
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _make_window_icon() -> QIcon:
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 0, 0, 0))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        glow = QRadialGradient(QPointF(32, 32), 30)
        glow.setColorAt(0.0, QColor(255, 183, 77, 190))
        glow.setColorAt(0.45, QColor(245, 158, 11, 110))
        glow.setColorAt(1.0, QColor(245, 158, 11, 0))
        painter.setPen(QColor(0, 0, 0, 0))
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(32, 32), 30, 30)

        painter.setPen(QColor("#ffcf70"))
        painter.setBrush(QColor("#f59e0b"))
        star = QPolygonF(
            [
                QPointF(32, 5),
                QPointF(39, 24),
                QPointF(59, 32),
                QPointF(39, 40),
                QPointF(32, 59),
                QPointF(25, 40),
                QPointF(5, 32),
                QPointF(25, 24),
            ]
        )
        painter.drawPolygon(star)
        painter.end()

        return QIcon(pixmap)

    # ── Folder browsing ────────────────────────────────────────────────────

    @Slot()
    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.txt_input.setText(folder)
            self._update_summary()

    @Slot()
    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.txt_output.setText(folder)
            self._update_summary()

    # ── UI reactions ───────────────────────────────────────────────────────

    @Slot(str)
    def _on_method_changed(self, method: str):
        self._set_k_sigma_visible("K-Sigma" in method)

    def _set_k_sigma_visible(self, visible: bool):
        for w in (self.lbl_iter, self.spin_iter, self.lbl_kappa, self.spin_kappa):
            w.setVisible(visible)

    @Slot()
    def _update_summary(self):
        folder = self.txt_input.text()
        if not folder or not Path(folder).is_dir():
            self.lbl_summary.setText("Select an input folder to begin.")
            self.btn_start.setEnabled(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            return

        fits_files = (
            list(Path(folder).glob("*.fits"))
            + list(Path(folder).glob("*.fit"))
            + list(Path(folder).glob("*.fts"))
            + list(Path(folder).glob("*.FITS"))
            + list(Path(folder).glob("*.FIT"))
            + list(Path(folder).glob("*.FTS"))
        )
        seen: set[str] = set()
        count = 0
        for f in fits_files:
            key = str(f).lower()
            if key not in seen:
                seen.add(key)
                count += 1

        gs = self.spin_group.value()
        full_groups = count // gs
        remainder = count % gs
        include_inc = self.chk_incomplete.isChecked()
        total_groups = full_groups + (1 if remainder > 0 and include_inc else 0)

        parts = [f"Files: {count}", f"Groups: {total_groups}"]
        if remainder > 0:
            if include_inc:
                parts.append(f"(last group: {remainder} frames)")
            else:
                parts.append(f"({remainder} skipped)")

        self.lbl_summary.setText("  |  ".join(parts))
        self.btn_start.setEnabled(count > 0 and bool(self.txt_output.text()))

    # ── Start / Cancel ─────────────────────────────────────────────────────

    @Slot()
    def _start(self):
        input_folder = self.txt_input.text()
        output_folder = self.txt_output.text()

        if not input_folder or not Path(input_folder).is_dir():
            QMessageBox.warning(self, "BatchStack", "Please select a valid input folder.")
            return
        if not output_folder:
            QMessageBox.warning(self, "BatchStack", "Please select an output folder.")
            return

        sort_by = "date-obs" if self.combo_sort.currentIndex() == 1 else "filename"
        naming = "range" if self.radio_range.isChecked() else "sequential"

        self.worker = StackWorker(
            input_folder=input_folder,
            output_folder=output_folder,
            group_size=self.spin_group.value(),
            method=self.combo_method.currentText(),
            kappa=self.spin_kappa.value(),
            iterations=self.spin_iter.value(),
            naming_style=naming,
            prefix=self.txt_prefix.text() or "stack",
            include_incomplete=self.chk_incomplete.isChecked(),
            sort_by=sort_by,
            dtype=self.combo_dtype.currentText(),
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.group_done.connect(self._on_group_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self._set_running(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.log_area.clear()
        self._log("Starting...")
        self.statusBar().showMessage("Stacking in progress...")
        self.worker.start()

    @Slot()
    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._log("Cancellation requested...")
            self.btn_cancel.setEnabled(False)

    # ── Worker signal handlers ─────────────────────────────────────────────

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self._log(message)

    @Slot(int, str)
    def _on_group_done(self, index: int, path: str):
        pass

    @Slot(int, float)
    def _on_finished(self, total: int, elapsed: float):
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m {secs:.1f}s" if mins >= 1 else f"{secs:.1f}s"
        self._log(f"Done! {total} group(s) stacked in {time_str}.")
        self.statusBar().showMessage(f"Completed: {total} groups in {time_str}")
        self._set_running(False)
        self._update_summary()

    @Slot(str)
    def _on_error(self, message: str):
        self._log(f"ERROR: {message}")
        self.statusBar().showMessage("Error occurred")
        self._set_running(False)
        self._update_summary()
        QMessageBox.critical(self, "BatchStack - Error", message)

    # ── About ──────────────────────────────────────────────────────────────

    @Slot()
    def _help(self):
        QDesktopServices.openUrl(QUrl("https://github.com/SahilPurabiya/BatchStacking"))

    @Slot()
    def _about(self):
        QMessageBox.about(
            self,
            "About BatchStack",
            "BatchStack v1.0\n"
            "Developed by Sahil Purabiya\n\n"
            "Batch FITS image stacker for\n"
            "astronomy & astrophotography.\n\n"
            "Stacks consecutive groups of FITS frames\n"
            "using Average, Maximum, Median,\n"
            "or K-Sigma methods.",
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        for w in (
            self.txt_input, self.txt_output,
            self.btn_browse_input, self.btn_browse_output,
            self.spin_group,
            self.combo_method, self.spin_iter, self.spin_kappa,
            self.combo_sort, self.radio_seq, self.radio_range,
            self.txt_prefix, self.chk_incomplete, self.combo_dtype,
        ):
            w.setEnabled(not running)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "BatchStack",
                "Stacking is still running. Cancel it and close BatchStack?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()

    def _log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.appendPlainText(f"[{ts}] {message}")
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )


# ── Standalone launch ──────────────────────────────────────────────────────────

def launch():
    """Create and show the BatchStack window."""
    app_instance = QApplication.instance()
    app = QApplication(sys.argv) if app_instance is None else cast(QApplication, app_instance)
    app.setWindowIcon(BatchStackWindow._make_window_icon())
    window = BatchStackWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
