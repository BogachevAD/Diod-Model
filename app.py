from __future__ import annotations

import csv
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULT_CALIBRATION_POINTS, DEFAULT_PARAMETERS
from model import CalibrationPoint, ModelParameters, angle_grid, moment_explanation, normalize_calibration_points, pad_amplitudes, parse_calibration_lines


class PhotodiodeWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Diod Model")
        self.default_params = ModelParameters(**DEFAULT_PARAMETERS.__dict__)
        self.default_calibration = normalize_calibration_points(DEFAULT_CALIBRATION_POINTS or self.default_params.calibration_points)
        self.params = ModelParameters(**self.default_params.__dict__)
        self.params.calibration_points = list(self.default_calibration)
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.scale = 120.0

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        main.addLayout(left, 2)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        left.addWidget(self.view, 3)
        self.position_label = QLabel()
        left.addWidget(self.position_label)
        left.addWidget(QLabel("Проверка по формулам моментов"))
        self.moment_text = QTextEdit()
        self.moment_text.setReadOnly(True)
        self.moment_text.setMinimumHeight(260)
        left.addWidget(self.moment_text, 2)

        right = QVBoxLayout()
        main.addLayout(right, 1)
        form = QFormLayout()
        right.addLayout(form)
        self.controls: dict[str, QDoubleSpinBox] = {}
        for key, title, value, minimum, maximum, step in [
            ("diode_diameter_mm", "Диаметр фотодиода, мм", 1.5, 0.1, 10.0, 0.1),
            ("gap_um", "Зазор, мкм", 100.0, 0.0, 2000.0, 10.0),
            ("beam_diameter_mm", "Диаметр пятна, мм", 1.0, 0.05, 10.0, 0.05),
            ("focal_length_mm", "Фокус, мм", 11.8, 0.1, 1000.0, 0.1),
            ("field_deg", "Поле, град", 5.0, 0.1, 180.0, 0.1),
            ("angle_step_deg", "Шаг, град", 0.1, 0.01, 10.0, 0.01),
            ("energy_fraction_diameter_part", "Доля диаметра", 0.9, 0.01, 1.0, 0.01),
            ("energy_fraction", "Доля энергии", 0.8, 0.01, 0.999, 0.01),
        ]:
            box = QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setSingleStep(step)
            box.setDecimals(3)
            box.setValue(getattr(self.params, key, value))
            box.valueChanged.connect(self.update_parameters)
            self.controls[key] = box
            form.addRow(title, box)

        self.x_slider = self._make_slider("Угол X")
        self.y_slider = self._make_slider("Угол Y")
        right.addWidget(QLabel("Известные значения: ax_deg, ay_deg, q1, q2, q3, q4"))
        self.manual_points = QTextEdit()
        self.manual_points.setPlaceholderText("Каждая строка: ax_deg, ay_deg, q1, q2, q3, q4\n0, 0, 1, 1, 1, 1\n1, 0, 0.2, 1.7, 1.7, 0.2")
        self.manual_points.setFixedHeight(110)
        right.addWidget(self.manual_points)
        apply_manual_button = QPushButton("Применить ручной список")
        apply_manual_button.clicked.connect(self.apply_manual_calibration)
        right.addWidget(apply_manual_button)
        load_button = QPushButton("Загрузить калибровку CSV")
        load_button.clicked.connect(self.load_calibration)
        right.addWidget(load_button)
        reset_button = QPushButton("Сбросить все")
        reset_button.clicked.connect(self.reset_defaults)
        right.addWidget(reset_button)
        export_button = QPushButton("Экспорт сетки 0.1° CSV")
        export_button.clicked.connect(self.export_grid)
        right.addWidget(export_button)

        self.table = QTableWidget(4, 2)
        self.table.setHorizontalHeaderLabels(["Площадка", "Амплитуда"])
        right.addWidget(self.table)
        self.update_parameters()

    def reset_defaults(self) -> None:
        for key, box in self.controls.items():
            box.blockSignals(True)
            box.setValue(getattr(self.default_params, key))
            box.blockSignals(False)
        self.params = ModelParameters(**self.default_params.__dict__)
        self.params.calibration_points = list(self.default_calibration)
        self.manual_points.clear()
        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.x_slider.setValue(0)
        self.y_slider.setValue(0)
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.update_parameters()

    def _make_slider(self, title: str) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.valueChanged.connect(self.update_angles)
        self.centralWidget().layout().itemAt(1).layout().addWidget(QLabel(title))
        self.centralWidget().layout().itemAt(1).layout().addWidget(slider)
        return slider

    def update_parameters(self) -> None:
        calibration = normalize_calibration_points(self.params.calibration_points)
        self.params = ModelParameters(**{key: box.value() for key, box in self.controls.items()})
        self.params.calibration_points = calibration
        maximum = round(self.params.half_field_deg / self.params.angle_step_deg)
        for slider in (self.x_slider, self.y_slider):
            slider.blockSignals(True)
            slider.setRange(-maximum, maximum)
            slider.blockSignals(False)
        self.update_angles()

    def update_angles(self) -> None:
        self.angle_x = self.x_slider.value() * self.params.angle_step_deg
        self.angle_y = self.y_slider.value() * self.params.angle_step_deg
        self.redraw()

    def redraw(self) -> None:
        self.scene.clear()
        radius = self.params.diode_radius_mm
        gap = self.params.gap_mm
        diode = QGraphicsEllipseItem(-radius * self.scale, -radius * self.scale, 2 * radius * self.scale, 2 * radius * self.scale)
        diode.setBrush(QBrush(QColor(210, 230, 255)))
        diode.setPen(QPen(Qt.GlobalColor.darkBlue, 2))
        self.scene.addItem(diode)

        vertical_gap = QGraphicsRectItem(-gap * self.scale / 2, -radius * self.scale, gap * self.scale, 2 * radius * self.scale)
        horizontal_gap = QGraphicsRectItem(-radius * self.scale, -gap * self.scale / 2, 2 * radius * self.scale, gap * self.scale)
        for gap_item in (vertical_gap, horizontal_gap):
            gap_item.setBrush(QBrush(QColor(245, 245, 245)))
            gap_item.setPen(QPen(Qt.GlobalColor.gray, 1))
            self.scene.addItem(gap_item)

        label_positions = [(-0.45 * radius, 0.45 * radius), (0.35 * radius, 0.45 * radius), (0.35 * radius, -0.45 * radius), (-0.45 * radius, -0.45 * radius)]
        for index, (label_x, label_y) in enumerate(label_positions, start=1):
            text = self.scene.addText(f"Q{index}")
            text.setPos(label_x * self.scale, -label_y * self.scale)

        cx, cy = self.params.spot_center_mm(self.angle_x, self.angle_y)
        radius = self.params.beam_diameter_mm / 2.0
        spot = QGraphicsEllipseItem((cx - radius) * self.scale, -(cy + radius) * self.scale, 2 * radius * self.scale, 2 * radius * self.scale)
        spot.setBrush(QBrush(QColor(255, 80, 80, 120)))
        spot.setPen(QPen(Qt.GlobalColor.red, 2))
        self.scene.addItem(spot)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30))

        amplitudes = pad_amplitudes(self.params, self.angle_x, self.angle_y)
        for row, value in enumerate(amplitudes):
            self.table.setItem(row, 0, QTableWidgetItem(f"Q{row + 1}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{value:.6f}"))
        self.moment_text.setPlainText(moment_explanation(self.params, amplitudes))
        self.position_label.setText(f"Угол: X={self.angle_x:.3f}°, Y={self.angle_y:.3f}°; центр пятна: x={cx:.3f} мм, y={cy:.3f} мм")

    def load_calibration(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "CSV калибровки", str(Path.cwd()), "CSV (*.csv)")
        if not file_name:
            return
        points: list[CalibrationPoint] = []
        with open(file_name, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                points.append(CalibrationPoint(float(row["ax_deg"]), float(row["ay_deg"]), (float(row["q1"]), float(row["q2"]), float(row["q3"]), float(row["q4"]))))
        self.params.calibration_points = points
        self.redraw()

    def apply_manual_calibration(self) -> None:
        text = self.manual_points.toPlainText().strip()
        if not text:
            self.params.calibration_points = []
            self.redraw()
            return
        self.params.calibration_points = parse_calibration_lines(text.splitlines())
        self.redraw()

    def export_grid(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(self, "Экспорт сетки", "diod_grid.csv", "CSV (*.csv)")
        if not file_name:
            return
        with open(file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ax_deg", "ay_deg", "q1", "q2", "q3", "q4"])
            for ax, ay, amps in angle_grid(self.params):
                writer.writerow([ax, ay, *amps])


def main() -> None:
    app = QApplication(sys.argv)
    window = PhotodiodeWindow()
    window.resize(1100, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()