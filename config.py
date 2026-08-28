"""User-editable startup values for the photodiode simulator.

Change the numbers here when you want a different default experiment before
starting the graphical interface with ``python run.py``.
"""

from __future__ import annotations

from model import CalibrationPoint, ModelParameters


DEFAULT_PARAMETERS = ModelParameters(
    diode_diameter_mm=1.5,
    gap_um=100.0,
    beam_diameter_mm=1.0,
    focal_length_mm=11.8,
    field_deg=5.0,
    angle_step_deg=0.1,
    energy_fraction_diameter_part=0.9,
    energy_fraction=0.8,
    # You may also put initial measurements here instead of DEFAULT_CALIBRATION_POINTS:
    # calibration_points=[CalibrationPoint(0, 0, (1, 1, 1, 1))],
)

# Startup calibration values are optional. The GUI can also load CSV or accept
# the same rows manually.
# Supported formats:
#   [CalibrationPoint(0, 0, (1, 1, 1, 1))]
#   [(0, 0, 1, 1, 1, 1), (1, 0, 0.2, 1.7, 1.7, 0.2)]
#   [0, 0, 1, 1, 1, 1]  # one flat row is also accepted
# Format: angle_x_deg, angle_y_deg, q1, q2, q3, q4.
# Pads are ordered clockwise from the upper-left pad:
# Q1 = upper-left, Q2 = upper-right, Q3 = lower-right, Q4 = lower-left.
DEFAULT_CALIBRATION_POINTS: list[CalibrationPoint] | list[tuple[float, float, float, float, float, float]] | list[float] = []
