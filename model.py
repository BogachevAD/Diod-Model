from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import exp, log, pi, radians, sqrt, tan


@dataclass(frozen=True)
class CalibrationPoint:
    """Known measured amplitudes at a beam angle."""

    angle_x_deg: float
    angle_y_deg: float
    amplitudes: tuple[float, float, float, float]


@dataclass
class ModelParameters:
    """Numerical parameters of the optical and diode model."""

    diode_diameter_mm: float = 1.5
    gap_um: float = 100.0
    beam_diameter_mm: float = 1.0
    focal_length_mm: float = 11.8
    field_deg: float = 5.0
    angle_step_deg: float = 0.1
    energy_fraction_diameter_part: float = 0.9
    energy_fraction: float = 0.8
    integration_step_mm: float = 0.025
    calibration_points: list[CalibrationPoint] | list[Sequence[float]] | list[float] = field(default_factory=list)

    @property
    def half_field_deg(self) -> float:
        return self.field_deg / 2.0

    @property
    def gap_mm(self) -> float:
        return self.gap_um / 1000.0

    @property
    def beam_sigma_mm(self) -> float:
        """Gaussian sigma where given diameter fraction contains known energy.

        For a circular 2D Gaussian the energy inside radius r is
        1 - exp(-r^2 / (2 sigma^2)). The configured fraction refers to the
        diameter part, so r equals beam_diameter * part / 2.
        """

        radius = self.beam_diameter_mm * self.energy_fraction_diameter_part / 2.0
        return radius / sqrt(-2.0 * log(1.0 - self.energy_fraction))

    def spot_center_mm(self, angle_x_deg: float, angle_y_deg: float) -> tuple[float, float]:
        return (
            self.focal_length_mm * tan(radians(angle_x_deg)),
            self.focal_length_mm * tan(radians(angle_y_deg)),
        )

    @property
    def diode_radius_mm(self) -> float:
        return self.diode_diameter_mm / 2.0

    def segment_bounds_mm(self) -> tuple[float, float, float, float]:
        """Return the common bounding box of the circular photosensitive area."""

        radius = self.diode_radius_mm
        return -radius, radius, -radius, radius


def _gaussian_density(x: float, y: float, center: tuple[float, float], sigma: float) -> float:
    cx, cy = center
    exponent = -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)
    return exp(exponent) / (2.0 * pi * sigma**2)


def _segment_index(x: float, y: float, half_gap: float, radius: float) -> int | None:
    if x * x + y * y > radius * radius:
        return None
    if abs(x) < half_gap or abs(y) < half_gap:
        return None
    if x < 0 and y > 0:
        return 0
    if x > 0 and y > 0:
        return 1
    if x > 0 and y < 0:
        return 2
    if x < 0 and y < 0:
        return 3
    return None


def segment_energy(params: ModelParameters, angle_x_deg: float, angle_y_deg: float) -> tuple[float, float, float, float]:
    """Numerically integrate Gaussian energy over four circular diode segments.

    The photodiode is one circular photosensitive area split by vertical and
    horizontal dead gaps. Segment order is clockwise from the upper-left pad:
    Q1 upper-left, Q2 upper-right, Q3 lower-right, Q4 lower-left.
    """

    radius = params.diode_radius_mm
    half_gap = params.gap_mm / 2.0
    step = params.integration_step_mm
    center = params.spot_center_mm(angle_x_deg, angle_y_deg)
    values = [0.0, 0.0, 0.0, 0.0]

    sample_count = max(1, int((2.0 * radius) / step))
    actual_step = (2.0 * radius) / sample_count
    start = -radius + actual_step / 2.0
    for ix in range(sample_count):
        x = start + ix * actual_step
        for iy in range(sample_count):
            y = start + iy * actual_step
            segment = _segment_index(x, y, half_gap, radius)
            if segment is not None:
                values[segment] += _gaussian_density(x, y, center, params.beam_sigma_mm)

    cell_area = actual_step * actual_step
    return tuple(value * cell_area for value in values)


def geometric_weights(params: ModelParameters, angle_x_deg: float, angle_y_deg: float) -> tuple[float, float, float, float]:
    """Compute normalized Gaussian energy overlap for each circular segment."""

    raw = segment_energy(params, angle_x_deg, angle_y_deg)
    total = sum(raw)
    return tuple(value / total for value in raw) if total > 0 else raw


def normalize_calibration_points(raw_points: object) -> list[CalibrationPoint]:
    """Convert supported calibration formats to CalibrationPoint objects.

    Supported inputs:
    - [CalibrationPoint(...), ...]
    - [(ax, ay, q1, q2, q3, q4), ...]
    - [[ax, ay, q1, q2, q3, q4], ...]
    - [ax, ay, q1, q2, q3, q4] for one point

    This keeps the application from crashing if startup values are edited as a
    plain numeric list in config.py.
    """

    if raw_points is None:
        return []
    if isinstance(raw_points, CalibrationPoint):
        return [raw_points]
    if not isinstance(raw_points, Sequence):
        raise TypeError("Calibration points must be a sequence")
    if not raw_points:
        return []

    if all(isinstance(value, (int, float)) for value in raw_points):
        if len(raw_points) % 6 != 0:
            raise ValueError("Flat calibration list length must be a multiple of 6")
        rows = [raw_points[index : index + 6] for index in range(0, len(raw_points), 6)]
    else:
        rows = raw_points

    points: list[CalibrationPoint] = []
    for row in rows:
        if isinstance(row, CalibrationPoint):
            points.append(row)
            continue
        if not isinstance(row, Sequence) or len(row) != 6:
            raise ValueError(f"Calibration row must contain 6 numbers: {row}")
        values = [float(value) for value in row]
        points.append(CalibrationPoint(values[0], values[1], (values[2], values[3], values[4], values[5])))
    return points


def interpolate_signal_gain(params: ModelParameters, angle_x_deg: float, angle_y_deg: float) -> float:
    """Interpolate optical/electrical gain with inverse-distance weighting.

    Calibration points determine an optical/electrical gain: measured total
    signal divided by modelled collected Gaussian energy at the calibration
    point. Without calibration, unit gain is used. Because pad_amplitudes uses
    absolute collected energy, signal naturally decreases when the spot leaves
    the circular photodiode.
    """

    points = normalize_calibration_points(params.calibration_points)
    if not points:
        return 1.0

    weighted_sum = 0.0
    weight_total = 0.0
    for point in points:
        measured_total = sum(point.amplitudes)
        model_total = sum(segment_energy(params, point.angle_x_deg, point.angle_y_deg))
        gain = measured_total / model_total if model_total > 0 else 0.0
        distance = sqrt((point.angle_x_deg - angle_x_deg) ** 2 + (point.angle_y_deg - angle_y_deg) ** 2)
        if distance < 1e-12:
            return float(gain)
        weight = 1.0 / distance**2
        weighted_sum += weight * gain
        weight_total += weight
    return weighted_sum / weight_total


def interpolate_total_amplitude(params: ModelParameters, angle_x_deg: float, angle_y_deg: float) -> float:
    """Predict the total signal collected by all diode segments."""

    return interpolate_signal_gain(params, angle_x_deg, angle_y_deg) * sum(segment_energy(params, angle_x_deg, angle_y_deg))


def pad_amplitudes(params: ModelParameters, angle_x_deg: float, angle_y_deg: float) -> tuple[float, float, float, float]:
    gain = interpolate_signal_gain(params, angle_x_deg, angle_y_deg)
    return tuple(value * gain for value in segment_energy(params, angle_x_deg, angle_y_deg))


def parse_calibration_lines(lines: Iterable[str]) -> list[CalibrationPoint]:
    """Parse manual calibration rows: ax, ay, q1, q2, q3, q4.

    Semicolons are accepted as separators too. A header line and empty/comment
    lines are ignored. Pad amplitudes are ordered clockwise from upper-left.
    """

    points: list[CalibrationPoint] = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#"):
            continue
        parts = [part.strip() for part in clean_line.replace(";", ",").split(",")]
        if parts[0].lower() in {"ax", "ax_deg", "angle_x", "angle_x_deg"}:
            continue
        if len(parts) != 6:
            raise ValueError(f"Calibration line must contain 6 numbers: {line}")
        values = [float(part) for part in parts]
        points.append(CalibrationPoint(values[0], values[1], (values[2], values[3], values[4], values[5])))
    return points


def angle_grid(params: ModelParameters) -> Iterable[tuple[float, float, tuple[float, float, float, float]]]:
    count = round(params.field_deg / params.angle_step_deg)
    for iy in range(count + 1):
        ay = -params.half_field_deg + iy * params.angle_step_deg
        for ix in range(count + 1):
            ax = -params.half_field_deg + ix * params.angle_step_deg
            yield ax, ay, pad_amplitudes(params, ax, ay)