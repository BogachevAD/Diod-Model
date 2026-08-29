from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import atan, degrees, exp, log, pi, radians, sqrt, tan


@dataclass(frozen=True)
class CalibrationPoint:
    """Known measured amplitudes at a beam angle."""

    angle_x_deg: float
    angle_y_deg: float
    amplitudes: tuple[float, float, float, float]


@dataclass(frozen=True)
class MomentResult:
    """Centroid/moment reconstruction from four pad amplitudes."""

    sum_signal: float
    normalized_x: float
    normalized_y: float
    x_mm: float
    y_mm: float
    angle_x_rad: float
    angle_y_rad: float
    angle_x_deg: float
    angle_y_deg: float
    model_angle_x_deg: float | None = None
    model_angle_y_deg: float | None = None


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


def moment_result(params: ModelParameters, amplitudes: Sequence[float]) -> MomentResult:
    """Reconstruct normalized moments, linear spot coordinates and angles.

    Amplitude order is clockwise from the upper-left segment:
    Q1 upper-left, Q2 upper-right, Q3 lower-right, Q4 lower-left.
    Normalized coordinates are computed from quadrant sums and then scaled by
    the active-area radius to get millimetres. Angles use atan(offset/focus).
    """

    if len(amplitudes) != 4:
        raise ValueError("Moment calculation requires exactly four amplitudes")
    q1, q2, q3, q4 = (float(value) for value in amplitudes)
    total = q1 + q2 + q3 + q4
    if total <= 0:
        return MomentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    normalized_x = ((q2 + q3) - (q1 + q4)) / total
    normalized_y = ((q1 + q2) - (q3 + q4)) / total
    x_mm = normalized_x * params.diode_radius_mm
    y_mm = normalized_y * params.diode_radius_mm
    angle_x_rad = atan(x_mm / params.focal_length_mm)
    angle_y_rad = atan(y_mm / params.focal_length_mm)
    return MomentResult(
        total,
        normalized_x,
        normalized_y,
        x_mm,
        y_mm,
        angle_x_rad,
        angle_y_rad,
        degrees(angle_x_rad),
        degrees(angle_y_rad),
    )


def moment_sensitivity_details(params: ModelParameters) -> dict[str, float]:
    """Return finite-difference values used for local moment sensitivity."""

    delta_angle = max(params.angle_step_deg / 10.0, 1e-4)
    x_plus = params.spot_center_mm(delta_angle, 0.0)[0]
    x_minus = params.spot_center_mm(-delta_angle, 0.0)[0]
    y_plus = params.spot_center_mm(0.0, delta_angle)[1]
    y_minus = params.spot_center_mm(0.0, -delta_angle)[1]
    mx_plus = moment_result(params, segment_energy(params, delta_angle, 0.0)).normalized_x
    mx_minus = moment_result(params, segment_energy(params, -delta_angle, 0.0)).normalized_x
    my_plus = moment_result(params, segment_energy(params, 0.0, delta_angle)).normalized_y
    my_minus = moment_result(params, segment_energy(params, 0.0, -delta_angle)).normalized_y
    kx = (mx_plus - mx_minus) / (x_plus - x_minus) if x_plus != x_minus else 0.0
    ky = (my_plus - my_minus) / (y_plus - y_minus) if y_plus != y_minus else 0.0
    return {
        "delta_angle_deg": delta_angle,
        "x_plus_mm": x_plus,
        "x_minus_mm": x_minus,
        "y_plus_mm": y_plus,
        "y_minus_mm": y_minus,
        "mx_plus": mx_plus,
        "mx_minus": mx_minus,
        "my_plus": my_plus,
        "my_minus": my_minus,
        "kx": kx,
        "ky": ky,
    }


def moment_sensitivity_coefficients(params: ModelParameters) -> tuple[float, float]:
    """Estimate local slopes dMx/dx and dMy/dy near the optical centre."""

    details = moment_sensitivity_details(params)
    return details["kx"], details["ky"]


def calibrated_linear_angles(params: ModelParameters, amplitudes: Sequence[float]) -> tuple[float, float, float, float]:
    """Return locally calibrated x/y offsets and angles from moments."""

    result = moment_result(params, amplitudes)
    kx, ky = moment_sensitivity_coefficients(params)
    x_mm = result.normalized_x / kx if kx else 0.0
    y_mm = result.normalized_y / ky if ky else 0.0
    return x_mm, y_mm, degrees(atan(x_mm / params.focal_length_mm)), degrees(atan(y_mm / params.focal_length_mm))


def reconstruct_angles_by_model(params: ModelParameters, amplitudes: Sequence[float]) -> tuple[float, float]:
    """Find angles whose simulated one-axis moments best match amplitudes."""

    target = moment_result(params, amplitudes)
    count = round(params.field_deg / params.angle_step_deg)

    best_x_error = float("inf")
    best_x = 0.0
    best_y_error = float("inf")
    best_y = 0.0
    for index in range(count + 1):
        angle = -params.half_field_deg + index * params.angle_step_deg
        simulated_x = moment_result(params, segment_energy(params, angle, 0.0))
        x_error = abs(simulated_x.normalized_x - target.normalized_x)
        if x_error < best_x_error:
            best_x_error = x_error
            best_x = angle

        simulated_y = moment_result(params, segment_energy(params, 0.0, angle))
        y_error = abs(simulated_y.normalized_y - target.normalized_y)
        if y_error < best_y_error:
            best_y_error = y_error
            best_y = angle
    return best_x, best_y


def linear_moment_explanation(params: ModelParameters, amplitudes: Sequence[float]) -> str:
    """Return one-line formulas and substitutions for the linear moment check."""

    q1, q2, q3, q4 = (float(value) for value in amplitudes)
    result = moment_result(params, (q1, q2, q3, q4))
    total = result.sum_signal
    if total <= 0:
        return "S = Q1 + Q2 + Q3 + Q4 = 0; линейный расчет невозможен."
    return "\n".join(
        [
            "Линейный блок: грубая шкала x = Mx · R, y = My · R.",
            "S = Q1 + Q2 + Q3 + Q4.",
            f"S = {q1:.6g} + {q2:.6g} + {q3:.6g} + {q4:.6g} = {total:.6g}.",
            "Mx = ((Q2 + Q3) - (Q1 + Q4)) / S.",
            f"Mx = (({q2:.6g} + {q3:.6g}) - ({q1:.6g} + {q4:.6g})) / {total:.6g} = {result.normalized_x:.6g}.",
            "My = ((Q1 + Q2) - (Q3 + Q4)) / S.",
            f"My = (({q1:.6g} + {q2:.6g}) - ({q3:.6g} + {q4:.6g})) / {total:.6g} = {result.normalized_y:.6g}.",
            "x = Mx · R; y = My · R.",
            f"x = {result.normalized_x:.6g} · {params.diode_radius_mm:.6g} = {result.x_mm:.6g} мм; y = {result.normalized_y:.6g} · {params.diode_radius_mm:.6g} = {result.y_mm:.6g} мм.",
            "αx = atan(x / f); αy = atan(y / f).",
            f"αx = atan({result.x_mm:.6g} / {params.focal_length_mm:.6g}) = {result.angle_x_deg:.3f}°; αy = atan({result.y_mm:.6g} / {params.focal_length_mm:.6g}) = {result.angle_y_deg:.3f}°.",
        ]
    )


def nonlinear_moment_explanation(params: ModelParameters, amplitudes: Sequence[float]) -> str:
    """Return detailed calibrated/nonlinear moment reconstruction text."""

    q1, q2, q3, q4 = (float(value) for value in amplitudes)
    result = moment_result(params, (q1, q2, q3, q4))
    details = moment_sensitivity_details(params)
    calibrated_x, calibrated_y, calibrated_ax, calibrated_ay = calibrated_linear_angles(params, (q1, q2, q3, q4))
    model_ax, model_ay = reconstruct_angles_by_model(params, (q1, q2, q3, q4))
    total = result.sum_signal
    if total <= 0:
        return "S = Q1 + Q2 + Q3 + Q4 = 0; нелинейный расчет невозможен."
    return "\n".join(
        [
            "Нелинейный блок: коэффициент K считается из производной около центра, затем выполняется обратный поиск по модели.",
            f"Исходные моменты: Mx = {result.normalized_x:.6g}; My = {result.normalized_y:.6g}.",
            "Kx = dMx/dx ≈ (Mx(+δ) - Mx(-δ)) / (x(+δ) - x(-δ)).",
            f"δ = {details['delta_angle_deg']:.6g}°; x(+δ) = {details['x_plus_mm']:.6g} мм; x(-δ) = {details['x_minus_mm']:.6g} мм; Mx(+δ) = {details['mx_plus']:.6g}; Mx(-δ) = {details['mx_minus']:.6g}.",
            f"Kx = ({details['mx_plus']:.6g} - {details['mx_minus']:.6g}) / ({details['x_plus_mm']:.6g} - {details['x_minus_mm']:.6g}) = {details['kx']:.6g} 1/мм.",
            "Ky = dMy/dy ≈ (My(+δ) - My(-δ)) / (y(+δ) - y(-δ)).",
            f"δ = {details['delta_angle_deg']:.6g}°; y(+δ) = {details['y_plus_mm']:.6g} мм; y(-δ) = {details['y_minus_mm']:.6g} мм; My(+δ) = {details['my_plus']:.6g}; My(-δ) = {details['my_minus']:.6g}.",
            f"Ky = ({details['my_plus']:.6g} - {details['my_minus']:.6g}) / ({details['y_plus_mm']:.6g} - {details['y_minus_mm']:.6g}) = {details['ky']:.6g} 1/мм.",
            "Калиброванная линейная формула: x = Mx / Kx; y = My / Ky.",
            f"x = {result.normalized_x:.6g} / {details['kx']:.6g} = {calibrated_x:.6g} мм; y = {result.normalized_y:.6g} / {details['ky']:.6g} = {calibrated_y:.6g} мм.",
            f"Угол по калиброванной линейной формуле: αx = {calibrated_ax:.3f}°; αy = {calibrated_ay:.3f}°.",
            "Обратный поиск: для каждого угла сетки моделируются Q1..Q4, считаются Mx/My, выбирается минимальная ошибка момента.",
            f"Угол по полной модели: αx = {model_ax:.3f}°; αy = {model_ay:.3f}°.",
        ]
    )


def moment_explanation(params: ModelParameters, amplitudes: Sequence[float]) -> str:
    """Return both linear and nonlinear moment explanations."""

    return linear_moment_explanation(params, amplitudes) + "\n\n" + nonlinear_moment_explanation(params, amplitudes)


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