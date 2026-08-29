from model import (
    CalibrationPoint,
    calibrated_linear_angles,
    ModelParameters,
    geometric_weights,
    interpolate_total_amplitude,
    linear_moment_explanation,
    moment_result,
    nonlinear_moment_explanation,
    normalize_calibration_points,
    pad_amplitudes,
    parse_calibration_lines,
    reconstruct_angles_by_model,
    segment_energy,
)


def test_centered_spot_splits_evenly_between_pads():
    params = ModelParameters()
    weights = geometric_weights(params, 0.0, 0.0)
    assert all(abs(a - b) < 1e-12 for a, b in zip(weights, [0.25, 0.25, 0.25, 0.25]))


def test_calibration_exact_point_controls_total_amplitude():
    params = ModelParameters(calibration_points=[CalibrationPoint(0.0, 0.0, (1.0, 2.0, 3.0, 4.0))])
    assert interpolate_total_amplitude(params, 0.0, 0.0) == 10.0
    assert abs(sum(pad_amplitudes(params, 0.0, 0.0)) - 10.0) < 1e-12


def test_angle_to_linear_position_uses_focal_length():
    params = ModelParameters(focal_length_mm=11.8)
    x, y = params.spot_center_mm(1.0, -1.0)
    assert x > 0
    assert y < 0
    assert abs(abs(x) - abs(y)) < 1e-12


def test_shifted_spot_favors_right_segments():
    params = ModelParameters(integration_step_mm=0.02)
    q1, q2, q3, q4 = segment_energy(params, 1.0, 0.0)
    assert q2 + q3 > q1 + q4


def test_total_signal_drops_when_spot_leaves_photodiode():
    params = ModelParameters(integration_step_mm=0.02)
    centered = sum(pad_amplitudes(params, 0.0, 0.0))
    outside = sum(pad_amplitudes(params, 8.0, 0.0))
    assert outside < centered


def test_moment_result_reconstructs_positive_x_angle():
    params = ModelParameters()
    result = moment_result(params, (1.0, 2.0, 2.0, 1.0))
    assert result.normalized_x > 0
    assert result.normalized_y == 0
    assert result.angle_x_rad > 0


def test_calibrated_linear_angle_is_closer_than_radius_scaling_near_center():
    params = ModelParameters(integration_step_mm=0.03)
    amplitudes = segment_energy(params, 0.0, 0.5)
    naive = moment_result(params, amplitudes).angle_y_deg
    _, _, _, calibrated = calibrated_linear_angles(params, amplitudes)
    assert abs(calibrated - 0.5) < abs(naive - 0.5)


def test_model_reconstruction_matches_simulated_axis_angle():
    params = ModelParameters(integration_step_mm=0.03)
    amplitudes = segment_energy(params, 0.0, 1.3)
    ax, ay = reconstruct_angles_by_model(params, amplitudes)
    assert abs(ax) <= params.angle_step_deg
    assert abs(ay - 1.3) <= params.angle_step_deg


def test_explanations_include_substitution_and_three_decimal_angles():
    params = ModelParameters(integration_step_mm=0.03)
    amplitudes = segment_energy(params, 0.0, 1.3)
    linear = linear_moment_explanation(params, amplitudes)
    nonlinear = nonlinear_moment_explanation(params, amplitudes)
    assert "My = ((" in linear
    assert "αy =" in linear
    assert "Ky = (" in nonlinear
    assert "Угол по полной модели" in nonlinear


def test_manual_calibration_parser_accepts_header_and_clockwise_values():
    points = parse_calibration_lines([
        "ax_deg,ay_deg,q1,q2,q3,q4",
        "0, 0, 1, 2, 3, 4",
    ])
    assert points == [CalibrationPoint(0.0, 0.0, (1.0, 2.0, 3.0, 4.0))]


def test_normalize_calibration_points_accepts_flat_numeric_list():
    points = normalize_calibration_points([0, 0, 1, 2, 3, 4])
    assert points == [CalibrationPoint(0.0, 0.0, (1.0, 2.0, 3.0, 4.0))]


def test_interpolation_accepts_flat_numeric_calibration_list():
    params = ModelParameters(calibration_points=[0, 0, 1, 2, 3, 4])
    assert interpolate_total_amplitude(params, 0.0, 0.0) == 10.0


if __name__ == "__main__":
    params = ModelParameters()
    amplitudes = pad_amplitudes(params, 0.0, 0.0)
    print("Проверочный расчет для ax=0°, ay=0°")
    print(f"Амплитуды Q1-Q4: {', '.join(f'{value:.6f}' for value in amplitudes)}")
    print("Для полной проверки запустите: python -m pytest")