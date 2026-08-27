import numpy as np
import pytest

# Update 'your_module' to match the file name where your calibrators are defined.
from nsbi_common_utils.calibration import (
    IsotonicCalibrator,
    HistogramCalibrator,
    PlattScalingCalibrator,
)

# =====================================================================
# Tests for IsotonicCalibrator
# =====================================================================


class TestIsotonicCalibrator:

    def test_fit_and_prediction_shape(self, binary_dataset):
        ratios, labels, weights = binary_dataset
        calibrator = IsotonicCalibrator(ratios, labels, weights)

        test_ratios = np.array([0.2, 1.0, 3.5])
        calibrated = calibrator.cali_pred(test_ratios)

        assert calibrated.shape == test_ratios.shape
        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated > 0)

    def test_monotonicity(self, binary_dataset):
        ratios, labels, weights = binary_dataset
        calibrator = IsotonicCalibrator(ratios, labels, weights)

        test_ratios = np.linspace(0.1, 5.0, 50)
        calibrated = calibrator.cali_pred(test_ratios)

        assert np.all(np.diff(calibrated) >= 0.0)

    def test_output_bounds_safety(self, binary_dataset):
        ratios, labels, weights = binary_dataset
        calibrator = IsotonicCalibrator(ratios, labels, weights)

        extreme_inputs = np.array([0.0, 1e-15, 1e15])
        calibrated = calibrator.cali_pred(extreme_inputs)

        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated > 0)


# =====================================================================
# Tests for HistogramCalibrator
# =====================================================================


class TestHistogramCalibrator:

    @pytest.mark.parametrize("mode", ["fixed", "dynamic", "dynamic_unweighted"])
    @pytest.mark.parametrize("method", ["direct", "indirect"])
    def test_binning_modes_and_methods(self, two_class_dataset, mode, method):
        data_num, data_den, w_num, w_den = two_class_dataset
        nbins = 10

        calibrator = HistogramCalibrator(
            data_num, data_den, w_num, w_den, mode=mode, nbins=nbins, method=method
        )

        # Evaluate within the range covered by calibration data
        test_inputs = np.array([0.5, 1.5, 3.0])
        calibrated = calibrator.cali_pred(test_inputs)

        assert len(calibrated) == len(test_inputs)
        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated >= 0)

    def test_invalid_mode_raises(self, two_class_dataset):
        data_num, data_den, w_num, w_den = two_class_dataset

        with pytest.raises(RuntimeError, match="Unknown mode"):
            HistogramCalibrator(data_num, data_den, w_num, w_den, mode="invalid_mode")

    def test_return_hist_structure(self, two_class_dataset):
        data_num, data_den, w_num, w_den = two_class_dataset
        calibrator = HistogramCalibrator(data_num, data_den, w_num, w_den, nbins=5)

        hist_num, hist_den, num_err, den_err, quant_binning = calibrator.return_hist()

        assert len(hist_num) == 5
        assert len(hist_den) == 5
        assert len(num_err) == 5
        assert len(den_err) == 5
        assert len(quant_binning) == 6

    def test_custom_histrange(self, two_class_dataset):
        data_num, data_den, w_num, w_den = two_class_dataset
        histrange = (0.0, 10.0)

        calibrator = HistogramCalibrator(
            data_num, data_den, w_num, w_den, mode="fixed", nbins=5, histrange=histrange
        )

        assert calibrator.range == histrange
        assert calibrator.edges[0] == 0.0
        assert calibrator.edges[-1] == 10.0


# =====================================================================
# Tests for PlattScalingCalibrator
# =====================================================================


class TestPlattScalingCalibrator:

    def test_fit_and_prediction(self, binary_dataset):
        ratios, labels, weights = binary_dataset
        calibrator = PlattScalingCalibrator(ratios, labels, weights)

        assert hasattr(calibrator, "a")
        assert hasattr(calibrator, "b")
        assert np.isfinite(calibrator.a)
        assert np.isfinite(calibrator.b)

        test_ratios = np.array([0.1, 1.0, 10.0])
        calibrated = calibrator.cali_pred(test_ratios)

        assert calibrated.shape == test_ratios.shape
        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated > 0)

    def test_extreme_zero_and_infinite_ratios(self, binary_dataset):
        ratios, labels, weights = binary_dataset
        calibrator = PlattScalingCalibrator(ratios, labels, weights, max_logr=50.0)

        extreme_ratios = np.array([0.0, 1e-40, 1e40, np.inf])
        calibrated = calibrator.cali_pred(extreme_ratios)

        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated > 0)


# =====================================================================
# Cross-Calibrator Consistency Test
# =====================================================================


def test_calibrator_api_contract(binary_dataset, two_class_dataset):
    ratios, labels, weights = binary_dataset
    data_num, data_den, w_num, w_den = two_class_dataset

    iso = IsotonicCalibrator(ratios, labels, weights)
    hist = HistogramCalibrator(data_num, data_den, w_num, w_den)
    platt = PlattScalingCalibrator(ratios, labels, weights)

    eval_data = np.array([0.5, 1.0, 2.5])

    res_iso = iso.cali_pred(eval_data)
    res_hist = hist.cali_pred(eval_data)
    res_platt = platt.cali_pred(eval_data)

    for res in (res_iso, res_hist, res_platt):
        assert isinstance(res, np.ndarray)
        assert res.shape == eval_data.shape
        assert np.all(np.isfinite(res))
        assert np.all(res >= 0)
