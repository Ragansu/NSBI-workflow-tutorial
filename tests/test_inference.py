import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CI execution
import matplotlib.pyplot as plt

# Import module under test (replace 'nsbi_common_utils' with the actual filename)
from nsbi_common_utils.inference import plot_NLL_scans

# =====================================================================
# Tests for plot_NLL_scans
# =====================================================================

class TestPlotNLLScans:

    def test_plot_nll_scans_creation(self):
        """Test creating a plot with explicit label and axes."""
        fig, ax = plt.subplots()
        
        scan_pts = [[0.0, 1.0, 2.0]]
        nll_vals = [[4.0, 0.0, 4.0]]
        labels = ["Stat + Syst"]
        linestyles = ["solid"]
        colors = ["black"]

        plot_NLL_scans(
            parameter_label=r"$\mu$",
            list_scan_points=scan_pts,
            list_nll_values=nll_vals,
            list_labels=labels,
            list_linestyles=linestyles,
            list_colors=colors,
            ax=ax,
        )

        assert ax.get_xlabel() == r"$\mu$"
        assert ax.get_ylabel() == r"$t_\mu$"
        assert ax.get_ylim()[0] == 0.0
        plt.close(fig)


def test_plot_nll_scans_fallback_parameter_name(monkeypatch):
    """
    Tests fallback behavior when parameter_label='' is passed,
    verifying it accesses `parameter_name` from calling scope.
    """
    import nsbi_common_utils.inference as inf_module

    # Inject 'parameter_name' into the module namespace where plot_NLL_scans looks for globals
    monkeypatch.setattr(inf_module, "parameter_name", "fallback_param", raising=False)

    fig, ax = plt.subplots()

    # Call the function from the module namespace so it sees the injected global
    inf_module.plot_NLL_scans(
        parameter_label="",
        list_scan_points=[[0.0, 1.0]],
        list_nll_values=[[1.0, 0.0]],
        list_labels=["Test"],
        list_linestyles=["-"],
        list_colors=["red"],
        ax=ax,
    )

    assert ax.get_xlabel() == "fallback_param"
    plt.close(fig)

# =====================================================================
# Tests for inference class
# =====================================================================

class TestInference:

    def test_initialization(self, dummy_inference_setup):
        """Verify attributes are assigned correctly on init."""
        engine = dummy_inference_setup
        assert engine.list_parameters == ["mu", "norm", "nuis_1"]
        assert engine.num_unconstrained_params == 2
        assert engine.pulls_global_fit is None

    def test_perform_fit_success(self, dummy_inference_setup):
        """Test that global fit converges near minimum (1.0, 2.0, 3.0)."""
        engine = dummy_inference_setup
        engine.perform_fit(fit_strategy=1)

        assert engine.pulls_global_fit is not None
        assert len(engine.pulls_global_fit) == 3
        # Check convergence to true minimum within tolerance
        np.testing.assert_allclose(engine.pulls_global_fit, [1.0, 2.0, 3.0], atol=1e-3)

    def test_perform_fit_with_frozen_params(self, dummy_inference_setup):
        """Test fixing a parameter during global fit."""
        engine = dummy_inference_setup
        engine.perform_fit(fit_strategy=1, freeze_params=["norm"])

        # 'norm' was initialized to 0.0 and frozen
        assert np.isclose(engine.pulls_global_fit[1], 0.0)

    def test_profile_scan_without_stat_only(self, dummy_inference_setup):
        """Test scanning profile likelihood without stat-only option."""
        engine = dummy_inference_setup
        
        scan_points, nll_values = engine.perform_profile_scan(
            parameter_name="mu",
            bound_range=(-1.0, 3.0),
            size=20,
            doStatOnly=False,
        )

        assert len(scan_points) == 20
        assert len(nll_values) == 20
        # Check minimum is near 0.0 at the best-fit point
        assert np.min(nll_values) == pytest.approx(0.0, abs=1e-2)

    def test_profile_scan_stat_only_raises_runtime_error(self, dummy_inference_setup):
        """
        Verify perform_profile_scan raises RuntimeError if doStatOnly=True 
        before perform_fit() has been executed.
        """
        engine = dummy_inference_setup
        
        with pytest.raises(RuntimeError, match="perform_fit\(\) must be called before doStatOnly=True"):
            engine.perform_profile_scan(
                parameter_name="mu",
                bound_range=(0.0, 2.0),
                doStatOnly=True,
            )

    def test_profile_scan_with_stat_only(self, dummy_inference_setup):
        """Test full profile scan returning Stat+Syst and Stat-Only arrays."""
        engine = dummy_inference_setup
        
        # 1. Execute global fit
        engine.perform_fit(fit_strategy=1)
        
        # 2. Execute profile scan with doStatOnly=True
        pts, nll, pts_stat, nll_stat = engine.perform_profile_scan(
            parameter_name="mu",
            bound_range=(-1.0, 3.0),
            size=15,
            doStatOnly=True,
        )

        assert len(pts) == 15
        assert len(nll) == 15
        assert len(pts_stat) == 15
        assert len(nll_stat) == 15