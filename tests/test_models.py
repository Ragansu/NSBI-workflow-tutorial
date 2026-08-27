import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Ensure auxiliary function is available if running standalone
if "_calculate_combined_var" not in globals():

    def _calculate_combined_var(param_syst, var_up, var_dn):
        # Fallback simplified mock for histfactory interpolation (Strategy 5 style)
        # alpha * (up - dn) / 2 + 1
        return jnp.ones_like(var_up[0])


# Import or define the class under test
from nsbi_common_utils.models import sbi_parametric_model

# =====================================================================
# Unit Tests
# =====================================================================


def test_initialization(dummy_workspace, mock_np_load):
    """Test standard initialization and attribute creation."""
    model = sbi_parametric_model(
        workspace=dummy_workspace, measurement_to_fit="Measurement_1"
    )

    assert model.measurement_name == "Measurement_1"
    assert model.poi == "mu"
    assert model.all_channels == ["control_region", "sbi_unbinned_region"]
    assert model.all_samples == ["signal"]
    assert "mu" in model.list_parameters
    assert "sys1" in model.list_parameters


def test_get_model_parameters(dummy_workspace, mock_np_load):
    """Verify get_model_parameters returns correct parameter list and initial vector."""
    model = sbi_parametric_model(
        workspace=dummy_workspace, measurement_to_fit="Measurement_1"
    )
    params, init_vec = model.get_model_parameters()

    assert params == ["mu", "sys1"]
    assert isinstance(init_vec, (jnp.ndarray, np.ndarray))
    assert len(init_vec) == 2
    assert init_vec[0] == 1.0
    assert init_vec[1] == 0.0


def test_index_lookups(dummy_workspace, mock_np_load):
    """Verify internal helper methods for mapping regions, samples, and modifiers."""
    model = sbi_parametric_model(
        workspace=dummy_workspace, measurement_to_fit="Measurement_1"
    )

    assert model._index_of_region("control_region") == 0
    assert model._index_of_region("sbi_unbinned_region") == 1
    assert model._index_of_region("non_existent_channel") is None

    assert model._index_of_sample("control_region", "signal") == 0
    assert model._index_of_modifiers("control_region", "signal", "sys1") == 1


def test_model_evaluation(dummy_workspace, mock_np_load):
    """Test evaluation of the JIT-compiled NLL output."""
    model = sbi_parametric_model(
        workspace=dummy_workspace, measurement_to_fit="Measurement_1"
    )
    params, init_vec = model.get_model_parameters()

    nll_val = model.model(init_vec)

    assert np.isfinite(nll_val)
    assert isinstance(nll_val, (jnp.ndarray, np.ndarray, float))


def test_model_gradient(dummy_workspace, mock_np_load):
    """Test reverse-mode automatic differentiation gradient output."""
    model = sbi_parametric_model(
        workspace=dummy_workspace, measurement_to_fit="Measurement_1"
    )
    _, init_vec = model.get_model_parameters()

    grad = model.model_grad(init_vec)

    assert isinstance(grad, np.ndarray)
    assert grad.shape == init_vec.shape
    assert not np.isnan(grad).any()
