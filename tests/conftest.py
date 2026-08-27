import pathlib
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import yaml
import json
import pytest

CONFIG_DIR = pathlib.Path(__file__).parent
from nsbi_common_utils.inference import inference


@pytest.fixture
def mock_np_load(tmp_path):
    """Fixture to intercept np.load calls for missing external numpy binary files."""

    def _dummy_load(file_path, *args, **kwargs):
        if "weights" in str(file_path):
            return np.array([1.0, 1.0, 1.0])
        if "ratio" in str(file_path):
            return np.array([1.2, 0.9, 1.1])
        return np.ones((3,))

    with patch("numpy.load", side_effect=_dummy_load):
        yield


@pytest.fixture
def dummy_workspace():
    """Builds a pyhf-like minimal workspace containing both binned and unbinned channels."""
    with open(CONFIG_DIR / "dummy_workspace.json", "r") as file:
        return json.load(file)


@pytest.fixture
def mock_config_dict():
    """Reads and parses the YAML config file directly into a Python dict."""
    config_file_path = CONFIG_DIR / "dummy_config.yaml"
    with open(config_file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_config_manager(mock_config_dict):
    """Creates a mocked ConfigManager instance."""
    mock = MagicMock()
    mock.config = mock_config_dict
    mock.get_index_unbinned_regions.return_value = 0
    mock.get_sample_index_unbinned_regions.return_value = 0
    mock.get_syst_index_unbinned_regions.return_value = 0
    return mock


@pytest.fixture
def mock_datasets():
    """Mocks the dataset loading and region filtering methods."""
    dummy_df = pd.DataFrame({"m_jj": [50.0, 150.0], "weights": [1.0, 1.0]})

    dataset_dict = {
        "Nominal": {"signal": dummy_df, "background": dummy_df},
        "sys1_Up": {"signal": dummy_df, "background": dummy_df},
        "sys1_Dn": {"signal": dummy_df, "background": dummy_df},
    }

    mock_ds_obj = MagicMock()
    mock_ds_obj.load_datasets_from_config.return_value = "dummy_datasets_incl"
    mock_ds_obj.filter_region_by_type.return_value = dataset_dict
    return mock_ds_obj


@pytest.fixture
def yaml_file(tmp_path, mock_config_dict):
    """Creates a temporary YAML configuration file on disk."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(mock_config_dict))
    return config_file


def quadratic_nll(params):
    """
    Simple 3D quadratic objective function centered at (1.0, 2.0, 3.0):
    NLL = (x - 1.0)^2 + (y - 2.0)^2 + (z - 3.0)^2
    """
    x, y, z = params
    return (x - 1.0) ** 2 + (y - 2.0) ** 2 + (z - 3.0) ** 2


def quadratic_grad(params):
    """Analytical gradient for quadratic_nll."""
    x, y, z = params
    return np.array([2.0 * (x - 1.0), 2.0 * (y - 2.0), 2.0 * (z - 3.0)])


@pytest.fixture
def dummy_inference_setup():
    """Provides a fully initialized inference engine instance."""
    initial_values = [0.0, 0.0, 0.0]
    list_parameters = ["mu", "norm", "nuis_1"]
    num_unconstrained = 2  # mu and norm are free; nuis_1 is constrained

    engine = inference(
        model_nll=quadratic_nll,
        initial_values=initial_values,
        list_parameters=list_parameters,
        num_unconstrained_params=num_unconstrained,
        model_grad=quadratic_grad,
    )
    return engine


@pytest.fixture
def binary_dataset():
    """Generates synthetic ratios, binary targets, and sample weights."""
    np.random.seed(42)
    n_samples = 300
    truth_labels = np.random.choice([0.0, 1.0], size=n_samples, p=[0.5, 0.5])
    ratios = np.where(
        truth_labels == 1.0,
        np.random.uniform(1.0, 5.0, n_samples),
        np.random.uniform(0.1, 2.0, n_samples),
    )
    weights = np.random.uniform(0.8, 1.2, n_samples)
    return ratios, truth_labels, weights


@pytest.fixture
def two_class_dataset():
    """Generates overlapping numerator and denominator samples so every bin receives events."""
    np.random.seed(42)
    n_num, n_den = 500, 500
    # Shared range ensuring denominator has support across the bin range
    data_num = np.random.uniform(0.1, 5.0, size=n_num)
    data_den = np.random.uniform(0.1, 5.0, size=n_den)
    w_num = np.ones(n_num)
    w_den = np.ones(n_den)
    return data_num, data_den, w_num, w_den
