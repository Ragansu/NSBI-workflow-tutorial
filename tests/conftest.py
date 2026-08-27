import pathlib
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import yaml
import json
import pytest

CONFIG_DIR = pathlib.Path(__file__).parent


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
        "Nominal": {"signal": dummy_df},
        "sys1_Up": {"signal": dummy_df},
        "sys1_Dn": {"signal": dummy_df},
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
