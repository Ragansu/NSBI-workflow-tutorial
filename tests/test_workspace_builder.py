import json
import pathlib
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from nsbi_common_utils.workspace_builder import WorkspaceBuilder

# =====================================================================
# Unit Tests
# =====================================================================


def test_init_and_poi_check(mock_config_dict, mock_config_manager):
    """Test that missing POI in ParametersToFit gets automatically added."""
    # Remove POI from ParametersToFit to test auto-insertion logic
    mock_config_dict["General"]["Measurement"]["ParametersToFit"] = ["sys1"]

    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")
        assert "mu" in builder.ParametersToFit


def test_normfactor_modifiers(mock_config_manager):
    """Test filtering of normfactors by region and sample."""
    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")

        # Matching region and sample
        mods = builder.normfactor_modifiers("region_1", "signal")
        assert len(mods) == 1
        assert mods[0]["name"] == "mu"
        assert mods[0]["type"] == "normfactor"

        # Non-matching region
        mods_empty = builder.normfactor_modifiers("wrong_region", "signal")
        assert len(mods_empty) == 0


def test_normplusshape_modifiers_binned(mock_config_dict, mock_config_manager):
    """Test building NormPlusShape modifiers for binned channels."""
    dummy_df = pd.DataFrame({"m_jj": [50.0, 150.0], "weights": [1.0, 1.0]})
    dataset = {"sys1_Up": {"signal": dummy_df}, "sys1_Dn": {"signal": dummy_df}}
    region = mock_config_dict["Regions"][0]
    sample = mock_config_dict["Samples"][0]
    systematic = mock_config_dict["Systematics"][0]
    nominal_data = np.array([1.0, 1.0])

    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")
        mods = builder.normplusshape_modifiers(
            dataset, region, sample, systematic, nominal_data, type_of_fit="binned"
        )
        assert len(mods) == 1
        assert mods[0]["name"] == "sys1"
        assert "hi_data" in mods[0]["data"]
        assert "lo_data" in mods[0]["data"]


def test_normplusshape_modifiers_unbinned(mock_config_dict, mock_config_manager):
    """Test building NormPlusShape modifiers for unbinned channels."""
    dummy_df = pd.DataFrame({"m_jj": [50.0, 150.0], "weights": [1.0, 1.0]})
    dataset = {"sys1_Up": {"signal": dummy_df}, "sys1_Dn": {"signal": dummy_df}}
    region = mock_config_dict["Regions"][0]
    sample = mock_config_dict["Samples"][0]
    systematic = mock_config_dict["Systematics"][0]
    nominal_data = np.array([1.0, 1.0])

    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")
        mods = builder.normplusshape_modifiers(
            dataset, region, sample, systematic, nominal_data, type_of_fit="unbinned"
        )
        assert len(mods) == 1
        assert mods[0]["data"]["hi_ratio"] == "/path/to/ratio_up.npy"
        assert mods[0]["data"]["lo_ratio"] == "/path/to/ratio_dn.npy"


def test_sys_modifiers_invalid_type(mock_config_dict, mock_config_manager):
    """Test that unsupported systematic types raise NotImplementedError."""
    mock_config_dict["Systematics"][0]["Type"] = "UnsupportedType"
    dummy_df = pd.DataFrame({"m_jj": [50.0, 150.0], "weights": [1.0, 1.0]})
    dataset = {"sys1_Up": {"signal": dummy_df}, "sys1_Dn": {"signal": dummy_df}}

    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")
        with pytest.raises(NotImplementedError):
            builder.sys_modifiers(
                dataset,
                mock_config_dict["Regions"][0],
                mock_config_dict["Samples"][0],
                np.array([1.0, 1.0]),
            )


def test_measurements(mock_config_dict, mock_config_manager):
    """Test extraction of measurements block."""
    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")
        meas = builder.measurements()

        assert len(meas) == 1
        assert meas[0]["name"] == "test_measurement"
        assert meas[0]["config"]["poi"] == "mu"
        param_names = [p["name"] for p in meas[0]["config"]["parameters"]]
        assert "mu" in param_names
        assert "sys1" in param_names


def test_build(mock_config_manager, mock_datasets):
    """Test full workspace construction via build()."""
    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ), patch("nsbi_common_utils.datasets.datasets", return_value=mock_datasets):

        builder = WorkspaceBuilder("dummy_config.yml")
        ws = builder.build()

        assert "measurements" in ws
        assert "channels" in ws
        assert "observations" in ws
        assert ws["version"] == "1.0.0"


def test_dump_and_load_workspace(tmp_path, mock_config_manager):
    """Test JSON serialization and deserialization using NumpyEncoder."""
    with patch(
        "nsbi_common_utils.configuration.ConfigManager",
        return_value=mock_config_manager,
    ):
        builder = WorkspaceBuilder("dummy_config.yml")

        # Test workspace with custom numpy types
        ws_dummy = {
            "version": "1.0.0",
            "int_val": np.int64(42),
            "float_val": np.float64(3.14),
            "arr_val": np.array([1, 2, 3]),
        }

        out_file = tmp_path / "workspace.json"
        builder.dump_workspace(ws_dummy, str(out_file))

        assert out_file.exists()

        loaded_ws = WorkspaceBuilder.load_workspace(str(out_file))
        assert loaded_ws["version"] == "1.0.0"
        assert loaded_ws["int_val"] == 42
        assert loaded_ws["float_val"] == 3.14
        assert loaded_ws["arr_val"] == [1, 2, 3]
