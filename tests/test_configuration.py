import pytest
import yaml

from nsbi_common_utils.configuration import ConfigError, ConfigManager


# =====================================================================
# Unit Tests
# =====================================================================

def test_init_file_missing_raises_error(tmp_path):
    """Test that initializing with a non-existent file raises ConfigError."""
    missing_file = tmp_path / "non_existent.yml"
    with pytest.raises(ConfigError, match="Config file does not exist"):
        ConfigManager(file_path_string=missing_file, create_if_missing=False)


def test_init_create_if_missing(tmp_path):
    """Test creating a new config file if it does not exist."""
    missing_file = tmp_path / "new_config.yml"
    template = {"Regions": [{"Name": "default_region"}]}
    
    cm = ConfigManager(file_path_string=missing_file, initial_template=template, create_if_missing=True)
    
    assert missing_file.exists()
    assert cm.config["Regions"][0]["Name"] == "default_region"


def test_load_and_save(yaml_file):
    """Test loading and saving configuration file."""
    cm = ConfigManager(file_path_string=yaml_file)
    assert cm.config["Regions"][0]["Name"] == "region_1"

    # Modify and save
    cm.config["Regions"][0]["Name"] = "updated_region_1"
    cm.save()

    # Reload to verify save worked
    cm_reloaded = ConfigManager(file_path_string=yaml_file)
    assert cm_reloaded.config["Regions"][0]["Name"] == "updated_region_1"


def test_invalid_yaml_raises_error(tmp_path):
    """Test that corrupted YAML files raise ConfigError during load."""
    bad_yaml = tmp_path / "bad.yml"
    bad_yaml.write_text("Regions: [invalid_yaml: : :")

    with pytest.raises(ConfigError, match="YAML parse error"):
        ConfigManager(file_path_string=bad_yaml)


def test_channel_management(yaml_file):
    """Test adding, listing, and removing channels."""
    cm = ConfigManager(file_path_string=yaml_file)

    # List channels
    assert cm.list_channels() == ["region_1", "region_2"]

    # Add channel without overwrite (error expected if duplicate)
    with pytest.raises(ConfigError, match="already exists"):
        cm.add_channel(channel_name="region_1", filter="pt > 300", observable="pt")

    # Add channel with overwrite
    cm.add_channel(channel_name="region_1", filter="pt > 300", observable="pt", overwrite=True)
    assert cm.get_channel_filters("region_1") == "pt > 300"

    # Add brand new channel
    cm.add_channel(channel_name="validation_region", filter="pt < 50", observable="m_jj")
    assert "validation_region" in cm.list_channels()

    # Remove channel
    assert cm.remove_channel("region_1") is True
    assert "region_1" not in cm.list_channels()

    # Remove non-existent channel returns False
    assert cm.remove_channel("non_existent") is False


def test_sample_queries(yaml_file):
    """Test retrieving basis, reference, and all sample names."""
    cm = ConfigManager(file_path_string=yaml_file)

    assert cm.get_basis_samples() == ["signal"]
    assert cm.get_reference_samples() == ["background"]
    assert cm.get_all_samples() == ["signal", "background"]


def test_get_training_features_and_region_cuts(yaml_file):
    """Test training features and region cut extractions."""
    cm = ConfigManager(file_path_string=yaml_file)

    features, to_standardize = cm.get_training_features()
    assert features == ["m_jj", "pt"]
    assert to_standardize == ["m_jj"]

    names, selections = cm.get_analysis_region_cuts()
    assert names == ["region_1", "region_2"]
    assert selections == ["pt > 200 && pt <= 200"]


def test_get_samples_in_syst_for_training(yaml_file):
    """Test retrieving sample lists for a given systematic variation."""
    cm = ConfigManager(file_path_string=yaml_file)

    sig_up = cm.get_samples_in_syst_for_training("sys1", "Up")
    assert sig_up == ["signal", "background"]

    sig_dn = cm.get_samples_in_syst_for_training("sys1", "Dn")
    assert sig_dn == ["signal"]

    assert cm.get_samples_in_syst_for_training("non_existent_sys", "Up") is None


def test_unbinned_region_indices_and_paths(yaml_file):
    """Test getting sample/systematic indices and paths for unbinned regions."""
    cm = ConfigManager(file_path_string=yaml_file)

    # Index lookups
    assert cm.get_index_unbinned_regions("region_2") == 0
    assert cm.get_index_unbinned_regions("missing_region") is None

    assert cm.get_sample_index_unbinned_regions("region_2", "signal") == 0
    assert cm.get_sample_index_unbinned_regions("region_2", "non_existent_sample") is None

    assert cm.get_syst_index_unbinned_regions("region_2", "signal", "sys1") == 0
    assert cm.get_syst_index_unbinned_regions("region_2", "signal", "missing_sys") is None

    # Asimov weight path lookup
    assert cm.get_channel_asimov_weight_path("region_2") == "/path/to/asimov_weights.npy"


def test_unbinned_region_missing_raises_error(yaml_file):
    """Test that querying an unbinned region that doesn't exist raises ConfigError."""
    cm = ConfigManager(file_path_string=yaml_file)

    with pytest.raises(ConfigError, match="Region missing_region not found"):
        cm.get_sample_index_unbinned_regions("missing_region", "signal")

    with pytest.raises(ConfigError, match="Region missing_region not found"):
        cm.get_syst_index_unbinned_regions("missing_region", "signal", "sys1")

    with pytest.raises(ConfigError, match="Region missing_region not found"):
        cm.get_channel_asimov_weight_path("missing_region")