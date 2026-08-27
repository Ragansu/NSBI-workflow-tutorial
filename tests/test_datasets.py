import os
import numpy as np
import pandas as pd
import awkward as ak
import pytest

from nsbi_common_utils.datasets import (
    datasets,
    load_dataframe_from_root,
    save_dataframe_as_root,
)

# =====================================================================
# Tests for datasets Initialization & Feature Append
# =====================================================================


class TestDatasetsInit:

    def test_init_registers_branches(self, datasets_instance):
        assert datasets_instance.branches_to_load == ["m_vis", "pt_1", "fold_index"]
        assert datasets_instance.branches_all == ["m_vis", "pt_1", "fold_index"]

    def test_add_appended_branches(self, datasets_instance):
        datasets_instance.add_appended_branches(["deltaR_tau_tau", "pt_1"])
        # Should add new branch while avoiding duplicate registration
        assert "deltaR_tau_tau" in datasets_instance.branches_all
        assert datasets_instance.branches_all.count("pt_1") == 1


# =====================================================================
# Tests for ROOT Data IO
# =====================================================================


class TestRootIO:

    def test_load_dataframe_from_root(self, datasets_instance, temp_root_file):
        df = datasets_instance._load_dataframe_from_root(
            file_path=temp_root_file,
            tree_name="Nominal",
            branches=["m_vis", "pt_1"],
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["m_vis", "pt_1"]
        assert len(df) == 100

    def test_load_missing_file_raises_error(self, datasets_instance):
        with pytest.raises(FileNotFoundError, match="ROOT file not found"):
            datasets_instance._load_dataframe_from_root(
                file_path="non_existent.root",
                tree_name="Nominal",
                branches=["m_vis"],
            )

    def test_load_missing_tree_raises_key_error(
        self, datasets_instance, temp_root_file
    ):
        with pytest.raises(KeyError, match="Tree 'InvalidTree' not found"):
            datasets_instance._load_dataframe_from_root(
                file_path=temp_root_file,
                tree_name="InvalidTree",
                branches=["m_vis"],
            )


# =====================================================================
# Tests for Dataset Loading & Filtering
# =====================================================================


class TestDatasetLoadingAndFiltering:

    def test_load_datasets_from_config(self, datasets_instance):
        data = datasets_instance.load_datasets_from_config(load_systematics=True)

        assert "Nominal" in data
        assert "TES_Up" in data
        assert "TES_Dn" in data
        assert "htautau" in data["Nominal"]

        df_htautau = data["Nominal"]["htautau"]
        assert "weights" in df_htautau.columns
        assert "sample_name" in df_htautau.columns
        assert (df_htautau["sample_name"] == "htautau").all()

    def test_filter_region_dataset(self, datasets_instance, dummy_dataframe):
        dataset_dict = {"htautau": dummy_dataframe.copy()}
        filtered = datasets_instance.filter_region_dataset(dataset_dict, region="sr")

        # Query filter defined in fixture is 'm_vis > 70.0'
        assert len(filtered["htautau"]) < len(dummy_dataframe)
        assert (filtered["htautau"]["m_vis"] > 70.0).all()


# =====================================================================
# Tests for Fold Splitting
# =====================================================================


class TestFoldSplitting:

    def test_split_by_fold_train_eval(self, dummy_dataframe):
        dummy_dataframe["sample_name"] = "htautau"
        dataset_dict = {"htautau": dummy_dataframe}

        train_dict = datasets.split_by_fold(
            dataset_dict, fold_index=0, num_folds=4, mode="train"
        )
        eval_dict = datasets.split_by_fold(
            dataset_dict, fold_index=0, num_folds=4, mode="eval"
        )

        assert (train_dict["htautau"]["fold_index"] != 0).all()
        assert (eval_dict["htautau"]["fold_index"] == 0).all()

    def test_split_by_fold_missing_column_raises(self):
        df_no_fold = pd.DataFrame({"m_vis": [90.0, 91.0]})
        with pytest.raises(KeyError, match="Column 'fold_index' not found"):
            datasets.split_by_fold({"htautau": df_no_fold}, fold_index=0, num_folds=2)


# =====================================================================
# Tests for Reference Priors Resolution & Training Set Generation
# =====================================================================


class TestTrainingDatasetPreparation:

    def test_resolve_reference_priors_numeric_and_none(self, dummy_dataframe):
        dummy_dataframe["weights"] = np.array([2.0] * len(dummy_dataframe))
        dataset_den = {"htautau": dummy_dataframe, "ztautau": dummy_dataframe}

        specs = {"htautau": None, "ztautau": 5.0}
        resolved = datasets._resolve_reference_priors(specs, dataset_den)

        # 'htautau' Auto-yield = sum of weights = 2.0 * 100 = 200.0
        assert resolved["htautau"] == 200.0
        assert resolved["ztautau"] == 5.0

    def test_resolve_reference_priors_cap_mode(self, dummy_dataframe):
        dummy_dataframe["weights"] = np.array([1.0] * len(dummy_dataframe))
        dataset_den = {"htautau": dummy_dataframe, "ztautau": dummy_dataframe}

        # Cap M=5.0 means cap_fraction = 1/5 = 0.2
        specs = {"htautau": None, "ztautau": {"cap": 5.0}}
        resolved = datasets._resolve_reference_priors(specs, dataset_den)

        assert "htautau" in resolved
        assert "ztautau" in resolved
        assert resolved["ztautau"] > 0

    def test_resolve_reference_priors_invalid_spec_raises(self, dummy_dataframe):
        dataset_den = {"htautau": dummy_dataframe}
        with pytest.raises(ValueError, match="booleans are not a valid spec"):
            datasets._resolve_reference_priors({"htautau": True}, dataset_den)

    def test_prepare_basis_training_dataset(self, datasets_instance, dummy_dataframe):
        df_num = dummy_dataframe.copy()
        df_num["weights"] = 1.0
        df_num["sample_name"] = "htautau"

        df_den = dummy_dataframe.copy()
        df_den["weights"] = 1.0
        df_den["sample_name"] = "ztautau"

        ds_num = {"htautau": df_num}
        ds_den = {"ztautau": df_den}

        mixed_df = datasets_instance.prepare_basis_training_dataset(
            dataset_numerator=ds_num,
            processes_numerator=["htautau"],
            dataset_denominator=ds_den,
            processes_denominator=["ztautau"],
        )

        assert "train_labels" in mixed_df.columns
        assert set(mixed_df["train_labels"].unique()) == {0, 1}


# =====================================================================
# Tests for Standalone Utility Functions
# =====================================================================


class TestStandaloneUtilities:

    def test_save_and_load_dataframe_root(self, dummy_dataframe, tmp_path):
        target_path = str(tmp_path / "test_out.root")
        tree_name = "TestTree"

        save_dataframe_as_root(dummy_dataframe, target_path, tree_name)
        assert os.path.exists(target_path)

        loaded_df = load_dataframe_from_root(
            target_path, tree_name, branches_to_load=["m_vis", "pt_1"]
        )

        print("*"*50)
        print(loaded_df)
        print(type(loaded_df))
        print("*"*50)

        assert isinstance(loaded_df, pd.DataFrame)        
        assert list(loaded_df.columns) == ["m_vis", "pt_1"]
        assert len(loaded_df) == len(dummy_dataframe)
