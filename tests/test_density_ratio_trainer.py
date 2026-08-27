import os
import logging
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Import from your training module (replace 'your_module' with your python file name)
from nsbi_common_utils.training.neural_ratio_estimation import configure_logging, density_ratio_trainer, logger


# =====================================================================
# Tests for Logging Configuration
# =====================================================================

class TestLoggingConfiguration:

    def test_configure_logging_levels(self):
        configure_logging(verbose_level=0)
        assert logger.level == logging.WARNING

        configure_logging(verbose_level=1)
        assert logger.level == logging.INFO

        configure_logging(verbose_level=2)
        assert logger.level == logging.DEBUG

    def test_configure_logging_adds_handler(self):
        # Clear existing handlers to test addition
        logger.handlers.clear()
        configure_logging(verbose_level=1)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)


# =====================================================================
# Tests for density_ratio_trainer Initialization
# =====================================================================

class TestTrainerInit:

    def test_init_creates_directories(self, sample_dataset, sample_training_inputs, temp_dirs):
        fig_dir, model_dir = temp_dirs
        weights, labels = sample_training_inputs

        trainer = density_ratio_trainer(
            dataset=sample_dataset,
            weights=weights,
            training_labels=labels,
            features=["m_vis", "pt_1"],
            features_scaling=["m_vis", "pt_1"],
            sample_name=["signal", "background"],
            output_name="test_run",
            path_to_figures=fig_dir,
            path_to_models=model_dir,
        )

        assert os.path.exists(fig_dir)
        assert os.path.exists(model_dir)
        assert trainer.use_log_loss is False

    def test_init_delete_existing_models(self, sample_dataset, sample_training_inputs, temp_dirs):
        fig_dir, model_dir = temp_dirs
        weights, labels = sample_training_inputs

        # Create dummy marker files
        marker_fig = os.path.join(fig_dir, "old_fig.png")
        marker_model = os.path.join(model_dir, "old_model.onnx")
        open(marker_fig, "w").close()
        open(marker_model, "w").close()

        trainer = density_ratio_trainer(
            dataset=sample_dataset,
            weights=weights,
            training_labels=labels,
            features=["m_vis", "pt_1"],
            features_scaling=["m_vis", "pt_1"],
            sample_name=["signal", "background"],
            output_name="test_run",
            path_to_figures=fig_dir,
            path_to_models=model_dir,
            delete_existing_models=True,
        )

        assert not os.path.exists(marker_fig)
        assert not os.path.exists(marker_model)
        assert os.path.exists(fig_dir)
        assert os.path.exists(model_dir)


# =====================================================================
# Tests for Training Execution & Workflows
# =====================================================================

class TestTrainerExecution:

    # @patch("pytorch_lightning.Trainer")
    # def test_train_fresh_model_execution(
    #     self,
    #     mock_lightning_trainer,
    #     sample_dataset,
    #     sample_training_inputs,
    #     temp_dirs,
    #     mock_nsbi_utils,
    # ):
    #     """Tests standard fresh training execution with PyTorch Lightning mocked."""
    #     fig_dir, model_dir = temp_dirs
    #     weights, labels = sample_training_inputs

    #     # Mock PyTorch Lightning Trainer instance
    #     mock_trainer_instance = MagicMock()
    #     mock_lightning_trainer.return_value = mock_trainer_instance

    #     trainer_obj = density_ratio_trainer(
    #         dataset=sample_dataset,
    #         weights=weights,
    #         training_labels=labels,
    #         features=["m_vis", "pt_1"],
    #         features_scaling=["m_vis", "pt_1"],
    #         sample_name=["signal", "background"],
    #         output_name="test_run",
    #         path_to_figures=fig_dir,
    #         path_to_models=model_dir,
    #     )

    #     trainer_obj.train(
    #         hidden_layers=2,
    #         neurons=32,
    #         number_of_epochs=1,
    #         batch_size=32,
    #         learning_rate=0.001,
    #         scalerType="StandardScaler",
    #         callback=False,
    #         verbose=1,
    #     )

    #     # Verify PyTorch Lightning Trainer fit was called
    #     assert mock_trainer_instance.fit.called
    #     # Verify saved split state metadata
    #     state_file = os.path.join(model_dir, "num_events_random_state_train_holdout_split.npy")
    #     assert os.path.exists(state_file)

    def test_train_load_existing_model(
        self, sample_dataset, sample_training_inputs, temp_dirs, mock_nsbi_utils
    ):
        """Tests that load_trained_models=True successfully loads saved artifacts."""
        fig_dir, model_dir = temp_dirs
        weights, labels = sample_training_inputs

        # Pre-create expected saved state files
        np.save(
            os.path.join(model_dir, "num_events_random_state_train_holdout_split.npy"),
            np.array([60, 42]),
        )
        open(os.path.join(model_dir, "model_scaler.bin"), "w").close()
        open(os.path.join(model_dir, "model.onnx"), "w").close()

        trainer_obj = density_ratio_trainer(
            dataset=sample_dataset,
            weights=weights,
            training_labels=labels,
            features=["m_vis", "pt_1"],
            features_scaling=["m_vis", "pt_1"],
            sample_name=["signal", "background"],
            output_name="test_run",
            path_to_figures=fig_dir,
            path_to_models=model_dir,
        )

        trainer_obj.train(
            hidden_layers=2,
            neurons=32,
            number_of_epochs=1,
            batch_size=32,
            learning_rate=0.001,
            scalerType="StandardScaler",
            load_trained_models=True,
        )

        # Ensure load_trained_model was invoked
        assert mock_nsbi_utils.training.utils.load_trained_model.called


# =====================================================================
# Tests for Diagnostic Plots
# =====================================================================

class TestTrainerPlots:

    @pytest.fixture
    def prepared_trainer(self, sample_dataset, sample_training_inputs, temp_dirs, mock_nsbi_utils):
        """Prepares a trained trainer instance with populated ratios for plot tests."""
        fig_dir, model_dir = temp_dirs
        weights, labels = sample_training_inputs

        trainer_obj = density_ratio_trainer(
            dataset=sample_dataset,
            weights=weights,
            training_labels=labels,
            features=["m_vis", "pt_1"],
            features_scaling=["m_vis", "pt_1"],
            sample_name=["signal", "background"],
            output_name="test_run",
            path_to_figures=fig_dir,
            path_to_models=model_dir,
        )

        # Populate internal ratio attributes directly to avoid full training loop execution
        n = len(sample_dataset) // 2
        trainer_obj.ratio_den_training = np.ones(n) * 0.8
        trainer_obj.ratio_num_training = np.ones(n) * 1.2
        trainer_obj.ratio_den_holdout = np.ones(n) * 0.85
        trainer_obj.ratio_num_holdout = np.ones(n) * 1.15

        trainer_obj.weight_den_training = np.ones(n)
        trainer_obj.weight_num_training = np.ones(n)
        trainer_obj.weight_den_holdout = np.ones(n)
        trainer_obj.weight_num_holdout = np.ones(n)

        return trainer_obj

    # def test_make_overfit_plots(self, prepared_trainer, mock_nsbi_utils):
    #     prepared_trainer.make_overfit_plots(ensemble_index=0)
    #     assert mock_nsbi_utils.plotting.plot_overfit_side_by_side.called

    # def test_make_calib_plots_score(self, prepared_trainer, mock_nsbi_utils):
    #     prepared_trainer.make_calib_plots(observable="score", nbins=10, ensemble_index=0)
    #     assert mock_nsbi_utils.plotting.plot_calibration_curve.called

    # def test_make_calib_plots_llr(self, prepared_trainer, mock_nsbi_utils):
    #     prepared_trainer.make_calib_plots(observable="llr", nbins=10, ensemble_index=0)
    #     assert mock_nsbi_utils.plotting.plot_calibration_curve_ratio.called

    def test_make_calib_plots_invalid_observable(self, prepared_trainer):
        with pytest.raises(Exception, match="observable not recognized"):
            prepared_trainer.make_calib_plots(observable="invalid_option")