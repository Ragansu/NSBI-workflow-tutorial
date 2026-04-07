Preselection Training
=====================

In many analyses, events need to be classified into different analysis regions — for example a signal-enriched region where unbinned SBI is performed, and one or more control regions used for constraining background processes and systematic uncertainties. The preselection network automates this classification step.


What the preselection network does
-----------------------------------

The preselection network is a multi-class classifier that learns to assign events to analysis regions based on kinematic features. Once trained, its output scores can be used to define region boundaries, replacing or supplementing traditional cut-based selections.

The number of output classes corresponds to the number of analysis regions. Each event receives a vector of softmax probabilities indicating its affinity to each region.


Using the fit configuration
---------------------------

The training features and data paths are typically read from the :doc:`fit configuration file <fit_config>` via :class:`~nsbi_common_utils.configuration.ConfigManager`:

.. code-block:: python

   from nsbi_common_utils import configuration, datasets

   config = configuration.ConfigManager(file_path_string="config_fit.yml")
   features, features_scaling = config.get_training_features()

   # Load all samples (including systematics for region label assignment)
   datasets_helper = datasets.datasets(config_path="config_fit.yml", branches_to_load=features)
   dataset_dict = datasets_helper.load_datasets_from_config(load_systematics=True)

These values can also be passed manually if you are using the training APIs independently of the configuration system.


How to use it
-------------

The ``preselection_network_trainer`` provides a simple interface for training, saving, loading, and evaluating the preselection classifier.

**Training** requires a DataFrame with events from all regions, a column of integer region labels, and a column of normalised per-event weights:

.. code-block:: python

   from nsbi_common_utils.training import preselection_network_trainer

   trainer = preselection_network_trainer(
       dataset=df,
       features=feature_list,
       features_scaling=feature_list,
   )

   trainer.train(
       hidden_layers=4,
       neurons=1000,
       epochs=20,
       batch_size=1024,
       learning_rate=0.1,
       path_to_save="saved_models/",
   )

**Loading a previously trained model** for evaluation without retraining:

.. code-block:: python

   trainer.assign_trained_model(path_to_models="saved_models/")

**Running inference** on new data to obtain per-event region probabilities:

.. code-block:: python

   predictions = trainer.predict(dataset=df)

The output is a 2D array of softmax probabilities with shape ``(n_events, n_regions)``. These scores can then be used downstream to filter events into the appropriate regions for density-ratio training and workspace construction.


Extending with custom models
----------------------------

The ``preselection_network_trainer`` uses the ``MultiClassLightning`` module by default, but the underlying training utilities (ONNX export, batched inference, feature scaling) are model-agnostic. You can substitute a different ``pl.LightningModule`` — for example a graph neural network or a transformer-based classifier — and still use the shared utilities in ``nsbi_common_utils.training.utils`` for serialisation and inference. See the :doc:`density ratio training page <density_ratio_training>` for more details on extending the training infrastructure.


Where it fits in the pipeline
-----------------------------

The preselection network is typically trained after data preprocessing and before density-ratio training. Its predictions need to be stored as a new column in the dataset, which can then be used by the fit configuration to define region filters for density-ratio training and workspace building steps.