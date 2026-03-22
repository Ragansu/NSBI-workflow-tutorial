Workspace Builder
=================

The workspace builder converts a human-readable YAML fit configuration into a
pyhf-like workspace dictionary that :class:`~nsbi_common_utils.models.sbi_parametric_model.sbi_parametric_model`
consumes directly. This is the bridge between your analysis definition and the
statistical model.

Workflow
--------

.. code-block:: python

   from nsbi_common_utils import workspace_builder, models, inference

   # 1. Build workspace from fit config
   ws = workspace_builder.WorkspaceBuilder(config_path="config_fit_nsbi.yml").build()

   # 2. Initialize the statistical model
   model = models.sbi_parametric_model(workspace=ws, measurement_to_fit="my_measurement")

   # 3. Fit
   params, init_vals = model.get_model_parameters()
   fitter = inference.inference(
       model_nll=model.model,
       list_parameters=params,
       initial_values=init_vals,
   )
   fitter.perform_fit()

Writing a fit configuration file
---------------------------------

The fit configuration is a YAML file with the following top-level sections:

General
^^^^^^^

Defines the measurement name, the parameter of interest (POI), and which
parameters to include in the fit.

.. code-block:: yaml

   General:
     Measurement:
       Name: higgs_tautau_signal_strength
       POI: mu_htautau
       ParametersToFit:
         - mu_htautau
         - mu_ztautau
         - mu_ttbar
         - TES
         - JES

Samples
^^^^^^^

Each physics process (signal, backgrounds) that enters the likelihood.
Samples are read from ROOT files via uproot.

.. code-block:: yaml

   Samples:
     - Name: htautau
       Tree: tree_htautau
       SamplePath: ./saved_datasets/dataset_nominal.root
       Weight: weights
       UseAsReference: True    # reference sample in the density ratio
       UseAsBasis: True        # used as a basis process for SBI

     - Name: ztautau
       Tree: tree_ztautau
       SamplePath: ./saved_datasets/dataset_nominal.root
       Weight: weights
       UseAsReference: False
       UseAsBasis: True

NormFactors
^^^^^^^^^^^

Free multiplicative scale factors (signal strengths, background normalizations).
Each norm factor applies to the listed samples and is unconstrained in the fit.

.. code-block:: yaml

   NormFactors:
     - Name: mu_htautau
       Samples: htautau
       Nominal: 1
       Bounds: [0, 10]
     - Name: mu_ztautau
       Samples: ztautau
       Nominal: 1
       Bounds: [0, 10]

Systematics
^^^^^^^^^^^

Shape + normalization uncertainties (``NormPlusShape``). Each systematic points
to the up/down varied ROOT files per sample. The workspace builder computes
variation ratios (varied / nominal) automatically.

.. code-block:: yaml

   Systematics:
     - Name: JES
       Type: NormPlusShape
       Nominal: 0
       Samples:
         - htautau
         - ztautau
         - ttbar
       Up:
         - SampleName: htautau
           Path: ./saved_datasets/dataset_JES_up.root
           Tree: tree_htautau
           Weight: weights
         - SampleName: ztautau
           Path: ./saved_datasets/dataset_JES_up.root
           Tree: tree_ztautau
           Weight: weights
       Dn:
         - SampleName: htautau
           Path: ./saved_datasets/dataset_JES_dn.root
           Tree: tree_htautau
           Weight: weights
         # ... etc.

Regions
^^^^^^^

Analysis regions define where events are counted. Each region can be
**binned** (template fit) or **unbinned** (SBI density-ratio fit).

.. code-block:: yaml

   Regions:
     # Binned control region
     - Name: CR
       Filter: presel_score < -1.0
       Variable: presel_score
       Type: binned
       Binning: [-8.75, -4.0, -2.5, -1.0]

     # Unbinned signal region (SBI)
     - Name: SR
       Filter: presel_score >= -1.0 & presel_score <= 4.5
       Type: unbinned
       Variable: null
       Binning: null
       AsimovWeights: ./saved_datasets/asimov_weights.npy
       TrainedModels:
         - SampleName: htautau
           Nominal:
             Models: ./saved_datasets/output_training_nominal/output_model_params_htautau/
             Ratios: ./saved_datasets/output_training_nominal/output_ratios_htautau/ratio_htautau.npy
           Systematics:
             - SystName: JES
               ModelsUp: ./saved_datasets/output_training_systematics/output_model_params_htautau_JES_Up/
               ModelsDn: ./saved_datasets/output_training_systematics/output_model_params_htautau_JES_Dn/
               RatiosUp: ./saved_datasets/output_training_systematics/output_ratios_htautau_JES_Up/ratio_htautau.npy
               RatiosDn: ./saved_datasets/output_training_systematics/output_ratios_htautau_JES_Dn/ratio_htautau.npy

For unbinned regions, ``TrainedModels`` points to the pre-trained density-ratio
networks and their evaluated ratio arrays. These are produced by the training
and evaluation pipeline steps before the workspace is built.

TrainingFeatures
^^^^^^^^^^^^^^^^

Features used to train the density-ratio networks. Also used by the dataset
loader to select branches from the ROOT files.

.. code-block:: yaml

   TrainingFeatures:
     - DER_mass_transverse_met_lep
     - log_DER_mass_vis
     - log_DER_pt_h
     # ...

   TrainingFeaturesToStandardize:
     - DER_mass_transverse_met_lep
     - log_DER_mass_vis
     # ...


API Reference
-------------

.. currentmodule:: nsbi_common_utils.workspace_builder

.. autoclass:: WorkspaceBuilder
   :members:
