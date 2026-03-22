Running the Workflow
====================

The full NSBI pipeline can be executed step-by-step or orchestrated via
HTCondor DAGMan on a cluster. This page describes both paths.

Pipeline overview
-----------------

.. image:: /_static/DAG_overview.svg
   :alt: NSBI Workflow Overview
   :align: center
   :width: 100%

Local (sequential) execution
-----------------------------

From the example directory
(``examples/FAIR_universe_Higgs_tautau/``):

.. code-block:: bash

   # 1. Load and preprocess data
   python scripts/data_loader.py --config config.pipeline.yaml
   python scripts/data_preprocessing.py --config config.pipeline.yaml

   # 2. Train preselection network (region classifier)
   python scripts/preselection_network.py --config config.pipeline.yaml

   # 3. Train nominal density-ratio ensembles (per process)
   python scripts/neural_likelihood_ratio_estimation.py \
       --config config.pipeline.yaml --process htautau --ensemble_index 0

   # 4. Train systematic variation networks
   python scripts/systematic_uncertainty_training.py \
       --config config.pipeline.yaml --process htautau --systematic JES --direction Up

   # 5. Evaluate all trained models on the Asimov dataset
   python scripts/data_nn_eval.py --config config.pipeline.yaml

   # 6. Build workspace and fit
   python scripts/parameter_fitting.py --config config.pipeline.yaml

Steps 3 and 4 are embarrassingly parallel across processes, ensemble members,
and systematic variations.

Cluster execution (HTCondor / DAGMan)
--------------------------------------

The ``htcondor/`` directory contains submit descriptions and DAG files that
orchestrate the pipeline on CHTC:

.. code-block:: text

   htcondor/
     stage_density_ratio_training.dag   # top-level DAG
     generate_training_dag.py           # generates train_ensemble.dag dynamically
     generate_systematics_dag.py        # generates train_systematics.dag dynamically
     train_ensemble.dag                 # one job per (process, ensemble_index)
     train_systematics.dag              # one job per (process, systematic, direction)
     job_density_ratio_training.sub     # submit file for nominal training
     job_systematics_training.sub       # submit file for systematic training
     job_density_ratio_eval.sub         # submit file for evaluation
     run_step.sh                        # wrapper script executed on the EP

Submit the full pipeline:

.. code-block:: bash

   condor_submit_dag examples/FAIR_universe_Higgs_tautau/htcondor/stage_density_ratio_training.dag

The full DAG structure, including ensemble parallelism:

.. image:: /_static/DAG_full_workflow.svg
   :alt: Full NSBI Workflow DAG
   :align: center
   :width: 100%

Stage 3 (density ratio training) in detail:

.. image:: /_static/DAG_stage3_ensemble_training.svg
   :alt: Stage 3 Ensemble Training Detail
   :align: center
   :width: 100%

DAGMan handles:

- **SCRIPT PRE** — dynamically generates the training DAGs by reading the
  pipeline config (number of ensemble members, systematic variations, etc.).
- **SUBDAG EXTERNAL** — submits the generated DAGs as nested sub-workflows.
- **PARENT/CHILD** — ensures evaluation runs only after all training completes.
- **RETRY** — automatically retries failed jobs (transient GPU errors, etc.).

File transfer
^^^^^^^^^^^^^

Each job transfers the source code and example directory to the execute point
via ``transfer_input_files``. Trained model outputs are transferred back
per-job to unique directories (keyed by process and ensemble index) to avoid
overwrites from concurrent jobs:

.. code-block:: text

   transfer_output_files = .../output_model_params_$(PROCESS_TYPE)$(ENSEMBLE_INDEX),
                           .../output_figures_$(PROCESS_TYPE)$(ENSEMBLE_INDEX),
                           .../output

The evaluation job (``data_nn_eval``) transfers the full ``saved_datasets/``
back since it is a single job with no concurrency risk.
