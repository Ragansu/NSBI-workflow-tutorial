Running the Workflow
====================

The full NSBI pipeline can be executed step-by-step on a single machine, or orchestrated as a parallel DAG on a cluster via `Snakemake <https://snakemake.readthedocs.io/>`_. Snakemake is infrastructure-agnostic — the same workflow definition runs on HTC (HTCondor), HPC (SLURM), Kubernetes, or a personal laptop just by swapping the executor profile.

Below is an example workflow using the FAIR Universe :math:`H\to \tau\tau` dataset.

All pipeline scripts are driven by a single configuration file, ``config.pipeline.yaml``, located at the root of each example directory (e.g. ``examples/FAIR_universe_Higgs_tautau/config.pipeline.yaml``). This file defines dataset paths, training hyperparameters, ensemble sizes, systematic variations, and fit settings. Inspect the example config to understand the available options.

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

Steps 3 and 4 are embarrassingly parallel across processes, ensemble members, k-fold splits, and systematic variations. Snakemake fans them out automatically when running on a cluster — see below.

Cluster execution (Snakemake on HTCondor)
------------------------------------------

A single ``Snakefile`` at the root of each example directory defines all pipeline rules; a profile under ``profiles/<cluster>/`` configures the executor and per-rule resource defaults. From the repository root:

.. code-block:: bash

   snakemake --snakefile examples/FAIR_universe_Higgs_tautau/Snakefile \
             --profile  examples/FAIR_universe_Higgs_tautau/profiles/chtc

That single command builds the full DAG (parameter_fitting at the leaf, all training and preprocessing as dependencies), submits each rule's jobs to HTCondor via the `snakemake-executor-plugin-htcondor <https://github.com/snakemake/snakemake-executor-plugin-htcondor>`_, and waits for completion.

File layout
^^^^^^^^^^^

.. code-block:: text

   examples/FAIR_universe_Higgs_tautau/
     Snakefile                            # all pipeline rules + wildcard fan-out
     config.pipeline.yaml                 # single source of truth for paths and hyperparameters
     profiles/
       chtc/
         config.yaml                      # snakemake profile: executor=htcondor, default resources
         job_wrapper.sh                   # per-job entrypoint inside the container
     scripts/                             # the same python scripts used by the local-execution path

Rule structure replaces DAGMan PRE / SUBDAG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each pipeline step is one Snakemake rule. Per-job fan-out (``process × fold × ensemble_index`` for nominal training; ``process × systematic × direction × fold × ensemble_index`` for systematics) is expressed as **wildcard expansion** at Snakefile-parse time, computed from the config:

.. code-block:: python

   BASIS_PROCESSES  = config["neural_likelihood_ratio_estimation"]["basis_processes_to_train"]
   N_ENSEMBLE       = config["neural_likelihood_ratio_estimation"]["num_ensemble_members_training"]
   NUM_FOLDS        = config.get("data_preprocessing", {}).get("num_folds", 1)

This replaces the legacy DAGMan pattern of (a) generating DAG files dynamically via ``SCRIPT PRE`` hooks, (b) submitting them as ``SUBDAG EXTERNAL`` nested workflows. K-fold cross-validation is just bumping ``num_folds`` in the config — no DAG regeneration step needed.

Sentinel-driven completion tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DAG dependency edges are tracked via per-rule sentinel files on shared storage (e.g. ``/projects/.../sentinels/.done_train_<process>_fold<F>_<E>``), not via HTCondor's ``transfer_output_files``. This decouples DAG state from HTCondor's job lifecycle and survives driver restarts cleanly. Sentinel paths and the shared-FS prefix are declared in the profile:

.. code-block:: yaml

   # profiles/chtc/config.yaml
   shared-fs-usage: none
   htcondor-shared-fs-prefixes: "/staging,/projects"

HTCondor resource specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-rule resource requests, ClassAds, and transfer specs are declared in the rule's ``resources:`` block. The htcondor plugin emits them verbatim into the submit description:

.. code-block:: python

   resources:
       request_memory          = "16GB",
       request_disk            = "32GB",
       request_gpus            = 1,
       gpus_minimum_capability = _r(7.0),
       classad_WantGPULab      = _r(True),
       classad_GPUJobLength    = "medium",
       requirements            = gpu_requirements(driver_version="12.4"),
       allowed_job_duration    = 9000,    # replaces periodic_hold + periodic_release
       max_retries             = 3,       # replaces DAGMan RETRY
       htcondor_transfer_input_files = COMMON_TRANSFER,

``allowed_job_duration`` + ``max_retries`` together replace the legacy ``periodic_hold`` / ``periodic_release`` retry idiom that the plugin doesn't expose.

Resuming and partial reruns
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sentinels make resumption trivial — Snakemake skips any rule whose output sentinel already exists. After a failed or killed driver, re-running the same command picks up where it left off and only submits the missing jobs. A few useful flags:

- ``--forcerun <rule>`` — re-execute a specific rule even if its sentinel exists. Does **not** re-run upstream rules unless their inputs are themselves missing.
- ``--rerun-triggers mtime`` — only re-run on file-modification-time changes, not on Snakefile/code edits. Pass this when you've edited the Snakefile for an unrelated reason and don't want every rule to re-trigger.
- ``--unlock`` — clear a stale ``LockException`` left by a previously killed driver.
- ``--cleanup-metadata <output paths>`` — mark previously-incomplete outputs as complete in the Snakemake metadata store. Needed if a previous driver died after the EP-side job touched its sentinel but before Snakemake recorded completion.
- ``--touch`` — bring outputs up-to-date in Snakemake's metadata without actually running anything. Useful for adopting an existing tree of artifacts.

Adapting to your cluster
^^^^^^^^^^^^^^^^^^^^^^^^

The Snakefile itself is portable. To target a different cluster, copy ``profiles/chtc/`` to ``profiles/<your-cluster>/``, change ``executor:`` to the appropriate plugin (``slurm``, ``cluster-generic``, ``kubernetes``, or just remove for local execution), and adjust ``default-resources``. Per-rule resource blocks (memory, threads, GPU requests) carry over unchanged.

Available executors are listed at https://snakemake.github.io/snakemake-plugin-catalog/.

For sites without HTCondor, the local sequential commands above still work and can be wrapped in your site's job-submission idiom directly.
