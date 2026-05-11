# =============================================================================
# Snakemake workflow: NSBI examples/FAIR Universe Higgs tautau
# Translated from DAGMan pipeline; mirrors the fan-out logic in
# examples/FAIR_universe_Higgs_tautau/htcondor/generate_{training,systematics}_dag.py
# =============================================================================

import os
import yaml

configfile: "examples/FAIR_universe_Higgs_tautau/config.pipeline.yaml"

EXAMPLE_DIR = "FAIR_universe_Higgs_tautau"
WORK_DIR    = f"examples/{EXAMPLE_DIR}"
CONFIG_FILE = "config.pipeline.yaml"

_nlre            = config["neural_likelihood_ratio_estimation"]
_syst_cfg        = config["systematic_uncertainty"]
BASIS_PROCESSES  = _nlre["basis_processes_to_train"]
N_ENSEMBLE       = _nlre["num_ensemble_members_training"]
ENSEMBLE_INDICES = list(range(N_ENSEMBLE))

N_ENSEMBLE_SYST  = _syst_cfg.get("num_ensemble_members_training", 1)
SYST_ENSEMBLE_INDICES = list(range(N_ENSEMBLE_SYST))

# K-fold cross-validation: num_folds lives under data_preprocessing in the pipeline config (set to 1 to disable). The fan-out below matches generate_training_dag.py and generate_systematics_dag.py: emit one job per (process, fold, ensemble_idx) for nominal training, and one per (process, systematic, direction, fold, ensemble_idx) for systematics. When NUM_FOLDS == 1 the fold dimension is dropped from sentinel filenames and `--fold_index` is omitted on the command line, matching what the training scripts expect to produce model paths without a `_fold{N}` suffix.
NUM_FOLDS    = int(config.get("data_preprocessing", {}).get("num_folds", 1))
FOLD_INDICES = list(range(NUM_FOLDS))
USE_KFOLD    = NUM_FOLDS > 1

# Read the NSBI fit config once to enumerate valid (process, systematic, direction) combos for the systematics fan-out — same filter logic as generate_systematics_dag.py.
_nsbi_fit_config_path = os.path.join(WORK_DIR, _syst_cfg["nsbi_fit_config"])
with open(_nsbi_fit_config_path) as _fh:
    _nsbi_fit_config = yaml.safe_load(_fh)

SYST_COMBOS = []  # list of (process, syst_name, direction) tuples
for _dict_syst in _nsbi_fit_config.get("Systematics", []) or []:
    if _dict_syst.get("Type") != "NormPlusShape":
        continue
    _syst_name = _dict_syst["Name"]
    for _process in BASIS_PROCESSES:
        if _process not in _dict_syst.get("Samples", []):
            continue
        for _direction in ["Up", "Dn"]:
            SYST_COMBOS.append((_process, _syst_name, _direction))

# Constrain wildcards so the two `train_ensemble_*` rules disambiguate cleanly (and likewise the two `systematic_uncertainty_training_*` rules). `fold` is digits only; `process`/`syst`/`direction` are alphanumeric tokens; `ensemble_idx` is digits.
wildcard_constraints:
    process      = r"[A-Za-z][A-Za-z0-9]*",
    syst         = r"[A-Za-z][A-Za-z0-9]*",
    direction    = r"Up|Dn",
    fold         = r"\d+",
    ensemble_idx = r"\d+",


# =============================================================================
# HTCondor requirements helpers (consumed by per-rule `resources:` blocks).
# The plugin reads `resources.requirements` and emits it verbatim into the
# submit description. Machine excludes vary per rule, driver-version floor
# bumps up for newer-CUDA jobs.
# =============================================================================

_BASE_GPU_EXCLUDES = [
    "vetsigian0001.chtc.wisc.edu",
    "vetsigian0000.chtc.wisc.edu",
    "gpulab2003.chtc.wisc.edu",
]


def gpu_requirements(driver_version="12.4", extra_excludes=()):
    """Build a HTCondor requirements expression for a GPU job: driver-version floor + HasCHTCStaging + machine excludes (base + per-rule extras)."""
    excludes = _BASE_GPU_EXCLUDES + list(extra_excludes)
    excl_clause = " && ".join(f'(Machine != "{m}")' for m in excludes)
    return f'(GPUs_DriverVersion >= {driver_version}) && (Target.HasCHTCStaging == true) && {excl_clause}'


def _r(v):
    """Wrap a literal in a zero-arg callable so Snakemake's resources-block type check accepts non-int/str values (bools, floats). The htcondor plugin invokes the callable and uses the returned value when emitting the submit description, so we can pass `True`/`False` for `+ClassAd` flags and `7.5` for `gpus_minimum_capability` without tripping the validator."""
    return lambda *_args, **_kwargs: v


# Static files that every rule needs on the EP. These are workflow assets (source, configs, scripts) — NOT DAG dependency edges. They live in `htcondor_transfer_input_files` so the plugin ships them to the EP; the `input:` directive of each rule lists only the upstream sentinels that form the actual DAG. Files under /staging or /projects are excluded by the profile's `htcondor-shared-fs-prefixes` and don't need to be listed here.
COMMON_TRANSFER = ",".join([
    "src",
    "pyproject.toml",
    "README.md",
    f"{WORK_DIR}/{CONFIG_FILE}",
    f"{WORK_DIR}/scripts",
    f"{WORK_DIR}/config_fit_nsbi.yml",
    f"{WORK_DIR}/config_fit_histogram.yml",
])


def train_sentinels():
    """All nominal-training sentinel paths, fanned out over (process, fold, ensemble_idx). Fold dimension is collapsed when NUM_FOLDS == 1."""
    if USE_KFOLD:
        return expand(
            f"{WORK_DIR}/.done_train_{{process}}_fold{{fold}}_{{ensemble_idx}}",
            process=BASIS_PROCESSES, fold=FOLD_INDICES, ensemble_idx=ENSEMBLE_INDICES,
        )
    return expand(
        f"{WORK_DIR}/.done_train_{{process}}_{{ensemble_idx}}",
        process=BASIS_PROCESSES, ensemble_idx=ENSEMBLE_INDICES,
    )


def syst_sentinels():
    """All systematic-training sentinel paths, fanned out over (process, syst, direction, fold, ensemble_idx)."""
    paths = []
    for process, syst, direction in SYST_COMBOS:
        for idx in SYST_ENSEMBLE_INDICES:
            if USE_KFOLD:
                for fold in FOLD_INDICES:
                    paths.append(f"{WORK_DIR}/.done_syst_{process}_{syst}_{direction}_fold{fold}_{idx}")
            else:
                paths.append(f"{WORK_DIR}/.done_syst_{process}_{syst}_{direction}_{idx}")
    return paths


rule all:
    input:
        f"{WORK_DIR}/.done_data_nn_eval"


# =============================================================================
# Stage 1: Data Processing
# =============================================================================

rule data_loader:
    # No `input:` — first stage of the DAG. All assets needed on the EP (src, pyproject, scripts, config) come via the static `htcondor_transfer_input_files` list in `resources:`.
    output:
        done = touch(f"{WORK_DIR}/.done_data_loader"),
    log:
        out = "logs/data_loader.out",
        err = "logs/data_loader.err",
    threads: 8
    resources:
        request_memory = "64GB",
        request_disk   = "64GB",
        requirements   = "(Target.HasCHTCStaging == true)",
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/data_loader.py --config {CONFIG_FILE} \
            1> ../{log.out} 2> ../{log.err}
        """


rule data_preprocessing:
    input:
        loader_done = f"{WORK_DIR}/.done_data_loader",
    output:
        done = touch(f"{WORK_DIR}/.done_data_preprocessing"),
    log:
        out = "logs/data_preprocessing.out",
        err = "logs/data_preprocessing.err",
    threads: 8
    resources:
        request_memory = "42GB",
        request_disk   = "42GB",
        request_gpus   = 1,
        requirements   = "(Target.HasCHTCStaging == true)",
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
        skip     = "--skip" if config.get("skip", False) else "",
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/data_preprocessing.py --config {CONFIG_FILE} \
            {params.skip} \
            1> ../{log.out} 2> ../{log.err}
        """


# =============================================================================
# Stage 2: Preselection Network
# =============================================================================

rule preselection_network:
    input:
        preprocessing_done = f"{WORK_DIR}/.done_data_preprocessing",
    output:
        done = touch(f"{WORK_DIR}/.done_preselection_network"),
    log:
        out = "logs/preselection_network.out",
        err = "logs/preselection_network.err",
    threads: 8
    resources:
        request_memory          = "32GB",
        request_disk            = "32GB",
        request_gpus            = 1,
        gpus_minimum_capability = _r(7.0),
        classad_WantGPULab      = _r(True),
        classad_GPUJobLength    = "short",
        requirements            = gpu_requirements(driver_version="12.0"),
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
        skip     = "--skip" if config.get("skip", False) else "",
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/preselection_network.py --config {CONFIG_FILE} \
            {params.skip} \
            1> ../{log.out} 2> ../{log.err}
        """


# =============================================================================
# Stage 3a: Density Ratio Training — parallel ensemble fan-out
#
# Replaces: generate_training_dag.py + SUBDAG EXTERNAL train_ensemble.dag
#
# Snakemake reads BASIS_PROCESSES, N_ENSEMBLE, and NUM_FOLDS from config at
# parse time and instantiates one job per (process, fold, ensemble_idx) when
# k-fold is on, or per (process, ensemble_idx) otherwise — mirroring the
# JOB/VARS pairs emitted by generate_training_dag.py.
# =============================================================================

rule train_ensemble_fold:
    """K-fold variant: one job per (process, fold, ensemble_idx). Only instantiated when NUM_FOLDS > 1; the fold sentinel pattern can't match anything when there's just one fold."""
    input:
        preselection_done = f"{WORK_DIR}/.done_preselection_network",
    output:
        done = touch(f"{WORK_DIR}/.done_train_{{process}}_fold{{fold}}_{{ensemble_idx}}"),
    log:
        out = "logs/train_ensemble_{process}_fold{fold}_{ensemble_idx}.out",
        err = "logs/train_ensemble_{process}_fold{fold}_{ensemble_idx}.err",
    threads: 8
    resources:
        request_memory          = "16GB",
        request_disk            = "32GB",
        request_gpus            = 1,
        gpus_minimum_capability = _r(7.0),
        classad_WantGPULab      = _r(True),
        classad_GPUJobLength    = "short",
        requirements            = gpu_requirements(driver_version="12.4", extra_excludes=["gpulab2001.chtc.wisc.edu", "dsigpu4001.chtc.wisc.edu"]),
        # Replaces legacy `periodic_hold` (2.5 h cap) + `periodic_release` retry loop. The plugin does not expose those primitives; `allowed_job_duration` kills the job at the deadline and `max_retries` re-submits up to N times.
        allowed_job_duration    = 9000,
        max_retries             = 3,
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/neural_likelihood_ratio_estimation.py \
            --config {CONFIG_FILE} \
            --process {wildcards.process} \
            --ensemble_index {wildcards.ensemble_idx} \
            --fold_index {wildcards.fold} \
            1> ../{log.out} 2> ../{log.err}
        """


rule train_ensemble_nofold:
    """Single-fold variant: one job per (process, ensemble_idx). The script omits the `_fold{N}` suffix from model paths when `--fold_index` is not passed, matching the htcondor non-kfold case."""
    input:
        preselection_done = f"{WORK_DIR}/.done_preselection_network",
    output:
        done = touch(f"{WORK_DIR}/.done_train_{{process}}_{{ensemble_idx}}"),
    log:
        out = "logs/train_ensemble_{process}_{ensemble_idx}.out",
        err = "logs/train_ensemble_{process}_{ensemble_idx}.err",
    threads: 8
    resources:
        request_memory          = "16GB",
        request_disk            = "32GB",
        request_gpus            = 1,
        gpus_minimum_capability = _r(7.0),
        classad_WantGPULab      = _r(True),
        classad_GPUJobLength    = "short",
        requirements            = gpu_requirements(driver_version="12.4", extra_excludes=["gpulab2001.chtc.wisc.edu", "dsigpu4001.chtc.wisc.edu"]),
        allowed_job_duration    = 9000,
        max_retries             = 3,
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/neural_likelihood_ratio_estimation.py \
            --config {CONFIG_FILE} \
            --process {wildcards.process} \
            --ensemble_index {wildcards.ensemble_idx} \
            1> ../{log.out} 2> ../{log.err}
        """


# =============================================================================
# Stage 3b: Systematic Uncertainty Training — parallel fan-out over
# (process, systematic, direction, fold, ensemble_idx).
#
# Replaces: generate_systematics_dag.py + SUBDAG EXTERNAL train_systematics.dag.
# SYST_COMBOS is built at parse time from the NSBI fit config (same filter
# logic: NormPlusShape systematics, restricted to listed Samples).
# =============================================================================

rule systematic_uncertainty_training_fold:
    """K-fold variant of the systematics training job."""
    input:
        preselection_done = f"{WORK_DIR}/.done_preselection_network",
    output:
        done = touch(f"{WORK_DIR}/.done_syst_{{process}}_{{syst}}_{{direction}}_fold{{fold}}_{{ensemble_idx}}"),
    log:
        out = "logs/syst_{process}_{syst}_{direction}_fold{fold}_{ensemble_idx}.out",
        err = "logs/syst_{process}_{syst}_{direction}_fold{fold}_{ensemble_idx}.err",
    threads: 8
    resources:
        request_memory          = "16GB",
        request_disk            = "32GB",
        request_gpus            = 1,
        gpus_minimum_capability = _r(7.5),
        classad_WantGPULab      = _r(True),
        classad_GPUJobLength    = "short",
        requirements            = gpu_requirements(driver_version="12.4", extra_excludes=["gpulab2001.chtc.wisc.edu"]),
        # Legacy `.sub` used a 1 h periodic_hold with 4 retries; plugin equivalent below.
        allowed_job_duration    = 3600,
        max_retries             = 3,
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/systematic_uncertainty_training.py \
            --config {CONFIG_FILE} \
            --process {wildcards.process} \
            --systematic {wildcards.syst} \
            --direction {wildcards.direction} \
            --ensemble_index {wildcards.ensemble_idx} \
            --fold_index {wildcards.fold} \
            --train \
            1> ../{log.out} 2> ../{log.err}
        """


rule systematic_uncertainty_training_nofold:
    """Single-fold variant of the systematics training job."""
    input:
        preselection_done = f"{WORK_DIR}/.done_preselection_network",
    output:
        done = touch(f"{WORK_DIR}/.done_syst_{{process}}_{{syst}}_{{direction}}_{{ensemble_idx}}"),
    log:
        out = "logs/syst_{process}_{syst}_{direction}_{ensemble_idx}.out",
        err = "logs/syst_{process}_{syst}_{direction}_{ensemble_idx}.err",
    threads: 8
    resources:
        request_memory          = "16GB",
        request_disk            = "32GB",
        request_gpus            = 1,
        gpus_minimum_capability = _r(7.5),
        classad_WantGPULab      = _r(True),
        classad_GPUJobLength    = "short",
        requirements            = gpu_requirements(driver_version="12.4", extra_excludes=["gpulab2001.chtc.wisc.edu"]),
        allowed_job_duration    = 3600,
        max_retries             = 3,
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/systematic_uncertainty_training.py \
            --config {CONFIG_FILE} \
            --process {wildcards.process} \
            --systematic {wildcards.syst} \
            --direction {wildcards.direction} \
            --ensemble_index {wildcards.ensemble_idx} \
            --train \
            1> ../{log.out} 2> ../{log.err}
        """


# =============================================================================
# Stage 4: Density Ratio Evaluation
# =============================================================================

rule data_nn_eval:
    input:
        # train_sentinels() and syst_sentinels() compute the full fan-out lists at parse time; their shape depends on NUM_FOLDS and on SYST_COMBOS (derived from the NSBI fit config). data_nn_eval cannot start until every individual training job lands.
        ensemble_done    = train_sentinels(),
        systematics_done = syst_sentinels(),
    output:
        done = touch(f"{WORK_DIR}/.done_data_nn_eval"),
    log:
        out = "logs/data_nn_eval.out",
        err = "logs/data_nn_eval.err",
    threads: 8
    resources:
        request_memory          = "16GB",
        request_disk            = "32GB",
        request_gpus            = 1,
        gpus_minimum_capability = _r(7.5),
        classad_WantGPULab      = _r(True),
        classad_GPUJobLength    = "short",
        requirements            = gpu_requirements(driver_version="12.6", extra_excludes=["gpulab2001.chtc.wisc.edu", "dsigpu4001.chtc.wisc.edu"]),
        htcondor_transfer_input_files = COMMON_TRANSFER,
    params:
        work_dir = WORK_DIR,
    shell:
        """
        set -e
        export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
        python -m pip install --no-deps --user -e .
        cd {params.work_dir}
        python -u scripts/data_nn_eval.py --config {CONFIG_FILE} \
            1> ../{log.out} 2> ../{log.err}
        """
