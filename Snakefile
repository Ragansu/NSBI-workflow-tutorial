# =============================================================================
# Snakemake workflow: NSBI FAIR Universe Higgs tautau
# Translated from DAGMan pipeline
# =============================================================================

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------
configfile: "examples/FAIR_universe_Higgs_tautau/config.pipeline.yaml"

EXAMPLE_DIR = "FAIR_universe_Higgs_tautau"
WORK_DIR    = f"examples/{EXAMPLE_DIR}"
CONFIG_FILE = "config.pipeline.yaml"    # path relative to WORK_DIR

# ---------------------------------------------------------------------------
# Read ensemble training parameters directly from config.
# This replaces the entire generate_training_dag.py PRE script — Snakemake
# does the fan-out at parse time instead of generating a child DAG at runtime.
# ---------------------------------------------------------------------------
_nlre           = config["neural_likelihood_ratio_estimation"]
BASIS_PROCESSES = _nlre["basis_processes_to_train"]
N_ENSEMBLE      = _nlre["num_ensemble_members_training"]
ENSEMBLE_INDICES = list(range(N_ENSEMBLE))


# ---------------------------------------------------------------------------
# Top-level target — pulling in the final step pulls in the whole pipeline
# ---------------------------------------------------------------------------
rule all:
    input:
        f"{WORK_DIR}/.done_data_nn_eval"


# =============================================================================
# Stage 1: Data Processing
# =============================================================================

rule data_loader:
    input:
        config    = f"{WORK_DIR}/{CONFIG_FILE}",
        src       = "src",
        pyproject = "pyproject.toml",
        readme    = "README.md",
    output:
        done = touch(f"{WORK_DIR}/.done_data_loader"),
    log:
        out = "logs/data_loader.out",
        err = "logs/data_loader.err",
    resources:
        cpus    = 8,
        mem_mb  = 64 * 1024,
        gpus    = 0,
        disk_mb = 64 * 1024,
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
        config      = f"{WORK_DIR}/{CONFIG_FILE}",
        src         = "src",
        pyproject   = "pyproject.toml",
        readme      = "README.md",
    output:
        done = touch(f"{WORK_DIR}/.done_data_preprocessing"),
    log:
        out = "logs/data_preprocessing.out",
        err = "logs/data_preprocessing.err",
    resources:
        cpus    = 8,
        mem_mb  = 42 * 1024,
        gpus    = 1,
        disk_mb = 42 * 1024,
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
        config             = f"{WORK_DIR}/{CONFIG_FILE}",
        src                = "src",
        pyproject          = "pyproject.toml",
        readme             = "README.md",
    output:
        done = touch(f"{WORK_DIR}/.done_preselection_network"),
    log:
        out = "logs/preselection_network.out",
        err = "logs/preselection_network.err",
    resources:
        cpus    = 8,
        mem_mb  = 32 * 1024,
        gpus    = 1,
        disk_mb = 32 * 1024,
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
# Snakemake reads BASIS_PROCESSES and N_ENSEMBLE from config at parse time
# and instantiates one job per (process, ensemble_idx) combination — exactly
# what generate_training_dag.py was building dynamically as JOB/VARS lines.
# No PRE script, no child DAG file, no condor_submit inside a script.
# =============================================================================

rule train_ensemble:
    input:
        preselection_done = f"{WORK_DIR}/.done_preselection_network",
        config            = f"{WORK_DIR}/{CONFIG_FILE}",
        src               = "src",
        pyproject         = "pyproject.toml",
        readme            = "README.md",
    output:
        # One sentinel per (process, ensemble_idx) — same granularity as the
        # individual JOB nodes that generate_training_dag.py was emitting
        done = touch(f"{WORK_DIR}/.done_train_{{process}}_{{ensemble_idx}}"),
    log:
        out = "logs/train_ensemble_{process}_{ensemble_idx}.out",
        err = "logs/train_ensemble_{process}_{ensemble_idx}.err",
    resources:
        cpus    = 8,
        mem_mb  = 16 * 1024,
        gpus    = 1,
        disk_mb = 32 * 1024,
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
            --ensemble_index {wildcards.ensemble_idx} \
            --process {wildcards.process} \
            1> ../{log.out} 2> ../{log.err}
        """


# =============================================================================
# Stage 3b: Systematic Uncertainty Training
# Runs in parallel with Stage 3a — both feed into data_nn_eval
# =============================================================================

rule systematic_uncertainty_training:
    input:
        preselection_done = f"{WORK_DIR}/.done_preselection_network",
        config            = f"{WORK_DIR}/{CONFIG_FILE}",
        src               = "src",
        pyproject         = "pyproject.toml",
        readme            = "README.md",
    output:
        done = touch(f"{WORK_DIR}/.done_systematic_uncertainty_training"),
    log:
        out = "logs/systematic_uncertainty_training.out",
        err = "logs/systematic_uncertainty_training.err",
    resources:
        cpus    = 8,
        mem_mb  = 16 * 1024,
        gpus    = 1,
        disk_mb = 32 * 1024,
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
            1> ../{log.out} 2> ../{log.err}
        """


# =============================================================================
# Stage 4: Density Ratio Evaluation
#
# Fan-in: waits for every ensemble member (all processes × all indices)
#         AND systematic_uncertainty_training — mirrors the two PARENT lines:
#   PARENT neural_likelihood_ratio_estimation CHILD data_nn_eval
#   PARENT systematic_uncertainty_training    CHILD data_nn_eval
# =============================================================================

rule data_nn_eval:
    input:
        # expand() produces the full list of sentinels, one per combination
        ensemble_done = expand(
            f"{WORK_DIR}/.done_train_{{process}}_{{ensemble_idx}}",
            process=BASIS_PROCESSES,
            ensemble_idx=ENSEMBLE_INDICES,
        ),
        systematics_done = f"{WORK_DIR}/.done_systematic_uncertainty_training",
        config           = f"{WORK_DIR}/{CONFIG_FILE}",
        src              = "src",
        pyproject        = "pyproject.toml",
        readme           = "README.md",
    output:
        done = touch(f"{WORK_DIR}/.done_data_nn_eval"),
    log:
        out = "logs/data_nn_eval.out",
        err = "logs/data_nn_eval.err",
    resources:
        cpus    = 8,
        mem_mb  = 16 * 1024,
        gpus    = 1,
        disk_mb = 32 * 1024,
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
