#!/bin/bash
# DAGMan SCRIPT POST — runs on the AP after each training job completes.
# Tars the job's output directories into a per-process archive on /staging,
# then removes the individual files to stay within inode quota.
#
# Usage: append_and_cleanup.sh <process> <ensemble_idx> <saved_data_path> <archive_dir>
#   $1 = process type (e.g. ttbar, ztautau, htautau)
#   $2 = ensemble index (e.g. 0, 1, 2, ...)
#   $3 = saved_data_path (e.g. /home/jsandesara/saved_datasets/output_training_nominal)
#   $4 = archive directory (e.g. /staging/jsandesara/model_archives)
#   $5 = DAGMan return code from the job (passed automatically by DAGMan)

PROCESS=$1
IDX=$2
SAVED_DATA_PATH=$3
ARCHIVE_DIR=$4
JOB_RETURN_CODE=$5

# Only archive if the job succeeded
if [ "$JOB_RETURN_CODE" != "0" ]; then
    echo "Job failed (rc=$JOB_RETURN_CODE), skipping archive for ${PROCESS}_${IDX}"
    exit 0
fi

ARCHIVE="${ARCHIVE_DIR}/models_${PROCESS}.tar"
LOCKFILE="${ARCHIVE_DIR}/.lock_${PROCESS}"
mkdir -p "$ARCHIVE_DIR"

MODEL_DIR="output_model_params_${PROCESS}${IDX}"
FIGURES_DIR="output_figures_${PROCESS}${IDX}"

cd "$SAVED_DATA_PATH" || exit 1

# Acquire per-process lock to prevent concurrent tar writes
exec 9>"$LOCKFILE"
flock 9

# Append model params to archive, then remove
if [ -d "$MODEL_DIR" ]; then
    tar -rf "$ARCHIVE" "$MODEL_DIR"
    rm -rf "$MODEL_DIR"
    echo "Archived and removed ${MODEL_DIR}"
fi

# Append figures to archive, then remove
if [ -d "$FIGURES_DIR" ]; then
    tar -rf "$ARCHIVE" "$FIGURES_DIR"
    rm -rf "$FIGURES_DIR"
    echo "Archived and removed ${FIGURES_DIR}"
fi

# Release lock
flock -u 9
