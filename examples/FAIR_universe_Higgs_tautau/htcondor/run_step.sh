#!/usr/bin/env bash
set -e

STEP=$1
CONFIG=$2
ENSEMBLE_INDEX=$3
PROCESS_TYPE=$4

WORK_DIR=FAIR_universe_Higgs_tautau

export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --no-deps --user -e .

echo "PATH=$PATH"
which python
python -c "import sys; print(sys.executable)"

cd $WORK_DIR

# Steps that read saved models: copy archives from /staging to local scratch and untar.
# Local scratch has no inode quota. The Python scripts read from saved_data_path
# which should be set to LOCAL_SAVED_DATA in the config for these steps.
ARCHIVE_DIR="/staging/jsandesara/model_archives"
LOCAL_SAVED_DATA="$(pwd)/local_saved_data"

if [ "$STEP" = "data_nn_eval" ] || [ "$STEP" = "parameter_fitting" ] || \
   { [ "$STEP" = "neural_likelihood_ratio_estimation" ] && [ -n "$ENSEMBLE_INDEX" ]; }; then

    mkdir -p "$LOCAL_SAVED_DATA"
    for archive in "${ARCHIVE_DIR}"/models_*.tar; do
        [ -f "$archive" ] || continue
        echo "Copying $(basename "$archive") to local scratch..."
        cp "$archive" "$LOCAL_SAVED_DATA/"
        echo "Extracting..."
        tar -xf "$LOCAL_SAVED_DATA/$(basename "$archive")" -C "$LOCAL_SAVED_DATA"
        rm -f "$LOCAL_SAVED_DATA/$(basename "$archive")"
    done
    echo "Models available at: ${LOCAL_SAVED_DATA}"
    export LOCAL_SAVED_DATA
fi

if [ "$STEP" = "neural_likelihood_ratio_estimation" ] && [ -n "$ENSEMBLE_INDEX" ]; then
    python -u scripts/${STEP}.py --config ${CONFIG} --ensemble_index ${ENSEMBLE_INDEX} --process ${PROCESS_TYPE}
else
    python -u scripts/${STEP}.py --config ${CONFIG}
fi
