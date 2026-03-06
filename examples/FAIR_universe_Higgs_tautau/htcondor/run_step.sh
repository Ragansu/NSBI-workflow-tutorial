#!/usr/bin/env bash
set -e

STEP=$1
CONFIG=$2
ENSEMBLE_INDEX=$3

WORK_DIR=FAIR_universe_Higgs_tautau

export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --no-deps --user -e .

echo "PATH=$PATH"
which python
python -c "import sys; print(sys.executable)"

cd $WORK_DIR

if [ "$STEP" = "neural_likelihood_ratio_estimation" ] && [ -n "$ENSEMBLE_INDEX" ]; then
    python -u scripts/${STEP}.py --config ${CONFIG} --ensemble_index ${ENSEMBLE_INDEX}
else
    python -u scripts/${STEP}.py --config ${CONFIG}
fi
