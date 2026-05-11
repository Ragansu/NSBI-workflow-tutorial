#!/bin/bash

# Fail early if there's an issue
set -e

# When .cache files are created, they need to know where HOME is to write there.
# In this case, that should be the HTCondor scratch dir the job is executing in.
export HOME=$(pwd)

# Workaround for a snakemake-executor-plugin-htcondor bug: the plugin tries to
# strip the literal prefix `"python -m snakemake "` from its args before passing
# them to a job_wrapper, but only matches a bare "python". When the AP's
# `sys.executable` is a full path (the pixi-env case:
# `/home/.../.pixi/envs/nsbi-env/bin/python3.12`), the removeprefix call doesn't
# match and "$@" arrives as either `<python-path> -m snakemake <real-args>` or
# `-m snakemake <real-args>`. Strip whichever form we got, then forward to
# snakemake. No-op if the plugin behaves correctly.
if [ "${2:-}" = "-m" ] && [ "${3:-}" = "snakemake" ]; then
    shift 3
elif [ "${1:-}" = "-m" ] && [ "${2:-}" = "snakemake" ]; then
    shift 2
fi

# Pass any arguments to Snakemake
snakemake "$@"
