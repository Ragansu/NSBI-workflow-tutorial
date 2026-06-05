FAIR Universe Dataset
--

The tabular dataset used in this demonstration is hosted on Zenodo (https://zenodo.org/records/15131565), and is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The dataset provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - good challenge to test the sensitivty of NSBI techniques.

## Download saved models and processed data

If you want to skip the data loading and preprocessing stages — and optionally use pre-trained ensemble networks — download the pre-processed bundle from [https://cernbox.cern.ch/s/mltUDvzKdisEEpJ](https://cernbox.cern.ch/s/mltUDvzKdisEEpJ) and extract it:

```bash
tar -xvf saved_datasets.tar.gz
```

This creates a `saved_datasets/` directory with the processed `.root` files (and optionally the trained models). **If you use this download, do not re-run notebooks 1 (data loader) and 2 (data preprocessing).** They regenerate exactly the artifacts you just downloaded and would overwrite them. Start from notebook 3 (or notebook 4 if you also pulled pre-trained models).

## Running the pipeline with Snakemake on HTCondor

The whole pipeline is one [Snakemake](https://snakemake.readthedocs.io/) workflow. From the repository root:

```bash
snakemake --snakefile examples/FAIR_universe_Higgs_tautau/Snakefile \
          --profile  examples/FAIR_universe_Higgs_tautau/profiles/chtc
```

That submits every required job to HTCondor (via the `snakemake-executor-plugin-htcondor`), waits for completion, and produces the final `parameter_fitting` outputs. Re-running the same command after a failure resumes from where it stopped — sentinels under `/projects/.../sentinels_FAIR_higgs/` track which (process, fold, ensemble) jobs are done.

To target a different cluster, copy `profiles/chtc/` to `profiles/<your-cluster>/`, change `executor:` to the appropriate snakemake plugin (`slurm`, `cluster-generic`, etc.) in `config.yaml`, and adjust `default-resources`.

See [`docs/basics/workflow.rst`](../../docs/basics/workflow.rst) for the full reference (rule structure, partial reruns, troubleshooting).

### Workflow chart

![NSBI workflow](../../docs/_images/toolkit_workflow_AGCstyle.png)
