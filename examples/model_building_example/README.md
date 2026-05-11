# Model Building and Fitting Example

This standalone example demonstrates how to build an unbinned statistical model and perform hypothesis testing using pre-computed density ratios. It is **independent** of the training pipeline — you only need:

1. A fit configuration YAML file (`config_fit_nsbi.yml`)
2. Pre-computed density ratio `.npy` files (nominal + systematic variations)
3. Asimov weights (or real data weights)
4. Input ROOT files (for binned histogram channels)

## Quick Start

```python
from nsbi_common_utils import workspace_builder, models, inference

# 1. Build workspace from config
ws = workspace_builder.WorkspaceBuilder(config_path="config_fit_nsbi.yml").build()

# 2. Initialize the statistical model (JAX JIT-compiled NLL)
model = models.sbi_parametric_model(workspace=ws, measurement_to_fit="my_measurement")

# 3. Fit
params, init_vals = model.get_model_parameters()
fitter = inference.inference(
    model_nll=model.model,
    initial_values=init_vals,
    list_parameters=params,
    num_unconstrained_params=model.num_unconstrained_param
)
fitter.perform_fit()
```

## What you need to provide

### Fit configuration (`config_fit_nsbi.yml`)

This YAML file defines:
- **Measurement**: which parameters to fit and the parameter of interest (POI)
- **Samples**: the physics processes (e.g. signal, backgrounds), with paths to ROOT files
- **NormFactors**: free normalization parameters per sample
- **Systematics**: nuisance parameters with paths to up/down variation ROOT files
- **Regions**: analysis regions — binned (control regions) and unbinned (signal region with density ratios)

The unbinned region references pre-computed density ratios:
```yaml
Regions:
- Name: SR
  Type: unbinned
  AsimovWeights: ./saved_datasets/asimov_weights.npy
  TrainedModels:
    - SampleName: signal
      Nominal:
        Ratios: ./saved_datasets/output_training_nominal/output_ratios_signal/ratio_signal.npy
      Systematics:
        - SystName: JES
          RatiosUp: ./saved_datasets/output_training_systematics/output_ratios_signal_JES_Up/ratio_signal.npy
          RatiosDn: ./saved_datasets/output_training_systematics/output_ratios_signal_JES_Dn/ratio_signal.npy
```

### Density ratio files

Each `.npy` file is a 1D array of shape `(n_events,)` containing the ensemble-averaged density ratio `p_c(x) / p_ref(x)` for each event in the dataset. These are produced by the training + evaluation pipeline (Stages 1-3), but can also be provided from any external source.

### Directory structure

```
model_building_example/
  config_fit_nsbi.yml          # Fit configuration
  saved_datasets/
    asimov_weights.npy         # Per-event weights for the unbinned region
    dataset_nominal.root       # Nominal MC (for binned channels)
    dataset_JES_up.root        # Systematic variation ROOT files
    ...
    output_training_nominal/
      output_ratios_<sample>/
        ratio_<sample>.npy     # Nominal density ratios per sample
    output_training_systematics/
      output_ratios_<sample>_<syst>_<dir>/
        ratio_<sample>.npy     # Systematic density ratios
```

## Notebook

See `parameter_fitting.ipynb` for a step-by-step walkthrough that:
1. Builds NSBI and histogram workspaces from config files
2. Initializes `sbi_parametric_model` for each
3. Performs NLL minimization via `iminuit`
4. Runs profile likelihood scans
5. Compares NSBI vs histogram-only sensitivity
