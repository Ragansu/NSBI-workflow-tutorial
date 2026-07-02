# ML4HEP-TIFR NSBI tutorial — Gaussian-mixture example

This tutorial walks through a full **Neural Simulation-Based Inference (NSBI)**
workflow for a signal-strength measurement: estimate per-event density ratios
with neural networks, build an unbinned statistical model, and fit the
parameter of interest $\mu$.

Generating the data
---
There is no download for this example — generate it locally:

```bash
python generate_distributions.py --n_bkg 5_000_000 --n_sig 5_000_000
```

Adjust `--n_bkg` / `--n_sig` to taste (larger = less per-event MC noise in the
fit). Parquets land in `./dataframes/` (`background`, `signal`, and an
independent `data` draw), diagnostic plots in `./plots/`, and the per-event
weights are auto-scaled so the total yields stay fixed regardless of the
requested statistics (`λ_bkg = 1e6`, `λ_sig ≈ 1100`).

Installation
---

- Clone the GitHub repository locally and check out this tutorial's branch using `GIT_LFS_SKIP_SMUDGE=1 git clone -b ml4hep_school_tutorial git@github.com:iris-hep/NSBI-workflow-tutorial.git --depth=1 && cd NSBI-workflow-tutorial`.
- Run `pixi install -e nsbi-env` if you are using CPU or Mac and `pixi install -e nsbi-env-gpu` if you have access to a CUDA-supported GPU.
- Install the kernel using `pixi run -e nsbi-env-gpu python -m ipykernel install --user --name nsbi-env-gpu --display-name "Python (pixi: nsbi-env)"` if you are running on GPU or `pixi run -e nsbi-env python -m ipykernel install --user --name nsbi-env --display-name "Python (pixi: nsbi-env)"` if you are running on CPU or Mac.
- Go to the tutorial directory `workshops/ml4hep_tifr/`, generate the data (above), and start running the notebooks. Make sure to select the kernel `Python (pixi: nsbi-env)` before you run.

Running the notebooks
---
Run them in order:

1. **`1_visualize_data.ipynb`** — inspect the features. See the multi-modal
   marginals and the 2D views that expose the correlated, mixture structure.
2. **`2a_SigvsRef_training.ipynb`** — train the signal-vs-reference density
   ratio `r_sig(x) = p_sig(x) / p_ref(x)`, then run the calibration and
   reweighting closure diagnostics.
3. **`2b_BkgvsRef_training.ipynb`** — train the background-vs-reference density
   ratio `r_bkg(x) = p_bkg(x) / p_ref(x)` with the same diagnostics.
4. **`3_parameter_fitting.ipynb`** — build the unbinned SBI workspace, evaluate
   the trained ratios on the Asimov dataset, and fit the signal strength $\mu$
   (including a profile-likelihood scan of $t_\mu$).

A small per-event bias in $\hat{\mu}$ is expected — it comes from the finite-statistics
MC noise in the per-event ratios. Reduce it by generating more data, training
larger ensembles, or increasing `N_TRAIN` in the training notebooks.

Files
---
- `generate_distributions.py` — Gaussian-mixture event generator (see the
  module docstring for the full design rationale).
- `utils.py` — shared helpers: the Gaussian-mixture definitions (`background_components`, `signal_components`, `mixture_density`) and the `split_train_inference` train/inference splitter.
- `dataframes/` — generated parquet samples (created by the generator).
- `plots*/`, `models*/`, `saved_*/` — outputs created while running the notebooks.
