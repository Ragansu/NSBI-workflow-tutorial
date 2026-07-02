# ML4HEP-TIF NSBI tutorial — Gaussian-mixture example

This tutorial walks through a full **Neural Simulation-Based Inference (NSBI)**
workflow for a signal-strength measurement: estimate per-event density ratios
with neural networks, build an unbinned statistical model, and fit the
parameter of interest `μ`.

It mirrors the [`nsbi_atlas_workshop`](../nsbi_atlas_workshop/) example, but the
toy data here is deliberately harder. Instead of each feature being a single
Gaussian, **every sample is a mixture of several correlated Gaussian
components**, giving multi-modal marginals and curved correlations. This makes
the joint density genuinely non-trivial to model — a good stress test for
flexible density estimators such as Normalizing Flows (a planned follow-up),
while the density-ratio-based workflow below handles it out of the box.

Numerical stability by design
---
Density-ratio estimation is only well behaved when the reference (denominator)
has support everywhere the numerator does. To guarantee this, **every** sample
(background, all signals, pseudo-data) contains a common broad component. Since
that component alone already covers the whole region of interest with
`p(x) > 0`, no phase-space pocket exists where one density vanishes while
another does not — so the trained ratios stay bounded and the classifier scores
stay away from 0 and 1. The reference hypothesis used for training is
`background + signal_0`, which covers both the background- and signal-like
regions.

Generating the data
---
There is no download for this example — generate it locally:

```bash
python generate_distributions.py --n_bkg 5_000_000 --n_sig 5_000_000
```

Adjust `--n_bkg` / `--n_sig` to taste (larger = less per-event MC noise in the
fit). Parquets land in `./dataframes/`, diagnostic plots in `./plots/`, and the
per-event weights are auto-scaled so the total yields stay fixed regardless of
the requested statistics (`λ_bkg = 1e6`, `λ_sig(v=10) ≈ 1100`). The `--nodes`
argument controls the signal basis parameter values used for Lagrange morphing
(default `0 5 10`).

Installation
---

- Clone the GitHub repository locally using `GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:iris-hep/nsbi-lhc-toolkit.git --depth=1 && cd nsbi-lhc-toolkit`.
- Run `pixi install -e nsbi-env` if you are using CPU or Mac and `pixi install -e nsbi-env-gpu` if you have access to a CUDA-supported GPU.
- Install the kernel using `pixi run -e nsbi-env-gpu python -m ipykernel install --user --name nsbi-env-gpu --display-name "Python (pixi: nsbi-env)"` if you are running on GPU or `pixi run -e nsbi-env python -m ipykernel install --user --name nsbi-env --display-name "Python (pixi: nsbi-env)"` if you are running on CPU or Mac.
- Go to the tutorial directory `workshops/ml4hep_tif/`, generate the data (above), and start running the notebooks. Make sure to select the kernel `Python (pixi: nsbi-env)` before you run.

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
   the trained ratios on the Asimov dataset, and fit the signal strength `μ`
   (including a profile-likelihood scan of `t_μ`).

A small per-event bias in `μ̂` is expected — it comes from the finite-statistics
MC noise in the per-event ratios. Reduce it by generating more data, training
larger ensembles, or increasing `N_TRAIN` in the training notebooks.

Files
---
- `generate_distributions.py` — Gaussian-mixture event generator (see the
  module docstring for the full design rationale).
- `utils.py` — shared helpers (colours, Lagrange morphing weights).
- `dataframes/` — generated parquet samples (created by the generator).
- `plots*/`, `models*/`, `saved_*/` — outputs created while running the notebooks.
