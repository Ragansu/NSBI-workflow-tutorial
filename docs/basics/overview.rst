Overview
========

What is NSBI?
-------------

Neural Simulation-Based Inference (NSBI) is a semi-parametric approach to
statistical inference in particle physics. Instead of relying entirely on
histogrammed templates (as in traditional HistFactory analyses) or fully
non-parametric methods, NSBI combines:

- **Binned template channels** — control regions and high-purity signal
  regions where histograms are sufficient.
- **Unbinned neural density-ratio channels** — signal regions where the
  per-event likelihood ratio is learned by a neural network, preserving
  the full kinematic information.

Both channel types share a common set of parameters (signal strengths and
nuisance parameters) and are combined into a single profiled negative
log-likelihood that is minimised with iminuit.

How the pieces fit together
---------------------------

The analysis pipeline has four stages:

1. **Data preparation** — Load ROOT ntuples, apply preselection, split
   into analysis regions (binned control regions, unbinned signal region).

2. **Training** — For each physics process (basis point), train an ensemble
   of density-ratio neural networks that learn
   :math:`r(x) = p_{\text{process}}(x) / p_{\text{reference}}(x)`.
   Systematic variations get their own networks.

3. **Evaluation** — Run the trained networks on the full Asimov dataset to
   produce per-event density-ratio arrays. These arrays, together with the
   binned histograms and systematic variation templates, are assembled into
   a **workspace** — a JSON-like dictionary that fully specifies the
   statistical model.

4. **Fitting** — The workspace is passed to
   :class:`~nsbi_common_utils.models.sbi_parametric_model.sbi_parametric_model`,
   which compiles a JAX-based NLL function. This is minimised by
   :class:`~nsbi_common_utils.inference.inference` to extract the
   best-fit parameters and profile-likelihood scans.

.. image:: /_static/DAG_overview.svg
   :alt: NSBI workflow overview
   :align: center
   :width: 100%
