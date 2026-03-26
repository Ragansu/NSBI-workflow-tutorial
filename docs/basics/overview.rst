Overview
========

What is Simulation-Based Inference?
-----------------------------------

Simulation-Based Inference (SBI) or Neural Simulation-Based Inference (NSBI) refers to set of statistical techniques that allow statistical inference directly using high-dimensional data. This circumvents the need to build low-dimensional summaries as is traditionally done and which can lose sensitive information. 

This toolkit helps facilitate the application of a type of SBI that is scalable for LHC-style analysis with high-dimensional parameter spaces, where the systematic uncertainty modeling is done via certain domain-specific assumptions. This is done via easy-to-use APIs for the various stages in the analysis as well as providing an end-to-end workflow orchestratation pipeline steered via human-readable configuration files.

More details to be added soon!

.. figure:: ../_images/toolkit_workflow_AGCstyle.png
   :width: 80%
   :align: center

   High-level overview of the SBI workflow and the various stages.


How the pieces fit together
---------------------------

The analysis pipeline has four stages:

1. **Data preparation** — Load ROOT ntuples, apply preselection, split
   into analysis regions (binned control regions, unbinned signal region).

2. **Training** — For each physics process (basis point), train an ensemble of density-ratio neural networks that learn :math:`r(x) = p_{\text{process}}(x) / p_{\text{reference}}(x)`. Systematic variations get their own ensemble models.

3. **Evaluation** — Run the trained networks on the full Asimov or real dataset to produce per-event density-ratio arrays. These arrays, together with the binned histograms and systematic variation templates, are assembled into a serialized **workspace** — a JSON-like dictionary that fully specifies the  statistical model.

4. **Fitting** — The workspace is passed to :class:`~nsbi_common_utils.models.sbi_parametric_model.sbi_parametric_model`, which compiles a JAX-based NLL function. This is minimised by :class:`~nsbi_common_utils.inference.inference` to extract the best-fit parameters and profile-likelihood scans. We will soon add the functionality to run Neyman Construction.

.. image:: /_static/DAG_overview.svg
   :alt: NSBI workflow overview
   :align: center
   :width: 100%
