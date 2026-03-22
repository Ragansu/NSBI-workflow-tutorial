Toolkit for Simulation-Based Inference
=======================================

``nsbi-common-utils`` provides building blocks for **Simulation-Based Inference (SBI)** analysis tailored to the statistical models typical at the ATLAS and CMS experiments. It implements semi-parametric approach to SBI where the statistical models are built using a combination of non-parametric and parametric methods targeting different parts. The toolkit has a modular structure, and offers APIs for dataset preparation, density-ratio estimation, model building and profiled-likelihood ratio fitting.

Key features
------------

- **Training pipeline** — density-ratio neural network training with PyTorch Lightning, ensemble support, and calibration and reweighting diagnostics.
- **Statistical models** — Models that can be written in pyhf-like workspace specification with JAX-compiled NLL functions supporting binned template-based & unbinned SBI-style analysis regions both with HistFactory-style systematic uncertainty models.
- **Inference engine** — profile-likelihood fits and NLL scans via iminuit, with plotting utilities.
- **Workflow integration** — Snakemake rules (under development) and HTCondor/DAGMan job descriptions for large-scale cluster submission (CHTC).

Getting started
---------------

To start with we provide documentation for the model building and fitting parts. Provided a workspace object, the statistical model can be built easily using the relevant APIs:

.. code-block:: python

   from nsbi_common_utils import models, inference

   model = models.sbi_parametric_model(workspace=ws, measurement_to_fit="my_meas")
   params, init_vals = model.get_model_parameters()

   fitter = inference.inference(
       model_nll=model.model,
       list_parameters=params,
       initial_parameter_values=init_vals,
   )
   fitter.perform_fit()

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/model
   api/inference
