Parameter Fitting and Hypothesis Testing
==========================================

Statistical inference engine wrapping iminuit for profiled negative log-likelihood minimisation and parameter scans.

When an analytical gradient function (``model_grad``) is supplied, iminuit uses it instead of finite-difference approximations.

.. currentmodule:: nsbi_common_utils.inference

.. autoclass:: inference
   :members:
   :undoc-members: False
   :show-inheritance:

.. autofunction:: plot_NLL_scans
