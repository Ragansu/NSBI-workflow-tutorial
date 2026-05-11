Statistical Models
==================

The statistical models available for NSBI. Defines the JIT-compiled negative log-likelihood (ratio) function written using JAX, which can be used by fitting algorithms.

The two main entry points for downstream code are:

* :meth:`~sbi_parametric_model.model` — the NLL callable (pass to ``inference`` as ``model_nll``).
* :meth:`~sbi_parametric_model.model_grad` — the NLL gradient callable (pass to ``inference`` as ``model_grad``).

.. currentmodule:: nsbi_common_utils.models.sbi_parametric_model

.. autoclass:: sbi_parametric_model
   :members:
   :undoc-members: False
   :show-inheritance:
