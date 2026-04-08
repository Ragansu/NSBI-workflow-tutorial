import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "nsbi-common-utils"
copyright = "2025, Jay Sandesara (IRIS-HEP, NSF OAC-1836650 / PHY-2323298)"
author = "Jay Sandesara"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
add_module_names = False
autodoc_mock_imports = [
    "numpy", "uproot", "ROOT", "torch", "pytorch_lightning", "onnx", "onnxruntime",
    "matplotlib", "mplhep", "hist", "coffea",
    "iminuit", "evermore", "flax",
    "jax", "jaxlib", "optax", "equinox",
    "yaml", "ruamel", "awkward", "pandas",
    "sklearn", "scipy", "h5py", "tqdm", "joblib",
]
autodoc_class_signature = "mixed"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "iminuit": ("https://scikit-hep.org/iminuit", None),
}

html_theme = "sphinx_rtd_theme"

exclude_patterns = ["_build"]
