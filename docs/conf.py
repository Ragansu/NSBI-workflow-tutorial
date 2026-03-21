import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "nsbi-common-utils"
copyright = "2024, FAIR Universe Collaboration"
author = "FAIR Universe Collaboration"

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
    "special-members": "__init__",
}
autodoc_typehints = "description"
add_module_names = False
autodoc_mock_imports = [
    "uproot", "ROOT", "torch", "pytorch_lightning", "onnx", "onnxruntime",
    "matplotlib", "mplhep", "hist", "coffea",
    "iminuit", "evermore", "flax",
    "jax", "jaxlib", "optax", "equinox",
    "yaml", "ruamel", "awkward", "pandas",
    "sklearn", "scipy", "h5py", "tqdm",
]
autodoc_class_signature = "mixed"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "iminuit": ("https://scikit-hep.org/iminuit", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/jsandesa/nsbi-common-utils",
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

exclude_patterns = ["_build"]
