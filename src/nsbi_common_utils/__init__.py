import importlib as _importlib

__all__ = [
    'configuration',
    'datasets',
    'training',
    'plotting',
    'inference',
    'workspace_builder',
    'models',
]

def __getattr__(name):
    if name in __all__:
        return _importlib.import_module(f"nsbi_common_utils.{name}")
    raise AttributeError(f"module 'nsbi_common_utils' has no attribute {name!r}")
