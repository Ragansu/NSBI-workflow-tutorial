import importlib as _importlib

__all__ = [
    "configuration",
    "datasets",
    "training",
    "plotting",
    "inference",
    "workspace_builder",
    "models",
]


def __getattr__(name):
    if name in __all__:
        return _importlib.import_module(f"nsbi_common_utils.{name}")
    raise AttributeError(f"module 'nsbi_common_utils' has no attribute {name!r}")


class Channel:
    def __init__(self, name, channel_type):
        self.name = name
        self.channel_type = channel_type
        self.samples = {}

    def add_sample(self, sample_data, parameters_in_measurement=None):
        sample_name = sample_data["name"]
        if parameters_to_fit is not None:
            modifiers = [
                modifier
                for modifier in sample_data["modifiers"]
                if modifier["name"] in parameters_in_measurement
            ]
            sample_data["modifiers"] = modifiers
        else:
            sample_data["modifiers"] = modifiers
            
        self.samples[sample_name] = Sample(name=sample_name, sample_data=sample_data)


class Sample:
    def __init__(self, name, sample_data):
        self.name = name
        self.modifiers = sample_data["modifiers"]
        self.data = sample_data["data"]
        self.normfactors = set()
        self.vandermonde_factors = {}
        self.shape_factors = set()
        for modifier in modifiers:
            if modifier["type"] == "normfactor":
                self.normfactors.add(modifier["name"])
            elif modifier["type"] == "vandermonde":
                self.vandermonde_factors[modifier["name"]] = modifier["coeff"]
            elif modifier["type"] == "normplusshape":
                self.shape_factors.add(modifier["name"])


class Measurement:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.parameters = {}
