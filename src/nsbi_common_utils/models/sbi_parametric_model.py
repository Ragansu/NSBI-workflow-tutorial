import numpy as np
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Dict, Union, Any, Optional, Literal
import evermore as evm


class sbi_parametric_model:
    """
    Statistical model for semi-parametric Simulation-Based Inference (SBI).
    Defines parameterized expected yields, density ratios, and the negative log-likelihood (NLL) passed to fitting algorithms. Supports both binned and unbinned channels with systematic uncertainties handled via evermore.


    Parameters
    ----------
    workspace : dict
         A workspace dictionary following the pyhf-like JSON schema. Must contain ``"measurements"`` and ``"channels"`` keys. Channels may be tagged with ``"type": "binned"`` or ``"type": "unbinned"``.
    
    measurement_to_fit : str
        Name of the measurement block inside ``workspace["measurements"]`` to use. Selects the parameter of interest (POI) and the list of parameters to fit.

    Attributes
    ----------
    parameters : dict of str to evm.Parameter
        Ordered mapping of parameter names to evermore :class:`~evermore.Parameter` (unconstrained) or :class:`~evermore.NormalParameter` (Gaussian-constrained) objects.
    list_parameters : list of str
        Ordered parameter names: POI first, then unconstrained norm factors, then constrained nuisance parameters.
    initial_parameter_values : jnp.ndarray
        Starting values for every parameter, in the same order as ``list_parameters``.
    num_unconstrained_param : int
        Number of leading parameters that are unconstrained (POI + free norm factors).
    expected_hist : jnp.ndarray
        Binned expected yields evaluated at the initial parameter values. Used as the Asimov observed data in the NLL.

    See Also
    --------
    nsbi_common_utils.inference.inference : Fits and scans this model.
    """

    def __init__(self,
                 workspace: Dict[Any, Any],
                 measurement_to_fit: str):

        self.workspace                                  = workspace
        self.measurements_dict: list[Dict[str, Any]]    = workspace["measurements"]
        for measurement in self.measurements_dict:
            measurement_name = measurement.get("name")
            if measurement_name == measurement_to_fit:
                self.measurement_name                   = measurement_name
                self.poi                                = measurement["config"]["poi"]
                self.measurement_param_dict             = measurement["config"]["parameters"]
                self.param_names                        = [p['name'] for p in measurement["config"]["parameters"]]
                break
        else:
            raise ValueError(f"Measurement '{measurement_to_fit}' not found in workspace. "
                             f"Available measurements: {[m.get('name') for m in self.measurements_dict]}")

        self.parameters_in_measurement, \
            self.initial_values_dict                    = self._get_parameters_to_fit()

        self.channels_binned                            = self._get_channel_list(type_of_fit="binned")
        self.channels_unbinned                          = self._get_channel_list(type_of_fit="unbinned")
        self.all_channels                               = self.channels_binned + self.channels_unbinned

        self.all_samples                                = self._get_samples_list()

        sorting_order                                   = {"normfactor": 0, "normplusshape": 1}
        self.list_parameters, \
            self.list_parameters_types, \
                self.num_unconstrained_param            = self._get_parameters(sorting_order)

        self.list_syst_normplusshape                    = self._get_list_syst_for_interp()
        self.list_normfactors, \
            self.norm_sample_map                        = self._get_norm_factors()

        self.has_normplusshape                          = len(self.list_syst_normplusshape) > 0

        self.initial_parameter_values                   = self._get_param_vec_initial()
        self.index_normparam_map                        = self._make_map_index_norm()

        # Build evermore parameter registry (Parameter for normfactors, NormalParameter for NPs)
        self.parameters: Dict[str, evm.Parameter]       = self._build_parameter_objects()

        self.yield_array_dict, _                        = self._get_nominal_expected_arrays(type_of_fit="binned")
        self.unbinned_total_dict, \
            self.ratios_array_dict                      = self._get_nominal_expected_arrays(type_of_fit="unbinned")

        self.combined_var_up_binned, \
            self.combined_var_dn_binned                 = self._get_systematic_data_binned()

        self.combined_var_up_unbinned, \
            self.combined_var_dn_unbinned, \
                self.combined_tot_up_unbinned, \
                    self.combined_tot_dn_unbinned       = self._get_systematic_data_unbinned()

        self.weight_arrays_unbinned                     = self._get_asimov_weights_array()

        self._finalize_to_device()
        self.expected_hist                              = self._get_expected_hist(param_vec=self.initial_parameter_values)

    def get_model_parameters(self):
        """
        Return parameter names and initial values for fitting.

        The returned order matches the convention expected by
        :class:`~nsbi_common_utils.inference.inference`: POI at index 0,
        followed by unconstrained norm factors, then constrained nuisance parameters.

        Returns
        -------
        list_parameters : list of str
            Ordered parameter names.
        initial_parameter_values : jnp.ndarray, shape (n_params,)
            Starting values aligned with ``list_parameters``.
        """
        return self.list_parameters, self.initial_parameter_values

    def model(self, param_array: Union[np.ndarray, jnp.ndarray, list]):
        """
        High-level API that returns the full negative log-likelihood for a parameter point.

        Computes the combined NLL in all channels defined by the input workspace — unbinned SBI and binned control and signal regions. Pass this callable to :class:`~nsbi_common_utils.inference.inference` as ``model_nll``.

        Parameters
        ----------
        param_array : array-like, shape (n_params,)
            Parameter values in the order defined by :meth:`get_model_parameters`.

        Returns
        -------
        nll : jnp.ndarray, scalar
            The negative log-likelihood value.
        """
        return self._nll_function(jnp.asarray(param_array))
    

    def _nll_function(self, param_vec: jnp.ndarray) -> jnp.ndarray:
        """Compute the full NLL at ``param_vec``."""
        alpha_vec      = param_vec[self.num_unconstrained_param:]
        norm_modifiers = self._calculate_norm_variations(param_vec)

        # Shape variations via evermore morphing (returns absolute varied histograms)
        hist_vars_binned, hist_vars_unbinned, ratio_vars_unbinned = self._compute_shape_variations(alpha_vec)

        nu_binned  = sum(norm_modifiers[p] * hist_vars_binned[p] for p in self.all_samples)
        llr_binned = -2.0 * evm.pdf.PoissonContinuous(lamb=nu_binned).log_prob(
            self.expected_hist
        ).sum()

        nu_tot_unbinned = sum(norm_modifiers[p] * hist_vars_unbinned[p] for p in self.all_samples)
        dnu_dx          = sum(norm_modifiers[p] * hist_vars_unbinned[p] * ratio_vars_unbinned[p]
                              for p in self.all_samples)
        llr_pe_unbinned = jnp.log(dnu_dx) - jnp.log(nu_tot_unbinned)

        # --- Gaussian constraints on nuisance parameters --------------------
        # Equivalent to -2 * sum(NormalParameter.log_prob) for Normal(0,1) priors
        llr_constraints = jnp.sum(alpha_vec ** 2)

        return (
            llr_binned
            - 2.0 * jnp.sum(self.weight_arrays_unbinned * llr_pe_unbinned)
            + llr_constraints
        )

    def _get_expected_hist(self, param_vec: jnp.ndarray) -> jnp.ndarray:
        """Compute binned expected yields at ``param_vec``. Builds the Asimov dataset."""
        alpha_vec      = param_vec[self.num_unconstrained_param:]
        norm_modifiers = self._calculate_norm_variations(param_vec)
        hist_vars_binned, _, _ = self._compute_shape_variations(alpha_vec)
        return sum(norm_modifiers[p] * hist_vars_binned[p] for p in self.all_samples)

    def _compute_shape_variations(
        self,
        alpha_vec: jnp.ndarray,
    ) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Compute systematic shape variations for all processes via evermore morphing.

        For each process and channel type, builds one :meth:`~evermore.NormalParameter.morphing` modifier per systematic and composes them with ``@``. Variation templates are derived from the stored ratio arrays as ``ratio * nominal``.

        Parameters
        ----------
        alpha_vec : jnp.ndarray, shape (n_syst,)
            Values of the constrained nuisance parameters.

        Returns
        -------
        hist_vars_binned : dict of str to jnp.ndarray
            Absolute varied binned yields per process.
        hist_vars_unbinned : dict of str to jnp.ndarray
            Absolute varied unbinned total yields per process.
        ratio_vars_unbinned : dict of str to jnp.ndarray
            Absolute varied density ratios per process.
        """
        hist_vars_binned     = {}
        hist_vars_unbinned   = {}
        ratio_vars_unbinned  = {}

        for process in self.all_samples:
            if self.has_normplusshape:
                mods_binned  = [
                    evm.NormalParameter(value=alpha_vec[i]).morphing(
                        up_template   = self.combined_var_up_binned[process][i]   * self.yield_array_dict[process],
                        down_template = self.combined_var_dn_binned[process][i]   * self.yield_array_dict[process],
                    )
                    for i in range(len(self.list_syst_normplusshape))
                ]
                mods_unbinned = [
                    evm.NormalParameter(value=alpha_vec[i]).morphing(
                        up_template   = self.combined_tot_up_unbinned[process][i] * self.unbinned_total_dict[process],
                        down_template = self.combined_tot_dn_unbinned[process][i] * self.unbinned_total_dict[process],
                    )
                    for i in range(len(self.list_syst_normplusshape))
                ]
                mods_ratio  = [
                    evm.NormalParameter(value=alpha_vec[i]).morphing(
                        up_template   = self.combined_var_up_unbinned[process][i] * self.ratios_array_dict[process],
                        down_template = self.combined_var_dn_unbinned[process][i] * self.ratios_array_dict[process],
                    )
                    for i in range(len(self.list_syst_normplusshape))
                ]

                hist_vars_binned[process]    = _compose_modifiers(mods_binned)(self.yield_array_dict[process])
                hist_vars_unbinned[process]  = _compose_modifiers(mods_unbinned)(self.unbinned_total_dict[process])
                ratio_vars_unbinned[process] = _compose_modifiers(mods_ratio)(self.ratios_array_dict[process])
            else:
                hist_vars_binned[process]    = self.yield_array_dict[process]
                hist_vars_unbinned[process]  = self.unbinned_total_dict[process]
                ratio_vars_unbinned[process] = self.ratios_array_dict[process]

        return hist_vars_binned, hist_vars_unbinned, ratio_vars_unbinned


    def _calculate_norm_variations(self, param_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        norm_var = {s: jnp.ones(()) for s in self.all_samples}
        for sample, params_sample in self.norm_sample_map.items():
            for param in params_sample:
                norm_var[sample] = norm_var[sample] * param_vec[self.index_normparam_map[param]]
        return norm_var

    def _build_parameter_objects(self) -> Dict[str, evm.Parameter]:
        """
        Build evermore parameter objects for each model parameter.

        Unconstrained parameters (POI, norm factors) become
        :class:`~evermore.Parameter`; constrained nuisance parameters become
        :class:`~evermore.NormalParameter` with a standard Normal prior.

        Returns
        -------
        dict of str to evm.Parameter
            Ordered mapping matching ``list_parameters``.
        """
        params = {}
        for name, ptype in zip(self.list_parameters, self.list_parameters_types):
            init_val = float(self.initial_values_dict[name])
            if ptype == "normfactor":
                params[name] = evm.Parameter(value=init_val)
            else:
                params[name] = evm.NormalParameter(value=init_val)
        return params

    def _make_map_index_norm(self) -> Dict[str, int]:
        """Map each normfactor name to its index in the parameter vector."""
        return {normfactor: self.list_parameters.index(normfactor)
                for normfactor in self.list_normfactors}

    def _get_param_vec_initial(self) -> jnp.ndarray:
        return jnp.asarray([self.initial_values_dict[p] for p in self.list_parameters])

    def _get_norm_factors(self) -> tuple[list, Dict[str, list]]:
        """Discover normfactors from the first channel (assumed shared across channels)."""
        dict_sample_normfactors = {s: [] for s in self.all_samples}
        list_all_norm_factors   = []
        for channel in self.all_channels[:1]:
            ch_idx = self._index_of_region(channel_name=channel)
            for sample in self.all_samples:
                s_idx         = self._index_of_sample(channel_name=channel, sample_name=sample)
                modifier_list = self.workspace["channels"][ch_idx]["samples"][s_idx]["modifiers"]
                for modifier in modifier_list:
                    if modifier["type"] == "normfactor":
                        name = modifier["name"]
                        if name not in list_all_norm_factors:
                            list_all_norm_factors.append(name)
                        if name not in dict_sample_normfactors[sample]:
                            dict_sample_normfactors[sample].append(name)

        list_all_norm_factors   = [p for p in list_all_norm_factors if p in self.param_names]
        dict_sample_normfactors = {k: v for k, v in dict_sample_normfactors.items()
                                   if any(p in self.param_names for p in v)}
        return list_all_norm_factors, dict_sample_normfactors

    def _get_parameters_to_fit(self) -> tuple[list[str], dict[str, float]]:
        """Return the list of parameters to fit and their initial values from the workspace."""
        parameters_to_fit    = []
        initial_value_params = {}
        for p in self.measurement_param_dict:
            parameters_to_fit.append(p["name"])
            initial_value_params[p["name"]] = p["inits"][0]
        return parameters_to_fit, initial_value_params

    def _get_list_syst_for_interp(self) -> list[str]:
        """Return the subset of parameters that require shape interpolation."""
        mask = (np.array(self.list_parameters_types) == "normplusshape")
        return np.array(self.list_parameters)[mask].tolist()

    def _get_channel_list(
        self, type_of_fit: Union[Literal["binned", "unbinned"], None] = None
    ) -> list[str]:
        """Return channel names filtered by type."""
        return [
            ch.get("name")
            for ch in self.workspace["channels"]
            if type_of_fit is None or ch.get("type") == type_of_fit
        ]

    def _get_samples_list(self) -> list[str]:
        """Return sample names from the first channel."""
        for ch in self.workspace["channels"]:
            return [s.get("name") for s in ch["samples"]]
        return []

    def _get_parameters(self, sorting_order: Dict[str, int]) -> tuple[list, list, int]:
        """Discover all parameters from workspace modifiers, sort, and place POI first."""
        list_param_names = []
        list_param_types = []
        for ch in self.workspace["channels"]:
            for sample in ch["samples"]:
                for modifier in sample["modifiers"]:
                    name = modifier.get("name")
                    if name not in self.parameters_in_measurement:
                        continue
                    if name not in list_param_names:
                        list_param_names.append(name)
                        list_param_types.append(modifier.get("type"))

        indices          = np.argsort([sorting_order.get(t, 999) for t in list_param_types])
        list_param_names = [list_param_names[i] for i in indices]
        list_param_types = [list_param_types[i] for i in indices]

        if self.poi not in list_param_names:
            raise ValueError(f"POI '{self.poi}' not found as a modifier in any channel sample. "
                             f"Found parameters: {list_param_names}")

        idx = list_param_names.index(self.poi)
        if idx != 0:
            list_param_names.insert(0, list_param_names.pop(idx))
            list_param_types.insert(0, list_param_types.pop(idx))

        num_unconstrained = sum(1 for t in list_param_types if t == "normfactor")
        return list_param_names, list_param_types, num_unconstrained

    def _get_asimov_weights_array(self) -> np.ndarray:
        """Concatenate Asimov weights across all unbinned channels."""
        weight_array = np.array([])
        for channel in self.channels_unbinned:
            ch_idx       = self._index_of_region(channel)
            weights      = np.load(self.workspace["channels"][ch_idx]["weights"])
            weight_array = np.append(weight_array, weights)
        return weight_array

    def _get_nominal_expected_arrays(
        self, type_of_fit: Literal["binned", "unbinned"]
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load nominal expected yields and (for unbinned) density ratios from the workspace."""
        data_expected  = {s: np.array([]) for s in self.all_samples}
        ratio_expected = {s: np.array([]) for s in self.all_samples}

        channels_list = self.channels_binned if type_of_fit == "binned" else self.channels_unbinned

        for sample_name in self.all_samples:
            for channel_name in channels_list:
                ch_idx = self._index_of_region(channel_name)
                s_idx  = self._index_of_sample(channel_name, sample_name)
                entry  = self.workspace["channels"][ch_idx]["samples"][s_idx]

                data_expected[sample_name] = np.append(
                    data_expected[sample_name], np.array(entry["data"])
                )
                if type_of_fit == "unbinned":
                    ratio_expected[sample_name] = np.append(
                        ratio_expected[sample_name], np.load(entry["ratios"])
                    )
        return data_expected, ratio_expected

    def _get_systematic_data_binned(
        self,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Build ``(N_syst × N_bins)`` variation-ratio arrays for binned channels.

        Returns
        -------
        combined_var_up : dict of str to np.ndarray, shape (n_syst, n_bins)
        combined_var_dn : dict of str to np.ndarray, shape (n_syst, n_bins)
        """
        n_syst = len(self.list_syst_normplusshape)
        n_bins = len(self.yield_array_dict[self.all_samples[0]])

        combined_var_up = {s: np.ones((n_syst, n_bins)) for s in self.all_samples}
        combined_var_dn = {s: np.ones((n_syst, n_bins)) for s in self.all_samples}

        for sample_name in self.all_samples:
            for count, syst_name in enumerate(self.list_syst_normplusshape):
                var_up = np.array([])
                var_dn = np.array([])
                for channel_name in self.channels_binned:
                    ch_idx = self._index_of_region(channel_name)
                    s_idx  = self._index_of_sample(channel_name, sample_name)
                    m_idx  = self._index_of_modifiers(channel_name, sample_name, syst_name)
                    mod    = self.workspace["channels"][ch_idx]["samples"][s_idx]["modifiers"][m_idx]
                    var_up = np.append(var_up, mod["data"]["hi_data"])
                    var_dn = np.append(var_dn, mod["data"]["lo_data"])
                combined_var_up[sample_name][count] = var_up
                combined_var_dn[sample_name][count] = var_dn

        return combined_var_up, combined_var_dn

    def _get_systematic_data_unbinned(
        self,
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """
        Build ``(N_syst × N_events)`` variation-ratio arrays for unbinned channels.

        Returns
        -------
        combined_var_up : dict, shape (n_syst, n_events) — ratio variation up
        combined_var_dn : dict, shape (n_syst, n_events) — ratio variation down
        combined_tot_up : dict, shape (n_syst, n_events) — total-yield variation up
        combined_tot_dn : dict, shape (n_syst, n_events) — total-yield variation down
        """
        n_syst   = len(self.list_syst_normplusshape)
        n_events = len(self.ratios_array_dict[self.all_samples[0]])
        n_tot    = len(self.unbinned_total_dict[self.all_samples[0]])

        combined_var_up = {s: np.ones((n_syst, n_events)) for s in self.all_samples}
        combined_var_dn = {s: np.ones((n_syst, n_events)) for s in self.all_samples}
        combined_tot_up = {s: np.ones((n_syst, n_tot))    for s in self.all_samples}
        combined_tot_dn = {s: np.ones((n_syst, n_tot))    for s in self.all_samples}

        for sample_name in self.all_samples:
            for count, syst_name in enumerate(self.list_syst_normplusshape):
                var_up = np.array([])
                var_dn = np.array([])
                tot_up = np.array([])
                tot_dn = np.array([])
                for channel_name in self.channels_unbinned:
                    ch_idx = self._index_of_region(channel_name)
                    s_idx  = self._index_of_sample(channel_name, sample_name)
                    m_idx  = self._index_of_modifiers(channel_name, sample_name, syst_name)
                    mod    = self.workspace["channels"][ch_idx]["samples"][s_idx]["modifiers"][m_idx]
                    var_up = np.append(var_up, np.load(mod["data"]["hi_ratio"]))
                    var_dn = np.append(var_dn, np.load(mod["data"]["lo_ratio"]))
                    tot_up = np.append(tot_up, mod["data"]["hi_data"])
                    tot_dn = np.append(tot_dn, mod["data"]["lo_data"])
                combined_var_up[sample_name][count] = var_up
                combined_var_dn[sample_name][count] = var_dn
                combined_tot_up[sample_name][count] = tot_up
                combined_tot_dn[sample_name][count] = tot_dn

        return combined_var_up, combined_var_dn, combined_tot_up, combined_tot_dn

    def _finalize_to_device(self):
        """Move all NumPy arrays to JAX device arrays."""
        self.yield_array_dict           = tree_map(jnp.asarray, self.yield_array_dict)
        self.unbinned_total_dict        = tree_map(jnp.asarray, self.unbinned_total_dict)
        self.ratios_array_dict          = tree_map(jnp.asarray, self.ratios_array_dict)

        self.combined_var_up_binned     = tree_map(jnp.asarray, self.combined_var_up_binned)
        self.combined_var_dn_binned     = tree_map(jnp.asarray, self.combined_var_dn_binned)

        self.combined_var_up_unbinned   = tree_map(jnp.asarray, self.combined_var_up_unbinned)
        self.combined_var_dn_unbinned   = tree_map(jnp.asarray, self.combined_var_dn_unbinned)
        self.combined_tot_up_unbinned   = tree_map(jnp.asarray, self.combined_tot_up_unbinned)
        self.combined_tot_dn_unbinned   = tree_map(jnp.asarray, self.combined_tot_dn_unbinned)

        self.weight_arrays_unbinned     = jnp.asarray(self.weight_arrays_unbinned)

    # ------------------------------------------------------------------ #
    # Workspace index helpers                                              #
    # ------------------------------------------------------------------ #

    def _index_of_region(self, channel_name: str) -> Optional[int]:
        """Return the list index of a channel in the workspace."""
        for i, ch in enumerate(self.workspace["channels"]):
            if ch.get("name") == channel_name:
                return i
        return None

    def _index_of_sample(self, channel_name: str, sample_name: str) -> Optional[int]:
        """Return the list index of a sample within a channel."""
        ch_idx = self._index_of_region(channel_name)
        for i, s in enumerate(self.workspace["channels"][ch_idx]["samples"]):
            if s.get("name") == sample_name:
                return i
        return None

    def _index_of_modifiers(self,
                             channel_name: str,
                             sample_name: str,
                             systematic_name: str) -> Optional[int]:
        """Return the list index of a modifier within a sample."""
        ch_idx = self._index_of_region(channel_name)
        s_idx  = self._index_of_sample(channel_name, sample_name)
        for i, m in enumerate(self.workspace["channels"][ch_idx]["samples"][s_idx]["modifiers"]):
            if m.get("name") == systematic_name:
                return i
        return None

def _compose_modifiers(modifiers: list) -> Any:
    """
    Compose a list of evermore modifiers left-to-right using the ``@`` operator.

    Parameters
    ----------
    modifiers : list of evm.Modifier
        Ordered list of modifiers to compose. Must be non-empty.

    Returns
    -------
    composed : evm.Modifier
        Single modifier equivalent to ``modifiers[0] @ modifiers[1] @ ...``.
    """
    composed = modifiers[0]
    for m in modifiers[1:]:
        composed = composed @ m
    return composed