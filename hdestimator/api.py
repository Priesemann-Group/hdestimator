import numpy as np
import logging

log = logging.getLogger("hdestimator")

from . import utils as utl
from . import embedding as emb

# from . import embedding_numba as emb_nb
from . import bbc_estimator as bbc
from . import shuffling_estimator as sh


def wrapper(
    spike_times,
    settings=None,
    seed=42,
):
    """

    # Paramters
    spike_times : array like,
        a (flat) list of spike times (one part),
        or a nested list/array of spike times for multiple parts.
        Each part will be prepared separately, see the `prepare_spike_times()` function
    seed : int or None
        the seed for the random number generator.
    settings : dict
        a dictionary of supported settings. Unspecified keys will be populated from
        our defaults. Below a list of common keys (with their defaults):
        - ...

    # Returns
    results : dict
    """

    # avoid accidentally modifying the provided settings dict, make a copy
    if settings is None:
        settings = dict()
    user_settings = settings.copy()

    # get default settings as dict and overwrite everything with user input
    settings = utl.get_default_settings()
    # overwrite some defaults, with more sensible ones for the python wrapper
    settings["persistent"] = False

    for key in user_settings.keys():
        if key not in settings.keys():
            raise ValueError(f"Unsupported settings key: {key}")
        # we might need to think again about the nested plot dict
        settings[key] = user_settings[key]

    # for now, the wrapper does not support all the bells and whistles.
    if settings["estimation_method"] not in ["bbc", "shuffling"]:
        raise NotImplementedError(
            "Here, we only support one estimation at a time, pick 'bbc' or 'shuffling'."
        )
    if settings["persistent"] is True:
        raise NotImplementedError(
            "Here, we do no support saving of intermediate results yet. "
            + "Set `persistent=False`."
        )

    # to be nice, throw a warning when the user provides a key that we do not use
    wrapper_keys = [
        "persistent",
        "estimation_method",
        "embedding_past_range_set",
        "embedding_step_size",
        "embedding_number_of_bins_set",
        "embedding_scaling_exponent_set",
        "block_length_l",
        "number_of_bootstraps_R_max",
        "number_of_bootstraps_R_tot",
    ]
    ignored_keys = [key for key in user_settings.keys() if key not in wrapper_keys]
    if len(ignored_keys) > 0:
        log.warning(
            f"The following keys of your settings will be ignored: {ignored_keys}"
        )

    log.debug(f"Python wrapper with settings: {settings}")

    # make results reproducible (we rely on bootstrapping)
    np.random.seed(seed)

    # create a copy of the provided spikes that matches our required format:
    # ensure spike_times is an array of arrays
    spike_times = utl.prepare_spike_times(spike_times)

    # pre-calculate some basics we will need:
    # recording length
    rls = [spt[-1] - spt[0] for spt in spike_times]
    recording_length = sum(rls)
    recording_length_sd = np.std(rls)

    # firing rate
    ess = settings["embedding_step_size"]
    frs = [utl.get_binned_firing_rate(spt, ess) for spt in spike_times]
    firing_rate = np.average(frs, weights=rls)
    firing_rate_sd = np.sqrt(np.average((frs - firing_rate) ** 2, weights=rls))

    # spiking entropy
    H_spiking = utl.get_shannon_entropy([firing_rate * ess, 1 - firing_rate * ess])

    # number of symbols in each block. if not specified by user we compute heuristically
    if settings["block_length_l"] is None:
        # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds,
        # for every second the number of symbols is 1/ embedding_step_size
        # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
        # (in the reponse, ignoring the past activity)
        settings["block_length_l"] = max(1, int(1 / (firing_rate * ess)))

    # get predictability for all embeddings, i.e. R as a function of T_D
    log.debug("Calculating history dependence for all embeddings...")
    embeddings_that_maximise_R, max_Rs = get_history_dependence_for_embedding_set(
        spike_times=spike_times,
        recording_length=recording_length,
        estimation_method=settings["estimation_method"],
        embedding_past_range_set=settings["embedding_past_range_set"],
        embedding_number_of_bins_set=settings["embedding_number_of_bins_set"],
        embedding_scaling_exponent_set=settings["embedding_scaling_exponent_set"],
        embedding_step_size=settings["embedding_step_size"],
    )

    # get the maximum history dependence and its embedding-tuple
    max_R, max_R_T = utl.get_max_R_T(max_Rs)
    number_of_bins_d, scaling_k = embeddings_that_maximise_R[max_R_T]
    max_R_embedding = (max_R_T, number_of_bins_d, scaling_k)

    # TODO:rng seed

    # get the uncertainty of max_R to adapt the threshold for T_D
    log.debug("Bootstrapping to estimate uncertainty of max_R...")
    R_max_replicas = utl.get_bootstrap_history_dependence(
        spike_times=spike_times,
        embedding=max_R_embedding,
        embedding_step_size=settings["embedding_step_size"],
        estimation_method=settings["estimation_method"],
        number_of_bootstraps=settings["number_of_bootstraps_R_max"],
        block_length_l=settings["block_length_l"],
    )

    # reudce the threshold
    max_R_sd = np.std(R_max_replicas)
    R_tot_thresh = max_R - max_R_sd

    # find the temporal depth T_D: the past range for the optimal embedding
    # in most cases, this will just be the peak position.
    log.debug("Finding temporal depth...")
    Ts = sorted([key for key in max_Rs.keys()])
    Rs = [max_Rs[T] for T in Ts]
    T_D = min(Ts)
    for R, T in zip(Rs, Ts):
        if R >= R_tot_thresh:
            T_D = T
            break

    # this gives R_tot, i.e. the (R at T_D)
    if not settings["return_averaged_R"]:
        R_tot = max_Rs[T_D]
    else:
        # but the esimtate gets more robust if we use an average over the T after T_D
        T_max = T_D
        for R, T in zip(Rs, Ts):
            if T < T_D:
                continue
            T_max = T
            if R < R_tot_thresh:
                break

        R_tot = np.average([R for R, T in zip(Rs, Ts) if T >= T_D and T < T_max])

    # get the timescale of R_tot
    tau_R = utl._get_information_timescale_tau_R(
        max_Rs=max_Rs, R_tot=R_tot, T_0=settings["timescale_minimum_past_range"]
    )

    # create the embedding-tuple for R_tot
    number_of_bins_d, scaling_k = embeddings_that_maximise_R[T_D]
    R_tot_embedding = (T_D, number_of_bins_d, scaling_k)

    # next, get a confidence interval for R_tot
    # start by getting the bootstrap replicates.
    log.debug("Bootstrapping to estimate uncertainty of R_tot...")
    R_tot_replicas = utl.get_bootstrap_history_dependence(
        spike_times=spike_times,
        embedding=R_tot_embedding,
        embedding_step_size=settings["embedding_step_size"],
        estimation_method=settings["estimation_method"],
        number_of_bootstraps=settings["number_of_bootstraps_R_tot"],
        block_length_l=settings["block_length_l"],
    )

    # prepare a dictionary to return the results (similar to a row in the statistics.csv)
    # this is only a subset of what the file-based version yields.
    # see also `utl.get_analysis_stats()`
    res = dict()

    res["firing_rate"] = firing_rate
    res["firing_rate_sd"] = firing_rate_sd
    res["recording_length"] = recording_length
    res["recording_length_sd"] = recording_length_sd
    res["H_spiking"] = H_spiking
    # the below stats depend on the estimation method, but we only accept one method
    # at a time, so we do not add another label to the variable.
    res["T_D"] = T_D
    res["tau_R"] = tau_R
    res["R_tot"] = R_tot
    res["AIS_tot"] = R_tot * H_spiking
    res["opt_scaling_k"] = scaling_k
    res["opt_number_of_bins_d"] = number_of_bins_d
    res["opt_first_bin_size"] = emb.get_first_bin_size_for_embedding(R_tot_embedding)
    # some useful additions by paul
    res["max_Rs"] = np.array(Rs)
    res["max_R_Ts"] = np.array(Ts)

    return res


def get_history_dependence(
    estimation_method,
    symbol_counts,
    number_of_bins_d,
    past_symbol_counts=None,
    bbc_tolerance=None,
    H_uncond=None,
    return_ais=False,
    **kwargs,
):
    """
    Get history dependence for binary random variable that takes
    into account outcomes with dimension d into the past, and dim 1
    at response, based on symbol counts.

    If no past_symbol_counts are provided, uses representation for
    symbols as given by emb.symbol_array_to_binary to obtain them.
    """

    # if no (unconditional) entropy of the response is provided,
    # assume it is a one-dimensional binary outcome (as in
    # a spike train) and compute it based on that assumption
    if H_uncond == None:
        H_uncond = utl.get_H_spiking(symbol_counts)

    if past_symbol_counts == None:
        past_symbol_counts = utl.get_past_symbol_counts(symbol_counts)

    alphabet_size_past = 2 ** int(number_of_bins_d)  # K for past activity
    alphabet_size = alphabet_size_past * 2  # K

    if estimation_method == "bbc":
        return bbc.bbc_estimator(
            symbol_counts,
            past_symbol_counts,
            alphabet_size,
            alphabet_size_past,
            H_uncond,
            bbc_tolerance=bbc_tolerance,
            return_ais=return_ais,
        )

    elif estimation_method == "shuffling":
        return sh.shuffling_estimator(
            symbol_counts, number_of_bins_d, H_uncond, return_ais=return_ais
        )


## below are functions for estimates on spike trains
def get_history_dependence_for_single_embedding(
    spike_times,
    recording_length,
    estimation_method,
    embedding,
    embedding_step_size,
    bbc_tolerance=None,
    **kwargs,
):
    """
    Apply embedding to spike_times to obtain symbol counts.
    Get history dependence from symbol counts.
    """

    past_range_T, number_of_bins_d, scaling_k = embedding

    # dimensionality check? cf. hde_utils ll 66
    spikes_are_flat = False
    try:
        len(spike_times[0])
    except TypeError:
        spikes_are_flat = True

    log.debug(f"Getting symbol counts... {spikes_are_flat=}")

    if spikes_are_flat:
        symbol_counts = emb.get_symbol_counts(spike_times, embedding, embedding_step_size)
    else:
        symbol_counts = utl.add_up_dicts(
            [
                emb.get_symbol_counts(spt, embedding, embedding_step_size)
                for spt in spike_times
            ]
        )

    if estimation_method == "bbc":
        history_dependence, bbc_term = get_history_dependence(
            estimation_method,
            symbol_counts,
            number_of_bins_d,
            bbc_tolerance=None,
            **kwargs,
        )

        if bbc_tolerance == None:
            return history_dependence, bbc_term

        if bbc_term >= bbc_tolerance:
            return None

    elif estimation_method == "shuffling":
        history_dependence = get_history_dependence(
            estimation_method, symbol_counts, number_of_bins_d, **kwargs
        )

    return history_dependence


def get_history_dependence_for_embedding_set(
    spike_times,
    recording_length,
    estimation_method,
    embedding_past_range_set,
    embedding_number_of_bins_set,
    embedding_scaling_exponent_set,
    embedding_step_size,
    bbc_tolerance=None,
    dependent_var="T",
    **kwargs,
):
    """
    Apply embeddings to spike_times to obtain symbol counts.
    For each T (or d), get history dependence R for the embedding for which
    R is maximised.
    """

    assert dependent_var in ["T", "d"]

    if bbc_tolerance == None:
        bbc_tolerance = np.inf

    max_Rs = {}
    embeddings_that_maximise_R = {}

    for embedding in emb.get_embeddings(
        embedding_past_range_set,
        embedding_number_of_bins_set,
        embedding_scaling_exponent_set,
    ):
        past_range_T, number_of_bins_d, scaling_k = embedding

        history_dependence = get_history_dependence_for_single_embedding(
            spike_times,
            recording_length,
            estimation_method,
            embedding,
            embedding_step_size,
            bbc_tolerance=bbc_tolerance,
            **kwargs,
        )
        if history_dependence == None:
            continue

        if dependent_var == "T":
            if (
                not past_range_T in embeddings_that_maximise_R
                or history_dependence > max_Rs[past_range_T]
            ):
                max_Rs[past_range_T] = history_dependence
                embeddings_that_maximise_R[past_range_T] = (number_of_bins_d, scaling_k)
        elif dependent_var == "d":
            if (
                not number_of_bins_d in embeddings_that_maximise_R
                or history_dependence > max_Rs[number_of_bins_d]
            ):
                max_Rs[number_of_bins_d] = history_dependence
                embeddings_that_maximise_R[number_of_bins_d] = (past_range_T, scaling_k)

    return embeddings_that_maximise_R, max_Rs


def get_CI_for_embedding(
    history_dependence,
    spike_times,
    estimation_method,
    embedding,
    embedding_step_size,
    number_of_bootstraps,
    block_length_l=None,
    bootstrap_CI_use_sd=True,
    bootstrap_CI_percentile_lo=2.5,
    bootstrap_CI_percentile_hi=97.5,
):
    """
    Compute confidence intervals for the history dependence estimate
    based on either the standard deviation or percentiles of
    bootstrap replications of R.
    """

    if block_length_l == None:
        # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds,
        # for every second the number of symbols is 1/ embedding_step_size
        # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
        # (in the reponse, ignoring the past activity)
        firing_rate = utl.get_binned_firing_rate(spike_times, embedding_step_size)
        block_length_l = max(1, int(1 / (firing_rate * embedding_step_size)))

    bs_history_dependence = utl.get_bootstrap_history_dependence(
        [spike_times],
        embedding,
        embedding_step_size,
        estimation_method,
        number_of_bootstraps,
        block_length_l,
    )

    return utl.get_CI_bounds(
        history_dependence,
        bs_history_dependence,
        bootstrap_CI_use_sd=bootstrap_CI_use_sd,
        bootstrap_CI_percentile_lo=bootstrap_CI_percentile_lo,
        bootstrap_CI_percentile_hi=bootstrap_CI_percentile_hi,
    )
