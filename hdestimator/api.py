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
        - TODO

    # Returns
    results : dict with (at least) the following keys:
        R_tot : float, total predictability
        R_tot_sd : float or None, standard deviation of R_tot, if calculated
        AIS_tot : float, active information storage (H*R_tot)
        T_D : float, temporal depth. The T where R_tot occurs
        tau_R : float, information timescale.

        T_vals : array, past range, i.e. T values where embeddings and R were calculated
        R_at_T_vals : array, maximum predictability found at T_vals
        number_of_bins_at_T_vals : array, number of bins that produce the max R at T
        scalings_at_T_vals : array, scaling k that produces the max R at T

        opt_first_bin_size : float
        opt_number_of_bins : float
        opt_scaling : float

        firing_rate : float
        firing_rate_sd : float
        recording_length : float
        recording_length_sd : float
        H_spiking : float

        settings : dict, starting from defaults + user input, what was used ultimately
    """

    # avoid accidentally modifying the provided settings dict, make a copy
    if settings is None:
        settings = dict()
    user_settings = settings.copy()

    # get default settings as dict and overwrite everything with user input
    settings = utl.get_default_settings()
    # overwrite some defaults, with more sensible ones for the python wrapper
    settings["persistent"] = False
    settings["ANALYSIS_DIR"] = None

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
        "timescale_minimum_past_range",
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

    # embeddings that maximise R
    # get predictability for all embeddings, i.e. R as a function of T_D.
    # think of embeddings as tuples of (T_D, number_of_bins_d, scaling_k)
    log.debug("Calculating history dependence for embedding set...")
    embeddings, R_of_T = get_history_dependence_for_embedding_set(
        spike_times=spike_times,
        recording_length=recording_length,
        estimation_method=settings["estimation_method"],
        embedding_past_range_set=settings["embedding_past_range_set"],
        embedding_number_of_bins_set=settings["embedding_number_of_bins_set"],
        embedding_scaling_exponent_set=settings["embedding_scaling_exponent_set"],
        embedding_step_size=settings["embedding_step_size"],
    )
    # results are dicts, mapping T_D -> (num_bins, scaling) and T_D -> R, respectively
    log.debug(f"{embeddings=}")
    log.debug(f"{R_of_T=}")

    # get the maximum history dependence and the temporal depth at which it occurs
    T_for_R_max = utl.get_min_key_for_max_value(R_of_T)
    R_max = R_of_T[T_for_R_max]
    R_max_embedding = (T_for_R_max, ) + embeddings[T_for_R_max] # merging tuples

    # the final R_tot is not R_max. by default, we average over a region around R_max
    # to be robust against fluctations in the tail.
    # get the uncertainty of R_max to adapt the threshold for the final R and T
    log.debug("Bootstrapping to estimate uncertainty of R_max...")
    R_max_replicas = utl.get_bootstrap_history_dependence(
        spike_times=spike_times,
        embedding=R_max_embedding,
        embedding_step_size=settings["embedding_step_size"],
        estimation_method=settings["estimation_method"],
        number_of_bootstraps=settings["number_of_bootstraps_R_max"],
        block_length_l=settings["block_length_l"],
    )

    # reudce the threshold
    R_max_sd = np.std(R_max_replicas)
    R_tot_thresh = R_max - R_max_sd

    # temporal depth T_D is simply how we call the T for R_tot.
    # in most cases, this will just be the peak position.
    log.debug("Finding temporal depth...")
    Ts = sorted([key for key in R_of_T.keys()])
    Rs = [R_of_T[T] for T in Ts]
    T_for_R_tot = min(Ts)
    for R, T in zip(Rs, Ts):
        if R >= R_tot_thresh:
            T_for_R_tot = T
            break

    if not settings["return_averaged_R"]:
        R_tot = R_of_T[T_for_R_tot]
    else:
        # but the esimtate gets more robust if we use an average over a few T
        T_max = T_for_R_tot
        for R, T in zip(Rs, Ts):
            if T < T_for_R_tot:
                continue
            T_max = T
            if R < R_tot_thresh:
                break

        R_tot = np.average([R for R, T in zip(Rs, Ts) if T >= T_for_R_tot and T < T_max])

    # from R_tot we can get the information timescale (similar to autocorrelation time)
    tau_R = utl._get_information_timescale_tau_R(
        max_Rs=R_of_T, R_tot=R_tot, T_0=settings["timescale_minimum_past_range"]
    )

    # create the embedding-tuple for R_tot, to return and for bootstrapping
    R_tot_embedding = (T_for_R_tot, ) + embeddings[T_for_R_tot]

    if settings["number_of_bootstraps_R_tot"] > 1:
        # start by getting the bootstrap replicates.
        # log.debug("Bootstrapping to estimate uncertainty of R_tot...")
        R_tot_replicas = utl.get_bootstrap_history_dependence(
            spike_times=spike_times,
            embedding=R_tot_embedding,
            embedding_step_size=settings["embedding_step_size"],
            estimation_method=settings["estimation_method"],
            number_of_bootstraps=settings["number_of_bootstraps_R_tot"],
            block_length_l=settings["block_length_l"],
        )

        R_tot_sd = np.std(R_tot_replicas)
    else:
        R_tot_sd = None

    # prepare a dictionary to return the results (similar to a row in the statistics.csv)
    # this is only a subset of what the file-based version yields.
    # see also `utl.get_analysis_stats()`
    res = dict()

    res["firing_rate"] = firing_rate
    res["firing_rate_sd"] = firing_rate_sd
    res["recording_length"] = recording_length
    res["recording_length_sd"] = recording_length_sd
    res["H_spiking"] = H_spiking
    # main results
    # the below stats depend on the estimation method, but we only accept one method
    # at a time (e.g. `bbc`), so we do not add another index to the variable label
    res["R_tot"] = R_tot
    res["R_tot_sd"] = R_tot_sd
    res["AIS_tot"] = R_tot * H_spiking
    res["T_D"] = T_for_R_tot
    res["tau_R"] = tau_R
    res["opt_first_bin_size"] = emb.get_first_bin_size_for_embedding(R_tot_embedding)
    res["opt_number_of_bins"] = R_tot_embedding[0]
    res["opt_scaling"] = R_tot_embedding[1]
    # some useful additions, so we can recreate the plots
    res["T_vals"] = np.array(Ts)
    res["R_at_T_vals"] = np.array(Rs)
    res["number_of_bins_at_T_vals"] = np.array([embeddings[T][0] for T in Ts])
    res["scalings_at_T_vals"] = np.array([embeddings[T][1] for T in Ts])
    # keep track of the settings used to produce the results
    res["settings"] = settings.copy()

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

    # log.debug(f"Getting symbol counts... {spikes_are_flat=}")

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

    # Returns
    embeddings_that_maximise_R : dict
    max_Rs : dict
        both dicts have the same keys. either the past_range `T` or
        the number_of_bins `d`, depending on your choice of `dependent_var` (default `T`)
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
