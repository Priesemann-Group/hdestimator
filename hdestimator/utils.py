# ------------------------------------------------------------------------------ #
# @Created:       2023-06-21 17:15:21
# @Last Modified: 2023-07-31 15:27:02
# ------------------------------------------------------------------------------ #
# One central idea is that analysis need precomputed data.
# Often we want to store this persistently, in a hdf5 file. This file is then
# used by many functions. To avoid disk io, it might also just be a dictionary
# in memory (as the h5 api is similar).
#
# This file or dict is the first argument `f` to some functions.
# ------------------------------------------------------------------------------ #

import logging
log = logging.getLogger("hdestimator")

import numpy as np
import hashlib

from collections import Counter

from . import embedding as emb

# a few functions still import get_history_dependence, but now locally
# from . import api as hapi

# these functions used to live in this file, I want to avoid breaking things
# while updating the dependencies.
from .input_output import *

# ------------------------------------------------------------------------------ #
# General
# ------------------------------------------------------------------------------ #

def get_default_settings():
    """
    Returns our default settings as a dictionary.

    # Note:
    Hardcoded here, do not change the yaml file for default settings on disk, manually!
    """
    settings = {'embedding_step_size' : 0.005,
            'embedding_past_range_set' : [float("{:.5f}".format(np.exp(x))) for x in np.arange(np.log(0.005), np.log(5.001), 0.05 * np.log(10))],
            'embedding_number_of_bins_set' : [int(x) for x in np.linspace(1,5,5)],
            'embedding_scaling_exponent_set' : {'number_of_scalings': 10,
                                                'min_first_bin_size' : 0.005,
                                                'min_step_for_scaling': 0.01},
            'estimation_method' : "shuffling",
            'bbc_tolerance' : 0.05,
            'cross_validated_optimization' : False,
            'return_averaged_R' : True,
            'timescale_minimum_past_range' : 0.01,
            'number_of_bootstraps_R_max' : 250,
            'number_of_bootstraps_R_tot' : 250,
            'number_of_bootstraps_nonessential' : 0,
            'block_length_l' : None,
            'bootstrap_CI_use_sd' : True,
            'bootstrap_CI_percentile_lo' : 2.5,
            'bootstrap_CI_percentile_hi' : 97.5,
            # 'number_of_permutations' : 100,
            'auto_MI_bin_size_set' : [0.005, 0.01, 0.025, 0.05, 0.25, 0.5],
            'auto_MI_max_delay' : 5,
            'label' : '',
            'ANALYSIS_DIR' : "./analysis",
            'persistent_analysis' : True,
            # 'verbose_output' : False,
            'plot_AIS' : False,
            'plot_settings' : {'figure.figsize' : [6.3, 5.5],
                                'axes.labelsize': 9,
                                'font.size': 9,
                                'legend.fontsize': 8,
                                'xtick.labelsize': 8,
                                'ytick.labelsize': 8,
                                'savefig.format': 'pdf'},
            'plot_color' : "#4da2e2"}

    return settings

def prepare_spike_times(spikes):
    """
    Prepares the provided spiketimes to match our needed format:
    - returns a copy (does not modify in place)
    - sorts spike times chronologically
    - rescales spike time to start at 0

    # Parameters:
    spikes : array like
        - a flat list/array is interpreted as a one part
        - a nested list (or list of arrays) is interpreted as multiple parts, each with separate spike times. Each part will be sorted and 0-aligned independent of the other parts.

    # Returns
    spikes : array like
        - if a flat list was provided, returns a 2dim array of shape (1, num_spikes)
        - if a nested list was provided, returns an array of arrays, with outer shape (num_parts) and respective inner shape (num_spikes)
    """

    input_spikes = spikes

    # start by checking if the provided data is nested
    is_flat = False
    try:
        len(input_spikes[0])
    except:
        is_flat = True

    log.debug(f"preparing spike times of type {type(input_spikes)}, {is_flat=}")

    if is_flat:
        first_spike = np.nanmin(input_spikes)
        # sort returns a copy, so we are good. float64 to comply with double type expected by cython
        res = np.array([np.sort(input_spikes - first_spike)], dtype=np.float64)
        is_ragged = False

    else:
        output_trains = []
        # we have to iterate and process each part separately
        for train in input_spikes:
            first_spike = np.nanmin(train)
            output_trains.append(np.sort(train - first_spike).astype(np.float64))

        # this list likely contains lists of varying size.
        # in this case, create the array of dtype object to avoid the deprication warning
        if len(output_trains) > 1:
            res = np.array(output_trains, dtype=object)
            is_ragged = True
        else:
            res = np.array(output_trains, dtype = np.float64)
            is_ragged = False

    log.debug(f"prepared spike train: length={len(res)} dtype={res.dtype}")
    log.debug(f"length varies" if is_ragged else f"with shape {res.shape}")
    return res

# ------------------------------------------------------------------------------ #
# Functions that need precomputed details via hdf5 or dict, `f`
# ------------------------------------------------------------------------------ #

def get_embeddings_that_maximise_R(f,
                                   estimation_method,
                                   embedding_step_size,
                                   bbc_tolerance=None,
                                   dependent_var="T",
                                   get_as_list=False,
                                   cross_val=None,
                                   **kwargs):
    """
    For each T (or d), get the embedding for which R is maximised.

    For the bbc estimator, here the bbc_tolerance is applied, ie
    get the unbiased embeddings that maximise R.
    """

    assert dependent_var in ["T", "d"]
    assert cross_val in [None, "h1", "h2"]

    if bbc_tolerance == None \
       or cross_val == "h2": # apply bbc only for optimization
        bbc_tolerance = np.inf

    if cross_val == None:
        root_dir = 'embeddings'
    else:
        root_dir = '{}_embeddings'.format(cross_val)

    max_Rs = {}
    embeddings_that_maximise_R = {}

    embedding_step_size_label = get_parameter_label(embedding_step_size)

    for past_range_T_label in f["{}/{}".format(root_dir, embedding_step_size_label)].keys():
        for number_of_bins_d_label in f["{}/{}/{}".format(root_dir,
                                                          embedding_step_size_label,
                                                          past_range_T_label)].keys():
            for scaling_k_label in f["{}/{}/{}/{}".format(root_dir,
                                                          embedding_step_size_label,
                                                          past_range_T_label,
                                                          number_of_bins_d_label)].keys():
                past_range_T = float(past_range_T_label)
                number_of_bins_d = int(float(number_of_bins_d_label))
                scaling_k = float(scaling_k_label)
                embedding = (past_range_T,
                             number_of_bins_d,
                             scaling_k)
                history_dependence = load_from_analysis_file(f,
                                                             "history_dependence",
                                                             embedding_step_size=embedding_step_size,
                                                             embedding=embedding,
                                                             estimation_method=estimation_method,
                                                             cross_val=cross_val)
                # if it has been estimated for one estimator, but not the other
                # it might be None. skip if this is the case
                if history_dependence == None:
                    continue

                if estimation_method == "bbc":
                    bbc_term = load_from_analysis_file(f,
                                                       "bbc_term",
                                                       embedding_step_size=embedding_step_size,
                                                       embedding=embedding,
                                                       estimation_method=estimation_method,
                                                       cross_val=cross_val)
                    if bbc_term >= bbc_tolerance:
                        continue

                if dependent_var == "T":
                    if not past_range_T in embeddings_that_maximise_R \
                       or history_dependence > max_Rs[past_range_T]:
                        max_Rs[past_range_T] = history_dependence
                        embeddings_that_maximise_R[past_range_T] = (number_of_bins_d,
                                                                    scaling_k)
                elif dependent_var == "d":
                    if not number_of_bins_d in embeddings_that_maximise_R \
                       or history_dependence > max_Rs[number_of_bins_d]:
                        max_Rs[number_of_bins_d] = history_dependence
                        embeddings_that_maximise_R[number_of_bins_d] = (past_range_T,
                                                                        scaling_k)

    if get_as_list:
        embeddings = []
        if dependent_var == "T":
            for past_range_T in embeddings_that_maximise_R:
                number_of_bins_d, scaling_k = embeddings_that_maximise_R[past_range_T]
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]
        elif dependent_var == "d":
            for number_of_bins_d in embeddings_that_maximise_R:
                past_range_T, scaling_k = embeddings_that_maximise_R[number_of_bins_d]
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]
        return embeddings
    else:
        return embeddings_that_maximise_R, max_Rs

def get_temporal_depth_T_D(f,
                           estimation_method,
                           bootstrap_CI_use_sd=True,
                           bootstrap_CI_percentile_lo=2.5,
                           bootstrap_CI_percentile_hi=97.5,
                           get_R_thresh=False,
                           **kwargs):
    """
    Get the temporal depth T_D, the past range for the
    'optimal' embedding parameters.

    Given the maximal history dependence R at each past range T,
    (cf get_embeddings_that_maximise_R), first find the smallest T at
    which R is maximised (cf get_max_R_T).  If bootstrap replications
    for this R are available, get the smallest T at which this R minus
    one standard deviation of the bootstrap estimates is attained.
    """

    # load data
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         estimation_method=estimation_method,
                                         **kwargs)

    Ts = sorted([key for key in max_Rs.keys()])
    Rs = [max_Rs[T] for T in Ts]

    # first get the max history dependence, and if available its bootstrap replications
    max_R, max_R_T = get_max_R_T(max_Rs)

    number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]
    bs_Rs = load_from_analysis_file(f,
                                    "bs_history_dependence",
                                    embedding_step_size=kwargs["embedding_step_size"],
                                    embedding=(max_R_T,
                                               number_of_bins_d,
                                               scaling_k),
                                    estimation_method=estimation_method,
                                    cross_val=kwargs['cross_val'])

    if isinstance(bs_Rs, np.ndarray):
        max_R_sd = np.std(bs_Rs)
    else:
        max_R_sd = 0

    R_tot_thresh = max_R - max_R_sd

    T_D = min(Ts)
    for R, T in zip(Rs, Ts):
        if R >= R_tot_thresh:
            T_D = T
            break

    if not get_R_thresh:
        return T_D
    else:
        return T_D, R_tot_thresh

def get_information_timescale_tau_R(f,
                                    estimation_method,
                                    **kwargs):
    """
    Get the information timescale tau_R, a characteristic
    timescale of history dependence similar to an autocorrelation
    time.
    """

    # load data
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         estimation_method=estimation_method,
                                         **kwargs)
    R_tot = get_R_tot(f,
                      estimation_method=estimation_method,
                      **kwargs)

    return _get_information_timescale_tau_R(max_Rs,
                                            R_tot,
                                            kwargs["timescale_minimum_past_range"],
                                            **kwargs)

def _get_information_timescale_tau_R(max_Rs,
                                     R_tot,
                                     T_0,
                                     **kwargs):

    Ts = np.array(sorted([key for key in max_Rs.keys()]))
    Rs = np.array([max_Rs[T] for T in Ts])

    # get dRs
    dRs = []
    R_prev = 0.

    # No values higher than R_tot are allowed,
    # otherwise the information timescale might be
    # misestimated because of spurious contributions
    # at large T
    for R, T in zip(Rs[Rs <= R_tot], Ts[Rs <= R_tot]):

        # No negative increments are allowed
        dRs += [np.amax([0.0, R - R_prev])]

        # The increment is taken with respect to the highest previous value of R
        if R > R_prev:
            R_prev = R

    dRs = np.pad(dRs, (0, len(Rs) - len(dRs)),
                 mode='constant', constant_values=0)


    # compute tau_R
    Ts_0 = np.append([0], Ts)
    dRs_0 = dRs[Ts_0[:-1] >= T_0]

    # Only take into considerations contributions beyond T_0
    Ts_0 = Ts_0[Ts_0 >= T_0]
    norm = np.sum(dRs_0)

    if norm == 0.:
        tau = 0.0
    else:
        Ts_0 -= Ts_0[0]
        tau = np.dot(((Ts_0[:-1] + Ts_0[1:]) / 2), dRs_0) / norm
    return tau

def get_R_tot(f,
              estimation_method,
              return_averaged_R=False,
              **kwargs):
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         estimation_method=estimation_method,
                                         **kwargs)

    if return_averaged_R:
        T_D, R_tot_thresh = get_temporal_depth_T_D(f,
                                                   estimation_method=estimation_method,
                                                   get_R_thresh=True,
                                                   **kwargs)

        Ts = sorted([key for key in max_Rs.keys()])
        Rs = [max_Rs[T] for T in Ts]

        T_max = T_D
        for R, T in zip(Rs, Ts):
            if T < T_D:
                continue
            T_max = T
            if R < R_tot_thresh:
                break

        return np.average([R for R, T in zip(Rs, Ts) if T >= T_D and T < T_max])

    else:
        temporal_depth_T_D = get_temporal_depth_T_D(f,
                                                    estimation_method=estimation_method,
                                                    **kwargs)

        return max_Rs[temporal_depth_T_D]

def compute_CIs(f,
                spike_times,
                estimation_method,
                embedding_step_size,
                block_length_l=None,
                target_R='R_max',
                **kwargs):
    """
    Compute bootstrap replications of the history dependence estimate
    which can be used to obtain confidence intervals.

    Load symbol counts, resample, then estimate entropy for each sample
    and save to file.

    :param target_R: One of 'R_max', 'R_tot' or 'nonessential'.
    If set to R_max, replications of R are produced for the T at which
    R is maximised.
    If set to R_tot, replications of R are produced for T = T_D (cf
    get_temporal_depth_T_D).
    If set to nonessential, replications of R are produced for each T
    (one embedding per T, cf get_embeddings_that_maximise_R).  These
    are not otherwise used in the analysis and are probably only useful
    if the resulting plot is visually inspected, so in most cases it can
    be set to zero.
    """

    assert target_R in ['nonessential', 'R_max', 'R_tot']

    number_of_bootstraps = kwargs['number_of_bootstraps_{}'.format(target_R)]

    if number_of_bootstraps == 0:
        return

    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         embedding_step_size=embedding_step_size,
                                         estimation_method=estimation_method,
                                         **kwargs)

    recording_length = load_from_analysis_file(f,
                                               "recording_length")

    firing_rate = load_from_analysis_file(f,
                                          "firing_rate")

    if block_length_l == None:
        # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds,
        # for every second the number of symbols is 1/ embedding_step_size
        # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
        # (in the reponse, ignoring the past activity)
        block_length_l = max(1, int(1 / (firing_rate * embedding_step_size)))

    if target_R == 'nonessential':
        # bootstrap R for unessential Ts (not required for the main analysis)
        embeddings = []

        for past_range_T in embedding_maximising_R_at_T:
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
            embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

    elif target_R == 'R_max':
        # bootstrap R for the max R, to get a good estimate for the standard deviation
        # which is used to determine R_tot
        max_R, max_R_T = get_max_R_T(max_Rs)
        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]

        embeddings = [(max_R_T, number_of_bins_d, scaling_k)]
    elif target_R == 'R_tot':
        T_D = get_temporal_depth_T_D(f,
                                     estimation_method,
                                     embedding_step_size=embedding_step_size,
                                     **kwargs)
        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[T_D]

        embeddings = [(T_D, number_of_bins_d, scaling_k)]

    for embedding in embeddings:
        stored_bs_Rs = load_from_analysis_file(f,
                                               "bs_history_dependence",
                                               embedding_step_size=embedding_step_size,
                                               embedding=embedding,
                                               estimation_method=estimation_method,
                                               cross_val=kwargs['cross_val'])
        if isinstance(stored_bs_Rs, np.ndarray):
            number_of_stored_bootstraps = len(stored_bs_Rs)
        else:
            number_of_stored_bootstraps = 0

        if not number_of_bootstraps > number_of_stored_bootstraps:
            continue

        bs_history_dependence \
            = get_bootstrap_history_dependence(spike_times,
                                               embedding,
                                               embedding_step_size,
                                               estimation_method,
                                               number_of_bootstraps - number_of_stored_bootstraps,
                                               block_length_l)

        save_to_analysis_file(f,
                              "bs_history_dependence",
                              embedding_step_size=embedding_step_size,
                              embedding=embedding,
                              estimation_method=estimation_method,
                              bs_history_dependence=bs_history_dependence,
                              cross_val=kwargs['cross_val'])

def compute_CIs_from_loaded(f,
                spike_times,
                estimation_method,
                embedding_step_size,
                block_length_l=None,
                target_R='R_max',
                **kwargs):
    """
    Compute bootstrap replications of the history dependence estimate
    which can be used to obtain confidence intervals.

    Load symbol counts, resample, then estimate entropy for each sample
    and save to file.

    :param target_R: One of 'R_max', 'R_tot' or 'nonessential'.
    If set to R_max, replications of R are produced for the T at which
    R is maximised.
    If set to R_tot, replications of R are produced for T = T_D (cf
    get_temporal_depth_T_D).
    If set to nonessential, replications of R are produced for each T
    (one embedding per T, cf get_embeddings_that_maximise_R).  These
    are not otherwise used in the analysis and are probably only useful
    if the resulting plot is visually inspected, so in most cases it can
    be set to zero.
    """

    assert target_R in ['nonessential', 'R_max', 'R_tot']

    number_of_bootstraps = kwargs['number_of_bootstraps_{}'.format(target_R)]

    if number_of_bootstraps == 0:
        return

    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         embedding_step_size=embedding_step_size,
                                         estimation_method=estimation_method,
                                         **kwargs)


    if block_length_l == None:
        # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds,
        # for every second the number of symbols is 1/ embedding_step_size
        # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
        # (in the reponse, ignoring the past activity)
        firing_rate = load_from_analysis_file(f, "firing_rate")
        block_length_l = max(1, int(1 / (firing_rate * embedding_step_size)))

    if target_R == 'nonessential':
        # bootstrap R for unessential Ts (not required for the main analysis)
        embeddings = []

        for past_range_T in embedding_maximising_R_at_T:
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
            embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

    elif target_R == 'R_max':
        # bootstrap R for the max R, to get a good estimate for the standard deviation
        # which is used to determine R_tot
        max_R, max_R_T = get_max_R_T(max_Rs)
        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]

        embeddings = [(max_R_T, number_of_bins_d, scaling_k)]
    elif target_R == 'R_tot':
        T_D = get_temporal_depth_T_D(f,
                                     estimation_method,
                                     embedding_step_size=embedding_step_size,
                                     **kwargs)
        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[T_D]

        embeddings = [(T_D, number_of_bins_d, scaling_k)]

    for embedding in embeddings:
        stored_bs_Rs = load_from_analysis_file(f,
                                               "bs_history_dependence",
                                               embedding_step_size=embedding_step_size,
                                               embedding=embedding,
                                               estimation_method=estimation_method,
                                               cross_val=kwargs['cross_val'])
        if isinstance(stored_bs_Rs, np.ndarray):
            number_of_stored_bootstraps = len(stored_bs_Rs)
        else:
            number_of_stored_bootstraps = 0

        if not number_of_bootstraps > number_of_stored_bootstraps:
            continue

        bs_history_dependence \
            = get_bootstrap_history_dependence(spike_times,
                                               embedding,
                                               embedding_step_size,
                                               estimation_method,
                                               number_of_bootstraps - number_of_stored_bootstraps,
                                               block_length_l)

        save_to_analysis_file(f,
                              "bs_history_dependence",
                              embedding_step_size=embedding_step_size,
                              embedding=embedding,
                              estimation_method=estimation_method,
                              bs_history_dependence=bs_history_dependence,
                              cross_val=kwargs['cross_val'])

def get_analysis_stats(f,
                       analysis_num,
                       estimation_method=None,
                       **kwargs):
    """
    Get statistics of the analysis, to export them to a csv file.
    """

    stats = {
        "analysis_num" : str(analysis_num),
        "label" : kwargs["label"],
        "tau_R_bbc" : "-",
        "T_D_bbc" : "-",
        "R_tot_bbc" : "-",
        "R_tot_bbc_CI_lo" : "-",
        "R_tot_bbc_CI_hi" : "-",
        # "R_tot_bbc_RMSE" : "-",
        # "R_tot_bbc_bias" : "-",
        "AIS_tot_bbc" : "-",
        "AIS_tot_bbc_CI_lo" : "-",
        "AIS_tot_bbc_CI_hi" : "-",
        # "AIS_tot_bbc_RMSE" : "-",
        # "AIS_tot_bbc_bias" : "-",
        "opt_number_of_bins_d_bbc" : "-",
        "opt_scaling_k_bbc" : "-",
        "opt_first_bin_size_bbc" : "-",
        # "asl_permutation_test_bbc" : "-",
        "tau_R_shuffling" : "-",
        "T_D_shuffling" : "-",
        "R_tot_shuffling" : "-",
        "R_tot_shuffling_CI_lo" : "-",
        "R_tot_shuffling_CI_hi" : "-",
        # "R_tot_shuffling_RMSE" : "-",
        # "R_tot_shuffling_bias" : "-",
        "AIS_tot_shuffling" : "-",
        "AIS_tot_shuffling_CI_lo" : "-",
        "AIS_tot_shuffling_CI_hi" : "-",
        # "AIS_tot_shuffling_RMSE" : "-",
        # "AIS_tot_shuffling_bias" : "-",
        "opt_number_of_bins_d_shuffling" : "-",
        "opt_scaling_k_shuffling" : "-",
        "opt_first_bin_size_shuffling" : "-",
        # "asl_permutation_test_shuffling" : "-",
        "embedding_step_size" : get_parameter_label(kwargs["embedding_step_size"]),
        "bbc_tolerance" : get_parameter_label(kwargs["bbc_tolerance"]),
        "timescale_minimum_past_range" : get_parameter_label(kwargs["timescale_minimum_past_range"]),
        "number_of_bootstraps_bbc" : "-",
        "number_of_bootstraps_shuffling" : "-",
        "bs_CI_percentile_lo" : "-",
        "bs_CI_percentile_hi" : "-",
        # "number_of_permutations_bbc" : "-",
        # "number_of_permutations_shuffling" : "-",
        "firing_rate" : get_parameter_label(load_from_analysis_file(f, "firing_rate")),
        "firing_rate_sd" : get_parameter_label(load_from_analysis_file(f, "firing_rate_sd")),
        "recording_length" : get_parameter_label(load_from_analysis_file(f, "recording_length")),
        "recording_length_sd" : get_parameter_label(load_from_analysis_file(f, "recording_length_sd")),
        "H_spiking" : "-",
    }

    if stats["label"] == "":
        stats["label"] = "-"

    H_spiking = load_from_analysis_file(f, "H_spiking")
    stats["H_spiking"] = get_parameter_label(H_spiking)

    for estimation_method in ["bbc", "shuffling"]:

        embedding_maximising_R_at_T, max_Rs \
            = get_embeddings_that_maximise_R(f,
                                             estimation_method=estimation_method,
                                             **kwargs)

        if len(embedding_maximising_R_at_T) == 0:
            continue

        tau_R = get_information_timescale_tau_R(f,
                                                estimation_method=estimation_method,
                                                **kwargs)

        temporal_depth_T_D = get_temporal_depth_T_D(f,
                                                    estimation_method=estimation_method,
                                                    **kwargs)

        R_tot = get_R_tot(f,
                          estimation_method=estimation_method,
                          **kwargs)
        opt_number_of_bins_d, opt_scaling_k \
            = embedding_maximising_R_at_T[temporal_depth_T_D]

        stats["tau_R_{}".format(estimation_method)] = get_parameter_label(tau_R)
        stats["T_D_{}".format(estimation_method)] = get_parameter_label(temporal_depth_T_D)
        stats["R_tot_{}".format(estimation_method)] = get_parameter_label(R_tot)
        stats["AIS_tot_{}".format(estimation_method)] = get_parameter_label(R_tot * H_spiking)
        stats["opt_number_of_bins_d_{}".format(estimation_method)] \
            = get_parameter_label(opt_number_of_bins_d)
        stats["opt_scaling_k_{}".format(estimation_method)] \
            = get_parameter_label(opt_scaling_k)

        stats["opt_first_bin_size_{}".format(estimation_method)] \
            = get_parameter_label(load_from_analysis_file(f,
                                                          "first_bin_size",
                                                          embedding_step_size\
                                                          =kwargs["embedding_step_size"],
                                                          embedding=(temporal_depth_T_D,
                                                                     opt_number_of_bins_d,
                                                                     opt_scaling_k),
                                                          estimation_method=estimation_method,
                                                          cross_val=kwargs['cross_val']))

        if not kwargs['return_averaged_R']:
            bs_Rs = load_from_analysis_file(f,
                                            "bs_history_dependence",
                                            embedding_step_size=kwargs["embedding_step_size"],
                                            embedding=(temporal_depth_T_D,
                                                       opt_number_of_bins_d,
                                                       opt_scaling_k),
                                            estimation_method=estimation_method,
                                            cross_val=kwargs['cross_val'])
            if isinstance(bs_Rs, np.ndarray):
                stats["number_of_bootstraps_{}".format(estimation_method)] \
                    = str(len(bs_Rs))

        if not stats["number_of_bootstraps_{}".format(estimation_method)] == "-":
            R_tot_CI_lo, R_tot_CI_hi = get_CI_bounds(R_tot,
                                                     bs_Rs,
                                                     kwargs["bootstrap_CI_use_sd"],
                                                     kwargs["bootstrap_CI_percentile_lo"],
                                                     kwargs["bootstrap_CI_percentile_hi"])
            stats["R_tot_{}_CI_lo".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_lo)
            stats["R_tot_{}_CI_hi".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_hi)
            stats["AIS_tot_{}_CI_lo".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_lo * H_spiking)
            stats["AIS_tot_{}_CI_hi".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_hi * H_spiking)

            # bias = estimate_bootstrap_bias(f,
            #                                estimation_method=estimation_method,
            #                                **kwargs)

            # variance = np.var(bs_Rs)

            # stats["R_tot_{}_RMSE".format(estimation_method)] \
            #     = get_parameter_label(np.sqrt(variance + bias**2))

            # stats["R_tot_{}_bias".format(estimation_method)] \
            #     = get_parameter_label(bias)

            # TODO RMSE, bias for AIS

        if kwargs["bootstrap_CI_use_sd"]:
            stats["bs_CI_percentile_lo"] = get_parameter_label(2.5)
            stats["bs_CI_percentile_hi"] = get_parameter_label(97.5)
        else:
            stats["bs_CI_percentile_lo"] = get_parameter_label(kwargs["bootstrap_CI_percentile_lo"])
            stats["bs_CI_percentile_hi"] = get_parameter_label(kwargs["bootstrap_CI_percentile_hi"])

        # pt_Rs = load_from_analysis_file(f,
        #                                 "pt_history_dependence",
        #                                 embedding_step_size=kwargs["embedding_step_size"],
        #                                 embedding=(temporal_depth_T_D,
        #                                            opt_number_of_bins_d,
        #                                            opt_scaling_k),
        #                                 estimation_method=estimation_method,
        #                                 cross_val=kwargs['cross_val'])

        # if isinstance(pt_Rs, np.ndarray):
        #     stats["asl_permutation_test_{}".format(estimation_method)] \
        #         = get_parameter_label(get_asl_permutation_test(pt_Rs, R_tot))

        #     stats["number_of_permutations_{}".format(estimation_method)] \
        #         = str(len(pt_Rs))

    return stats

def get_histdep_data(f,
                     analysis_num,
                     estimation_method,
                     **kwargs):
    """
    Get R values for each T, as needed for the plots, to export them
    to a csv file.
    """

    histdep_data = {
        "T" : [],
        "max_R_bbc" : [],
        "max_R_bbc_CI_lo" : [],
        "max_R_bbc_CI_hi" : [],
        # "max_R_bbc_CI_med" : [],
        "max_AIS_bbc" : [],
        "max_AIS_bbc_CI_lo" : [],
        "max_AIS_bbc_CI_hi" : [],
        # "max_AIS_bbc_CI_med" : [],
        "number_of_bins_d_bbc" : [],
        "scaling_k_bbc" : [],
        "first_bin_size_bbc" : [],
        "max_R_shuffling" : [],
        "max_R_shuffling_CI_lo" : [],
        "max_R_shuffling_CI_hi" : [],
        # "max_R_shuffling_CI_med" : [],
        "max_AIS_shuffling" : [],
        "max_AIS_shuffling_CI_lo" : [],
        "max_AIS_shuffling_CI_hi" : [],
        # "max_AIS_shuffling_CI_med" : [],
        "number_of_bins_d_shuffling" : [],
        "scaling_k_shuffling" : [],
        "first_bin_size_shuffling" : [],
    }

    for estimation_method in ['bbc', 'shuffling']:
        # kwargs["estimation_method"] = estimation_method

        embedding_maximising_R_at_T, max_Rs \
            = get_embeddings_that_maximise_R(f,
                                             estimation_method=estimation_method,
                                             **kwargs)

        if len(embedding_maximising_R_at_T) == 0:
            if estimation_method == 'bbc':
                embedding_maximising_R_at_T_bbc = {}
                max_Rs_bbc = []
                max_R_bbc_CI_lo = {}
                max_R_bbc_CI_hi = {}
                # max_R_bbc_CI_med = {}
            elif estimation_method == 'shuffling':
                embedding_maximising_R_at_T_shuffling = {}
                max_Rs_shuffling = []
                max_R_shuffling_CI_lo = {}
                max_R_shuffling_CI_hi = {}
                # max_R_shuffling_CI_med = {}
            continue

        max_R_CI_lo = {}
        max_R_CI_hi = {}
        # max_R_CI_med = {}

        for past_range_T in embedding_maximising_R_at_T:
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
            embedding = (past_range_T, number_of_bins_d, scaling_k)

            bs_Rs = load_from_analysis_file(f,
                                            "bs_history_dependence",
                                            embedding_step_size=kwargs["embedding_step_size"],
                                            embedding=embedding,
                                            estimation_method=estimation_method,
                                            cross_val=kwargs['cross_val'])

            if isinstance(bs_Rs, np.ndarray):
                max_R_CI_lo[past_range_T], max_R_CI_hi[past_range_T] \
                    = get_CI_bounds(max_Rs[past_range_T],
                                    bs_Rs,
                                    kwargs["bootstrap_CI_use_sd"],
                                    kwargs["bootstrap_CI_percentile_lo"],
                                    kwargs["bootstrap_CI_percentile_hi"])
                # max_R_CI_med[past_range_T] \
                #     = np.median(bs_Rs)
            else:
                max_R_CI_lo[past_range_T] \
                    = max_Rs[past_range_T]

                max_R_CI_hi[past_range_T] \
                    = max_Rs[past_range_T]

                # max_R_CI_med[past_range_T] \
                #     = max_Rs[past_range_T]


        if estimation_method == 'bbc':
            embedding_maximising_R_at_T_bbc = embedding_maximising_R_at_T.copy()
            max_Rs_bbc = max_Rs.copy()
            max_R_bbc_CI_lo = max_R_CI_lo.copy()
            max_R_bbc_CI_hi = max_R_CI_hi.copy()
            # max_R_bbc_CI_med = max_R_CI_med.copy()
        elif estimation_method == 'shuffling':
            embedding_maximising_R_at_T_shuffling = embedding_maximising_R_at_T.copy()
            max_Rs_shuffling = max_Rs.copy()
            max_R_shuffling_CI_lo = max_R_CI_lo.copy()
            max_R_shuffling_CI_hi = max_R_CI_hi.copy()
            # max_R_shuffling_CI_med = max_R_CI_med.copy()

    Ts = sorted(np.unique(np.hstack(([R for R in max_Rs_bbc],
                                      [R for R in max_Rs_shuffling]))))
    H_spiking = load_from_analysis_file(f,
                                       "H_spiking")

    for T in Ts:
        histdep_data["T"] += [get_parameter_label(T)]
        if T in max_Rs_bbc:
            number_of_bins_d = embedding_maximising_R_at_T_bbc[T][0]
            scaling_k = embedding_maximising_R_at_T_bbc[T][1]
            first_bin_size = emb.get_first_bin_size_for_embedding((T,
                                                                  number_of_bins_d,
                                                                  scaling_k))

            histdep_data["max_R_bbc"] \
                += [get_parameter_label(max_Rs_bbc[T])]
            histdep_data["max_R_bbc_CI_lo"] \
                += [get_parameter_label(max_R_bbc_CI_lo[T])]
            histdep_data["max_R_bbc_CI_hi"] \
                += [get_parameter_label(max_R_bbc_CI_hi[T])]
            # histdep_data["max_R_bbc_CI_med"] \
            #     += [get_parameter_label(max_R_bbc_CI_med[T])]
            histdep_data["max_AIS_bbc"] \
                += [get_parameter_label(max_Rs_bbc[T] * H_spiking)]
            histdep_data["max_AIS_bbc_CI_lo"] \
                += [get_parameter_label(max_R_bbc_CI_lo[T] * H_spiking)]
            histdep_data["max_AIS_bbc_CI_hi"] \
                += [get_parameter_label(max_R_bbc_CI_hi[T] * H_spiking)]
            # histdep_data["max_AIS_bbc_CI_med"] \
            #     += [get_parameter_label(max_R_bbc_CI_med[T] * H_spiking)]
            histdep_data["number_of_bins_d_bbc"] \
                += [get_parameter_label(number_of_bins_d)]
            histdep_data["scaling_k_bbc"] \
                += [get_parameter_label(scaling_k)]
            histdep_data["first_bin_size_bbc"] \
                += [get_parameter_label(first_bin_size)]
        else:
            for key in histdep_data:
                if 'bbc' in key:
                    histdep_data[key] += ['-']
        if T in max_Rs_shuffling:
            number_of_bins_d = embedding_maximising_R_at_T_shuffling[T][0]
            scaling_k = embedding_maximising_R_at_T_shuffling[T][1]
            first_bin_size = emb.get_first_bin_size_for_embedding((T,
                                                                  number_of_bins_d,
                                                                  scaling_k))
            histdep_data["max_R_shuffling"] \
                += [get_parameter_label(max_Rs_shuffling[T])]
            histdep_data["max_R_shuffling_CI_lo"] \
                += [get_parameter_label(max_R_shuffling_CI_lo[T])]
            histdep_data["max_R_shuffling_CI_hi"] \
                += [get_parameter_label(max_R_shuffling_CI_hi[T])]
            # histdep_data["max_R_shuffling_CI_med"] \
            #     += [get_parameter_label(max_R_shuffling_CI_med[T])]
            histdep_data["max_AIS_shuffling"] \
                += [get_parameter_label(max_Rs_shuffling[T] * H_spiking)]
            histdep_data["max_AIS_shuffling_CI_lo"] \
                += [get_parameter_label(max_R_shuffling_CI_lo[T] * H_spiking)]
            histdep_data["max_AIS_shuffling_CI_hi"] \
                += [get_parameter_label(max_R_shuffling_CI_hi[T] * H_spiking)]
            # histdep_data["max_AIS_shuffling_CI_med"] \
            #     += [get_parameter_label(max_R_shuffling_CI_med[T] * H_spiking)]
            histdep_data["number_of_bins_d_shuffling"] \
                += [get_parameter_label(number_of_bins_d)]
            histdep_data["scaling_k_shuffling"] \
                += [get_parameter_label(scaling_k)]
            histdep_data["first_bin_size_shuffling"] \
                += [get_parameter_label(first_bin_size)]
        else:
            for key in histdep_data:
                if 'shuffling' in key:
                    histdep_data[key] += ['-']
    return histdep_data

def get_auto_MI_data(f,
                     analysis_num,
                     **kwargs):
    """
    Get auto MI values for each delay, as needed for the plots, to export
    them to a csv file.
    """

    auto_MI_data = {
        "auto_MI_bin_size" : [],
        "delay" : [],
        "auto_MI" : []
    }

    for auto_MI_bin_size in kwargs["auto_MI_bin_size_set"]:
        auto_MIs = load_from_analysis_file(f,
                                           "auto_MI",
                                           auto_MI_bin_size=auto_MI_bin_size)

        if isinstance(auto_MIs, np.ndarray):
            for delay, auto_MI in enumerate(auto_MIs):
                    auto_MI_data["auto_MI_bin_size"] += [get_parameter_label(auto_MI_bin_size)]
                    auto_MI_data["delay"] += [get_parameter_label(delay * auto_MI_bin_size)]
                    auto_MI_data["auto_MI"] += [get_parameter_label(auto_MI)]

    return auto_MI_data

def analyse_auto_MI(f,
                    spike_times,
                    auto_MI_bin_size_set,
                    auto_MI_max_delay,
                    **settings):
    """
    Get the auto MI for the spike times.  If it is available from file, load
    it, else compute it.
    """

    for auto_MI_bin_size in auto_MI_bin_size_set:
        number_of_delays = int(auto_MI_max_delay / auto_MI_bin_size) + 1

        auto_MI = load_from_analysis_file(f,
                                          "auto_MI",
                                          auto_MI_bin_size=auto_MI_bin_size)

        if isinstance(auto_MI, np.ndarray) and len(auto_MI) >= number_of_delays:
            continue

        # if no analysis found or new analysis includes more delays:
        # perform the analysis

        auto_MI = get_auto_MI(spike_times, auto_MI_bin_size, number_of_delays)

        save_to_analysis_file(f,
                              "auto_MI",
                              auto_MI_bin_size=auto_MI_bin_size,
                              auto_MI=auto_MI)

# ------------------------------------------------------------------------------ #
# the ones below are called `save`, but above we also save sometimes.
# all actual writing uses `save_to_analysis_file`, see `input_output.py`
# ------------------------------------------------------------------------------ #

def save_spike_times_stats(f, spike_times, embedding_step_size, **kwargs):
    """
    Save some statistics about the spike times.
    """

    recording_length = load_from_analysis_file(f, "recording_length")
    if recording_length == None:
        recording_lengths = [spt[-1] - spt[0] for spt in spike_times]
        recording_length = sum(recording_lengths)
        recording_length_sd = np.std(recording_lengths)

        save_to_analysis_file(f, "recording_length", recording_length=recording_length)
        save_to_analysis_file(
            f, "recording_length_sd", recording_length_sd=recording_length_sd
        )

    firing_rate = load_from_analysis_file(f, "firing_rate")
    if firing_rate == None:
        firing_rates = [
            get_binned_firing_rate(spt, embedding_step_size) for spt in spike_times
        ]
        recording_lengths = [spt[-1] - spt[0] for spt in spike_times]
        recording_length = sum(recording_lengths)

        firing_rate = np.average(firing_rates, weights=recording_lengths)
        firing_rate_sd = np.sqrt(
            np.average((firing_rates - firing_rate) ** 2, weights=recording_lengths)
        )

        save_to_analysis_file(f, "firing_rate", firing_rate=firing_rate)
        save_to_analysis_file(f, "firing_rate_sd", firing_rate_sd=firing_rate_sd)

    H_spiking = load_from_analysis_file(f, "H_spiking")

    if H_spiking == None:
        H_spiking = get_shannon_entropy(
            [firing_rate * embedding_step_size, 1 - firing_rate * embedding_step_size]
        )

        save_to_analysis_file(f, "H_spiking", H_spiking=H_spiking)

def save_history_dependence_for_embeddings(
    f,
    spike_times,
    estimation_method,
    embedding_past_range_set,
    embedding_number_of_bins_set,
    embedding_scaling_exponent_set,
    embedding_step_size,
    **kwargs,
):
    """
    Apply embeddings to spike times to obtain symbol counts.  Estimate
    the history dependence for each embedding.  Save results to file.
    """

    from .api import get_history_dependence

    if kwargs["cross_val"] == None or kwargs["cross_val"] == "h1":
        embeddings = emb.get_embeddings(
            embedding_past_range_set,
            embedding_number_of_bins_set,
            embedding_scaling_exponent_set,
        )
    elif kwargs["cross_val"] == "h2":
        # here we set cross_val to h1, because we load the
        # embeddings that maximise R from the optimisation step
        embeddings = get_embeddings_that_maximise_R(
            f,
            estimation_method,
            embedding_step_size,
            bbc_tolerance=kwargs["bbc_tolerance"],
            get_as_list=True,
            cross_val="h1",
        )

    for embedding in embeddings:
        past_range_T = embedding[0]
        number_of_bins_d = embedding[1]
        first_bin_size = emb.get_first_bin_size_for_embedding(embedding)

        symbol_counts = load_from_analysis_file(
            f,
            "symbol_counts",
            embedding_step_size=embedding_step_size,
            embedding=embedding,
            cross_val=kwargs["cross_val"],
        )
        if symbol_counts == None:
            symbol_counts = add_up_dicts(
                [
                    emb.get_symbol_counts(spt, embedding, embedding_step_size)
                    for spt in spike_times
                ]
            )
            save_to_analysis_file(
                f,
                "symbol_counts",
                embedding_step_size=embedding_step_size,
                embedding=embedding,
                symbol_counts=symbol_counts,
                cross_val=kwargs["cross_val"],
            )

        if estimation_method == "bbc":
            history_dependence = load_from_analysis_file(
                f,
                "history_dependence",
                embedding_step_size=embedding_step_size,
                embedding=embedding,
                estimation_method="bbc",
                cross_val=kwargs["cross_val"],
            )

            if history_dependence == None:
                history_dependence, bbc_term = get_history_dependence(
                    estimation_method, symbol_counts, number_of_bins_d
                )
                save_to_analysis_file(
                    f,
                    "history_dependence",
                    embedding_step_size=embedding_step_size,
                    embedding=embedding,
                    first_bin_size=first_bin_size,
                    estimation_method="bbc",
                    history_dependence=history_dependence,
                    bbc_term=bbc_term,
                    cross_val=kwargs["cross_val"],
                )

        elif estimation_method == "shuffling":
            history_dependence = load_from_analysis_file(
                f,
                "history_dependence",
                embedding_step_size=embedding_step_size,
                embedding=embedding,
                estimation_method="shuffling",
                cross_val=kwargs["cross_val"],
            )
            if history_dependence == None:
                history_dependence = get_history_dependence(
                    estimation_method, symbol_counts, number_of_bins_d
                )
                save_to_analysis_file(
                    f,
                    "history_dependence",
                    embedding_step_size=embedding_step_size,
                    embedding=embedding,
                    first_bin_size=first_bin_size,
                    estimation_method="shuffling",
                    history_dependence=history_dependence,
                    cross_val=kwargs["cross_val"],
                )

# ------------------------------------------------------------------------------ #
# Lower-level calculalations, no file needed.
# ------------------------------------------------------------------------------ #

def get_CI_bounds(R,
                  bs_Rs,
                  bootstrap_CI_use_sd=True,
                  bootstrap_CI_percentile_lo=2.5,
                  bootstrap_CI_percentile_hi=97.5):
    """
    Given bootstrap replications bs_Rs of the estimate for R,
    obtain the lower and upper bound of a 95% confidence
    interval based on the standard deviation; or an arbitrary
    confidence interval based on percentiles of the replications.
    """

    if bootstrap_CI_use_sd:
        sigma_R = np.std(bs_Rs)
        CI_lo = R - 2 * sigma_R
        CI_hi = R + 2 * sigma_R
    else:
        CI_lo = np.percentile(bs_Rs, bootstrap_CI_percentile_lo)
        CI_hi = np.percentile(bs_Rs, bootstrap_CI_percentile_hi)
    return (CI_lo, CI_hi)

def get_max_R_T(max_Rs):
    """
    Get R and T for which R is maximised.

    If R is maximised at several T, get the
    smallest respective T.
    """

    max_R_T = get_min_key_for_max_value(max_Rs)
    max_R = max_Rs[max_R_T]
    return max_R, max_R_T

def get_bootstrap_history_dependence(spike_times,
                                     embedding,
                                     embedding_step_size,
                                     estimation_method,
                                     number_of_bootstraps,
                                     block_length_l):
    """
    For a given embedding, return bootstrap replications for R.
    """

    from .api import get_history_dependence

    past_range_T, number_of_bins_d, scaling_k = embedding

    # compute total number of symbols in original data:
    # this is the amount of symbols we want to replicate
    min_num_symbols = 1 + int((min([spt[-1] - spt[0] for spt in spike_times])
                               - (past_range_T + embedding_step_size))
                              / embedding_step_size)

    symbol_block_length = int(block_length_l)

    if symbol_block_length >= min_num_symbols:
        print("Warning. Block length too large given number of symbols. Skipping.")
        return []

    # compute the bootstrap replications

    bs_Rs = np.zeros(number_of_bootstraps)

    symbols_array \
        = [get_symbols_array(spt, embedding, embedding_step_size)
           for spt in spike_times]

    for rep in range(number_of_bootstraps):
        bs_symbol_counts \
            = add_up_dicts([get_bootstrap_symbol_counts_from_symbols_array(symbols_array[i],
                                                                           symbol_block_length)
                            for i in range(len(symbols_array))])

        bs_history_dependence = get_history_dependence(estimation_method,
                                                            bs_symbol_counts,
                                                            number_of_bins_d,
                                                            bbc_tolerance=np.inf)

        bs_Rs[rep] = bs_history_dependence

    return bs_Rs

def get_symbols_array(spike_times, embedding, embedding_step_size):
    """
    Apply an embedding to a spike train and get the resulting symbols.
    """

    past_range_T, number_of_bins_d, scaling_k = embedding
    first_bin_size = emb.get_first_bin_size_for_embedding(embedding)

    raw_symbols = emb.get_raw_symbols(spike_times,
                                      embedding,
                                      first_bin_size,
                                      embedding_step_size)

    median_number_of_spikes_per_bin = emb.get_median_number_of_spikes_per_bin(raw_symbols)

    # symbols_array: array containing symbols
    # symbol_array: array of spikes representing symbol
    symbols_array = np.zeros(len(raw_symbols))

    for symbol_index, raw_symbol in enumerate(raw_symbols):
        # both are arrays, the comparison is element-wise, giving a bool array
        symbol_array = raw_symbol > median_number_of_spikes_per_bin
        symbol = emb.symbol_array_to_binary(symbol_array)
        symbols_array[symbol_index] = symbol

    return symbols_array

def get_bootstrap_symbol_counts_from_symbols_array(symbols_array,
                                                   symbol_block_length):
    """
    Given an array of symbols (cf get_symbols_array), get bootstrap
    replications of the symbol counts.
    """

    num_symbols = len(symbols_array)

    rand_indices = np.random.randint(0, num_symbols - (symbol_block_length - 1),
                                     size=int(num_symbols/ symbol_block_length))

    symbol_counts = Counter()

    for rand_index in rand_indices:
        for symbol in symbols_array[rand_index:rand_index + symbol_block_length]:
            symbol_counts[symbol] += 1

    residual_block_length = num_symbols - sum(symbol_counts.values())

    if residual_block_length > 0:
        rand_index_residual = np.random.randint(0, num_symbols - (residual_block_length - 1))

        for symbol in symbols_array[rand_index_residual:rand_index_residual + residual_block_length]:
            symbol_counts[symbol] += 1

    return symbol_counts

def get_shannon_entropy(probabilities):
    """
    Get the entropy of a random variable based on the probabilities
    of its outcomes.
    """

    return - sum((p * np.log(p) for p in probabilities if not p == 0))

def get_H_spiking(symbol_counts):
    """
    Get the (unconditional) entropy of a spike train, based
    on its outcomes, as stored in the symbol_counts dictionary.

    For each symbol, determine what the response was (spike/ no spike),
    and obtain the spiking probability. From that compute the entropy.
    """

    number_of_spikes = 0
    number_of_symbols = 0
    for symbol, counts in symbol_counts.items():
        number_of_symbols += counts
        if symbol % 2 == 1:
            number_of_spikes += counts

    p_spike = number_of_spikes / number_of_symbols
    return get_shannon_entropy([p_spike,
                                1 - p_spike])

def get_binned_neuron_activity(spike_times, bin_size, relative_to_median_activity=False):
    """
    Get an array of 0s and 1s representing the spike train.
    """

    number_of_bins = int(spike_times[-1] / bin_size) + 1
    binned_neuron_activity = np.zeros(number_of_bins, dtype=int)
    if relative_to_median_activity:
        for spike_time in spike_times:
            binned_neuron_activity[int(spike_time / bin_size)] += 1
        median_activity = np.median(binned_neuron_activity)
        spike_indices = np.where(binned_neuron_activity > median_activity)
        binned_neuron_activity = np.zeros(number_of_bins, dtype=int)
        binned_neuron_activity[spike_indices] = 1
    else:
        binned_neuron_activity[[int(spike_time / bin_size) for spike_time in spike_times]] = 1
    return binned_neuron_activity

def get_binned_firing_rate(spike_times, bin_size):
    """
    Get the firing rate of a spike train, as obtained after binning the activity.
    """

    binned_neuron_activity = get_binned_neuron_activity(spike_times, bin_size)
    number_of_bins = int(spike_times[-1] / bin_size) + 1
    return sum(binned_neuron_activity) / (number_of_bins * bin_size)

def get_smoothed_neuron_activity(spt,
                                 averaging_time,
                                 binning_time=0.005):
    """
    Get a smoothed version of the neuron activity, for visualization.

    cf https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    binned_neuron_activity \
        = get_binned_neuron_activity(spt,
                                     binning_time)
    smoothing_window_len = int(averaging_time / binning_time)

    s = np.r_[binned_neuron_activity[smoothing_window_len-1:0:-1],
              binned_neuron_activity,
              binned_neuron_activity[-2:-smoothing_window_len-1:-1]]

    return np.convolve(np.hanning(smoothing_window_len) /
                       np.hanning(smoothing_window_len).sum(), s,
                       mode='valid') / binning_time

def get_past_symbol_counts(symbol_counts, merge=True):
    """
    Get symbol_counts for the symbols excluding the response.
    """

    past_symbol_counts = [Counter(), Counter()]
    for symbol in symbol_counts:
        response = int(symbol % 2)
        past_symbol = symbol // 2
        past_symbol_counts[response][past_symbol] = symbol_counts[symbol]

    if merge:
        merged_past_symbol_counts = Counter()
        for response in [0, 1]:
            for symbol in past_symbol_counts[response]:
                merged_past_symbol_counts[symbol] += past_symbol_counts[response][symbol]
        return merged_past_symbol_counts
    else:
        return past_symbol_counts

def get_auto_MI(spike_times, bin_size, number_of_delays):
    """
    Compute the auto mutual information in the neuron's activity, a
    measure closely related to history dependence.
    """

    binned_neuron_activity = []

    for spt in spike_times:
        # represent the neural activity as an array of 0s (no spike) and 1s (spike)
        binned_neuron_activity += [get_binned_neuron_activity(spt,
                                                              bin_size,
                                                              relative_to_median_activity=True)]

    p_spike = sum([sum(bna)
                  for bna in binned_neuron_activity]) / sum([len(bna)
                                                             for bna in binned_neuron_activity])
    H_spiking = get_shannon_entropy([p_spike,
                                    1 - p_spike])

    auto_MIs = []

    # compute auto MI
    for delay in range(number_of_delays):

        symbol_counts = []
        for bna in binned_neuron_activity:
            number_of_symbols = len(bna) - delay - 1

            symbols = np.array([2 * bna[i] + bna[i + delay + 1]
                                for i in range(number_of_symbols)])

            symbol_counts += [dict([(unq_symbol, len(np.where(symbols==unq_symbol)[0]))
                                    for unq_symbol in np.unique(symbols)])]

        symbol_counts = add_up_dicts(symbol_counts)
        number_of_symbols = sum(symbol_counts.values())
        # number_of_symbols = sum([len(bna) - delay - 1 for bna in binned_neuron_activity])

        H_joint = get_shannon_entropy([number_of_occurrences / number_of_symbols
                                       for number_of_occurrences in symbol_counts.values()])

        # I(X : Y) = H(X) - H(X|Y) = H(X) - (H(X,Y) - H(Y)) = H(X) + H(Y) - H(X,Y)
        # auto_MI = 2 * H_spiking - H_joint
        auto_MI = 2 - H_joint/ H_spiking # normalized auto MI = auto MI / H_spiking

        auto_MIs += [auto_MI]

    return auto_MIs

# ------------------------------------------------------------------------------ #
# Helpers, lowest level
# ------------------------------------------------------------------------------ #

def get_parameter_label(parameter):
    """
    Get a number in a unified format as label for the hdf5 file.
    """

    return "{:.5f}".format(parameter)

def find_existing_parameter(new_parameter, existing_parameters, tolerance=1e-5):
    """
    Search for a parameter value in a list, return label for
    the hdf5 file and whether an existing one was found.

    Tolerance should be no lower than precision in get_parameter_label.
    """

    new_parameter = float(new_parameter)
    if not isinstance(existing_parameters, list):
        existing_parameters = [existing_parameters]
    for existing_parameter in existing_parameters:
        if np.abs(float(existing_parameter) - new_parameter) <= tolerance:
            return existing_parameter, True
    return get_parameter_label(new_parameter), False

def get_hash(spike_times):
    """
    Get hash representing the spike times, for bookkeeping.
    """

    if len(spike_times) == 1:
        m = hashlib.sha256()
        m.update(str(spike_times[0]).encode('utf-8'))
        return m.hexdigest()
    else:
        ms = []
        for spt in spike_times:
            m = hashlib.sha256()
            m.update(str(spt).encode('utf-8'))
            ms += [m.hexdigest()]
        m = hashlib.sha256()
        m.update(str(sorted(ms)).encode('utf-8'))
        return m.hexdigest()

def is_float(x):
  try:
    float(x)
    return True
  except ValueError:
    return False

def remove_key(d, key):
    """
    Remove an entry from a dictionary .
    """

    r = d.copy()
    del r[key]
    return r

def add_up_dicts(dicts):
    return sum((Counter(dict(d)) for d in dicts),
               Counter())

def get_min_key_for_max_value(d):
    """
    For a dictionary d, get the key for the largest value.
    If largest value is attained several times, get the
    smallest respective key.
    """

    sorted_keys = sorted([key for key in d.keys()])
    values = [d[key] for key in sorted_keys]
    max_value = max(values)
    for key, value in zip(sorted_keys, values):
        if value == max_value:
            return key
