import numpy as np
import hde_utils as utl
import hde_embedding as emb
import hde_bbc_estimator as bbc
import hde_shuffling_estimator as sh

def get_history_dependence(estimation_method,
                           symbol_counts,
                           number_of_bins_d,
                           past_symbol_counts=None,
                           bbc_tolerance=None,
                           H_uncond=None,
                           return_ais=False,
                           **kwargs):
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

    alphabet_size_past = 2 ** int(number_of_bins_d) # K for past activity
    alphabet_size = alphabet_size_past * 2          # K

    if estimation_method == "bbc":
        return bbc.bbc_estimator(symbol_counts,
                                 past_symbol_counts,
                                 alphabet_size,
                                 alphabet_size_past,
                                 H_uncond,
                                 bbc_tolerance=bbc_tolerance,
                                 return_ais=return_ais)

    elif estimation_method == "shuffling":
        return sh.shuffling_estimator(symbol_counts,
                                      number_of_bins_d,
                                      H_uncond,
                                      return_ais=return_ais)



## below are functions for estimates on spike trains
    
def get_history_dependence_for_single_embedding(spike_times,
                                                recording_length,
                                                estimation_method,
                                                embedding,
                                                embedding_step_size,
                                                bbc_tolerance=None,
                                                **kwargs):
    """
    Apply embedding to spike_times to obtain symbol counts.
    Get history dependence from symbol counts.
    """

    past_range_T, number_of_bins_d, scaling_k = embedding

    symbol_counts = emb.get_symbol_counts(spike_times, embedding, embedding_step_size)

    if estimation_method == 'bbc':
        history_dependence, bbc_term = get_history_dependence(estimation_method,
                                                              symbol_counts,
                                                              number_of_bins_d,
                                                              bbc_tolerance=None,
                                                              **kwargs)

        if bbc_tolerance == None:
            return history_dependence, bbc_term
        
        if bbc_term >= bbc_tolerance:
            return None
      
    elif estimation_method == 'shuffling':
        history_dependence = get_history_dependence(estimation_method,
                                                    symbol_counts,
                                                    number_of_bins_d,
                                                    **kwargs)

    return history_dependence

def get_history_dependence_for_embedding_set(spike_times,
                                               recording_length,
                                               estimation_method,
                                               embedding_past_range_set,
                                               embedding_number_of_bins_set,
                                               embedding_scaling_exponent_set,
                                               embedding_step_size,
                                               bbc_tolerance=None,
                                               dependent_var="T",
                                               **kwargs):
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
    
    for embedding in emb.get_embeddings(embedding_past_range_set,
                                        embedding_number_of_bins_set,
                                        embedding_scaling_exponent_set):
        past_range_T, number_of_bins_d, scaling_k = embedding

        history_dependence = get_history_dependence_for_single_embedding(spike_times,
                                                                         recording_length,
                                                                         estimation_method,
                                                                         embedding,
                                                                         embedding_step_size,
                                                                         bbc_tolerance=bbc_tolerance,
                                                                         **kwargs)
        if history_dependence == None:
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

    return embeddings_that_maximise_R, max_Rs

# FIXME don't use percentiles, use std per default
def get_CI_for_embedding(spike_times,
                         recording_length,
                         estimation_method,
                         embedding,
                         embedding_step_size,
                         number_of_bootstraps,
                         block_length_l=None,
                         bootstrap_CI_use_sd=True,
                         bootstrap_CI_percentile_lo=2.5,
                         bootstrap_CI_percentile_hi=97.5):
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
    
    bs_history_dependence \
            = utl.get_bootstrap_history_dependence(spike_times,
                                                   recording_length,
                                                   embedding,
                                                   embedding_step_size,
                                                   estimation_method,
                                                   number_of_bootstraps,
                                                   block_length_l)

    if bootstrap_CI_use_sd:
        mu = np.average(bs_history_dependence)
        sigma = np.std(bs_history_dependence)
        CI_lo = mu - 2 * sigma
        CI_hi = mu + 2 * sigma
    else:
        CI_lo = np.percentile(bs_history_dependence, bootstrap_CI_percentile_lo)
        CI_hi = np.percentile(bs_history_dependence, bootstrap_CI_percentile_hi)

    return (CI_lo, CI_hi)
