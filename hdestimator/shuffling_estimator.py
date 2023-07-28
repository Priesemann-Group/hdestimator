import numpy as np
from collections import Counter
from . import utils as utl
from . import embedding as emb

import logging
log = logging.getLogger("hdestimator")


def get_P_X_uncond(number_of_symbols):
    """
    Compute P(X), the probability of the current activity using
    the plug-in estimator.
    """

    return [number_of_symbols[0] / sum(number_of_symbols),
            number_of_symbols[1] / sum(number_of_symbols)]

def get_P_X_past_uncond(past_symbol_counts, number_of_symbols):
    """
    Compute P(X_past), the probability of the past activity using
    the plug-in estimator.
    """

    P_X_past_uncond = {}
    for response in [0, 1]:
        for symbol in past_symbol_counts[response]:
            if symbol in P_X_past_uncond:
                P_X_past_uncond[symbol] += past_symbol_counts[response][symbol]
            else:
                P_X_past_uncond[symbol] = past_symbol_counts[response][symbol]
    number_of_symbols_uncond = sum(number_of_symbols)

    for symbol in P_X_past_uncond:
        P_X_past_uncond[symbol] /= number_of_symbols_uncond
    return P_X_past_uncond

def get_P_X_past_cond_X(past_symbol_counts, number_of_symbols):
    """
    Compute P(X_past | X), the probability of the past activity conditioned
    on the response X using the plug-in estimator.
    """

    P_X_past_cond_X = [{}, {}]
    for response in [0, 1]:
        for symbol in past_symbol_counts[response]:
            P_X_past_cond_X[response][symbol] \
                = past_symbol_counts[response][symbol] / number_of_symbols[response]
    return P_X_past_cond_X

def get_H0_X_past_cond_X_eq_x(marginal_probabilities, number_of_bins_d):
    """
    Compute H_0(X_past | X = x), cf get_H0_X_past_cond_X.
    """
    return utl.get_shannon_entropy(marginal_probabilities) \
        + utl.get_shannon_entropy(1 - marginal_probabilities)

def get_H0_X_past_cond_X(marginal_probabilities, number_of_bins_d, P_X_uncond):
    """
    Compute H_0(X_past | X), the estimate of the entropy for the past
    symbols given a response, under the assumption that activity in
    the past contributes independently towards the response.
    """
    H0_X_past_cond_X_eq_x = [0, 0]
    for response in [0, 1]:
        H0_X_past_cond_X_eq_x[response] \
            = get_H0_X_past_cond_X_eq_x(marginal_probabilities[response],
                                        number_of_bins_d)
    return sum([P_X_uncond[response] * H0_X_past_cond_X_eq_x[response] for response in [0, 1]])

def get_H_X_past_uncond(P_X_past_uncond):
    """
    Compute H(X_past), the plug-in estimate of the entropy for the past symbols, given
    their probabilities.
    """

    return utl.get_shannon_entropy(P_X_past_uncond.values())

def get_H_X_past_cond_X(P_X_uncond, P_X_past_cond_X):
    """
    Compute H(X_past | X), the plug-in estimate of the conditional entropy for the past
    symbols, conditioned on the response X,  given their probabilities.
    """

    return sum((P_X_uncond[response] * get_H_X_past_uncond(P_X_past_cond_X[response])
                for response in [0, 1]))

def get_marginal_frequencies_of_spikes_in_bins(symbol_counts, number_of_bins_d):
    """
    Compute for each past bin 1...d the sum of spikes found in that bin across all
    observed symbols.
    """
    return np.array(sum((emb.symbol_binary_to_array(symbol, number_of_bins_d)
                         * symbol_counts[symbol]
                         for symbol in symbol_counts)), dtype=int)

def get_shuffled_symbol_counts(symbol_counts, past_symbol_counts, number_of_bins_d,
                               number_of_symbols):
    """
    Simulate new data by, for each past bin 1...d, permutating the activity
    across all observed past_symbols (for a given response X). The marginal
    probability of observing a spike given the response is thus preserved for
    each past bin.
    """
    number_of_spikes = sum(past_symbol_counts[1].values())

    marginal_frequencies = [get_marginal_frequencies_of_spikes_in_bins(past_symbol_counts[response],
                                                                       number_of_bins_d)
                            for response in [0, 1]]

    shuffled_past_symbols = [np.zeros(number_of_symbols[response]) for response in [0, 1]]

    for i in range(0, number_of_bins_d):
        for response in [0, 1]:
            shuffled_past_symbols[response] \
                += 2 ** (number_of_bins_d - i - 1) \
                * np.random.permutation(np.hstack((np.ones(marginal_frequencies[response][i]),
                                                   np.zeros(number_of_symbols[response] \
                                                            - marginal_frequencies[response][i]))))

    for response in [0, 1]:
        shuffled_past_symbols[response] = np.array(shuffled_past_symbols[response], dtype=int)

    shuffled_past_symbol_counts = [Counter(), Counter()]

    for response in [0, 1]:
        for past_symbol in shuffled_past_symbols[response]:
            shuffled_past_symbol_counts[response][past_symbol] += 1

    marginal_probabilities = [marginal_frequencies[response] / number_of_symbols[response]
                              for response in [0, 1]]

    return shuffled_past_symbol_counts, marginal_probabilities



def shuffling_MI(symbol_counts, number_of_bins_d):
    """
    Estimate the mutual information between current and past activity
    in a spike train using the shuffling estimator.

    To obtain the shuffling estimate, compute the plug-in estimate and
    a correction term to reduce its bias.

    For the plug-in estimate:
    - Extract the past_symbol_counts from the symbol_counts.
    - I_plugin = H(X_past) - H(X_past | X)

    Notation:
    X: current activity, aka response
    X_past: past activity

    P_X_uncond: P(X)
    P_X_past_uncond: P(X_past)
    P_X_past_cond_X: P(X_past | X)

    H_X_past_uncond: H(X_past)
    H_X_past_cond_X: H(X_past | X)

    I_plugin: plugin estimate of I(X_past; X)


    For the correction term:
    - Simulate additional data under the assumption that activity
    in the past contributes independently towards the current activity.
    - Compute the entropy under the assumptions of the model, which
    due to its simplicity is easy to sample and the estimate unbiased
    - Compute the entropy using the plug-in estimate, whose bias is
    similar to that of the plug-in estimate on the original data
    - Compute the correction term as the difference between the
    unbiased and biased terms

    Notation:
    P0_sh_X_past_cond_X: P_0,sh(X_past | X), equiv. to P(X_past | X)
                         on the shuffled data

    H0_X_past_cond_X: H_0(X_past | X), based on the model of independent
    contributions
    H0_sh_X_past_cond_X: H_0,sh(X_past | X), based on
    P0_sh_X_past_cond_X, ie the plug-in estimate

    I_corr: the correction term to reduce the bias of I_plugin


    :param symbol_counts: the activity of a spike train is embedded into symbols,
    whose occurences are counted (cf emb.get_symbol_counts)
    :param number_of_bins_d: the number of bins of the embedding
    """

    # plug-in estimate
    past_symbol_counts = utl.get_past_symbol_counts(symbol_counts, merge=False)
    number_of_symbols = [sum(past_symbol_counts[response].values()) for response in [0, 1]]

    P_X_uncond = get_P_X_uncond(number_of_symbols)
    P_X_past_uncond = get_P_X_past_uncond(past_symbol_counts, number_of_symbols)
    P_X_past_cond_X = get_P_X_past_cond_X(past_symbol_counts, number_of_symbols)

    H_X_past_uncond = get_H_X_past_uncond(P_X_past_uncond)
    H_X_past_cond_X = get_H_X_past_cond_X(P_X_uncond, P_X_past_cond_X)

    I_plugin = H_X_past_uncond - H_X_past_cond_X

    # correction term
    shuffled_past_symbol_counts, marginal_probabilities \
        = get_shuffled_symbol_counts(symbol_counts, past_symbol_counts, number_of_bins_d,
                                     number_of_symbols)

    P0_sh_X_past_cond_X = get_P_X_past_cond_X(shuffled_past_symbol_counts, number_of_symbols)

    H0_X_past_cond_X = get_H0_X_past_cond_X(marginal_probabilities, number_of_bins_d, P_X_uncond)
    H0_sh_X_past_cond_X = get_H_X_past_cond_X(P_X_uncond, P0_sh_X_past_cond_X)

    I_corr = H0_X_past_cond_X - H0_sh_X_past_cond_X

    # shuffling estimate
    return I_plugin - I_corr

def shuffling_estimator(symbol_counts, number_of_bins_d, H_uncond,
                        return_ais=False):
    """
    Estimate the history dependence in a spike train using the shuffling estimator.

    :param symbol_counts: the activity of a spike train is embedded into symbols,
    whose occurences are counted (cf emb.get_symbol_counts)
    :param number_of_bins_d: the number of bins of the embedding
    :param H_uncond: the (unconditional) spiking entropy of the spike train
    :param return_ais: define whether to return the unnormalized mutual information,
    aka active information storage (ais), instead of the history dependence
    """

    I_sh = shuffling_MI(symbol_counts,
                        number_of_bins_d)

    if return_ais:
        return I_sh
    else:
        return I_sh / H_uncond
