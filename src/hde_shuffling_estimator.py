import numpy as np
import hde_utils as utl
import hde_embedding as emb

def P_r(number_of_symbols):
    return [number_of_symbols[0] / sum(number_of_symbols),
            number_of_symbols[1] / sum(number_of_symbols)]

def P_s(past_symbol_counts, number_of_symbols):
    P_stimulus_uncond = {}
    for response in [0, 1]:
        for symbol in past_symbol_counts[response]:
            if symbol in P_stimulus_uncond:
                P_stimulus_uncond[symbol] += past_symbol_counts[response][symbol]
            else:
                P_stimulus_uncond[symbol] = past_symbol_counts[response][symbol]
    number_of_symbols_uncond = sum(number_of_symbols)
    for symbol in P_stimulus_uncond:
        P_stimulus_uncond[symbol] /= number_of_symbols_uncond
    return P_stimulus_uncond

def P_s_r(past_symbol_counts, number_of_symbols):
    P_cond = [{}, {}]
    for response in [0, 1]:
        for symbol in past_symbol_counts[response]:
            P_cond[response][symbol] \
                = past_symbol_counts[response][symbol] / number_of_symbols[response]
    return P_cond

def H0_s(marginal_probabilities, i, number_of_bins_d, sm, prd):
    sm1 = 0
    sm2 = 0
    if not marginal_probabilities[i] == 0:
       sm1 = np.log(marginal_probabilities[i])
    if not marginal_probabilities[i] == 1:
       sm2 = np.log(1 - marginal_probabilities[i])
    
    if i == number_of_bins_d - 1:
        return (sm - sm1) * (prd * marginal_probabilities[i]) \
            + (sm - sm2) * (prd * (1 - marginal_probabilities[i]))
    return H0_s(marginal_probabilities,
                i + 1,
                number_of_bins_d,
                sm - sm1,
                prd*marginal_probabilities[i]) + H0_s(marginal_probabilities,
                                                      i + 1,
                                                      number_of_bins_d,
                                                      sm - sm2,
                                                      prd*(1 - marginal_probabilities[i]))

def H0_s_r(marginal_probabilities, number_of_bins_d, P_uncond):
    H0_r = [0, 0]
    for response in [0, 1]:
        H0_r[response] = H0_s(marginal_probabilities[response], 0, number_of_bins_d, 0, 1)
    return sum([P_uncond[response] * H0_r[response] for response in [0, 1]])

def H(X):
    return utl.get_shannon_entropy(X)

def H_s_r(P_uncond, P_cond):
    return sum((P_uncond[response] * H(P_cond[response].values()) for response in [0, 1]))

def get_marginal_frequencies_of_spikes_in_bins(symbol_counts, number_of_bins_d):
    return np.array(sum((emb.symbol_binary_to_array(symbol, number_of_bins_d)
                         * symbol_counts[symbol]
                         for symbol in symbol_counts)), dtype=int)

def get_shuffled_symbol_counts(symbol_counts, number_of_bins_d):
    past_symbol_counts = utl.get_past_symbol_counts(symbol_counts, merge=False)

    number_of_symbols = [sum(past_symbol_counts[response].values()) for response in [0, 1]]
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

    shuffled_past_symbol_counts = [{}, {}]

    for response in [0, 1]:
        for past_symbol in shuffled_past_symbols[response]:
            if past_symbol in shuffled_past_symbol_counts[response]:
                shuffled_past_symbol_counts[response][past_symbol] += 1
            else:
                shuffled_past_symbol_counts[response][past_symbol] = 1

    marginal_probabilities = [marginal_frequencies[response] / number_of_symbols[response]
                              for response in [0, 1]]

    return past_symbol_counts, number_of_symbols, shuffled_past_symbol_counts, marginal_probabilities



def shuffling_MI(symbol_counts, number_of_bins_d):
    past_symbol_counts, number_of_symbols, shuffled_past_symbol_counts, marginal_probabilities \
        = get_shuffled_symbol_counts(symbol_counts, number_of_bins_d)
    
    P_uncond = P_r(number_of_symbols)
    P_stimulus_uncond = P_s(past_symbol_counts, number_of_symbols)

    P_cond = P_s_r(past_symbol_counts, number_of_symbols)
    P0_sh_cond = P_s_r(shuffled_past_symbol_counts, number_of_symbols)

    H_uncond = H(P_stimulus_uncond.values())
    H_cond = H_s_r(P_uncond, P_cond)
    H0_cond = H0_s_r(marginal_probabilities, number_of_bins_d, P_uncond)
    H0_sh_cond = H_s_r(P_uncond, P0_sh_cond)

    return H0_sh_cond - H_cond - H0_cond + H_uncond

def shuffling_estimator(symbol_counts, number_of_bins_d, H_uncond,
                        return_ais=False):
    I_sh = shuffling_MI(symbol_counts,
                        number_of_bins_d)

    if return_ais:
        return I_sh
    else:
        return I_sh / H_uncond
