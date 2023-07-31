# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-07-27 15:08:56
# @Last Modified: 2023-07-28 14:50:28
# ------------------------------------------------------------------------------ #
# Reworked embeddings to use numba, if available. this saves a lot of redundant
# code, and cython.
# ------------------------------------------------------------------------------ #


import numpy as np
from scipy.optimize import newton
from collections import Counter
from sys import stderr, exit
from typing import Tuple
import logging

log = logging.getLogger("hdestimator")
# log.setLevel("DEBUG")

try:
    from numba import jit, prange

    # raise ImportError
    log.info("Using numba for parallelizable functions")
    FAST_EMBEDDING_AVAILABLE = True

except ImportError:
    log.info("Numba not available, skipping parallelization")
    FAST_EMBEDDING_AVAILABLE = False

    # replace numba functions if numba not available: we only use jit and prange.
    def parametrized(dec):
        def layer(*args, **kwargs):
            def repl(f):
                return dec(f, *args, **kwargs)

            return repl

        return layer

    @parametrized
    def jit(func, **kwargs):
        return func

    def prange(*args):
        return range(*args)


def get_symbol_counts(spike_times, embedding, embedding_step_size):
    """
    Apply embedding to the spike times to obtain the symbol counts.
    """

    first_bin_size = get_first_bin_size_for_embedding(embedding)

    raw_symbols = get_raw_symbols(
        spike_times, embedding, first_bin_size, embedding_step_size
    )

    symbol_counts = count_raw_symbols(raw_symbols)

    return Counter(symbol_counts)


def get_first_bin_size_for_embedding(embedding):
    """
    Get size of first bin for the embedding, based on the parameters
    T, d and k.
    """

    past_range_T, number_of_bins_d, scaling_k = embedding
    return newton(
        lambda first_bin_size: get_past_range(number_of_bins_d, first_bin_size, scaling_k)
        - past_range_T,
        0.005,
        tol=1e-03,
        maxiter=100,
    )


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def get_raw_symbols(
    spike_times,
    embedding,
    first_bin_size,
    embedding_step_size,
):
    """
    Get the raw symbols (in which the number of spikes per bin are counted,
    ie not necessarily binary quantity), as obtained by applying the
    embedding.

    # Parameters
    spike_times: array of float
    """

    past_range_T, number_of_bins_d, scaling_k = embedding

    # the window is the embedding plus the response,
    # ie the embedding and one additional bin of size embedding_step_size
    window_delimiters = get_window_delimiters(
        number_of_bins_d, scaling_k, first_bin_size, embedding_step_size
    )
    window_length = window_delimiters[-1]
    num_spike_times = len(spike_times)
    last_spike_time = spike_times[-1]

    num_symbols = int((last_spike_time - window_length) / embedding_step_size)
    raw_symbols = np.empty((num_symbols, number_of_bins_d + 1), dtype=np.int64)

    time = 0
    spike_index_lo = 0

    # prealloc
    embedding_bin_index = 0
    spike_index_hi = 0
    spikes_in_window = np.zeros(number_of_bins_d + 1, dtype=np.int64)

    for sdx in range(num_symbols):

        while spike_index_lo < num_spike_times and spike_times[spike_index_lo] < time:
            spike_index_lo += 1

        spike_index_hi = spike_index_lo
        while (
            spike_index_hi < num_spike_times
            and spike_times[spike_index_hi] < time + window_length
        ):
            spike_index_hi += 1

        spikes_in_window[:] = 0
        embedding_bin_index = 0
        for spike_index in range(spike_index_lo, spike_index_hi):
            while (
                spike_times[spike_index] > time + window_delimiters[embedding_bin_index]
            ):
                embedding_bin_index += 1
            spikes_in_window[embedding_bin_index] += 1

        raw_symbols[sdx] = spikes_in_window

        time += embedding_step_size

    return raw_symbols


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def count_raw_symbols(raw_symbols):
    """
    Count occurences of precalculated raw symbols.

    # Parameters
    raw_symbols : 2d array of type int,
        shape (num_symbols, num_bins)

    # Returns
    symbol_counts : dict
        symbols -> counts
    """

    # number_of_bins here is number_of_bins_d + 1,
    # as it here includes not only the bins of the embedding but also the response

    num_symbols, num_bins = raw_symbols.shape
    bins = np.arange(0, num_bins)

    median_num_spikes_per_bin = get_median_number_of_spikes_per_bin(raw_symbols)
    assert len(median_num_spikes_per_bin) == num_bins

    symbol_counts = dict()

    # log.debug(f"{median_num_spikes_per_bin=}")
    # log.debug(f"{num_bins=}")

    for raw_symbol in raw_symbols:
        # both are np arrays, so the comparison is elementwise
        symbol_array = raw_symbol > median_num_spikes_per_bin

        # symbol = symbol_array_to_binary(symbol_array)
        symbol = np.sum(
            np.power(2, num_bins - bins - 1) * symbol_array[bins],
            dtype=np.int64,
        )

        if symbol in symbol_counts:
            symbol_counts[symbol] += 1
        else:
            symbol_counts[symbol] = 1

    return symbol_counts


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def get_past_range(number_of_bins_d, first_bin_size, scaling_k):
    """
    Get the past range T of the embedding, based on the parameters d, tau_1 and k.
    """
    i = np.arange(1, number_of_bins_d + 1)
    return np.sum(first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k))


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def get_window_delimiters(
    number_of_bins_d, scaling_k, first_bin_size, embedding_step_size
):
    """
    Get delimiters of the window, used to describe the embedding. The
    window includes both the past embedding and the response.

    The delimiters are times, relative to the first bin, that separate
    two consequent bins.
    """

    i = np.arange(1, number_of_bins_d + 1)
    bin_sizes = first_bin_size * np.power(10, ((number_of_bins_d - i) * scaling_k))

    window_delimiters = np.cumsum(bin_sizes)

    # Append the last element + embedding_step_size to get the final window_delimiters
    window_delimiters = np.append(
        window_delimiters, window_delimiters[-1] + embedding_step_size
    )

    return window_delimiters


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def get_median_number_of_spikes_per_bin(raw_symbols):
    """
    Given raw symbols (in which the number of spikes per bin are counted,
    ie not necessarily binary quantity), get the median number of spikes
    for each bin, among all symbols obtained by the embedding.

    this is the same as np.median(raw_symbols, axis=0), but numba does not like
    the axis argument.
    """

    # return np.median(raw_symbols, axis=0) # numba....
    num_bins = len(raw_symbols[0])
    median_num_spikes_per_bin = np.zeros(num_bins, dtype=np.int64)
    for i in range(num_bins):
        median_num_spikes_per_bin[i] = np.median(raw_symbols[:, i])

    return median_num_spikes_per_bin


def get_set_of_scalings(
    past_range_T,
    number_of_bins_d,
    number_of_scalings,
    min_first_bin_size,
    min_step_for_scaling,
):
    """
    Get scaling exponents such that the uniform embedding as well as
    the embedding for which the first bin has a length of
    min_first_bin_size (in seconds), as well as linearly spaced
    scaling factors in between, such that in total
    number_of_scalings scalings are obtained.
    """

    min_scaling = 0
    if past_range_T / number_of_bins_d <= min_first_bin_size or number_of_bins_d == 1:
        max_scaling = 0
    else:
        # for the initial guess assume the largest bin dominates, so k is approx. log(T) / d

        max_scaling = newton(
            lambda scaling: get_past_range(number_of_bins_d, min_first_bin_size, scaling)
            - past_range_T,
            np.log10(past_range_T / min_first_bin_size) / (number_of_bins_d - 1),
            tol=1e-04,
            maxiter=500,
        )

    while (
        np.linspace(min_scaling, max_scaling, number_of_scalings, retstep=True)[1]
        < min_step_for_scaling
    ):
        number_of_scalings -= 1

    return np.linspace(min_scaling, max_scaling, number_of_scalings)


def get_embeddings(
    embedding_past_range_set, embedding_number_of_bins_set, embedding_scaling_exponent_set
):
    """
    Get all combinations of parameters T, d, k, based on the
    sets of selected parameters.
    """

    embeddings = []
    for past_range_T in embedding_past_range_set:
        for number_of_bins_d in embedding_number_of_bins_set:
            if not isinstance(number_of_bins_d, int) or number_of_bins_d < 1:
                log.warning(
                    f"numer of bins {number_of_bins_d} is not a positive integer."
                    " Skipping."
                )
                continue

            if type(embedding_scaling_exponent_set) == dict:
                scaling_set_given_T_and_d = get_set_of_scalings(
                    past_range_T, number_of_bins_d, **embedding_scaling_exponent_set
                )
            else:
                scaling_set_given_T_and_d = embedding_scaling_exponent_set

            for scaling_k in scaling_set_given_T_and_d:
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

    return embeddings


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def symbol_binary_to_array(symbol_binary, number_of_bins_d):
    """
    Given a binary representation of a symbol (cf symbol_array_to_binary),
    convert it back into its array-representation.
    """

    spikes_in_window = np.zeros(number_of_bins_d, dtype=np.int64)
    for i in range(0, number_of_bins_d):
        b = np.power(2, (number_of_bins_d - 1 - i))
        if b <= symbol_binary:
            spikes_in_window[i] = 1
            symbol_binary -= b
    return spikes_in_window


@jit(nopython=True, parallel=False, fastmath=True, cache=True, error_model="numpy")
def symbol_array_to_binary(spikes_in_window: np.ndarray) -> int:
    """
    Given an array of 1s and 0s, representing spikes and the absence
    thereof, read the array as a binary number to obtain a
    (base 10) integer.
    """

    num_bins = len(spikes_in_window)

    i = np.arange(0, num_bins)
    res = np.sum(np.power(2, num_bins - i - 1) * spikes_in_window[i], dtype=np.int64)
    return res
