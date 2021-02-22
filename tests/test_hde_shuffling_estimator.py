from estimate import parse_arguments
import hde_utils as utl
import hde_shuffling_estimator as sh

from sys import path
from os.path import realpath, dirname
path.insert(1, dirname(realpath(__file__)))
import expected_output as exp

import numpy as np

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

spike_times_file_name = 'sample_data/spike_times.dat'
defined_tasks = ["history-dependence",
                 "confidence-intervals",
                 # "permutation-test",
                 "auto-mi",
                 "csv-files",
                 "plots",
                 "full-analysis"]
defined_estimation_methods = ['bbc', 'shuffling', 'all']

class estimator_env():    
    spike_times = None
    settings = None

def test_setup_env():
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_utils.yaml']

    task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
        analysis_num, settings = parse_arguments(arguments,
                                                 defined_tasks,
                                                 defined_estimation_methods)

    estimator_env.spike_times = spike_times
    estimator_env.settings = settings


## consistency checks that everything returns the same as in previous versions

def test_shuffling_estimator():
    past_range_T, number_of_bins_d, scaling_k = exp.embedding
    
    alphabet_size_past = 2 ** int(number_of_bins_d) # K for past activity
    alphabet_size = alphabet_size_past * 2          # K

    history_dependence = sh.shuffling_estimator(exp.symbol_counts,
                                                number_of_bins_d,
                                                H_uncond=utl.get_H_spiking(exp.symbol_counts),
                                                return_ais=False)
    
    assert np.isclose(history_dependence, exp.R_tot_shuffling,
                      atol=1e-3)
