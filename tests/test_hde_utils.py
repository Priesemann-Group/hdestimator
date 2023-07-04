from estimate import parse_arguments
from ..hdestimator import utils as utl

from sys import path
from os.path import realpath, dirname
path.insert(1, dirname(realpath(__file__)))
import expected_output as exp

from pathlib import Path
import ast
import yaml
import h5py
import pytest
import numpy as np

np.random.seed(42)

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
    spike_times_optimization = None
    spike_times_validation = None
    analysis_file = None
    csv_stats_file = None
    csv_histdep_data_file = None
    csv_auto_MI_data_file = None
    analysis_num = None
    settings = None

def test_get_spike_times_from_file():
    estimator_env.spike_times = utl.get_spike_times_from_file(spike_times_file_name)
    assert len(estimator_env.spike_times) > 0

def test_find_existing_analysis():
    analysis_dir, analysis_num, existing_analysis_found \
        = utl.get_or_create_analysis_dir(estimator_env.spike_times,
                                         spike_times_file_name,
                                         'analysis')

    assert existing_analysis_found

def test_create_default_settings():
    utl.create_default_settings_file(ESTIMATOR_DIR="tests/")
    settings_file_path = Path("tests/settings/default.yaml")
    assert settings_file_path.is_file()

    with open("tests/settings/default.yaml", 'r') as created_settings_file:
        created_settings = yaml.load(created_settings_file, Loader=yaml.BaseLoader)

    with open("settings/default.yaml", 'r') as default_settings_file:
        default_settings = yaml.load(default_settings_file, Loader=yaml.BaseLoader)

    assert created_settings == default_settings

def test_argument_parser():
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_utils.yaml']

    task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
        analysis_num, settings = parse_arguments(arguments,
                                                 defined_tasks,
                                                 defined_estimation_methods)

    assert task == "full-analysis"
    assert np.isclose(spike_times, estimator_env.spike_times).all()
    assert utl.get_hash(spike_times) == utl.get_hash(estimator_env.spike_times)
    assert type(analysis_file) == h5py.File
    for f in [csv_stats_file,
              csv_histdep_data_file,
              csv_auto_MI_data_file]:
        p = Path(f.name)
        assert p.is_file()

    assert settings["embedding_past_range_set"] == exp.embedding_past_range_set
    assert settings["embedding_number_of_bins_set"] == exp.embedding_number_of_bins_set
    assert settings["embedding_scaling_exponent_set"] == exp.embedding_scaling_exponent_set
    assert settings["estimation_method"] == exp.estimation_method

    estimator_env.spike_times_optimization = spike_times_optimization
    estimator_env.spike_times_validation = spike_times_validation
    estimator_env.analysis_file = analysis_file
    estimator_env.csv_stats_file = csv_stats_file
    estimator_env.csv_histdep_data_file = csv_histdep_data_file
    estimator_env.csv_auto_MI_data_file = csv_auto_MI_data_file
    estimator_env.analysis_num = analysis_num
    estimator_env.settings = settings

    # assert error catching
    # no arguments
    arguments = []

    with pytest.raises(SystemExit) as pytest_e:
        task, spike_times, spike_times_optimization, spike_times_validation, \
            analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
            analysis_num, settings = parse_arguments(arguments,
                                                     defined_tasks,
                                                     defined_estimation_methods)
    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 2 # might want to remove this line

    # wrong task (non-existing spike times file)
    arguments = ['tests/asdfg.dat']

    with pytest.raises(SystemExit) as pytest_e:
        task, spike_times, spike_times_optimization, spike_times_validation, \
            analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
            analysis_num, settings = parse_arguments(arguments,
                                                     defined_tasks,
                                                     defined_estimation_methods)
    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == EXIT_FAILURE

    # wrong task (too short -> not unique, could be confidence-intervals or csv-files)
    arguments = [spike_times_file_name,
                 '-t', 'c',
                 '--settings-file', 'tests/settings/test_utils.yaml']

    with pytest.raises(SystemExit) as pytest_e:
        task, spike_times, spike_times_optimization, spike_times_validation, \
            analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
            analysis_num, settings = parse_arguments(arguments,
                                                     defined_tasks,
                                                     defined_estimation_methods)
    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == EXIT_FAILURE

    # max number of bins d too large, has to be < 63
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_max_num_bins.yaml']

    with pytest.raises(SystemExit) as pytest_e:
        task, spike_times, spike_times_optimization, spike_times_validation, \
            analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
            analysis_num, settings = parse_arguments(arguments,
                                                     defined_tasks,
                                                     defined_estimation_methods)
    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == EXIT_FAILURE


## consistency checks that everything returns the same as in previous versions

def test_save_spike_times_stats():
    utl.save_spike_times_stats(estimator_env.analysis_file,
                               estimator_env.spike_times,
                               **estimator_env.settings)


    assert np.isclose(utl.load_from_analysis_file(estimator_env.analysis_file,
                                                  "recording_length"), exp.recording_length)
    assert np.isclose(utl.load_from_analysis_file(estimator_env.analysis_file,
                                                  "firing_rate"), exp.firing_rate)
    assert np.isclose(utl.load_from_analysis_file(estimator_env.analysis_file,
                                                  "H_spiking"), exp.H_spiking)

def test_get_past_symbol_counts():
    past_symbol_counts = utl.get_past_symbol_counts(exp.symbol_counts)
    assert past_symbol_counts == exp.past_symbol_counts

def test_history_dependence_estimation():
    estimator_env.settings['cross_val'] = None

    for estimation_method in ['bbc', 'shuffling']:
        estimator_env.settings['estimation_method'] = estimation_method

        utl.save_history_dependence_for_embeddings(estimator_env.analysis_file,
                                                   estimator_env.spike_times,
                                                   **estimator_env.settings)
        utl.compute_CIs(estimator_env.analysis_file,
                        estimator_env.spike_times, target_R='R_max',
                        **estimator_env.settings)

    utl.create_CSV_files(estimator_env.analysis_file,
                         estimator_env.csv_stats_file,
                         estimator_env.csv_histdep_data_file,
                         estimator_env.csv_auto_MI_data_file,
                         estimator_env.analysis_num,
                         **estimator_env.settings)

    check_parameters()

def check_parameters():
    # H_spiking is approx 0.1, so we use this as a factor for the tolerance

    # for the bbc estimator results should also be the same -> low error tolerance
    assert np.isclose(exp.tau_R_bbc, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                            "tau_R_bbc"))
    assert np.isclose(exp.T_D_bbc, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                          "T_D_bbc"))
    assert np.isclose(exp.R_tot_bbc, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                            "R_tot_bbc"))
    assert np.isclose(exp.AIS_tot_bbc, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                              "AIS_tot_bbc"))
    assert np.isclose(exp.opt_number_of_bins_d_bbc,
                      utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                             "opt_number_of_bins_d_bbc"))
    assert np.isclose(exp.opt_scaling_k_bbc,
                      utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                             "opt_scaling_k_bbc"))
    assert np.isclose(exp.opt_first_bin_size_bbc,
                      utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                             "opt_first_bin_size_bbc"))

    # the shuffling estimator has some stochasticity -> be a bit more tolerant
    assert np.isclose(exp.tau_R_shuffling, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                                  "tau_R_shuffling"))
    assert np.isclose(exp.T_D_shuffling, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                                "T_D_shuffling"))
    assert np.isclose(exp.R_tot_shuffling, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                                  "R_tot_shuffling"),
                      atol=1e-3)
    assert np.isclose(exp.AIS_tot_shuffling, utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                                                    "AIS_tot_shuffling"),
                      atol=1e-4)
    assert np.isclose(exp.opt_number_of_bins_d_shuffling,
                      utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                             "opt_number_of_bins_d_shuffling"))
    assert np.isclose(exp.opt_scaling_k_shuffling,
                      utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                             "opt_scaling_k_shuffling"))
    assert np.isclose(exp.opt_first_bin_size_shuffling,
                      utl.load_from_CSV_file(estimator_env.csv_stats_file,
                                             "opt_first_bin_size_shuffling"))

