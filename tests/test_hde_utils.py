from estimate import parse_arguments
import hde_utils as utl

from pathlib import Path
import ast
import yaml
import h5py
import pytest

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
                 '--settings-file', 'tests/settings/test_utils.yaml',
                 '--output', 'tests/test_output.pdf']

    task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
        analysis_num, settings = parse_arguments(arguments,
                                                 defined_tasks,
                                                 defined_estimation_methods)

    assert task == "full-analysis"
    assert utl.get_hash(spike_times) == utl.get_hash(estimator_env.spike_times)
    assert type(analysis_file) == h5py.File
    for f in [csv_stats_file,
              csv_histdep_data_file,
              csv_auto_MI_data_file]:
        p = Path(f.name)
        assert p.is_file()

    assert settings["embedding_past_range_set"] == [0.005, 0.00998, 0.15811, 1.25594, 5.0]
    assert settings["embedding_number_of_bins_set"] == [1, 3, 5]
    assert settings["estimation_method"] == 'all'

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
                 '--settings-file', 'tests/settings/test_utils.yaml',
                 '--output', 'tests/test_output.pdf']
    
    with pytest.raises(SystemExit) as pytest_e:
        task, spike_times, spike_times_optimization, spike_times_validation, \
            analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
            analysis_num, settings = parse_arguments(arguments,
                                                     defined_tasks,
                                                     defined_estimation_methods)
    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == EXIT_FAILURE

