from estimate import main

from os import mkdir, listdir
from pathlib import Path
import pytest

## setup test environment

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
analysis_dir = 'tests/analysis'
output_file_name = 'tests/test_output.pdf'
spike_times_file_name = 'sample_data/spike_times.dat'

# we don't create analysis dir using tempfile
# so that we can reference it from the settings
p = Path(analysis_dir)
if not p.is_dir():
    mkdir(analysis_dir)
ls = listdir(analysis_dir)
assert ls == ['ANALYSIS0000'] or ls == []
for d in [D for D in p.iterdir() if D.is_dir()]:
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
assert listdir(analysis_dir) == []

output_file_path = Path(output_file_name)
output_file_path.unlink(missing_ok=True)


## perform tests

def test_main():
    output_file_path.unlink(missing_ok=True)
    
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf']

    assert main(arguments) == EXIT_SUCCESS
    output_file_path.unlink(missing_ok=False)

def test_command_line_params():
    output_file_path.unlink(missing_ok=True)

    # bbc estimator
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'bbc']

    assert main(arguments) == EXIT_SUCCESS
    output_file_path.unlink(missing_ok=False)

    # hdf5 data
    arguments = ['sample_data/spike_times.h5',
                 '-h5', 'spt',
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'shuffling']

    assert main(arguments) == EXIT_SUCCESS
    output_file_path.unlink(missing_ok=False)

    # each task individually
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'shuffling',
                 '-t', 'history-dependence']
    assert main(arguments) == EXIT_SUCCESS

    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'shuffling',
                 '-t', 'confidence-intervals']
    assert main(arguments) == EXIT_SUCCESS

    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'shuffling',
                 '-t', 'auto-mi']
    assert main(arguments) == EXIT_SUCCESS

    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'shuffling',
                 '-t', 'csv-files']
    assert main(arguments) == EXIT_SUCCESS

    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml',
                 '--output', 'tests/test_output.pdf',
                 '-e', 'shuffling',
                 '-t', 'plots']
    assert main(arguments) == EXIT_SUCCESS
    output_file_path.unlink(missing_ok=False)

def test_crossval(): # also tests persistent_analysis: False
    output_file_path.unlink(missing_ok=True)
    
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_crossval.yaml',
                 '--output', 'tests/test_output.pdf']

    assert main(arguments) == EXIT_SUCCESS
    output_file_path.unlink(missing_ok=False)

def test_non_averaged_R():
    output_file_path.unlink(missing_ok=True)
    
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_non_averaged_R.yaml',
                 '--output', 'tests/test_output.pdf']

    assert main(arguments) == EXIT_SUCCESS
    output_file_path.unlink(missing_ok=False)

def test_error_no_analysis_dir():
    ## cleanup test environment

    ls = listdir(analysis_dir)
    p = Path(analysis_dir)
    for d in [D for D in p.iterdir() if D.is_dir()]:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    p.rmdir()

    assert not p.is_dir()

    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_main.yaml']

    with pytest.raises(SystemExit) as pytest_e:
        main(arguments)
    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == EXIT_FAILURE
