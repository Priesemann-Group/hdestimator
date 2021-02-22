from estimate import parse_arguments
import hde_utils as utl
import hde_embedding as emb

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
from collections import Counter

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

    FAST_EMBEDDING_AVAILABLE = False

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

def test_cython_implementation_available():
    assert emb.FAST_EMBEDDING_AVAILABLE

    if(emb.FAST_EMBEDDING_AVAILABLE):
        estimator_env.FAST_EMBEDDING_AVAILABLE = True

def test_get_embeddings():
    embeddings = emb.get_embeddings(estimator_env.settings["embedding_past_range_set"],
                                    estimator_env.settings["embedding_number_of_bins_set"],
                                    estimator_env.settings["embedding_scaling_exponent_set"])
    assert len(embeddings) == exp.num_embeddings
    assert exp.embedding in embeddings


class TestGetSymbolCounts():
    def test_cython_implementation(self):
        # skip if cython module is not available,
        assert estimator_env.FAST_EMBEDDING_AVAILABLE
        emb.FAST_EMBEDDING_AVAILABLE = True

        symbol_counts = emb.get_symbol_counts(estimator_env.spike_times[0],
                                              exp.embedding,
                                              estimator_env.settings["embedding_step_size"])
        
        assert sum(symbol_counts.values()) == sum(exp.symbol_counts.values())
        assert symbol_counts == exp.symbol_counts

    def test_pure_python_implementation(self):
        emb.FAST_EMBEDDING_AVAILABLE = False

        symbol_counts = emb.get_symbol_counts(estimator_env.spike_times[0],
                                              exp.embedding,
                                              estimator_env.settings["embedding_step_size"])

        assert sum(symbol_counts.values()) == sum(exp.symbol_counts.values())
        assert symbol_counts == exp.symbol_counts
