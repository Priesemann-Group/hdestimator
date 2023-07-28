from sys import path
from os.path import realpath, dirname
path.insert(1, dirname(realpath(__file__)))

from estimate import parse_arguments
from hdestimator import utils as utl
from hdestimator import api as hapi

import expected_output as exp

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
    settings = None

def test_setup_env():
    arguments = [spike_times_file_name,
                 '--settings-file', 'tests/settings/test_utils.yaml']

    task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, \
        analysis_num, settings = parse_arguments(arguments,
                                                 defined_tasks,
                                                 defined_estimation_methods)

    estimator_env.spike_times = spike_times[0]
    estimator_env.settings = settings

## consistency checks that everything returns the same as in previous versions

class TestGetHistoryDependence():
    def test_bbc(self):
        past_range_T, number_of_bins_d, scaling_k = exp.embedding

        assert np.isclose([exp.R_tot_bbc, exp.bbc_term],
                          hapi.get_history_dependence("bbc",
                                                      exp.symbol_counts,
                                                      number_of_bins_d)).all()
    def test_shuffling(self):
        past_range_T, number_of_bins_d, scaling_k = exp.embedding

        assert np.isclose(exp.R_tot_shuffling,
                          hapi.get_history_dependence("shuffling",
                                                      exp.symbol_counts,
                                                      number_of_bins_d),
                          atol=1e-3)


class TestGetHistoryDependenceForSingleEmbedding():
    def test_bbc(self):
        assert np.isclose([exp.R_tot_bbc, exp.bbc_term],
                          hapi.get_history_dependence_for_single_embedding(estimator_env.spike_times,
                                                                           exp.recording_length,
                                                                           "bbc",
                                                                           exp.embedding,
                                                                           estimator_env.settings["embedding_step_size"])).all()

    def test_shuffling(self):
        assert np.isclose(exp.R_tot_shuffling,
                          hapi.get_history_dependence_for_single_embedding(estimator_env.spike_times,
                                                                           exp.recording_length,
                                                                           "shuffling",
                                                                           exp.embedding,
                                                                           estimator_env.settings["embedding_step_size"]),
                          atol=1e-3)

class TestGetHistoryDependenceForEmbeddingSet():
    def test_bbc(self):
        embeddings_that_maximise_R, max_Rs \
            = hapi.get_history_dependence_for_embedding_set(estimator_env.spike_times,
                                                            exp.recording_length,
                                                            "bbc",
                                                            exp.embedding_past_range_set,
                                                            exp.embedding_number_of_bins_set,
                                                            exp.embedding_scaling_exponent_set,
                                                            estimator_env.settings["embedding_step_size"])

        max_R, max_R_T = utl.get_max_R_T(max_Rs)
        number_of_bins_d, scaling_k = embeddings_that_maximise_R[max_R_T]

        assert len(max_Rs) == len(exp.embedding_past_range_set)
        assert len(embeddings_that_maximise_R) == len(exp.embedding_past_range_set)
        for T in max_Rs:
            assert T in exp.max_Rs_bbc
            assert np.isclose(max_Rs[T], exp.max_Rs_bbc[T])
        assert np.isclose(max_R, exp.R_tot_bbc)
        assert np.all(
            np.isclose((max_R_T, number_of_bins_d, scaling_k), exp.embedding, atol=1e-8)
        )

    def test_shuffling(self):
        embeddings_that_maximise_R, max_Rs \
            = hapi.get_history_dependence_for_embedding_set(estimator_env.spike_times,
                                                            exp.recording_length,
                                                            "shuffling",
                                                            exp.embedding_past_range_set,
                                                            exp.embedding_number_of_bins_set,
                                                            exp.embedding_scaling_exponent_set,
                                                            estimator_env.settings["embedding_step_size"])

        max_R, max_R_T = utl.get_max_R_T(max_Rs)
        number_of_bins_d, scaling_k = embeddings_that_maximise_R[max_R_T]

        assert len(max_Rs) == len(exp.embedding_past_range_set)
        assert len(embeddings_that_maximise_R) == len(exp.embedding_past_range_set)
        for T in max_Rs:
            assert T in exp.max_Rs_shuffling
            assert np.isclose(max_Rs[T], exp.max_Rs_shuffling[T], atol=1e-3)
        assert np.isclose(max_R, exp.R_tot_shuffling, atol=1e-3)
        assert np.all(
            np.isclose((max_R_T, number_of_bins_d, scaling_k), exp.embedding, atol=1e-8)
        )



class TestGetCIForEmbedding():
    def test_bbc(self):
        CI_lo, CI_hi = hapi.get_CI_for_embedding(exp.R_tot_bbc,
                                                 estimator_env.spike_times,
                                                 "bbc",
                                                 exp.embedding,
                                                 estimator_env.settings["embedding_step_size"],
                                                 100)

        assert np.isclose([CI_lo, CI_hi], [exp.CI_lo, exp.CI_hi], atol=1e-3).all()

    def test_shuffling(self):
        CI_lo, CI_hi = hapi.get_CI_for_embedding(exp.R_tot_bbc,
                                                 estimator_env.spike_times,
                                                 "shuffling",
                                                 exp.embedding,
                                                 estimator_env.settings["embedding_step_size"],
                                                 100)

        assert np.isclose([CI_lo, CI_hi], [exp.CI_lo, exp.CI_hi], atol=1e-3).all()
