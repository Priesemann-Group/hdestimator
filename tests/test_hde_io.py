from sys import path
from os.path import realpath, dirname

path.insert(1, dirname(realpath(__file__)))

from pathlib import Path
import yaml
import h5py
import pytest
import numpy as np
import numbers
import tempfile

np.random.seed(42)

from hdestimator import input_output as io
from hdestimator import utils as utl
from hdestimator import api as hapi
from estimate import do_main_analysis

import expected_output as exp

spike_times_file_name = "sample_data/spike_times.dat"


def test_hdf5_persistent_vs_dict():
    """
    The option to avoid all the disk io was added later. check things are consistent
    """

    # settings for some fast results
    settings = utl.get_default_settings()
    settings["ANALYSIS_DIR"] = tempfile.mkdtemp()
    settings["estimation_method"] = "bbc"
    settings["cross_val"] = None
    settings["number_of_bootstraps_R_max"] = 0
    settings["number_of_bootstraps_R_tot"] = 0
    settings["embedding_past_range_set"] = [0.005, 0.00561, 0.00629, 0.00706, 0.00792]

    settings["persistent_analysis"] = True

    spike_times = io.get_spike_times_from_file(spike_times_file_name)
    # boilerplate, needed just to get it running
    spike_times_optimization = np.array([], dtype=object)
    spike_times_validation = np.array([], dtype=object)

    f = utl.get_analysis_file(
        persistent_analysis=True,
        analysis_dir=settings["ANALYSIS_DIR"]
    )

    assert isinstance(f, h5py.File)

    # this guy should also modify analysis_file in place!
    res_via_disk = do_main_analysis(
        spike_times,
        spike_times_optimization,
        spike_times_validation,
        f,
        settings,
    )

    # non-persistent
    settings = settings.copy()
    settings["persistent_analysis"] = False
    d = utl.get_analysis_file(
        persistent_analysis=False, analysis_dir=None
    )

    assert isinstance(d, dict)

    res_via_dict = do_main_analysis(
        spike_times,
        spike_times_optimization,
        spike_times_validation,
        d,
        settings,
    )


    wrong_keys = _find_missmatches(res_via_dict, res_via_disk)
    assert len(wrong_keys) == 0



def _find_missmatches(d1, d2):
    """
    Take two dict-like objects and compare them recursively.

    returns keys that are not equal
    """

    wrong_keys = []
    # compare keys
    if d1.keys() != d2.keys():
        # return the keys that are not equal
        for k in d1.keys():
            if k not in d2.keys():
                wrong_keys.extend(k)
                print(f"key {k} not in d2")
        for k in d2.keys():
            if k not in d1.keys():
                wrong_keys.extend(k)
                print(f"key {k} not in d1")

    # compare values
    for k in d1.keys():
        if isinstance(d1[k], dict):
            wrong_here = _find_missmatches(d1[k], d2[k])
            if len(wrong_here) > 0:
                print(f"missmatch in subdir of {k}")
            wrong_keys.extend(wrong_here)
        else:
            # if value is numeric, do a numerical comparison
            if isinstance(d1[k], numbers.Number):
                if not np.isclose(d1[k], d2[k]):
                    wrong_keys.extend(k)
                    print(f"numeric missmatch for {k}")
            else:
                if type(d1[k]) != type(d2[k]) and not isinstance(d1[k], (np.ndarray, h5py.Dataset)):
                    # arrays and h5 datasets are okay
                    wrong_keys.extend(k)
                    print(f"type missmatch for {k}")
                elif np.any(d1[k] != d2[k]):
                    wrong_keys.extend(k)
                    print(f"== missmatch for {k}:")
                    print(f"  {d1[k]} vs. {d2[k]}")

    return wrong_keys
