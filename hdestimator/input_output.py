# ------------------------------------------------------------------------------ #
# @Created:       2023-07-31 10:52:29
# @Last Modified: 2023-07-31 17:25:56
# ------------------------------------------------------------------------------ #
# all things that do disk io are here:
# - the hdf5 or dict for the analysis details (`f`)
# - csv files with summaries
# ------------------------------------------------------------------------------ #

import logging

log = logging.getLogger("hdestimator")

import numpy as np
import h5py
import tempfile
import yaml
import io

from os import listdir, mkdir, replace
from os.path import isfile, abspath
from collections import Counter
from ast import literal_eval

# dont import from utils on top level to avoid circular imports
# from .utils import (
#     get_hash,
#     is_float,
#     get_default_settings,
#     find_existing_parameter,
# )


# ------------------------------------------------------------------------------ #
# spikes, directory structure, settings
# ------------------------------------------------------------------------------ #


def get_spike_times_from_file(file_names, hdf5_datasets=None):
    """
    Get spike times from a file (either one spike time per line, or
    a dataset in a hdf5 file.).

    Ignore lines that don't represent times, sort the spikes
    chronologically, shift spike times to start at 0 (remove any silent
    time at the beginning..).

    It is also possible to import spike times from several non-contiguous
    parts.  These can be located either from many file names or many
    hdf5 datasets within one file.  It is also possible to provide them
    from one file only by using '----------' as a delimiter.
    """

    parts_delimiter = "----------"
    spike_times_raw = []

    if not hdf5_datasets == None:
        if type(file_names) == list or type(hdf5_datasets) == list:
            if not type(file_names) == list:
                file_names = [file_names] * len(hdf5_datasets)
            if not type(hdf5_datasets) == list:
                hdf5_datasets = [hdf5_datasets] * len(file_names)
            if not len(file_names) == len(hdf5_datasets):
                log.error(
                    "Number of hdf filenames and datasets do not match. Please provide"
                    " them in a 1:n, n:1 or n:n relation."
                )
                return None
        else:
            file_names = [file_names]
            hdf5_datasets = [hdf5_datasets]

        for file_name, hdf5_dataset in zip(file_names, hdf5_datasets):
            f = h5py.File(file_name, "r")

            if not hdf5_dataset in f:
                log.error(
                    "Dataset {} not found in file {}.".format(hdf5_dataset, file_name)
                )
                return None

            spike_times_part = f[hdf5_dataset][()]
            if len(spike_times_part) > 0:
                spike_times_raw += [spike_times_part]

            f.close()

    else:
        if not type(file_names) == list:
            file_names = [file_names]

        for file_name in file_names:
            spike_times_part = []

            with open(file_name, "r") as f:
                for line in f.readlines():
                    try:
                        spike_times_part += [float(line)]
                    except:
                        if line.strip() == parts_delimiter:
                            if len(spike_times_part) > 0:
                                spike_times_raw += [spike_times_part]
                                spike_times_part = []
                        continue

            if len(spike_times_part) > 0:
                spike_times_raw += [spike_times_part]
                spike_times_part = []

            f.close()

    spike_times = []
    if len(spike_times_raw) > 0:
        for spike_times_part in spike_times_raw:
            spike_times += [np.array(sorted(spike_times_part)) - min(spike_times_part)]

        # if len(spike_times) == 1:
        #     return spike_times[0]
        # else:
        return np.array(spike_times)
    else:
        return np.array([])


def get_or_create_analysis_dir(spike_times, spike_times_file_names, root_analysis_dir):
    """
    Search for existing folder in our default location, containing associated analysis.
    """

    from .utils import get_hash

    analysis_num = -1
    analysis_dir_prefix = "ANALYSIS"
    prefix_len = len(analysis_dir_prefix)
    analysis_id_file_name = ".associated_spike_times_file"
    analysis_id = {
        "path": "\n".join(
            [
                abspath(spike_times_file_name).strip()
                for spike_times_file_name in spike_times_file_names
            ]
        ),
        "hash": get_hash(spike_times),
    }
    existing_analysis_found = False

    for d in sorted(listdir(root_analysis_dir)):
        if not d.startswith(analysis_dir_prefix):
            continue

        try:
            analysis_num = int(d[prefix_len:])
        except:
            continue

        if isfile("{}/{}/{}".format(root_analysis_dir, d, analysis_id_file_name)):

            with open(
                "{}/{}/{}".format(root_analysis_dir, d, analysis_id_file_name), "r"
            ) as analysis_id_file:
                lines = analysis_id_file.readlines()
                for line in lines:
                    if line.strip() == analysis_id["hash"]:
                        existing_analysis_found = True
            analysis_id_file.close()
        else:
            continue

        if existing_analysis_found:
            break

    if not existing_analysis_found:
        analysis_num += 1

    # if several dirs are attempted to be created in parallel
    # this might create a race condition -> test for success
    successful = False

    while not successful:
        analysis_num_label = str(analysis_num)
        if len(analysis_num_label) < 4:
            analysis_num_label = (4 - len(analysis_num_label)) * "0" + analysis_num_label

        analysis_dir = "{}/ANALYSIS{}".format(root_analysis_dir, analysis_num_label)

        if not existing_analysis_found:
            try:
                mkdir(analysis_dir)
            except:
                analysis_num += 1
                continue
            with open(
                "{}/{}".format(analysis_dir, analysis_id_file_name), "w"
            ) as analysis_id_file:
                analysis_id_file.write(
                    "{}\n{}\n".format(analysis_id["path"], analysis_id["hash"])
                )
            analysis_id_file.close()
            successful = True
        else:
            successful = True

    return analysis_dir, analysis_num, existing_analysis_found


def create_default_settings_file(ESTIMATOR_DIR="."):
    """
    Create the  default settings/parameters file, in case the one
    shipped with the tool is missing.
    """

    from .utils import get_default_settings

    settings = get_default_settings()

    with open("{}/settings/default.yaml".format(ESTIMATOR_DIR), "w") as settings_file:
        for setting_name in settings:
            if isinstance(settings[setting_name], dict):
                settings_file.write("{} :\n".format(setting_name))
                for s in settings[setting_name]:
                    if isinstance(settings[setting_name][s], str):
                        settings_file.write(
                            "    '{}' : '{}'\n".format(s, settings[setting_name][s])
                        )
                    else:
                        settings_file.write(
                            "    '{}' : {}\n".format(s, settings[setting_name][s])
                        )
            else:
                if isinstance(settings[setting_name], str):
                    settings_file.write(
                        "{} : '{}'\n".format(setting_name, settings[setting_name])
                    )
                else:
                    settings_file.write(
                        "{} : {}\n".format(setting_name, settings[setting_name])
                    )
    settings_file.close()


def load_settings_from_file(file_name):
    """
    Settings are stored as yaml,
    this is just a thin wrapper to load them.
    """

    # we need to fix some none-conformities that some of our past code relies on
    def fix_val(v):
        if isinstance(v, str):
            # yamls wants `null`, we have `None` ...
            if v.lower() == "none":
                return None
            # strip redundant quotes
            elif v.startswith("'") and v.endswith("'"):
                return v[1:-1]
            elif v.startswith('"') and v.endswith('"'):
                return v[1:-1]
            else:
                return v
        else:
            return v

    def fix_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = fix_dict(v)
            else:
                d[k] = fix_val(v)
        return d

    with open(file_name, "r") as file:
        res = yaml.load(file, Loader=yaml.SafeLoader)
        res = fix_dict(res)
        return res


# ------------------------------------------------------------------------------ #
# hdf5
# ------------------------------------------------------------------------------ #


def get_analysis_file(persistent_analysis, analysis_dir=None):
    """
    Get the hdf5 file to store the analysis in (either
    temporarily or persistently.)
    """

    analysis_file_name = "analysis_data.h5"
    if analysis_dir is None:
        assert persistent_analysis is False

    if not persistent_analysis:
        return dict()

    analysis_file = h5py.File("{}/{}".format(analysis_dir, analysis_file_name), "a")

    return analysis_file


def load_from_analysis_file(f, data_label, **data):
    """
    Load data from hdf5 file if it exists.
    """

    data_dir = get_or_create_data_directory_in_file(f, data_label, get_only=True, **data)

    if data_dir == None or data_label not in data_dir:
        return None
    elif data_label == "symbol_counts":
        try:
            symbol_counts = data_dir[data_label][()]
            # convert back from 2-col np array to Counter
            res = Counter()
            for idx in range(len(symbol_counts)):
                k = symbol_counts[idx, 0]
                v = symbol_counts[idx, 1]
                res[k] = v
            return res
        except:
            # try to keep compatibility with the old format (dict as str)
            if type(symbol_counts) == bytes:
                return Counter(literal_eval(symbol_counts.decode("utf-8")))
            else:
                return Counter(literal_eval(symbol_counts))
    else:
        return data_dir[data_label][()]


def save_to_analysis_file(f, data_label, estimation_method=None, **data):
    """
    Sava data to hdf5 file, overwrite or expand as necessary.

    # Parameters
    f : an open hdf5 file
    """

    data_dir = get_or_create_data_directory_in_file(
        f, data_label, estimation_method=estimation_method, **data
    )

    if data_label in [
        "firing_rate",
        "firing_rate_sd",
        "H_spiking",
        "recording_length",
        "recording_length_sd",
    ]:
        if not data_label in data_dir:
            _create_dataset(data_dir, data_label, data=data[data_label])

    # we might want to update the auto mutual information
    # so if already stored, delete it first
    elif data_label == "auto_MI":
        if not data_label in data_dir:
            _create_dataset(data_dir, data_label, data=data[data_label])
        else:
            del data_dir[data_label]
            _create_dataset(data_dir, data_label, data=data[data_label])

    elif data_label == "symbol_counts":
        if not data_label in data_dir:
            # the Counter is essentially a dict. and we should have
            # only integers for keys and values! -> save as a 2-column np array
            ctr = data[data_label]
            keys, values = zip(*ctr.items())
            keys = np.array(keys, dtype=np.int64)
            values = np.array(values, dtype=np.int64)
            _create_dataset(
                data_dir, data_label, data=np.array([keys, values], dtype=np.int64).T
            )

    elif data_label == "history_dependence":
        if not data_label in data_dir:
            _create_dataset(data_dir, data_label, data=data[data_label])
        if estimation_method == "bbc" and not "bbc_term" in data_dir:
            _create_dataset(data_dir, "bbc_term", data=data["bbc_term"])
        if not "first_bin_size" in data_dir:
            _create_dataset(data_dir, "first_bin_size", data=data["first_bin_size"])

    # bs: bootstrap, pt: permutation test
    # each value is stored, so that addition re-draws can be made and
    # the median/ CIs re-computed
    elif data_label in ["bs_history_dependence", "pt_history_dependence"]:
        if not data_label in data_dir:
            _create_dataset(data_dir, data_label, data=data[data_label])
        else:
            # this _should_ work for dicts ands hdf5 files.
            new_and_old_data_joint = np.hstack(
                (data_dir[data_label][()], data[data_label])
            )
            del data_dir[data_label]
            _create_dataset(data_dir, data_label, data=new_and_old_data_joint)


def get_or_create_data_directory_in_file(
    f,
    data_label,
    embedding_step_size=None,
    embedding=None,
    estimation_method=None,
    auto_MI_bin_size=None,
    get_only=False,
    cross_val=None,
    **kwargs,
):
    """
    Search for directory in hdf5, optionally create it if nonexistent
    and return it.

    # Parameters
    f: hdf5 file, or dict
        if a dict is passed, directories are created as dictionaries.
        (i.e. relevant when not saving peristently)
    """

    from .utils import find_existing_parameter

    if data_label in [
        "firing_rate",
        "firing_rate_sd",
        "H_spiking",
        "recording_length",
        "recording_length_sd",
    ]:
        root_dir = "other"
    elif data_label == "auto_MI":
        root_dir = "auto_MI"
    elif not cross_val == None:
        root_dir = "{}_embeddings".format(cross_val)
    else:
        root_dir = "embeddings"

    try:
        f.keys()
    except AttributeError:
        log.debug(
            "get_or_create_data_directory_in_file was called with non-inialized file. "
            + "Using non-persistent storage."
        )
        f = get_analysis_file(persistent_analysis=False)

    if not root_dir in f.keys():
        if get_only:
            return None
        else:
            _create_group(f, root_dir)

    data_dir = f[root_dir]

    if data_label in [
        "firing_rate",
        "firing_rate_sd",
        "H_spiking",
        "recording_length",
        "recording_length_sd",
    ]:
        return data_dir
    elif data_label == "auto_MI":
        bin_size_label, found = find_existing_parameter(
            auto_MI_bin_size, [key for key in data_dir.keys()]
        )
        if found:
            data_dir = data_dir[bin_size_label]
        else:
            if get_only:
                return None
            else:
                data_dir = _create_group(data_dir, bin_size_label)
        return data_dir
    else:
        past_range_T = embedding[0]
        number_of_bins_d = embedding[1]
        scaling_k = embedding[2]
        for parameter in [embedding_step_size, past_range_T, number_of_bins_d, scaling_k]:
            parameter_label, found = find_existing_parameter(
                parameter, [key for key in data_dir.keys()]
            )

            if found:
                data_dir = data_dir[parameter_label]
            else:
                if get_only:
                    return None
                else:
                    data_dir = _create_group(data_dir, parameter_label)

        if data_label == "symbol_counts":
            return data_dir
        else:
            if not estimation_method in data_dir:
                if get_only:
                    return None
                else:
                    _create_group(data_dir, estimation_method)
            return data_dir[estimation_method]


# thin wrappers to work with dicts or hdf5 files
def _create_dataset(dir, key, data):
    if isinstance(dir, dict):
        dir[key] = data
    else:
        if not isinstance(key, str):
            # hdf5 does not like int keys. stay aware of this when loading back!
            key = str(key)
        try:
            dir.create_dataset(key, data=data, compression="gzip")
        except TypeError:
            # some datasets do not support compression
            dir.create_dataset(key, data=data)
    return dir[key]


def _create_group(d, key):
    if isinstance(d, dict):
        d[key] = dict()
    else:
        d.create_group(key)
    return d[key]


def check_h5py_version(version, required_version):
    """
    Check version (of h5py module).
    """

    for i, j in zip(version.split("."), required_version.split(".")):
        try:
            i = int(i)
            j = int(j)
        except:
            log.warning(f"Could not check version {version} against {required_version}")
            return False

        if i > j:
            return True
        elif i == j:
            continue
        elif i < j:
            return False
    return True


# ------------------------------------------------------------------------------ #
# csv
# ------------------------------------------------------------------------------ #


def create_CSV_files(
    f,
    f_csv_stats,
    f_csv_histdep_data,
    f_csv_auto_MI_data,
    analysis_num,
    **kwargs,
):
    """
    Create three files per neuron (one for summary stats, one for
    detailed data for the history dependence plots and one for the
    auto mutual information plot), and write the respective data.

    # Parameters
    f : h5py.File or dict with analysis data
    f_csv_ ... : file handles for the three csv files, use `get_CSV_files`

    """

    from .utils import get_analysis_stats, get_histdep_data, get_auto_MI_data

    stats = get_analysis_stats(f, analysis_num, **kwargs)

    f_csv_stats.write("#{}\n".format(",".join(stats.keys())))
    f_csv_stats.write("{}\n".format(",".join(stats.values())))

    histdep_data = get_histdep_data(f, analysis_num, **kwargs)

    f_csv_histdep_data.write("#{}\n".format(",".join(histdep_data.keys())))
    histdep_data_m = np.array([vals for vals in histdep_data.values()])
    for line_num in range(np.size(histdep_data_m, axis=1)):
        f_csv_histdep_data.write("{}\n".format(",".join(histdep_data_m[:, line_num])))

    auto_MI_data = get_auto_MI_data(f, analysis_num, **kwargs)

    f_csv_auto_MI_data.write("#{}\n".format(",".join(auto_MI_data.keys())))
    auto_MI_data_m = np.array([vals for vals in auto_MI_data.values()])
    for line_num in range(np.size(auto_MI_data_m, axis=1)):
        f_csv_auto_MI_data.write("{}\n".format(",".join(auto_MI_data_m[:, line_num])))


def get_CSV_files(task, persistent_analysis, analysis_dir):
    """
    Create csv files for create_CSV_files, back up existing ones.
    """

    if not persistent_analysis:
        f_csv_stats = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".csv", prefix="statistics"
        )
        f_csv_histdep_data = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".csv", prefix="histdep_data"
        )
        f_csv_auto_MI_data = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".csv", prefix="auto_MI_data"
        )
        return f_csv_stats, f_csv_histdep_data, f_csv_auto_MI_data

    csv_stats_file_name = "statistics.csv"
    csv_histdep_data_file_name = "histdep_data.csv"
    csv_auto_MI_data_file_name = "auto_MI_data.csv"

    if task == "csv-files" or task == "full-analysis":
        file_mode = "w+"
        # backup existing files (overwrites old backups)
        for f_csv in [
            csv_stats_file_name,
            csv_histdep_data_file_name,
            csv_auto_MI_data_file_name,
        ]:
            if isfile("{}/{}".format(analysis_dir, f_csv)):
                replace(
                    "{}/{}".format(analysis_dir, f_csv),
                    "{}/{}.old".format(analysis_dir, f_csv),
                )
    elif task == "plots":
        file_mode = "r"

        files_missing = False
        for f_csv in [
            csv_stats_file_name,
            csv_histdep_data_file_name,
            csv_auto_MI_data_file_name,
        ]:
            if not isfile("{}/{}".format(analysis_dir, f_csv)):
                files_missing = True
        if files_missing:
            return None, None, None
    else:
        return None, None, None

    f_csv_stats = open("{}/{}".format(analysis_dir, csv_stats_file_name), file_mode)
    f_csv_histdep_data = open(
        "{}/{}".format(analysis_dir, csv_histdep_data_file_name), file_mode
    )
    f_csv_auto_MI_data = open(
        "{}/{}".format(analysis_dir, csv_auto_MI_data_file_name), file_mode
    )

    return f_csv_stats, f_csv_histdep_data, f_csv_auto_MI_data


def load_from_CSV_file(csv_file, data_label):
    """
    Get all data of a column in a csv file.
    """

    from .utils import is_float

    csv_file.seek(0)  # jump to start of file

    lines = (line for line in csv_file.readlines())
    header = next(lines)

    data_index = get_data_index_from_CSV_header(header, data_label)

    data = []
    for line in lines:
        datum = line.split(",")[data_index]
        if is_float(datum):
            data += [float(datum)]
        elif data_label == "label":
            data += [datum]
        else:
            data += [np.nan]

    if len(data) == 1:
        return data[0]
    else:
        return data


def load_auto_MI_data(f_csv_auto_MI_data):
    """
    Load the data from the auto MI csv file as needed for the plot.
    """

    auto_MI_bin_sizes = load_from_CSV_file(f_csv_auto_MI_data, "auto_MI_bin_size")
    delays = np.array(load_from_CSV_file(f_csv_auto_MI_data, "delay"))
    auto_MIs = np.array(load_from_CSV_file(f_csv_auto_MI_data, "auto_MI"))

    auto_MI_data = {}
    for auto_MI_bin_size in np.unique(auto_MI_bin_sizes):
        indices = np.where(auto_MI_bin_sizes == auto_MI_bin_size)
        auto_MI_data[auto_MI_bin_size] = (
            [float(delay) for delay in delays[indices]],
            [float(auto_MI) for auto_MI in auto_MIs[indices]],
        )

    return auto_MI_data


def get_data_index_from_CSV_header(header, data_label):
    """
    Get column index to reference data within a csv file.
    """

    header = header.strip()
    if header.startswith("#"):
        header = header[1:]
    labels = header.split(",")
    for index, label in enumerate(labels):
        if label == data_label:
            return index
    return np.nan
