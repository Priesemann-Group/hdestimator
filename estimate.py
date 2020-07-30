import argparse
from sys import exit, stderr, argv, path
from os.path import isfile, isdir, realpath, dirname, exists
import ast
import yaml
import numpy as np

ESTIMATOR_DIR = dirname(realpath(__file__))
path.insert(1, '{}/src'.format(ESTIMATOR_DIR))

import hde_utils as utl
import hde_visualization as vsl

__version__ = "unknown"
from _version import __version__

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

def do_main_analysis(spike_times, spike_times_optimization, spike_times_validation,
                     analysis_file, settings):
    """
    Determine the history dependence of a neuron's activity based on
    spike time data.
    """

    utl.save_spike_times_stats(analysis_file, spike_times, **settings)
    
    if settings['cross_validated_optimization']:
        settings['cross_val'] = 'h1' # first half of the data
        utl.save_history_dependence_for_embeddings(analysis_file,
                                                   spike_times_optimization,
                                                   **settings)

        settings['cross_val'] = 'h2' # second half of the data
        utl.save_history_dependence_for_embeddings(analysis_file,
                                                   spike_times_validation,
                                                   **settings)
    else:
        settings['cross_val'] = None
        utl.save_history_dependence_for_embeddings(analysis_file,
                                                   spike_times, **settings)

def compute_CIs(spike_times, analysis_file, settings):
    """
    Compute bootstrap replications of the history-dependence estimate
    which can be used to obtain confidence intervals.
    """

    if settings['cross_validated_optimization']:
        settings['cross_val'] = 'h2' # second half of the data
    else:
        settings['cross_val'] = None
    
    utl.compute_CIs(analysis_file, spike_times, **settings)

# def perform_permutation_test(analysis_file, settings):
#     """
#     Perform a permutation test to check whether the history dependece 
#     in the target neuron is significantly different from zero.
#     """

#     utl.perform_permutation_test(analysis_file, **settings)

def analyse_auto_MI(spike_times, analysis_file, settings):
    """
    Compute the auto mutual information in the neuron's activity, a 
    measure closely related to history dependence.
    """

    utl.analyse_auto_MI(analysis_file, spike_times, **settings)

def create_CSV_files(analysis_file,
                     csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                     analysis_num, settings):
    """
    Export the data resulting from the analysis as csv files.
    """

    if settings['cross_validated_optimization']:
        settings['cross_val'] = 'h2' # second half of the data
    else:
        settings['cross_val'] = None
    
    utl.create_CSV_files(analysis_file,
                         csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                         analysis_num, **settings)

def produce_plots(spike_times, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                  settings):
    """
    Produce plots that visualize the results.
    """
    
    vsl.produce_plots(spike_times,
                      csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                      **settings)
    
# parse arguments received via the command line and check for validity
def parse_arguments(defined_tasks, defined_estimation_methods):
    """
    Parse the arguments passed to the script via the command line.

    Import settings from file, do some sanity checks to avoid faulty runs.
    """
    
    # parse arguments
    parser = argparse.ArgumentParser(description=
        """
    History dependence estimator, v. {}

    Estimate the history dependence and temporal depth of a single
    neuron, based on information-theoretical measures for spike time
    data, as presented in (Rudelt et al, in prep.).  Parameters can be
    passed via the command line or through files, where command line
    options are prioritised over those passed by file.  (If none are
    supplied, settings are read from the 'default.yaml' file.)  A user
    new to this tool is encouraged to run

      python3 {} sample_data/spike_times.dat -o sample_output.pdf \\
        -s settings/test.yaml

    to test the functionality of this tool.  A more detailed
    description can be found in the guide provided with the tool.
        """.format(__version__, argv[0]), formatter_class=argparse.RawDescriptionHelpFormatter)
    optional_arguments = parser._action_groups.pop()
    
    required_arguments = parser.add_argument_group("required arguments")
    required_arguments.add_argument('spike_times_file', action="store", help="Define file from which to read spike times and on which to perform the analysis.  The file should contain one spike time per line.")

    optional_arguments.add_argument("-t", "--task", metavar="TASK", action="store", help="Define task to be performed.  One of {}.  Per default, the full analysis is performed.".format(defined_tasks),
                                    default="full-analysis")
    optional_arguments.add_argument("-e", "--estimation-method", metavar="EST_METHOD", action="store", help="Specify estimation method for the analysis, one of {}.".format(defined_estimation_methods))
    optional_arguments.add_argument("-h5", "--hdf5-dataset", action="store", help="Load data stored in a dataset in a hdf5 file.")
    optional_arguments.add_argument("-o", "--output", metavar="IMAGE_FILE", action="store", help="Save the output image to file.")
    optional_arguments.add_argument("-p", "--persistent", action="store_true", help="Save the analysis to file.  If an existing analysis is found, read it from file.")
    optional_arguments.add_argument("-s", "--settings-file", metavar="SETTINGS_FILE", action="store", help="Specify yaml file from which to load custom settings.")
    optional_arguments.add_argument("-l", "--label", metavar="LABEL", action="store", help="Include a label in the output to classify the analysis.")
    optional_arguments.add_argument("-v", "--verbose", action="store_true", help="Print more info at run time.")
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    # check that parsed arguments are valid

    task = args.task.lower()
    spike_times_file_name = args.spike_times_file

    task_found = False
    task_full_name = ""
    for defined_task in defined_tasks:
        if defined_task.startswith(task):
            if not task_found:
                task_found = True
                task_full_name = defined_task
            else:
                print("Task could not be uniquely determined.  Task must be one of {}.  Aborting.".format(defined_tasks), file=stderr, flush=True)
                exit(EXIT_FAILURE)

    task = task_full_name
                
    if not task in defined_tasks:
        print("Task must be one of {}.  Aborting.".format(defined_tasks), file=stderr, flush=True)
        exit(EXIT_FAILURE)

    if not exists(spike_times_file_name):
        print("Spike times file {} not found.  Aborting.".format(spike_times_file_name),
              file=stderr, flush=True)
        exit(EXIT_FAILURE)

    spike_times = utl.get_spike_times_from_file(spike_times_file_name,
                                                args.hdf5_dataset)

    if not isinstance(spike_times, np.ndarray):
        print("Error loading spike times. Aborting.",
              file=stderr, flush=True)
        exit(EXIT_FAILURE)
    
    #
    # PARSE SETTINGS
    #
        
    # create default settings file if it does not exist:
    if not isfile('{}/settings/default.yaml'.format(ESTIMATOR_DIR)):
        utl.create_default_settings_file(ESTIMATOR_DIR)

    # load default settings
    with open('{}/settings/default.yaml'.format(ESTIMATOR_DIR), 'r') as default_settings_file:
        settings = yaml.load(default_settings_file, Loader=yaml.BaseLoader)

    # overwrite default settings with custom ones
    if not args.settings_file is None:
        if not isfile(args.settings_file):
            print("Error: Settings file {} not found. Aborting.".format(args.settings_file),
                  file=stderr, flush=True)
            exit(EXIT_FAILURE)
        with open(args.settings_file, 'r') as custom_settings_file:
            custom_settings = yaml.load(custom_settings_file, Loader=yaml.BaseLoader)
        for setting_key in settings:
            if setting_key in custom_settings:
                settings[setting_key] = custom_settings[setting_key]

    if args.persistent:
        settings['persistent_analysis'] = "True"
    if args.verbose:
        settings['verbose_output'] = "True"

    if not args.estimation_method is None:
        settings['estimation_method'] = args.estimation_method

    if not 'block_length_l' in settings:
        settings['block_length_l'] = "None"

    # check that required settings are defined
    required_parameters = ['embedding_past_range_set', 'embedding_number_of_bins_set',
                           'embedding_scaling_exponent_set', 'embedding_step_size',
                           'bbc_tolerance',
                           'number_of_bootstraps', 'number_of_bootstraps_nonessential',
                           'block_length_l',
                           'bootstrap_CI_percentile_lo',
                           'bootstrap_CI_percentile_hi',
                           # 'number_of_permutations',
                           'auto_MI_bin_size_set',
                           'auto_MI_max_delay']
    
    required_settings = ['estimation_method', 'plot_AIS',
                         'ANALYSIS_DIR', 'persistent_analysis',
                         'cross_validated_optimization',
                         'bootstrap_CI_use_sd',
                         'verbose_output',
                         'plot_settings', 'plot_color'] + required_parameters
    
    for required_setting in required_settings:
        if not required_setting in settings:
            print("Error in settings file: {} is not defined. Aborting.".format(required_setting),
                  file=stderr, flush=True)
            exit(EXIT_FAILURE)

    # sanity check for the settings
    if not settings['estimation_method'] in defined_estimation_methods:
        print("Error: estimation_method must be one of {}. Aborting.".format(defined_estimation_methods),
              file=stderr, flush=True)
        exit(EXIT_FAILURE)

    # evaluate settings (turn strings into booleans etc if applicable)
    for setting_key in ['persistent_analysis',
                        'verbose_output',
                        'cross_validated_optimization',
                        'bootstrap_CI_use_sd',
                        'plot_AIS']:
        settings[setting_key] = ast.literal_eval(settings[setting_key])
    for plot_setting in settings['plot_settings']:
        try:
            settings['plot_settings'][plot_setting] \
                = ast.literal_eval(settings['plot_settings'][plot_setting])
        except:
            continue

    for parameter_key in required_parameters:
        if isinstance(settings[parameter_key], list):
            settings[parameter_key] = [ast.literal_eval(element)
                                       for element in settings[parameter_key]]
        elif parameter_key == 'embedding_scaling_exponent_set' \
             and isinstance(settings['embedding_scaling_exponent_set'], dict):
            # embedding_scaling_exponent_set can be passed either as a
            # list, in which case it is evaluated as such or it can be
            # passed by specifying three parameters that determine how
            # many scaling exponents should be used.  In the latter case, the
            # uniform embedding as well as the embedding for which
            # the first bin has a length of min_first_bin_size (in
            # seconds) are used, as well as linearly spaced scaling
            # factors in between, such that in total
            # number_of_scalings scalings are used

            for key in settings['embedding_scaling_exponent_set']:
                settings['embedding_scaling_exponent_set'][key] \
                    = ast.literal_eval(settings['embedding_scaling_exponent_set'][key])
        else:
            settings[parameter_key] = ast.literal_eval(settings[parameter_key])

    # if the user specifies a file in which to store output image:
    # store this in settings
    if not args.output is None:
        settings['output_image'] = args.output

        
    # if the user wants to store the data, do so in a dedicated directory below the
    # ANALYSIS_DIR passed via settings (here it is also checked whether there is an
    # existing analysis, for which the hash sum of the content of the spike times
    # file must match).
    #
    # If the user does not want to store the data, a temporary file is created and
    # then deleted after the program finishes
    #
    # For most tasks an existing analysis file is expected

    if settings['persistent_analysis']:
        if not isdir(settings['ANALYSIS_DIR']):
            print("Error: {} not found. Aborting.".format(settings['ANALYSIS_DIR']),
                  file=stderr, flush=True)
            exit(EXIT_FAILURE)

        analysis_dir, analysis_num, existing_analysis_found \
            = utl.get_or_create_analysis_dir(spike_times,
                                             spike_times_file_name,
                                             settings['ANALYSIS_DIR'])

    
        settings['ANALYSIS_DIR'] = analysis_dir
    else:
        analysis_num = "temp"
            
    analysis_file = utl.get_analysis_file(settings['persistent_analysis'],
                                          settings['ANALYSIS_DIR'])

    # sanity check for tasks

    if not task == "full-analysis" and not settings['persistent_analysis']:
        print("Error.  Setting 'persistent_analysis' is set to 'False' and task is not 'full-analysis'.  This would produce no output.  Aborting.", file=stderr, flush=True)
        exit(EXIT_FAILURE)

    if task in ["confidence-intervals",
                # "permutation-test",
                "csv-files"]:
        if not "embeddings" in analysis_file.keys():
            print("Error.  No existing analysis found.  Please run the 'history-dependence' task first.  Aborting.", file=stderr, flush=True)
            exit(EXIT_FAILURE)

    csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file \
        = utl.get_CSV_files(task,
                            settings['persistent_analysis'],
                            settings['ANALYSIS_DIR'])

    if task == "plots":
        for csv_file in [csv_stats_file,
                         csv_histdep_data_file,
                         csv_auto_MI_data_file]:
            if csv_file == None:
                print("Error.  CSV files not found and needed to produce plots.  Please run the 'csv-files' task first.  Aborting.", file=stderr, flush=True)
                exit(EXIT_FAILURE)
                                                
    
    # label for the output
    if not args.label is None:
        settings['label'] = args.label
    else:
        if not 'label' in settings:
            settings['label'] = ""
    if "," in settings['label']:
        new_label = ""
        for char in settings['label']:
            if not char == ",":
                new_label += char
            else:
                new_label += ";"
        settings['label'] = new_label
        print("Warning: Invalid label '{}'. It may not contain any commas, as this conflicts with the CSV file format.  The commas have been replaced by semicolons.".format(settings['label']),
              file=stderr, flush=True)


    # for cross-validation
    # split up data in two halves
    if settings['cross_validated_optimization']:
        spike_times_half_time = (spike_times[-1] - spike_times[0]) / 2
        spike_times_optimization = spike_times[spike_times < spike_times_half_time]
        spike_times_validation = spike_times[spike_times >= spike_times_half_time] \
            - spike_times_half_time
    else:
        spike_times_optimization = spike_times
        spike_times_validation = spike_times

    return task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, analysis_num, \
        settings
            
def main():
    """
    Parse arguments and settings and then run selected tasks.
    """
    
    # definitions
    defined_tasks = ["history-dependence",
                     "confidence-intervals",
                     # "permutation-test",
                     "auto-mi",
                     "csv-files",
                     "plots",
                     "full-analysis"]
    
    defined_estimation_methods = ['bbc', 'shuffling', 'all']
    
    # get task and target (parse arguments and check for validity)
    task, spike_times, spike_times_optimization, spike_times_validation, \
        analysis_file, csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file, analysis_num, \
        settings = parse_arguments(defined_tasks,
                                   defined_estimation_methods)

    if settings['estimation_method'] == 'all':
        estimation_methods = ['bbc', 'shuffling']
    else:
        estimation_methods = [settings['estimation_method']]
        
    # now perform tasks as specified by the parsed arguments
    
    for estimation_method in estimation_methods:
        settings['estimation_method'] = estimation_method

        if task == "history-dependence" or task == "full-analysis":
            do_main_analysis(spike_times, spike_times_optimization, spike_times_validation,
                             analysis_file, settings)

        if task == "confidence-intervals" or task == "full-analysis":
            compute_CIs(spike_times_validation, analysis_file, settings)

        # if task == "permutation-test" or task == "full-analysis":
        #     perform_permutation_test(analysis_file, settings)


    if task == "auto-mi" or task == "full-analysis":
        analyse_auto_MI(spike_times, analysis_file, settings)
        
    if task == "csv-files" or task == "full-analysis":
        create_CSV_files(analysis_file,
                         csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                         analysis_num, settings)
        

    if task == "plots" or task == "full-analysis":
        produce_plots(spike_times,
                      csv_stats_file, csv_histdep_data_file, csv_auto_MI_data_file,
                      settings)


    for f in [analysis_file,
              csv_stats_file,
              csv_histdep_data_file,
              csv_auto_MI_data_file]:
        if not f == None:
            f.close()    
        
    return EXIT_SUCCESS

if __name__ == "__main__":
    if len(argv) == 1:
        argv += ["-h"]
    exit(main())

