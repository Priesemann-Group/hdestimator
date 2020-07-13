import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import date
import hde_utils as utl
__version__ = "unknown"
from _version import __version__

def format_x_label(x, pos):
    return "{:.0f}".format(x)

def format_x_label_in_ms(x, pos):
    return "{:.0f}".format(x * 1000)
    
def format_y_label(y, pos):
    if y > 0 and y <= 0.02:
        ret = "{:.3f}".format(y)
        return ret[1:]
    return "{:.2f}".format(y)

def make_plot_pretty(ax):
    ax.tick_params(axis = 'x', top=False)
    ax.tick_params(axis = 'y', right=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis = 'x', direction = 'out')
    ax.tick_params(axis = 'y', length = 0)

    for spine in ax.spines.values():
        spine.set_position(('outward', 5))

    ax.set_axisbelow(True)

    ax.grid(axis = 'y', color='0.9', linestyle='-', linewidth=1)

def make_twin_plot_pretty(axv):
    axv.tick_params(axis = 'x', top=False)
    axv.tick_params(axis = 'y', left=False)
    axv.spines['top'].set_visible(False)
    axv.spines['right'].set_visible(False)
    axv.spines['left'].set_visible(False)
    axv.spines['bottom'].set_visible(False)
    axv.get_yaxis().tick_right()
    axv.tick_params(axis = 'y', length = 0)

def plot_neuron_activity(ax, spike_times, recording_length, averaging_time,
                         color):
    """
    Visualize the moving average of some neural spiking activity.
    """
    
    spike_density = np.zeros(int(recording_length) - averaging_time)
    spike_index = 0
    for t in range(len(spike_density)):
        while spike_index < len(spike_times) and spike_times[spike_index] < t:
            spike_index += 1
        spike_index_2 = 0

        while spike_index + spike_index_2 < len(spike_times) \
              and spike_times[spike_index + spike_index_2] < t + averaging_time:
            spike_density[t] += 1
            spike_index_2 += 1        

    ax.plot(range(len(spike_density)), spike_density / averaging_time, color=color)
    
    ax.set_xlabel(r"time $t$ " + "[s]")
    ax.set_ylabel("fir. rate ({}s avg.) [Hz]".format(averaging_time))

    make_plot_pretty(ax)

def plot_auto_mutual_information(ax, auto_MI_data, color):
    """
    Visualize the auto mutual information for a set of delays and
    bin sizes.

    As with the history dependence R, the auto MI is normalized to
    be in the range [0, 1] (unless data is scarce and there is
    a bias in the estimation). 
    """
    
    line_styles = ["-", ":", "-.", "--"]
    marker_types = [None, "^", ">", "v", "<"]
    
    for i, auto_MI_bin_size in enumerate(sorted(auto_MI_data.keys())):
        delays, auto_MIs = auto_MI_data[auto_MI_bin_size]

        ls = line_styles[i % len(line_styles)]
        marker = marker_types[i // len(line_styles)]

        ax.plot(delays, np.array(auto_MIs), ls=ls, marker=marker,
                color=color, label='{:.0f}'.format(auto_MI_bin_size * 1000))

    legend = ax.legend(title="bin size [ms]:",
                       fancybox=False,
                       loc="center right", bbox_to_anchor=(1.62, 0.5))
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_x_label_in_ms))
    ax.set_xlabel(r"time $t$ " + "[ms]")
    ax.set_ylabel("normalized Auto MI")

    make_plot_pretty(ax)

def plot_history_dependence(ax,
                            csv_stats_file,
                            csv_histdep_data_file,
                            estimation_method,
                            color,
                            plot_AIS=False):
    """
    Plot history dependence (or AIS) estimates for the BBC or 
    shuffling estimator.
    """

    if plot_AIS:
        Y_label = 'AIS'
    else:
        Y_label = 'R'
    
    Ts = utl.load_from_CSV_file(csv_histdep_data_file,
                                 "T")
    Ys = utl.load_from_CSV_file(csv_histdep_data_file,
                                "max_{}_{}".format(Y_label, estimation_method))
    Ys_CI_lo = utl.load_from_CSV_file(csv_histdep_data_file,
                                      "max_{}_{}_CI_lo".format(Y_label,
                                                               estimation_method))
    Ys_CI_hi = utl.load_from_CSV_file(csv_histdep_data_file,
                                      "max_{}_{}_CI_hi".format(Y_label,
                                                               estimation_method))
    # Ys_CI_med = utl.load_from_CSV_file(csv_histdep_data_file,
    #                                    "max_{}_{}_CI_med".format(Y_label,
    #                                                              estimation_method))
    T_D = utl.load_from_CSV_file(csv_stats_file,
                                 "T_D_{}".format(estimation_method))

    bs_CI_percentile_lo = utl.load_from_CSV_file(csv_stats_file,
                                                 "bs_CI_percentile_lo")
    bs_CI_percentile_hi = utl.load_from_CSV_file(csv_stats_file,
                                                 "bs_CI_percentile_hi")
    CI = int(bs_CI_percentile_hi - bs_CI_percentile_lo)

    # history dependence R (or AIS) as a function of the past range T
    ax.plot(Ts, Ys, '-', color=color)

    # vertical line marking T_D
    ax.axvline(x=T_D, color='k', ls='--', label=r"$T_D$")

    # bootstrap confidence intervals
    ax.fill_between(Ts, Ys_CI_lo, Ys_CI_hi,
                    alpha=0.25, linewidth=0, color=color, label='{}% CI'.format(CI))
    
    make_plot_pretty(ax)
    ax.set_xscale('log')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_x_label_in_ms))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_label))

    ax.set_xlabel("past range " + r"$T$ " + "[ms]")
    if plot_AIS:
        ax.set_ylabel("Active Info. Storage (AIS)")
    else:
        ax.set_ylabel("History Dependence " + r"$R$")

    # shared legend between the subplots
    if estimation_method == 'bbc':
        legend = ax.legend(fancybox=False, loc="center right", bbox_to_anchor=(1.6, 0.5))
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')


def produce_plots(spike_times,
                  csv_stats_file,
                  csv_histdep_data_file,
                  csv_auto_MI_data_file,
                  plot_color,
                  plot_AIS,
                  **kwargs):
    """
    Visualize the results of the history dependence analysis.

    Plot neural activity.
    Plot auto mutual information.
    Plot history dependence estimates for the BBC and 
    shuffling estimators.
    Print a table with the main results.
    """

    #
    # define figure, axes, update settings
    #

    plt.rcParams.update(kwargs["plot_settings"])
    # fig0, ((ax0l, ax0r),
    #        (ax1l, ax1r),
    #        (ax2l, ax2r)) = plt.subplots(3, 2)
    fig0 = plt.figure()
    ax0l = plt.subplot(321)
    ax0r = plt.subplot(322)
    ax1l = plt.subplot(323)
    ax1r = plt.subplot(324)
    ax2l = plt.subplot(325)
    ax2r = plt.subplot(326, sharex=ax2l, sharey=ax2l)
    fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1, wspace=1, hspace=0.8)
    
    #
    # ax0l
    # plot (5s moving-average) of firing rate
    #

    recording_length = utl.load_from_CSV_file(csv_stats_file,
                                              "recording_length")
    averaging_time = 5
    plot_neuron_activity(ax0l, spike_times, recording_length, averaging_time,
                         plot_color)

    #
    # ax1l
    # plot auto mutual information
    #

    auto_MI_data = utl.load_auto_MI_data(csv_auto_MI_data_file)
    
    plot_auto_mutual_information(ax1l, auto_MI_data, plot_color)



    for ax2, estimation_method in zip([ax2l, ax2r],
                                      ['shuffling', 'bbc']):
        #
        # ax2
        # plot history-dependence as fn of T with bootstrap confidence interval
        #

        plot_history_dependence(ax2,
                                csv_stats_file,
                                csv_histdep_data_file,
                                estimation_method,
                                plot_color,
                                plot_AIS)

    # shared x axis betw. auto MI and histdep for easy comparison

    x_autoMI_lo, x_autoMI_hi = ax1l.get_xlim()
    x_R_lo, x_R_hi = ax2l.get_xlim()

    x_lo = min(x_autoMI_lo, x_R_lo)
    x_hi = max(x_autoMI_hi, x_R_hi)
    
    ax1l.set_xlim(x_lo, x_hi)

    #
    # tweak the looks of the plots
    #

    ax0r.axis('off')
    ax1r.axis('off')

    ax0l.set_title("Neuron activity")
    ax1l.set_title("Auto Mutual Information")

    if plot_AIS:
        ax2l.set_title("AIS: Shuffling estimate")
        ax2r.set_title("AIS: BBC estimate")
    else:
        ax2l.set_title("Hist. Dep.: Shuffling estimate")
        ax2r.set_title("Hist. Dep.: BBC estimate")

    #
    # add the stats of the analysis to the plot
    #
    stats_fontsize = 8

    if plot_AIS:
        Y_label = 'AIS'
    else:
        Y_label = 'R'

    try:
        analysis_num = str(int(utl.load_from_CSV_file(csv_stats_file,
                                                      "analysis_num")))
        analysis_num = "0"*(4-len(analysis_num)) + analysis_num
    except:
        analysis_num = ""
    analysis_label = utl.load_from_CSV_file(csv_stats_file,
                                            "label")
    firing_rate = utl.load_from_CSV_file(csv_stats_file,
                                         "firing_rate")
    recording_length = utl.load_from_CSV_file(csv_stats_file,
                                              "recording_length")

    T_D_bbc = utl.load_from_CSV_file(csv_stats_file,
                                     "T_D_bbc")
    Y_tot_bbc = utl.load_from_CSV_file(csv_stats_file,
                                       "{}_tot_bbc".format(Y_label))
    Y_tot_bbc_CI_lo = utl.load_from_CSV_file(csv_stats_file,
                                             "{}_tot_bbc_CI_lo".format(Y_label))
    Y_tot_bbc_CI_hi = utl.load_from_CSV_file(csv_stats_file,
                                             "{}_tot_bbc_CI_hi".format(Y_label))
    opt_number_of_bins_d_bbc = utl.load_from_CSV_file(csv_stats_file,
                                                      "opt_number_of_bins_d_bbc")
    opt_scaling_k_bbc = utl.load_from_CSV_file(csv_stats_file,
                                               "opt_scaling_k_bbc")
    opt_first_bin_size_bbc = utl.load_from_CSV_file(csv_stats_file,
                                                    "opt_first_bin_size_bbc")
    # asl_permutation_test_bbc = utl.load_from_CSV_file(csv_stats_file,
    #                                                   "asl_permutation_test_bbc")

    T_D_shuffling = utl.load_from_CSV_file(csv_stats_file,
                                           "T_D_shuffling")
    Y_tot_shuffling = utl.load_from_CSV_file(csv_stats_file,
                                             "{}_tot_shuffling".format(Y_label))
    Y_tot_shuffling_CI_lo = utl.load_from_CSV_file(csv_stats_file,
                                                   "{}_tot_shuffling_CI_lo".format(Y_label))
    Y_tot_shuffling_CI_hi = utl.load_from_CSV_file(csv_stats_file,
                                                   "{}_tot_shuffling_CI_hi".format(Y_label))
    opt_number_of_bins_d_shuffling = utl.load_from_CSV_file(csv_stats_file,
                                                            "opt_number_of_bins_d_shuffling")
    opt_scaling_k_shuffling = utl.load_from_CSV_file(csv_stats_file,
                                                     "opt_scaling_k_shuffling")
    opt_first_bin_size_shuffling = utl.load_from_CSV_file(csv_stats_file,
                                                          "opt_first_bin_size_shuffling")
    # asl_permutation_test_shuffling = utl.load_from_CSV_file(csv_stats_file,
    #                                                         "asl_permutation_test_shuffling")

    bs_CI_percentile_lo = utl.load_from_CSV_file(csv_stats_file,
                                                 "bs_CI_percentile_lo")
    bs_CI_percentile_hi = utl.load_from_CSV_file(csv_stats_file,
                                                 "bs_CI_percentile_hi")
    CI = int(bs_CI_percentile_hi - bs_CI_percentile_lo)

    
    ax0r.text(-0.5, 0.975, "analysis: {}".format(analysis_num),
              fontsize=stats_fontsize)
    ax0r.text(0.35, 0.975, "hde v. {}, {}".format(__version__, date.today()),
              fontsize=stats_fontsize)
    ax0r.text(-0.5, 0.775, analysis_label,
              fontsize=stats_fontsize)
    ax0r.text(-0.3, 0.475, "recording length: {:.1f}s".format(recording_length),
              fontsize=stats_fontsize)
    ax0r.text(-0.3, 0.275, "firing rate: {:.1f}Hz".format(firing_rate),
              fontsize=stats_fontsize)

    if plot_AIS:
        ax0r.text(-0.3, -.025, "Active Information Storage:",
              fontsize=stats_fontsize)
    else:
        ax0r.text(-0.3, -.025, "History Dependence:",
                  fontsize=stats_fontsize)
    shuffling_h_pos = 0.14
    bbc_h_pos = 0.68
    ax0r.text(shuffling_h_pos, -.225, "Shuffling",
              fontsize=stats_fontsize)
    ax0r.text(bbc_h_pos, -.225, "BBC",
              fontsize=stats_fontsize)
    ax0r.text(-0.3, -.3, "---------------------------------------------------------------",
              fontsize=stats_fontsize)
    current_v_pos = -.425

    # these are used further below to print results to terminal
    bbc_results = []
    shuffling_results = []
    
    for label, shuffling_value, bbc_value in [("$\hat{{T}}_D\,$[s]:", T_D_shuffling, T_D_bbc),
                                              ("$\hat{{{}}}_{{tot}}$:".format(Y_label),
                                               Y_tot_shuffling, Y_tot_bbc),
                                              ("$\hat{{{}}}_{{tot}}$, {}% CI:".format(Y_label, CI),
                                               "[{:.3f}, {:.3f}]".format(Y_tot_shuffling_CI_lo,
                                                                         Y_tot_shuffling_CI_hi),
                                              "[{:.3f}, {:.3f}]".format(Y_tot_bbc_CI_lo,
                                                                        Y_tot_bbc_CI_hi)),
                                              ("opt. $d$:",
                                               "{:.0f}".format(opt_number_of_bins_d_shuffling),
                                               "{:.0f}".format(opt_number_of_bins_d_bbc)),
                                              ("opt. $\kappa$:",
                                               opt_scaling_k_shuffling,
                                               opt_scaling_k_bbc),
                                              (r"opt. $\tau_0\,$[s]:",
                                               opt_first_bin_size_shuffling,
                                               opt_first_bin_size_bbc)]:
                                              # ("ASL perm. test:",
                                              #  asl_permutation_test_shuffling,
                                              #  asl_permutation_test_bbc)]:
        if not type(shuffling_value) == str:
            shuffling_value = "{:.3f}".format(shuffling_value)
        if not type(bbc_value) == str:
            bbc_value = "{:.3f}".format(bbc_value)
                
        ax0r.text(-0.3, current_v_pos, label, fontsize=stats_fontsize)
        ax0r.text(shuffling_h_pos, current_v_pos, shuffling_value, fontsize=stats_fontsize)
        ax0r.text(bbc_h_pos, current_v_pos, bbc_value, fontsize=stats_fontsize)

        current_v_pos -= 0.2

        label = label.replace("$", "").replace("\\,", " ").replace("\\", "").replace("{", "").replace("}", "").replace("hat", "")
        
        shuffling_results += ["{} {}".format(label,
                                             shuffling_value)]
        bbc_results += ["{} {}".format(label,
                                       bbc_value)]

    # also print results to terminal
    print("analysis: {}, hde v. {}, {}".format(analysis_num, __version__, date.today()))
    print(analysis_label)
    print("recording length: {:.1f}s".format(recording_length))
    print("firing rate: {:.1f}Hz".format(firing_rate))
    print()

    print("Shuffling")
    print("---------------------------------------------------------------")
    print('\n'.join(shuffling_results))
    print()
    print("BBC")
    print("---------------------------------------------------------------")
    print('\n'.join(bbc_results))

    if 'output_image' in kwargs:
        plt.savefig(kwargs['output_image'],
                    bbox_inches='tight', dpi=300)
    elif kwargs['persistent_analysis']:
        plt.savefig("{}/{}".format(kwargs['ANALYSIS_DIR'],
                                   analysis_num),
                    bbox_inches='tight', dpi=300)
        print("Saved image to {}/{}.{}".format(kwargs['ANALYSIS_DIR'],
                                               analysis_num,
                                               plt.rcParams['savefig.format']))
    else:
        plt.show()
