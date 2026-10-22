import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import json
import glob
import pickle

import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import ttest_ind
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
import tensorflow as tf
from collections import OrderedDict
from pathlib import Path
from scipy.stats import ttest_ind, ks_2samp
import seaborn as sns

plt.ioff() # prevents windows to pop up when figs and plots are created

from _analysis import clustering  # , variance
from _training import apply_density_threshold
from _network import Model
import _tools
from _tools import rule_name, load_pickle

# attention: script is outdated for evaluating task representations
#  use _multipleNetworkAnalysis.py or _hyperparameterOverview.py (for preprocessing) instead

'''
- Plots train/test performance and cost for one defined model 
- Computes and visualizes functional and structural (model weights) correlation matrices
- Creates and visualizes task variance and lesioning plots - can compare subjects
- Creates topological markers and saves as dicts (outdated)
'''

selected_hp_keys = ['participant', 'rnn_type', 'data', 'multiLayer', 'n_rnn_per_layer', 'activations_per_layer',
                    'activation', 'optimizer', 'loss_type', 'batch_size', 'l1_h', 'l2_h', 'l1_weight', 'l2_weight',
                    'learning_rate', 'learning_rate_mode', 'n_rnn', 'sigma_rec', 'sigma_x', 'w_rec_init',
                    'c_mask_responseValue', 'errorBalancingValue', 'p_weight_train',
                    'w_mask_value']  # Replace with the keys you want info: 'data' only exists from 15.01.25 on

rule_color = {
    # **DM tasks (Dark Purple - High Contrast)**
    'DM': '#0d0a29',  # Deep Black-Purple
    'DM_Anti': '#271258',  # Dark Blue-Purple

    # **EF tasks (Purple-Pink Family - High Contrast)**
    'EF': '#491078',  # Muted Indigo
    'EF_Anti': '#671b80',  # Dark Magenta-Purple

    # **RP tasks (Pink/Red Family - High Contrast)**
    'RP': '#862781',  # Rich Magenta
    'RP_Anti': '#a6317d',  # Strong Pink
    'RP_Ctx1': '#c53c74',  # Bright Pinkish-Red
    'RP_Ctx2': '#e34e65',  # Vivid Red

    # **WM tasks (Red-Orange/Yellow Family - High Contrast)**
    'WM': '#f66c5c',  # Warm Reddish-Orange
    'WM_Anti': '#fc9065',  # Vibrant Orange
    'WM_Ctx1': '#feb67c',  # Pastel Orange
    'WM_Ctx2': '#fdda9c'  # Light Yellow
}

comparison = False

def smoothed(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    # Calculate how many points we need to add to match the length of the original data
    padding_length = len(data) - len(smoothed_data)
    if padding_length > 0:
        last_value = smoothed_data[-1]
        smoothed_data = np.concatenate((smoothed_data, [last_value] * padding_length))
    return smoothed_data

def create_legend_image():
    """ Generate and return a properly formatted legend image. """
    legend_fig = plt.figure(figsize=(2, 0.6))  # Decrease figure size
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")

    ordered_rules = list(currentHP['rule_prob_map'].keys())

    # Create handles with appropriate transparency
    rule_handles = [
        plt.Line2D([0], [0], color=rule_color[r], lw=3, alpha=1.0 if currentHP['rule_prob_map'][r] > 1e-10 else 0.5)
        for r in ordered_rules
    ]

    # Labels remain in the same order
    rule_labels = [rule_name[r] for r in ordered_rules]

    # Add legend with optimized spacing
    legend = legend_ax.legend(
        handles=rule_handles, labels=rule_labels, ncol=2, loc="center", fontsize=5,
        frameon=True, edgecolor="black",
        columnspacing=1, handletextpad=0.7
    )
    legend.get_title().set_fontsize(10)  # Slightly larger title

    # Ensure text fits properly
    plt.tight_layout()

    # Convert figure to an image (Ensure full legend is captured)
    canvas = FigureCanvas(legend_fig)
    canvas.draw()

    maskDirectory = os.path.join(os.getcwd(), 'pngs', 'legend.png')
    # Save as buffer to prevent cropping
    legend_fig.savefig(maskDirectory, format='png', dpi=200, bbox_inches='tight', pad_inches=0.05, transparent=True)

    # Load the properly saved image
    legend_img = plt.imread(maskDirectory)

    # Close the figure
    plt.close(legend_fig)

    return legend_img

def fig_to_array(figure):
    """ Convert Matplotlib figure to a NumPy array (RGB image). """
    canvas = FigureCanvas(figure)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    # plt.close(fig)
    return img

def define_data_folder(split_parts):
    # Predefined categories or flags to look for
    possible_keys = [
        'highDim', 'lowDim',
        'correctOnly', 'correctOnly_3stimTC', '3stimTC', 'correctOnly_timeCompressed',
        'lowCognition', 'noCognition', 'timeCompressed',
        'all_beRNNs'
    ]

    # Filter parts that match any known key
    found_keys = [key for key in possible_keys if key in split_parts]

    # If at least one key is found, join them with underscores
    if found_keys:
        data_folder = 'data_' + '_'.join(found_keys)
    else:
        data_folder = 'data_unknown'

    return data_folder


########################################################################################################################
# head. Performance - Individual network
########################################################################################################################
def plot_performanceprogress_test_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Evaluation Progress
    log = _tools.load_log(model_dir)
    # log = _tools.load_log(currentModelDirectory)
    hp = _tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']
    x_plot = np.array(trials) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    fig_eval = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = fig_eval.add_axes([0.12, 0.1, 0.75, 0.65])  # [left, bottom, width, height]
    lines = list()
    labels = list()

    # if rule_plot == None:
    #     # rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        # co: add [::2] if you want to have only every second validation values
        line = ax.plot(x_plot, log['perf_' + rule], color=rule_color[rule], linewidth=3)
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 80000])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')  # info: Add title

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_test.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_test.png'), format='png', dpi=300)

    # plt.show()
    img_eval = fig_to_array(fig_eval)
    plt.close(fig_eval)

    return img_eval

def plot_performanceprogress_train_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Training Progress
    log = _tools.load_log(model_dir)
    hp = _tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']  # info: There is an entry every 40 trials for each task
    x_plot = (np.array(trials)) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    fig_train = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = fig_train.add_axes([0.12, 0.1, 0.75, 0.65])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    # if rule_plot is None:
    # rule_plot = hp['rules']
    # rule_plot = ['DM', 'DM_Anti']

    for i, rule in enumerate(rule_plot):
        y_perf = log['perf_train_' + rule][::int((len(log['perf_train_' + rule]) / len(x_plot)))][:len(x_plot)]

        window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation

        # y_cost_smoothed = smoothed(y_cost, window_size=window_size)
        y_perf_smoothed = smoothed(y_perf, window_size=window_size)

        # Ensure the lengths match
        y_perf_smoothed = y_perf_smoothed[:len(x_plot)]

        line = ax.plot(x_plot, y_perf_smoothed, color=rule_color[rule], linewidth=3)

        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    # lg = perf_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.2),
    #                       fontsize=fs, labelspacing=0.3, loc=6, frameon=False) # info: first value influences horizontal position of legend
    # plt.setp(lg.get_title(), fontsize=fs)

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_train.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_train.png'), format='png', dpi=300)

    # plt.show()
    img_train = fig_to_array(fig_train)
    plt.close(fig_train)

    return img_train

def plot_cost_test_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Evaluation Progress
    log = _tools.load_log(model_dir)
    # log = _tools.load_log(currentModelDirectory)
    hp = _tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']
    x_plot = np.array(trials) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    cost_eval = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = cost_eval.add_axes([0.12, 0.1, 0.75, 0.65])  # [left, bottom, width, height]
    lines = list()
    labels = list()

    # if rule_plot == None:
    #     # rule_plot = hp['rules']

    # fix cost plotting, seems to be wrongly normalized in some cases w. ceiling effect
    # Collect all log-transformed costs for scaling
    all_costs_log = ([np.log10(np.array(log['cost_train_' + rule]) + 1e-8) for rule in rule_plot]
                     + [np.log10(np.array(log['cost_' + rule]) + 1e-8) for rule in rule_plot])

    # Determine global min and max on the log-transformed scale
    cost_log_min = min(np.min(c) for c in all_costs_log)
    cost_log_max = max(np.max(c) for c in all_costs_log)

    for i, rule in enumerate(rule_plot):
        # co: add [::2] if you want to have only every second validation values
        y_cost = log['cost_' + rule]
        # Safe log transform (shift to avoid log(0))
        y_cost_log = np.log10(list(map(lambda x: x + 1e-8, y_cost)))
        # Normalize log costs
        y_cost_log_norm = (y_cost_log - cost_log_min) / (cost_log_max - cost_log_min) # fix: It seems there is some issue here

        line = ax.plot(x_plot, y_cost_log_norm, color=rule_color[rule], linewidth=3)
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 80000])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Normalized Log10(Cost)', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')  # info: Add title

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_test.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_test.png'), format='png', dpi=300)

    # plt.show()
    img_eval = fig_to_array(cost_eval)
    plt.close(cost_eval)

    return img_eval

def plot_cost_train_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Training Progress
    log = _tools.load_log(model_dir)
    hp = _tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']  # info: There is an entry every 40 trials for each task
    x_plot = (np.array(trials)) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    cost_train = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = cost_train.add_axes([0.12, 0.1, 0.75, 0.65])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    # if rule_plot is None:
    # rule_plot = hp['rules']

    # Collect all log-transformed costs for scaling
    all_costs_log = ([np.log10(np.array(log['cost_train_' + rule]) + 1e-8) for rule in rule_plot]
                     + [np.log10(np.array(log['cost_' + rule]) + 1e-8) for rule in rule_plot])

    # Determine global min and max on the log-transformed scale
    cost_log_min = min(np.min(c) for c in all_costs_log)
    cost_log_max = max(np.max(c) for c in all_costs_log)

    for i, rule in enumerate(rule_plot):
        # y_cost = log['cost_train_' + rule][::int((len(log['cost_train_' + rule]) / len(x_plot)))][:len(x_plot)]
        y_cost = log['cost_train_' + rule]
        y_cost_smoothed = smoothed(y_cost, window_size=5)[:len(trials)]

        # window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation
        # y_cost_smoothed = smoothed(y_cost, window_size=window_size)
        # Ensure the lengths match
        # y_cost_smoothed = y_cost_smoothed[:len(x_plot)]
        # Safe log transform (shift to avoid log(0))
        y_cost_log = np.log10(list(map(lambda x: x + 1e-8, y_cost_smoothed)))
        # Normalize log costs
        y_cost_log_norm = (y_cost_log - cost_log_min) / (cost_log_max - cost_log_min)

        # line = ax.plot(x_plot, np.log10(y_cost_smoothed), color=rule_color[rule])
        line = ax.plot(x_plot, y_cost_log_norm, color=rule_color[rule], linewidth=3)

        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Normalized Log10(Cost)', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    # lg = perf_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.2),
    #                       fontsize=fs, labelspacing=0.3, loc=6, frameon=False) # info: first value influences horizontal position of legend
    # plt.setp(lg.get_title(), fontsize=fs)

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_train.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_train.png'), format='png', dpi=300)

    # plt.close()
    img_train = fig_to_array(cost_train)
    plt.close(cost_train)

    return img_train


########################################################################################################################
# head. Functional & Structural Correlation  - Individual networks
########################################################################################################################
def compute_functionalCorrelation(model_dir, threshold, monthsConsidered, mode, figurePath, analysis):
    # different correlation function used in all published results - see analysis/variance.py line 156/157
    # correlation = analysis.get_dotProductCorrelation() # legacy
    correlation = analysis.h_corr_all
    # path = os.path.join(figurePath,'functionalCorrelation_npy')

    # if not os.path.exists(path):
    #     os.makedirs(path)

    # modelName = model_dir.split('\\')[-1]
    # np.save(os.path.join(figurePath,f'{modelName}_functionalCorrelation'),correlation)

    # Set up the figure
    corr_fig = plt.figure(figsize=(8, 8))

    # Create the main similarity matrix plot
    matrix_left = 0.11
    matrix_bottom = 0.1
    matrix_width = 0.75
    matrix_height = 0.75

    ax_matrix = corr_fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
    im = ax_matrix.imshow(correlation, cmap='magma', interpolation='nearest', vmin=-1, vmax=1)

    # Add title
    model = '_'.join(model_dir.split("\\")[-1].split('_')[0:4])
    ax_matrix.set_title(f'{model}', fontsize=26, fontweight='bold', pad=20)
    # Add x-axis and y-axis labels
    ax_matrix.set_xlabel('Hidden units', fontsize=24, labelpad=15)
    ax_matrix.set_ylabel('Hidden units', fontsize=24, labelpad=15)

    # Remove x and y ticks
    ax_matrix.set_xticks([])  # Disable x-ticks
    ax_matrix.set_yticks([])  # Disable y-ticks

    # Create the colorbar on the right side, aligned with the matrix
    colorbar_left = matrix_left + matrix_width + 0.02
    colorbar_width = 0.04

    ax_cb = corr_fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_ticks([-1, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Correlation', fontsize=24, labelpad=0)

    # Compute topological marker
    functionalCorrelation_thresholded = apply_density_threshold(correlation, threshold)

    # Function to apply a threshold to the matrix
    G_sparse = nx.from_numpy_array(functionalCorrelation_thresholded)

    if G_sparse.number_of_edges() == 0 or G_sparse.number_of_nodes() < 2:
        print(f"Skipping modularity calculation for {model_dir} — graph has no edges. Setting mod_value = 0.")
        avg_clustering = 0
        mod_value_sparse = 0
        avg_pc = 0
    else:
        try:
            # clustering
            clustering = nx.clustering(G_sparse)
            avg_clustering = np.mean(list(clustering.values()))
            # modularity
            communities_sparse = greedy_modularity_communities(G_sparse)
            mod_value_sparse = modularity(G_sparse, communities_sparse)
            # participation
            pc_dict = _tools.participation_coefficient(G_sparse, communities_sparse)
            avg_pc = np.mean(list(pc_dict.values()))

        except Exception as e:
            print(f"Greedy modularity failed for {model_dir}. Setting mod_value = 0. ({e})")
            avg_clustering = 0
            mod_value_sparse = 0
            avg_pc = 0

    # Show the top Markers within the func Correlation
    ax_matrix.text(0.5, 0.6, f'Modularity: {mod_value_sparse:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    ax_matrix.text(0.5, 0.5, f'Clustering: {avg_clustering:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    ax_matrix.text(0.5, 0.4, f'Participation: {avg_pc:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    return corr_fig, mod_value_sparse, avg_clustering, avg_pc
    # plt.show()
    # plt.close()

def visualize_meanMatrix_singleModel(pkl_file):
    # pkl_file must be directly defined to corr pkl file
    meanMatrix_taskwise = load_pickle(pkl_file)
    h_mean_all = meanMatrix_taskwise['h_mean_all']
    rdm, rdm_vector = clustering.compute_rdm(h_mean_all, 'cosine')

    plt.figure(figsize=(8, 6))

    # 1. Heatmap zuerst zeichnen
    ax = sns.heatmap(h_mean_all.T, vmin=0, vmax=0.5, cmap="Greys")

    # 2. Ticks definieren (bis 257, damit 256 inkludiert ist)
    tick_x_locations = np.arange(0, 257, step=32)
    tick_y_locations = np.arange(0, 12, step=1)

    # 3. Ticks anwenden
    # Wir addieren 0.5, um den Tick exakt in der Mitte der Zelle zu platzieren
    plt.yticks(tick_y_locations + 0.5, labels=['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2'], rotation=0)
    plt.xticks(tick_x_locations + 0.5, labels=tick_x_locations, rotation=0)

    # plt.title("h_mean_all")
    plt.show()

def visualize_rdMatrix_singleModel(pkl_file):
    # pkl_file must be directly defined to mean pkl file
    meanMatrix_taskwise = load_pickle(pkl_file)

    h_mean_all = meanMatrix_taskwise['h_mean_all']

    rdm, rdm_vector = clustering.compute_rdm(h_mean_all, 'cosine')

    mask = np.tril(np.ones_like(rdm, dtype=bool))

    plt.figure(figsize=(8, 6)) # Höhe etwas erhöht für bessere Lesbarkeit

    # 1. Heatmap zuerst zeichnen
    ax = sns.heatmap(-rdm, mask = mask, vmin=-0.5, vmax=0.5, cmap="Greys")

    # 2. Ticks definieren (bis 257, damit 256 inkludiert ist)
    tick_x_locations = np.arange(0, 12, step=1)
    tick_y_locations = np.arange(0, 12, step=1)

    # 3. Ticks anwenden (nach dem Heatmap-Aufruf)
    # Wir addieren 0.5, um den Tick exakt in der Mitte der Zelle zu platzieren
    plt.yticks(tick_y_locations + 0.5, labels=['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2'], rotation=0)
    plt.xticks(tick_x_locations + 0.5, labels=['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2'], rotation=0)

    # plt.title("h_mean_all")
    plt.show()

    # optional correlation of two rdm_vectors e.g. for paper visualizations
    rdm1_vector = rdm_vector
    rdm2_vector = rdm_vector

    rho = scipy.stats.spearmanr(rdm1_vector, rdm2_vector).correlation
    plt.figure(figsize=(8, 6))  # Höhe etwas erhöht für bessere Lesbarkeit
    ax = sns.heatmap(np.array([[1 - rho]]), vmin=0, vmax=0.5, cmap="viridis")
    plt.show()

def compute_structuralCorrelation(model_dir, figurePath, monthsConsidered, mode):
    for month in monthsConsidered:
        _analysis = clustering.Analysis(data_dir, os.path.join(model_dir, f'model_{month}'), layer, 'cosine', mode,
                                        monthsConsidered, 'rule', True)
        correlationRecurrent = _analysis.easy_connectivity_plot_recurrentWeightsOnly(
            os.path.join(model_dir, f'model_{month}'))
        # correlationExcitatoryGates = _analysis.easy_connectivity_plot_excitatoryGatedWeightsOnly(model_dir)
        # correlationInhibitoryGates = _analysis.easy_connectivity_plot_inhibitoryGatedWeightsOnly(model_dir)

        path = os.path.join(figurePath, 'structuralCorrelation_npy')

        if not os.path.exists(path):
            os.makedirs(path)

        correlationNames = ['CorrelationRecurrent']  # , 'CorrelationInhibitoryGates', 'CorrelationExcitatoryGates']

        correlationDict = {'CorrelationRecurrent': correlationRecurrent}
        # 'CorrelationInhibitoryGates': correlationInhibitoryGates,
        # 'CorrelationExcitatoryGates': correlationExcitatoryGates}

        for correlationName in correlationNames:
            modelName = model_dir.split('beRNN_')[-1]
            np.save(os.path.join(path, f'structural{correlationName}_{modelName}_{mode}.npy'),
                    correlationDict[correlationName])

            # Set up the figure
            fig = plt.figure(figsize=(10, 10))

            # Create the main similarity matrix plot
            matrix_left = 0.1
            matrix_bottom = 0.3
            matrix_width = 0.6
            matrix_height = 0.6

            ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
            im = ax_matrix.imshow(correlationDict[correlationName], cmap='coolwarm', interpolation='nearest', vmin=-1,
                                  vmax=1)  # info: change here

            # Add title
            # subject = '_'.join(model_dir.split("\\")[-1].split('_')[0:4])
            ax_matrix.set_title(f'Structural Correlation - {month}', fontsize=22, pad=20)  # info: change here

            # Add x-axis and y-axis labels
            ax_matrix.set_xlabel('Hidden weights', fontsize=16, labelpad=15)
            ax_matrix.set_ylabel('Hidden weights', fontsize=16, labelpad=15)

            # Remove x and y ticks
            ax_matrix.set_xticks([])  # Disable x-ticks
            ax_matrix.set_yticks([])  # Disable y-ticks

            # Create the colorbar on the right side, aligned with the matrix
            colorbar_left = matrix_left + matrix_width + 0.02
            colorbar_width = 0.03

            ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
            cb = plt.colorbar(im, cax=ax_cb)
            cb.set_ticks([-1, 1])
            cb.outline.set_linewidth(0.5)
            cb.set_label('Correlation', fontsize=18, labelpad=0)  # info: change here

            # # Set the title above the similarity matrix, centered
            # if mode == 'Training':
            #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
            # elif mode == 'Evaluation':
            #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'

            # ax_matrix.set_title(title, fontsize=14, pad=20)
            # Save the figure with a tight bounding box to ensure alignment
            # save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNNmodels\\Visuals\\Similarity\\finalReport',
            #                          model_dir.split("\\")[-1] + '_' + 'Similarity' + '.png')
            # save_path = os.path.join(
            #     'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\Visuals\\CorrelationStructure\\BarnaModels',
            #     model_dir.split("\\")[-1] + '_' + 'CorrelationStructure' + '.png')
            plt.savefig(os.path.join(figurePath, f'{correlationName}_{modelName}_{month}_{mode}.png'), format='png',
                        dpi=300, bbox_inches='tight')  # info: change here

            plt.show()
            # plt.close()

# info: Adapted Yang functions #########################################################################################
class TaskSetAnalysis(object):
    """Analyzing the representation of tasks."""
    def __init__(self, model_dir, rules=None):
        """Initialization.

        Args:
            model_dir: str, model directory
            rules: None or a list of rules
        """
        # Stimulus-averaged traces
        h_stimavg_byrule = OrderedDict()
        h_stimavg_byepoch = OrderedDict()
        # Last time points of epochs
        h_lastt_byepoch = OrderedDict()

        model = Model(model_dir)
        hp = model.hp

        if rules is None:  # Default value - all tasks
            rules = hp['rules']

        n_rules = len(rules)

        # Define main path
        path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'

        # Define data path
        preprocessedData_path = os.path.join(path, 'Data', hp['participant'], hp['data'])  # pandora

        with tf.Session() as sess:
            model.restore()

            for rule in rules:
                month = hp['monthsConsidered'][-1]
                train_data, test_data = _tools.createSplittedDatasets(hp, preprocessedData_path, month)

                x, y, y_loc, response = _tools.load_trials(hp['rng'], rule, 'test', hp['batch_size'], test_data,
                                                          False)  # y_loc is participantResponse_perfEvalForm
                c_mask = _tools.create_cMask(y, response, hp, 'test')
                feed_dict = _tools.gen_feed_dict(model, x, y, c_mask, hp)

                h = sess.run(model.h, feed_dict=feed_dict)  # info: Trainables are actualized - train_step should represent the step in _training.py and the global_step in network.py

                # c_lsq, c_reg, y_hat_test = sess.run([model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict)

                # Average across trials
                h_stimavg = h.mean(axis=1)

                # dt_new = 50
                # every_t = int(dt_new/hp['dt'])

                t_start = int(500 / hp['dt'])  # Important: Ignore the initial transition
                # Extract epoch of interest - most often response epoch
                h_stimavg_byrule[rule] = h_stimavg[t_start:, :]

                fixation_steps, response_steps = _tools.getEpochSteps(y)

                # Take epoch
                e_time_start = fixation_steps
                e_time_end = fixation_steps + response_steps
                e_time = [e_time_start, e_time_end]
                e_name = 'response'

                e_time_start = e_time[0] - 1 if e_time[0] > 0 else 0
                h_stimavg_byepoch[(rule, e_name)] = h_stimavg[e_time_start:e_time[1], :]
                # Take last time point from epoch
                # h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[0]:e_time[1],:,:][-1], axis=1)
                h_lastt_byepoch[(rule, e_name)] = h[e_time[1] - 1, :, :]

        self.rules = rules
        self.h_stimavg_byrule = h_stimavg_byrule
        self.h_stimavg_byepoch = h_stimavg_byepoch
        self.h_lastt_byepoch = h_lastt_byepoch
        self.model_dir = model_dir

    @staticmethod  # utility function within class, no method
    def filter(h, rules=None, epochs=None, non_rules=None, non_epochs=None, get_lasttimepoint=True, get_timeaverage=False, **kwargs):
        # h should be a dictionary
        # get a new dictionary containing keys from the list of rules and epochs
        # And avoid epochs from non_rules and non_epochs
        # h_new = OrderedDict([(key, val) for key, val in h.items() if key[1] in epochs])

        if get_lasttimepoint:
            print('Analyzing last time points of epochs')
        if get_timeaverage:
            print('Analyzing time-averaged activities of epochs')

        h_new = OrderedDict()
        for key in h:
            rule, epoch = key

            include_key = True
            if rules is not None:
                include_key = include_key and (rule in rules)

            if epochs is not None:
                include_key = include_key and (epoch in epochs)

            if non_rules is not None:
                include_key = include_key and (rule not in non_rules)

            if non_epochs is not None:
                include_key = include_key and (epoch not in non_epochs)

            if include_key:
                if get_lasttimepoint:
                    h_new[key] = h[key][np.newaxis, -1, :]
                elif get_timeaverage:
                    h_new[key] = np.mean(h[key], axis=0, keepdims=True)
                else:
                    h_new[key] = h[key]

        return h_new

    def compute_taskspace(self, fname, dim_reduction_type, epochs, **kwargs):
        # Only get last time points for each epoch
        h = self.filter(self.h_stimavg_byepoch, epochs=epochs, rules=self.rules, **kwargs)

        # Concatenate across rules to create dataset
        data = np.concatenate(list(h.values()), axis=0)
        data = data.astype(dtype='float64')

        # First reduce dimension to dimension of data points
        from sklearn.decomposition import PCA
        n_comp = int(np.min([data.shape[0], data.shape[1]]) - 1)
        model = PCA(n_components=n_comp)
        data = model.fit_transform(data)

        if dim_reduction_type == 'PCA':
            model = PCA(n_components=2)

        elif dim_reduction_type == 'MDS':
            from sklearn.manifold import MDS
            model = MDS(n_components=2, metric=True, random_state=0)

        elif dim_reduction_type == 'TSNE':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=2, init='pca',
                         verbose=1, method='exact', learning_rate=100, perplexity=5)

        elif dim_reduction_type == 'IsoMap':
            from sklearn.manifold import Isomap
            model = Isomap(n_components=2)

        else:
            raise ValueError('Unknown dim_reduction_type')

        # Transform data
        data_trans = model.fit_transform(data)

        # Package back to dictionary
        h_trans = OrderedDict()
        i_start = 0
        for key, val in h.items():
            i_end = i_start + val.shape[0]
            h_trans[key] = data_trans[i_start:i_end, :]
            i_start = i_end

        # save file
        with open(fname, "wb") as f:
            pickle.dump(h_trans, f)

        return h_trans

    def plot_taskspace(self, h_trans, directory, sort_variable, mode, fig_name, level, plot_example=False, lxy=None, plot_arrow=True, **kwargs):
        figsize = (5, 5)
        fs = 7  # fontsize
        dim0, dim1 = (0, 1)  # plot dimensions
        i_example = 0  # index of the example to plot

        texts = list()

        maxv0, maxv1 = -1, -1

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.2, 0.2, 0.65, 0.65])

        for key, val in h_trans.items():
            rule, epoch = key
            # Default coloring by rule_color
            color = rule_color[rule]

            if plot_example:
                xplot, yplot = val[i_example, dim0], val[i_example, dim1]
            else:
                xplot, yplot = val[:, dim0], val[:, dim1]

            # ax.plot(xplot, yplot, 'o', color=color, mec=color, mew=1.0, ms=2)
            ax.plot(xplot, yplot, 'o', color=color, mec=color, mew=0.5, ms=5, alpha=0.4)

            xtext = np.mean(val[:, dim0])
            if np.mean(val[:, dim1]) > 0:
                ytext = np.max(val[:, dim1])
                va = 'bottom'
            else:
                ytext = np.min(val[:, dim1])
                va = 'top'

            texts.append(ax.text(xtext * 1.1, ytext * 1.1, _tools.rule_name[rule],
                                 fontsize=6, color=color, alpha=0.6,
                                 horizontalalignment='center', verticalalignment=va))

            maxv0 = np.max([maxv0, np.max(abs(val[:, dim0]))])
            maxv1 = np.max([maxv1, np.max(abs(val[:, dim1]))])

        if lxy is None:
            lx = np.ceil(maxv0)
            ly = np.ceil(maxv1)
        else:
            lx, ly = lxy

        ax.tick_params(axis='both', which='major', labelsize=fs)
        # plt.locator_params(nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.margins(0.1)
        # plt.axis('equal')
        plt.xlim([-lx, lx])
        plt.ylim([-ly, ly])
        ax.plot([0, 0], [-ly, ly], '--', color='gray')
        ax.plot([-lx, lx], [0, 0], '--', color='gray')
        ax.set_xticks([-lx, lx])
        ax.set_yticks([-ly, ly])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        pc_name = 'rPC'
        ax.set_xlabel(pc_name + ' {:d}'.format(dim0 + 1), fontsize=fs, labelpad=-5)
        ax.set_ylabel(pc_name + ' {:d}'.format(dim1 + 1), fontsize=fs, labelpad=-5)

        finalDirectory = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'compositionalRepresentation')
        os.makedirs(finalDirectory, exist_ok=True)
        if level == 'individual':
            plt.savefig(os.path.join(finalDirectory, fig_name + '.png'), transparent=True, dpi=300)
        elif level == 'group':
            plt.savefig(os.path.join(finalDirectory, '2DtaskRepresentation_overview.png'), transparent=True, dpi=300)
        # plt.show()

    def collect_h_trans(self, h_trans, h_trans_all, model):
        # Collect h_trans values
        h_trans_values = list(h_trans.values())

        # rotation_matrix, clock wise
        get_angle = lambda vec: np.arctan2(vec[1], vec[0])
        theta = get_angle(h_trans_values[0][0])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        for key, val in h_trans.items():
            val_rot = np.dot(val, rot_mat)
            if key in h_trans_all:
                h_trans_all[key] = np.concatenate((h_trans_all[key], val_rot), axis=0)
            else:
                h_trans_all[key] = val_rot

        h_trans_values = list(h_trans_all.values())
        if h_trans_values[1][0][1] < 0:
            for key, val in h_trans_all.items():
                h_trans_all[key] = val * np.array([1, -1])

        return h_trans_all

def plot_taskVariance_and_lesioning(directory, mode, sort_variable, rdm_metric, robustnessTest, batch, numberOfModels):
    # Colors used for clusters
    kelly_colors = \
        [np.array([0.94901961, 0.95294118, 0.95686275]),
         np.array([0.13333333, 0.13333333, 0.13333333]),
         np.array([0.95294118, 0.76470588, 0.]),
         np.array([0.52941176, 0.3372549, 0.57254902]),
         np.array([0.95294118, 0.51764706, 0.]),
         np.array([0.63137255, 0.79215686, 0.94509804]),
         np.array([0.74509804, 0., 0.19607843]),
         np.array([0.76078431, 0.69803922, 0.50196078]),
         np.array([0.51764706, 0.51764706, 0.50980392]),
         np.array([0., 0.53333333, 0.3372549]),
         np.array([0.90196078, 0.56078431, 0.6745098]),
         np.array([0., 0.40392157, 0.64705882]),
         np.array([0.97647059, 0.57647059, 0.4745098]),
         np.array([0.37647059, 0.30588235, 0.59215686]),
         np.array([0.96470588, 0.65098039, 0.]),
         np.array([0.70196078, 0.26666667, 0.42352941]),
         np.array([0.8627451, 0.82745098, 0.]),
         np.array([0.53333333, 0.17647059, 0.09019608]),
         np.array([0.55294118, 0.71372549, 0.]),
         np.array([0.39607843, 0.27058824, 0.13333333]),
         np.array([0.88627451, 0.34509804, 0.13333333]),
         np.array([0.16862745, 0.23921569, 0.14901961])]

    if robustnessTest == False:
        txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f'bestModels_{sort_variable}_{mode}.txt')
    else:
        txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', batch, f'bestModels_{sort_variable}_{mode}_{batch}.txt')

    with open(txtFile, "r") as file:
        lines = file.read().splitlines()
    cleaned_lines = [line.strip().strip('\'",') for line in lines]

    for model in range(1, numberOfModels + 1):
        best_model_dir = cleaned_lines[model]  # Choose model of interest, starting with [1]

        dataFolder = define_data_folder(best_model_dir.split('_'))

        participant = [i for i in best_model_dir.split('\\') if 'beRNN_' in i][0]
        data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data', participant, dataFolder)

        hp = _tools.load_hp(best_model_dir)
        # _tools.save_hp(hp_copy, best_model_dir) # pop out rng main reason for saving again
        # hp = _tools.load_hp(best_model_dir)
        layer = [1 if hp['multiLayer'] == False else 3][0]

        # info: Create TaskVariance Plot
        knowledgeBase = clustering.Analysis(data_dir, best_model_dir, layer, rdm_metric, 'test', hp['monthsConsidered'],'rule', True)
        figurePath = os.path.join(directory, 'visuals')

        # Plot task variance anyway
        knowledgeBase.plot_variance(best_model_dir, os.path.join(directory, 'visuals'), mode_=f'{model}_{sort_variable}_{mode}')
        # But skip multiLayer
        if hp['multiLayer'] == True:
            continue
        else:
            knowledgeBase.plot_lesions(data_dir, best_model_dir, figurePath, mode_=f'{model}_{sort_variable}_{mode}')

def plot_group_rdm_mds(directory, mode, sort_variable, rdm_metric, numberOfModels, ruleset):
    def plot_rdm_heatmap(rdm, metric, task_labels=None, title='RDM Heatmap'):
        n_tasks = rdm.shape[0]
        fig_size = max(6, n_tasks * 0.5)  # auto-scale figure for more tasks

        plt.figure(figsize=(fig_size, fig_size * 0.85))
        ax = sns.heatmap(rdm, annot=False, cmap='viridis', square=True,
                         xticklabels=task_labels, yticklabels=task_labels,
                         cbar_kws={'shrink': 0.7})

        plt.title(title + ' - ' + metric, fontsize=12)
        plt.xlabel('Tasks', fontsize=10)
        plt.ylabel('Tasks', fontsize=10)

        # Rotate x-axis labels and adjust font sizes
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)

        plt.tight_layout()
        # plt.show()

        return plt

    def plot_rdm_mds(rdm, metric, task_keys, rule_color, title='RDM MDS'):
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42) # fix: check if right scipy package was imported
        coords = mds.fit_transform(rdm)

        plt.figure(figsize=(5.2, 4.5))
        ax = plt.gca()

        for i, key in enumerate(task_keys):
            x, y = coords[i]
            color = rule_color.get(key, 'gray')
            label = key

            # Dot
            ax.scatter(x, y, color=color, s=60, edgecolor='black', zorder=3)

            # Label outside the dot, same color
            ax.text(x + 0.035, y, label, fontsize=6.5,
                    color=color, ha='left', va='center', zorder=4)

        # Coordinate system (axes on, ticks small and clean)
        ax.tick_params(axis='both', which='major', labelsize=6, length=2)
        ax.set_xlabel('MDS Dimension 1', fontsize=7.5)
        ax.set_ylabel('MDS Dimension 2', fontsize=7.5)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        plt.title(title + ' - ' + metric, fontsize=9)
        plt.tight_layout()
        # plt.show()

    all_coords = []
    all_keys = []
    all_models = []
    # Apply non-linear dimensionality reduction that maps element-wise distance as good as possible into n-dim space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)  # euclidean distance based by default

    if robustnessTest == False: # fix: legacy stuff - might not work anymore esp. w. cleaned_lines list
        txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f'bestModels_{sort_variable}_{mode}.txt')
    else:
        txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', batch, f'bestModels_{sort_variable}_{mode}_{batch}.txt')

    with open(txtFile, "r") as file:
        lines = file.read().splitlines()
    cleaned_lines = [line.strip().strip('\'",') for line in lines]

    for model in range(1, numberOfModels+1):  # Best of x models - max. 256 - always start with 1
        best_model_dir = cleaned_lines[model]

        if best_model_dir == '':
            print('No best model saved in text file.')
            exit()

        dataFolder = define_data_folder(best_model_dir.split('_'))

        participant = [i for i in best_model_dir.split('\\') if 'beRNN_' in i and len(i) == 8][0]

        data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data', participant, dataFolder)

        hp = _tools.load_hp(best_model_dir)
        layer = [1 if hp['multiLayer'] == False else 3][0]

        # Create task variance matrix for current model in loop
        knowledgeBase = clustering.Analysis(data_dir, best_model_dir, layer, rdm_metric, 'test', hp['monthsConsidered'],'rule', True)

        # # Skip dummy RDMs
        # if np.allclose(knowledgeBase.rdm, knowledgeBase.rdm[0, 0]):
        #     print(f"Skipping model {model} from final plot due to dummy RDM (constant dissimilarity).")
        #     continue

        if model == 1:
            coords_ref = mds.fit_transform(knowledgeBase.rdm)
            coords_aligned = coords_ref
        else:
            coords_model = mds.fit_transform(knowledgeBase.rdm)
            R, _ = orthogonal_procrustes(coords_model, coords_ref)
            coords_aligned = coords_model @ R

        all_coords.append(coords_aligned)
        all_keys.append(knowledgeBase.keys)
        all_models.append(knowledgeBase.model_dir)

        # # info: Plot also single performance plots for each of the chosen models
        # rule_plot = [i for i in hp['rule_prob_map'] if hp['rule_prob_map'][i] > 0]
        # performanceTest_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'performanceTest')
        # performanceTrain_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'performanceTrain')
        if robustnessTest == True:
            representationalDissimilarity_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', str(batch),
                                                             f'representationalDissimilarity_{rdm_metric}_{ruleset}')
        else:
            representationalDissimilarity_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}',
                                                             f'representationalDissimilarity_{rdm_metric}_{ruleset}')

        # os.makedirs(performanceTest_dir, exist_ok=True)
        # os.makedirs(performanceTrain_dir, exist_ok=True)
        os.makedirs(representationalDissimilarity_dir, exist_ok=True)

        # save rdm
        np.save(os.path.join(representationalDissimilarity_dir,
                             f'{model}_{sort_variable}_{mode}' + '_' + 'batch_' + best_model_dir.split("\\")[-5] + '_' +
                             best_model_dir.split("\\")[-3].split('_')[
                                 -4] + f'_rdmArray_{knowledgeBase.rdm_metric}_{ruleset}.npy'), knowledgeBase.rdm)

        # info: Create RDM Heatmaps and 2D representations
        label_list = [_tools.rule_name[key] for key in knowledgeBase.keys]
        fig_rdm = plot_rdm_heatmap(knowledgeBase.rdm, knowledgeBase.rdm_metric, task_labels=label_list)
        fig_rdm.savefig(os.path.join(representationalDissimilarity_dir,
                                     f'{model}_{sort_variable}_{mode}' + '_' + 'batch_' + best_model_dir.split("\\")[
                                         -5] + '_' + best_model_dir.split("\\")[-3].split('_')[
                                         -4] + f'_representationalDissimilarity_{knowledgeBase.rdm_metric}_{ruleset}.png'),
                        format='png', dpi=300)
        fig_rdm_mds = plot_rdm_mds(knowledgeBase.rdm, knowledgeBase.rdm_metric, task_keys=knowledgeBase.keys,
                                   rule_color=rule_color)
        fig_rdm.savefig(os.path.join(representationalDissimilarity_dir,
                                     f'{model}_{sort_variable}_{mode}' + '_' + 'batch_' + best_model_dir.split("\\")[
                                         -5] + '_' + best_model_dir.split("\\")[-3].split('_')[
                                         -4] + f'_representationalDissimilarity_{knowledgeBase.rdm_metric}_{ruleset}_2DspaceGeometry.png'),
                        format='png', dpi=300)

    if numberOfModels > 1:
        plt.figure(figsize=(5.2, 4.5))
        ax = plt.gca()

        # Track which keys have already been labeled
        labeled_keys = set()

        for coords, keys in zip(all_coords, all_keys):
            for x, y, key in zip(coords[:, 0], coords[:, 1], keys):
                color = rule_color.get(key, 'gray')
                # Only set label for the first occurrence
                label = key if key not in labeled_keys else None

                ax.scatter(x, y, color=color, s=60, edgecolor='black', zorder=3, alpha=0.6)
                ax.text(x + 0.035, y, label, fontsize=6.5,
                        color=color, ha='left', va='center', zorder=4)

        # Only show unique legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6.5, loc='best')

        ax.tick_params(axis='both', which='major', labelsize=6, length=2)
        ax.set_xlabel('MDS Dimension 1', fontsize=7.5)
        ax.set_ylabel('MDS Dimension 2', fontsize=7.5)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        plt.title(f'RDM MDS – {knowledgeBase.rdm_metric} – Multiple Aligned Models', fontsize=9)
        plt.tight_layout()
        # plt.show()

        plt.savefig(os.path.join(representationalDissimilarity_dir, best_model_dir.split("\\")[-3].split('_')[
            -4] + f'_representationalDissimilarity_{knowledgeBase.rdm_metric}_{ruleset}_2DspaceGeometry_modelAlignment_{numberOfModels}.png'), format='png', dpi=300)
        plt.close()
# info: Adapted Yang functions #########################################################################################

if __name__ == "__main__":
    ########################################################################################################################
    # head. Create Overview and topologcial Marker
    ########################################################################################################################
    def figureSceletton(model_column_widths):
        total_cols = sum(model_column_widths)
        fig = plt.figure(figsize=(3 * total_cols, 16))  # Adjust width proportionally
        gs = fig.add_gridspec(5, total_cols, height_ratios=[1, 1, 1, 1, 0.4])

        fig.text(0.4975, 0.955, 'TRAIN', fontsize=12, fontweight='bold',
                 verticalalignment='center', horizontalalignment='center', color='black')
        fig.text(0.4975, 0.52, 'TEST', fontsize=12, fontweight='bold',
                 verticalalignment='center', horizontalalignment='center', color='black')

        axs = [[] for _ in range(5)]  # rows 0 to 4
        current_col = 0

        for model_width in model_column_widths:
            # Row 0: train cost/perf (2 cols)
            axs[0].append(fig.add_subplot(gs[0, current_col:current_col + model_width]))
            # Row 1: train functional correlation (n layers)
            axs[1].extend([fig.add_subplot(gs[1, current_col + i]) for i in range(model_width)])
            # Row 2: test cost/perf (2 cols)
            axs[2].append(fig.add_subplot(gs[2, current_col:current_col + model_width]))
            # Row 3: test functional correlation (n layers)
            axs[3].extend([fig.add_subplot(gs[3, current_col + i]) for i in range(model_width)])

            current_col += model_width

        # Row 4: legend and HP (2 subplots only)
        axs[4] = [
            fig.add_subplot(gs[4, 0:total_cols // 2]),
            fig.add_subplot(gs[4, total_cols // 2:total_cols])
        ]

        return fig, axs

    # fix: sort all model_month_XX into one folder for each participant x dataset
    folder, dataType, participant, batch, months = '_fs_mT_beRNN_04_highDimCorrects_256_hp_9__1-12', 'highDim_correctOnly', 'beRNN_04', '1', ['month_1', 'month_2', 'month_6']
    _finalPath = Path('C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels', f'{folder}/{dataType}/{participant}/{batch}')
    _data_dir = Path('C:/Users/oliver.frank/Desktop/PyProjects/Data')
    topMarker_saving = True
    # threshold = 0.1

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

        # Define paths
        participant_id = participant  # Change this for each participant
        topMarker_path = os.path.join(_finalPath, 'overviews', 'all_topologicalMarker_files')
        os.makedirs(topMarker_path, exist_ok=True)
        # Create overview folder
        overviewFolder = Path(_finalPath, 'overviews')
        os.makedirs(overviewFolder, exist_ok=True)
        # Create model list for every iteration
        _model_list = os.listdir(_finalPath)
        _model_list = [i for i in _model_list if 'beRNN' in i]

        # fix: Add list of top Marker lists here
        modList_list = []
        clusteringList_list = []
        participationList_list = []

        for _model in _model_list:
            try:
                # Re-Initialize for each model
                model_column_widths = []
                _model_paths = []

                model_dir = os.path.join(_finalPath, _model)
                model_list = sorted([i for i in os.listdir(model_dir) if 'model' in i])

                # Loop through all models per months for each model
                for model in model_list:
                    currentModelDirectory = os.path.join(model_dir, model)
                    currentHP = _tools.load_hp(currentModelDirectory)

                    if currentHP.get('multiLayer'):
                        n_layers = len(currentHP['n_rnn_per_layer'])
                    else:
                        n_layers = 1

                    block_width = max(n_layers, 3)
                    model_column_widths.append(block_width)
                    _model_paths.append((_model, model))

                # Create dynamic figure
                fig, axs = figureSceletton(model_column_widths)

                # Now process each model and plot
                col_start = 0
                for model_idx, (_model, model) in enumerate(_model_paths):
                    model_dir = os.path.join(_finalPath, _model)
                    currentModelDirectory = os.path.join(model_dir, model)
                    currentHP = _tools.load_hp(currentModelDirectory)

                    dataFolder = define_data_folder(_model.split('_'))

                    data_dir = os.path.join(_data_dir, participant, dataFolder)
                    rule_plot = [i for i in currentHP['rule_prob_map'] if currentHP['rule_prob_map'][i] > 0]

                    # Plot performance and cost
                    img_perf_test = plot_performanceprogress_test_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)
                    img_perf_train = plot_performanceprogress_train_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)
                    img_cost_test = plot_cost_test_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)
                    img_cost_train = plot_cost_train_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)

                    # Plot correlations
                    n_layers = len(currentHP['n_rnn_per_layer']) if currentHP.get('multiLayer') else 1
                    funcTrainList, funcTestList = [], []
                    for layer in range(n_layers):
                        try:
                            analysis_train = clustering.Analysis(data_dir, currentModelDirectory, layer,  'cosine', 'train', currentHP['monthsConsidered'], 'rule', True)
                            analysis_test = clustering.Analysis(data_dir, currentModelDirectory, layer, 'cosine','test', currentHP['monthsConsidered'], 'rule', True)

                            func_train, *_ = compute_functionalCorrelation(currentModelDirectory, threshold, currentHP['monthsConsidered'], 'train', None, analysis_train)
                            func_test, avg_mod_test, clustering_test, participation_test = compute_functionalCorrelation(currentModelDirectory, threshold, currentHP['monthsConsidered'], 'test', None, analysis_test)

                            # info: Append test top. Markers into a list and save them in folder #######################################
                            modList = []
                            clusteringList = []
                            participationList = []
                            modList.append(avg_mod_test)
                            clusteringList.append(clustering_test)
                            participationList.append(participation_test)

                            # info: meta lists
                            modList_list.append(avg_mod_test)
                            clusteringList_list.append(clustering_test)
                            participationList_list.append(participation_test)

                            topMarkerList = [clusteringList, modList, participationList]
                            topMarkerNamesList = ['clusteringList', 'modList', 'participationList']
                            for i in range(0, len(topMarkerNamesList)):
                                mean_value = np.mean(topMarkerList[i])
                                variance_value = np.var(topMarkerList[i])
                                # mean_variance = np.array([mean_value, variance_value])
                                np.save(os.path.join(topMarker_path, f'{topMarkerNamesList[i]}_{_model}_{model}_layer{layer}.npy'), topMarkerList[i])
                            # info: ####################################################################################################

                            funcTrainList.append(fig_to_array(func_train))
                            funcTestList.append(fig_to_array(func_test))

                            plt.close(func_train)
                            plt.close(func_test)

                        except Exception as e:
                            print(f"Skipping layer {layer} for model {currentModelDirectory} due to error: {e}")
                            # You can optionally append blank images or zero arrays to keep the grid layout consistent
                            placeholder = np.zeros((currentHP['n_rnn_per_layer'][layer], currentHP['n_rnn_per_layer'][layer], 3), dtype=np.uint8)  # black placeholder RGB image
                            funcTrainList.append(placeholder)
                            funcTestList.append(placeholder)

                    # Place plots into skeleton
                    combined_train = np.concatenate((img_cost_train, img_perf_train), axis=1)
                    axs[0][model_idx].imshow(combined_train)
                    combined_test = np.concatenate((img_cost_test, img_perf_test), axis=1)
                    axs[2][model_idx].imshow(combined_test)

                    for i in range(n_layers):
                        axs[1][col_start + i].imshow(funcTrainList[i])
                        axs[3][col_start + i].imshow(funcTestList[i])

                    # Clean up axes
                    for row in range(4):
                        for i in range(model_column_widths[model_idx]):
                            if row == 0 or row == 2:
                                axs[row][model_idx].axis('off')
                            else:
                                axs[row][col_start + i].axis('off')

                    col_start += model_column_widths[model_idx]

                    # Add legend
                    # legend_img = create_legend_image()
                    # axs[4][0].imshow(legend_img)
                    # axs[4][0].axis("off")

                    # Add hyperparameter box
                    hp_lines = [f"{key}: {currentHP[key]}" for key in selected_hp_keys if key in currentHP]
                    num_columns = 3
                    hp_text_formatted = "\n".join(["   ".join(hp_lines[i:i + num_columns]) for i in range(0, len(hp_lines), num_columns)])
                    # fix: fontsize decides size of box and should be individual for number of layers - 3: 9
                    axs[4][1].text(0.5, 0.5, hp_text_formatted, fontsize=9, verticalalignment='center',
                                   horizontalalignment='center', color='black',
                                   bbox=dict(facecolor='white', alpha=1, edgecolor='black'))
                    axs[4][1].axis("off")

                plt.tight_layout()
                overview_path = os.path.join(overviewFolder, f'{_model}_OVERVIEW.png')
                plt.savefig(overview_path, dpi=150, bbox_inches='tight')
                # plt.close(fig)

                print(f"Overview saved to: {overview_path}")

            except Exception as e:
                print("An exception occurred with model number: ", model_dir, model)
                print("Error: ", e)


        #################################################################################################################
        # head. Distribution plot
        #################################################################################################################

        # Collect all top Marker files into one destination
        _iterationList = os.listdir(_finalPath) # info: Folder of several iterations for one training batch of one participant
        iterationList = [os.path.join(_finalPath, iteration, 'topologicalMarker')
                         for iteration in _iterationList if 'beRNN' in iteration]

        npy_files = glob.glob(os.path.join(topMarker_path, "*.npy"))
        # Extract markers and months from filenames
        topMarkers = ['modList', 'clusteringList', 'participationList']

        num_rows = len(topMarkers)
        num_columns = len(months)

        # Group files based on (marker, month)
        groups = {marker: {month: [] for month in months} for marker in topMarkers}
        for file in npy_files:
            for marker in topMarkers:
                for month in months:
                    if marker in file and month in file:
                        groups[marker][month].append(os.path.join(topMarker_path, file))

        # Create figure and axes
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(6, 1.5 * num_rows), sharex=False, sharey=False)

        # Ensure axes is always a 2D array
        if num_rows == 1:
            axes = np.array([axes])
        if num_columns == 1:
            axes = np.expand_dims(axes, axis=1)

        # Directory to save distributions
        distribution_dir = os.path.join(_finalPath, 'overviews', "topologicalMarker_lists")
        os.makedirs(distribution_dir, exist_ok=True)

        # Dictionary to store t-test results
        t_test_results = {}

        # Process and plot distributions
        for row, marker in enumerate(topMarkers):
            t_test_results[marker] = {}

            for col, month in enumerate(months):
                files = groups[marker][month]
                ax = axes[row, col]

                if files:
                    all_data = []
                    for file in files:
                        try:
                            data = np.load(file)
                            all_data.append(data)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")

                    if all_data:
                        # Convert to NumPy arrays and filter out empty data
                        all_data = [np.asarray(arr).flatten() for arr in all_data if arr.size > 0]

                        if all_data:
                            if len(all_data) == 1:
                                valid_data = all_data[0]
                            else:
                                valid_data = np.concatenate(all_data)

                            # Save distribution for later comparisons
                            np.save(os.path.join(distribution_dir, f"{marker}_{month}.npy"), valid_data)

                            mean, variance = np.mean(valid_data), np.var(valid_data)

                            # Plot histogram
                            ax.hist(valid_data, bins=20, density=False, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.axvline(mean, color='red', linestyle='dashed', linewidth=1.5,
                                       label=f'Mean: {mean:.2f}\nVar: {variance:.2f}')
                            ax.set_title(f"{marker} - {month}", fontsize=12)
                            ax.legend(fontsize=6)

                            ax.tick_params(axis='both', which='major', labelsize=6)  # Adjust size as needed
                            ax.tick_params(axis='both', which='minor', labelsize=6)  # Even smaller for minor ticks

                            # Store t-test results
                            if col > 0:
                                prev_files = groups[marker][months[col - 1]]

                                prev_data_list = []
                                for prev_file in prev_files:
                                    try:
                                        prev_data_list.append(np.load(prev_file))
                                    except Exception as e:
                                        print(f"Error loading {prev_file}: {e}")

                                # Ensure previous data is valid
                                if prev_data_list:
                                    prev_data_list = [np.asarray(arr).flatten() for arr in prev_data_list if arr.size > 0]
                                    if prev_data_list:
                                        prev_data = np.concatenate(prev_data_list) if len(prev_data_list) > 1 else prev_data_list[0]

                                        if len(prev_data) > 1 and len(valid_data) > 1:
                                            t_stat, p_value = ttest_ind(prev_data, valid_data, equal_var=False)
                                            t_test_results[marker][(months[col - 1], month)] = (t_stat, p_value)

                                            if p_value < 0.05:
                                                ax.annotate(f'* p={p_value:.2e}', xy=(0.5, 0.55), xycoords='axes fraction',
                                                            fontsize=6, fontweight='bold', ha='center', color='red')
                        else:
                            ax.set_title(f"{marker} - {month} (No Valid Data)", fontsize=10)
                    else:
                        ax.set_title(f"{marker} - {month} (No Data)", fontsize=10)
                else:
                    ax.set_title(f"{marker} - {month} (No Data)", fontsize=10)

        # Adjust layout and save plot
        plt.tight_layout()
        plt.suptitle(f"Distributions for {participant_id}", fontsize=12, fontweight='bold', y=1.02)

        # Save the figure
        plot_path = os.path.join(_finalPath, 'overviews', f"topologicalMarkers_distribution_{participant_id}_{dataType}_{batch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.show()

        print(f"Plot saved at: {plot_path}")
        print(f"Distributions saved for participant {participant_id} in {distribution_dir}")


        if comparison == True:
            # #################################################################################################################
            # # head: Comparison - Only apply after _analysis
            # #################################################################################################################

            # Define variables for topological marker distribution comparison
            participant1, batch1 = 'beRNN_01', '32'
            participant2, batch2 = 'beRNN_03', '32'

            def load_distributions(distribution_dir,topMarkers,months):
                """
                Load saved distributions for a given participant.
                """
                distributions = {}

                if not os.path.exists(distribution_dir):
                    print(f"No distributions found for {distribution_dir}")
                    return None

                for file in os.listdir(distribution_dir):
                    if file.endswith(".npy"):
                        for marker in topMarkers:
                            for month in months:
                                # Safely initialize nested dict
                                if marker not in distributions:
                                    distributions[marker] = {}
                                distributions[marker][month] = np.load(os.path.join(distribution_dir, file))

                return distributions

            def compare_participants(dist_1, dist_2, participant_1, participant_2, destination_dir):
                """
                Compare the distributions of two participants and display significance.
                """
                p_values = {}  # Store p-values for visualization

                for marker in dist_1.keys():
                    p_values[marker] = {}

                    for month in dist_1[marker].keys():
                        if marker in dist_2 and month in dist_2[marker]:  # Ensure both have data
                            data_1 = dist_1[marker][month]
                            data_2 = dist_2[marker][month]

                            if len(data_1) > 1 and len(data_2) > 1:
                                # Perform statistical tests
                                t_stat, p_ttest = ttest_ind(data_1, data_2, equal_var=False)
                                ks_stat, p_ks = ks_2samp(data_1, data_2)

                                p_values[marker][month] = min(p_ttest, p_ks)  # Store min p-value
                            else:
                                p_values[marker][month] = 1.0  # No valid comparison

                # Convert to DataFrame for visualization
                p_df = pd.DataFrame(p_values).T  # Transpose so markers are rows, months are columns

                # Prepare text annotations with significance levels
                def format_p_value(p):
                    if p < 0.001:
                        return f"$\\bf{{{p:.3f}}}$***"  # Bold + ***
                    elif p < 0.01:
                        return f"$\\bf{{{p:.3f}}}$**"  # Bold + **
                    elif p < 0.05:
                        return f"$\\bf{{{p:.3f}}}$*"  # Bold + *
                    else:
                        return f"{p:.3f}"  # No bold

                annotations = p_df.applymap(format_p_value)

                # Plot heatmap of p-values
                plt.figure(figsize=(10, 6))
                ax = sns.heatmap(
                    p_df.astype(float),
                    annot=annotations,
                    fmt="",
                    cmap="magma",  # Reverse "magma" so low p-values are lighter
                    vmin=0.001,
                    vmax=1.0,
                    center=0.05,
                    cbar_kws={"shrink": 1.0},  # Fix legend error
                    annot_kws={"fontsize": 10, "color": "white"},  # Ensure all text is white
                )

                plt.title(f"Statistical Comparison: {participant_1} vs {participant_2}")
                plt.xlabel("Months")
                plt.ylabel("Topological Markers")

                # Save and show the plot
                plot_path = os.path.join(destination_dir, '_topologicalMarkerComparison')
                os.makedirs(plot_path, exist_ok=True)
                plt.savefig(os.path.join(plot_path, f"topMarkerComparison_{participant_1}_{participant_2}.png"), dpi=300, bbox_inches='tight')
                plt.show()


            destination_dir = Path('C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels', f'{folder}/{dataType}')
            os.makedirs(destination_dir, exist_ok=True)

            distributions_dir_participant_01 = f"{destination_dir}\\{participant1}\\{batch1}\\overviews\\topologicalMarker_lists"
            distributions_dir_participant_02 = f"{destination_dir}\\{participant2}\\{batch2}\\overviews\\topologicalMarker_lists"

            dist_1 = load_distributions(distributions_dir_participant_01,topMarkers,months)
            dist_2 = load_distributions(distributions_dir_participant_02,topMarkers,months)

            compare_participants(dist_1, dist_2, participant1, participant2, destination_dir)

            # optional
            # visualize_meanMatrix_singleModel(r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_robustnessTest_multiTask_beRNN_03_highDimCorrects_256_hp_2\highDim_correctOnly\beRNN_03\2\beRNN_03_AllTask_4-6_highDim_correctOnly_iter1_LeakyRNN_diag_256_relu\model_month_6\mean_test_lay1_rule_all.pkl')
            # optional
            # visualize_rdMatrix_singleModel(r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_robustnessTest_multiTask_beRNN_05_highDimCorrects_256_hp_2\highDim_correctOnly\beRNN_05\2\beRNN_05_AllTask_4-6_highDim_correctOnly_iter1_LeakyRNN_diag_256_relu\model_month_6\mean_test_lay1_rule_all.pkl')
            # optional
            # compute_structuralCorrelation(
            #     r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\robustnessTest\highDim_correctOnly\beRNN_01\0\beRNN_01_AllTask_4-6_data_highDim_correctOnly_iteration1_LeakyRNN_diag_256_softplus',
            #     r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\robustnessTest\highDim_correctOnly\beRNN_01\visuals\performance_test\structuralCorrelation',
            #     ['month_4', 'month_5', 'month_6'], 'test')
            # optional: adapted Yang functions

        # info: Specific saving of topological marker values for beRNN-brain comparison ****************************************
        if topMarker_saving == True:
            print(np.round(modList_list, 3))
            print(np.round(clusteringList_list, 3))
            print(np.round(participationList_list, 3))

            topologicalMarker_dict_beRNN = {
                "modularity": modList_list,
                "clustering": clusteringList_list,
                "participation": participationList_list
            }

            with open(os.path.join(r"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists", f'topologicalMarker_dict_{participant}_{dataType}_{threshold}.json'), 'w') as fp:
                json.dump(topologicalMarker_dict_beRNN, fp)

# # info: Specific creation of meta topologicalMarker_dict_beRNN
# participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
# # threshold = 1.0
#
# for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     meta_topologicalMarker_dict_beRNN = {}
#
#     for participant in participantList:
#         with open(os.path.join(r"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists", f'topologicalMarker_dict_{participant}_highDim_correctOnly_{threshold}.json'), 'r') as fp:
#             topologicalMarker_dict_beRNN = json.load(fp)
#         meta_topologicalMarker_dict_beRNN[participant] = topologicalMarker_dict_beRNN
#
#     with open(os.path.join(r"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists", f'topologicalMarker_dict_beRNN_highDim_correctOnly_{threshold}.json'), 'w') as fp:
#         json.dump(meta_topologicalMarker_dict_beRNN, fp)


