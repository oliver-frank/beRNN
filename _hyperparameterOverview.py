from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import defaultdict
from collections import OrderedDict
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import _tools
# from _analysis import variance
from _analysis import clustering
from _singleNetworkAnalysis import define_data_folder
# from _analysis import standard_analysis

import networkx as nx
from _training import apply_density_threshold
from networkx.algorithms.community import greedy_modularity_communities, modularity

'''
- Creates grid search hp overview for one network size 
- Create histogramms for each hyperparameter seperatly
- Evaluate top. Marker metrics for finding best hp for paper
'''

########################################################################################################################
# head: Create histogramms to visualize and investigate interrelations of hyperparameter, modularity and performance ###
########################################################################################################################
def compute_n_cluster(model_dirs, mode):
    successful_model_dirs = []

    for model_dir in model_dirs:
        print('')
        print('********************************************************************************************************')
        print(model_dir)
        try:
            hp = tools.load_hp(model_dir)
            log = tools.load_log(model_dir)
            dataFolder = define_data_folder(model_dir.split('_'))
            # participant = [i for i in model_dir.split('\\') if 'beRNN_' in i][0]
            participant = '_'.join(['beRNN', [string for string in model_dir.split('_') if '0' in string and len(string) == 2][0]])
            layer = [1 if hp['multiLayer'] == False else 3][0]

            # Important overwriting of incongruent information in hp between single and multiLayer architecture
            hp['n_rnn'] = hp['n_rnn_per_layer'][0] if hp.get('multiLayer') else hp['n_rnn']
            hp['activation'] = hp['activations_per_layer'][0] if hp.get('multiLayer') else hp['activation']
            tools.save_hp(hp, model_dir)


            # Average performance for training at the last time point ##################################################
            totalPerformanceTraining = 0
            totalPerformanceTesting = 0
            numberOfTasks = 0

            tasksToTakeIntoAccount = [i for i in hp['rule_prob_map'] if hp['rule_prob_map'][i] > 0]

            for key in log.keys():
                if 'perf_train' in key and 'avg' not in key and any(task in key for task in
                                                                    tasksToTakeIntoAccount):  # Only rule_prob_map > 0 are saved during training
                    totalPerformanceTraining += log[key][-1]
                    numberOfTasks += 1
            averageTotalPerformanceTraining = totalPerformanceTraining / numberOfTasks

            for key in log.keys():
                if 'perf_' in key and 'avg' not in key and any(task for task in tasksToTakeIntoAccount if
                                                               task == key.split('perf_')[
                                                                   -1]):  # All tasks in rule_prob_map are saved during training
                    totalPerformanceTesting += log[key][-1]
            averageTotalPerformanceTesting = totalPerformanceTesting / numberOfTasks

            log['avg_perf_train'] = averageTotalPerformanceTraining
            log['avg_perf_test'] = averageTotalPerformanceTesting
            # log['avg_perf_test'] = log['perf_avg'][-1]
            # Average performance for training at the last time point ##################################################


            # Define right data
            data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data', participant, dataFolder)
            rdm_metric = 'cosine'
            if mode == 'test':
                analysis = clustering.Analysis(data_dir, model_dir, layer, rdm_metric,'test', hp['monthsConsidered'], 'rule', True) # test performance
            elif mode == 'train':
                analysis = clustering.Analysis(data_dir, model_dir, layer, rdm_metric,'train', hp['monthsConsidered'], 'rule', True) # train performance

            log['n_cluster'] = 0  # Yang legacy - dummy value
            log['score'] = 0  # Yang legacy - dummy value
            # log['rdm'] = analysis.rdm.tolist() # not ideal for saving as json
            log['model_dir'] = model_dir
            tools.save_log(log)

            # except IOError:
                # Training never finished
                # assert log['perf_min'][-1] <= hp['target_perf']

            # _analysis.plot_example_unit()
            # _analysis.plot_variance()
            # _analysis.plot_2Dvisualization()

            successful_model_dirs.append(model_dir)
            print("done")
            print('********************************************************************************************************')
            print('')

        except Exception as e:
            print(f"An exception occurred in compute_n_cluster: {e}")
            # Overwrite existing log with fallback variables - never save_log without load_log as empirical data will be overwritten
            log = tools.load_log(model_dir)
            log['model_dir'] = model_dir # info. important for overwriting directories of server trained models
            # log = tools.load_log(r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_gridSearch_domainTask-DM_beRNN_03_highDim_16\highDim\beRNN_03\3\beRNN_03_AllTask_4-6_data_highDim_trainingBatch3_iteration2_LeakyRNN_16_relu\model_month_6\log.json')
            # hp = tools.load_hp(model_dir)

            # Create fallback rdm
            # try:
            #     rules = [key for key in hp["rule_prob_map"].keys() if hp["rule_prob_map"][key] != 0]
            #     n_tasks = len(rules)
            # except Exception as e:
            #     print(f"An exception occured when loading hp: {e}. n_tasks = 12")
            #     n_tasks = 12
            #
            # rdm = np.full((n_tasks, n_tasks), 0.5, dtype=float)
            # np.fill_diagonal(rdm, 0.0)
            # log['rdm'] = rdm.tolist() # not ideal for saving as json

            # Create other fallback variables
            log['n_cluster'] = 0 # Yang legacy
            log['score'] = 0 # Yang legacy


            # Average performance for training at the last time point ##################################################
            totalPerformanceTraining = 0
            totalPerformanceTesting = 0
            numberOfTasks = 0

            tasksToTakeIntoAccount = [i for i in hp['rule_prob_map'] if hp['rule_prob_map'][i] > 0]

            for key in log.keys():
                if 'perf_train' in key and 'avg' not in key and any(task in key for task in
                                                                    tasksToTakeIntoAccount):  # Only rule_prob_map > 0 are saved during training
                    totalPerformanceTraining += log[key][-1]
                    numberOfTasks += 1
            averageTotalPerformanceTraining = totalPerformanceTraining / numberOfTasks

            for key in log.keys():
                if 'perf_' in key and 'avg' not in key and any(task for task in tasksToTakeIntoAccount if
                                                               task == key.split('perf_')[
                                                                   -1]):  # All tasks in rule_prob_map are saved during training
                    totalPerformanceTesting += log[key][-1]
            averageTotalPerformanceTesting = totalPerformanceTesting / numberOfTasks

            log['avg_perf_train'] = averageTotalPerformanceTraining
            log['avg_perf_test'] = averageTotalPerformanceTesting
            # log['avg_perf_test'] = log['perf_avg'][-1]
            # Average performance for training at the last time point ##################################################


            tools.save_log(log)

            successful_model_dirs.append(model_dir)
            print("fallback done - dummy log created")
            print('********************************************************************************************************')
            print('')

    return successful_model_dirs

def get_n_clusters(model_dirs, density):
    # model_dirs = tools.valid_model_dirs(root_dir)
    hp_list = list()
    n_clusters = list()
    silhouette_score = list()
    avg_perf_train_list = list()
    avg_perf_test_list = list()
    avg_clustering_list = list()
    modularity_list_sparse = list()
    participation_coefficient_list = list()

    # info. add lists here for voxel analysis list *****************************************************************
    n_modules_list = list()
    global_efficiency_list = list()
    node_degree_list = list()
    betweenness_centrality_list = list()
    eigenvector_centrality_list = list()
    # info. add lists here for voxel analysis list *****************************************************************

    for i, model_dir in enumerate(model_dirs):
        if i % 50 == 0:
            print('Analyzing model {:d}/{:d}'.format(i, len(model_dirs)))
        print(model_dir)
        hp = tools.load_hp(model_dir)
        log = tools.load_log(model_dir)

        # Handle internal matplotlib issue with None values for plotting legend
        if hp['learning_rate_mode'] is None:
            hp['learning_rate_mode'] = 'constant'
            print('None overwritten with "constant"')
        hp['rnn_type'] = 'MultiLayer' if hp.get('multiLayer') else hp['rnn_type']

        # check if performance exceeds target
        # if log['perf_min'][-1] > hp['target_perf']:
        hp_list.append(hp)
        n_clusters.append(log['n_cluster'])
        silhouette_score.append(log['score'])
        avg_perf_train_list.append(log['avg_perf_train'])
        avg_perf_test_list.append(log['avg_perf_test'])

        if hp.get('multiLayer') == False:

            if 'fundamentals' in model_dir or 'fm' in model_dir:
                pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_fundamentals.pkl'
                pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_fundamentals.pkl'
            # elif 'domainTask' in model_dir or 'singleTask' in model_dir:
            #     pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_taskSubset.pkl'
            #     pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_taskSubset.pkl'
            elif '1task' in model_dir:
                pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_1task.pkl'
                pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_1task.pkl'
            elif'4task' in model_dir:
                pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_4task.pkl'
                pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_4task.pkl'
            elif '8task' in model_dir:
                pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_8task.pkl'
                pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_8task.pkl'
            elif 'bench' in model_dir:
                pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_all_benchmark.pkl'
                pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_all_benchmark.pkl'
            elif 'multiTask' in model_dir or 'AllTask' in model_dir or 'multi':
                pkl_beRNN3 = rf'{model_dir}\var_test_lay1_rule_all.pkl'
                pkl_beRNN2 = rf'{model_dir}\corr_test_lay1_rule_all.pkl'
            else:
                print('No .pkl file found!')

            # info. variance filtering
            # h_mean_all as basis for thresholding dead neurons as h_corr_all can result in high values for dead neurons
            # leading to surpassing the threshold
            res3 = tools.load_pickle(pkl_beRNN3)
            h_var_all_ = res3['h_var_all']
            activityThreshold = 1e-3
            ind_active = np.where(h_var_all_.sum(axis=1) >= activityThreshold)[0]

            # h_corr_all as representative for modularity _analysis reflecting similar neuronal behavior over time and trials
            res2 = tools.load_pickle(pkl_beRNN2)
            h_corr_all = res2['h_corr_all']
            h_corr_all_ = h_corr_all.mean(axis=2)  # average over all tasks

            # info. variance filtering
            numberOfHiddenUnits = hp['n_rnn']

            if ind_active.shape[0] < h_corr_all_.shape[0] and ind_active.shape[0] < h_corr_all_.shape[1] and ind_active.shape[0] > 1:
                h_corr_all_ = h_corr_all_[ind_active, :]
                h_corr_all = h_corr_all_[:, ind_active]
                # Apply threshold
                functionalCorrelation_density = apply_density_threshold(h_corr_all, density=density)
            else:
                functionalCorrelation_density = np.zeros((numberOfHiddenUnits, numberOfHiddenUnits))

            # functionalCorrelation_density = apply_density_threshold(h_corr_all, density=density)

            # Compute modularity
            np.fill_diagonal(functionalCorrelation_density, 0)  # prevent self-loops
            G_sparse = nx.from_numpy_array(functionalCorrelation_density)

            if G_sparse.number_of_edges() == 0 or G_sparse.number_of_nodes() < 2:
                print(f"Skipping modularity calculation for {model_dir} — graph has no edges. Setting mod_value = 0.")
                avg_clustering = 0
                avg_clustering_list.append(avg_clustering)
                mod_value_sparse = 0
                modularity_list_sparse.append(mod_value_sparse)
                avg_pc = 0
                participation_coefficient_list.append(avg_pc)

                # info. add lists here for voxel analysis list *****************************************************************
                n_modules_list.append(0)
                global_efficiency_list.append(0)
                node_degree_list.append(0)
                betweenness_centrality_list.append(0)
                eigenvector_centrality_list.append(0)
                # info. add lists here for voxel analysis list *****************************************************************

            else:
                try:
                    # clustering
                    clustering = nx.clustering(G_sparse)
                    avg_clustering = np.mean(list(clustering.values()))
                    avg_clustering_list.append(avg_clustering)
                    # modularity
                    communities_sparse = greedy_modularity_communities(G_sparse)
                    mod_value_sparse = modularity(G_sparse, communities_sparse)
                    modularity_list_sparse.append(mod_value_sparse)
                    # participation
                    pc_dict = tools.participation_coefficient(G_sparse, communities_sparse)
                    avg_pc = np.mean(list(pc_dict.values()))
                    participation_coefficient_list.append(avg_pc)

                    # info. add lists here for voxel analysis list *****************************************************************
                    # number of modularity modules
                    n_modules = len(communities_sparse)
                    n_modules_list.append(n_modules)
                    # global efficiency
                    global_efficiency = nx.global_efficiency(G_sparse)
                    global_efficiency_list.append(global_efficiency)
                    # node degree
                    node_degree = [d for n, d in G_sparse.degree()]
                    node_degree_list.append(node_degree)
                    # betweenness centrality
                    betweenness_centrality = list(nx.betweenness_centrality(G_sparse).values())
                    betweenness_centrality_list.append(betweenness_centrality)
                    # eigenvector centrality
                    eigenvector_centrality = list(nx.eigenvector_centrality(G_sparse, max_iter=1000).values())
                    eigenvector_centrality_list.append(eigenvector_centrality)
                    # info. add lists here for voxel analysis list *****************************************************************

                except Exception as e:
                    print(f"Greedy modularity failed for {model_dir}. Setting mod_value = 0. ({e})")
                    avg_clustering = 0
                    avg_clustering_list.append(avg_clustering)
                    mod_value_sparse = 0
                    modularity_list_sparse.append(mod_value_sparse)
                    avg_pc = 0
                    participation_coefficient_list.append(avg_pc)

                    # info. add lists here for voxel analysis list *****************************************************************
                    n_modules_list.append(0)
                    global_efficiency_list.append(0)
                    node_degree_list.append(0)
                    betweenness_centrality_list.append(0)
                    eigenvector_centrality_list.append(0)
                    # info. add lists here for voxel analysis list *****************************************************************

            # # info: Alternatively take already calculatded mod value w. density threshold .1
            # try:
            #     modularity_list_sparse.append(log['modularity_sparse'][-1])
            # except Exception as e:
            #     modularity_list_sparse.append(0)

        else:
            modularity_list_sparse = []

    return n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, n_modules_list, global_efficiency_list, node_degree_list, betweenness_centrality_list, eigenvector_centrality_list

def plot_histogram():
    initdict = defaultdict(list)
    initdictother = defaultdict(list)
    initdictotherother = defaultdict(list)

    for model_dir in model_dirs:
        hp = tools.load_hp(model_dir)
        # check if performance exceeds target
        log = tools.load_log(model_dir)
        # if log['perf_avg'][-1] > hp['target_perf']:
        if log['perf_min'][-1] > hp['target_perf']: # attention: define well - maybe conservative th: 0.5 & liberal th: 0.7
            print('no. of clusters', log['n_cluster'])
            n_clusters.append(log['n_cluster'])
            hp_list.append(hp)

            initdict[hp['w_rec_init']].append(log['n_cluster'])
            initdict[hp['activation']].append(log['n_cluster'])

            # initdict[hp['rnn_type']].append(log['n_cluster'])
            # if hp['activation'] != 'tanh': fix: why no tanh??
            initdict[hp['rnn_type']].append(log['n_cluster'])
            initdictother[hp['rnn_type'] + hp['activation']].append(log['n_cluster'])
            initdictotherother[hp['rnn_type'] + hp['activation'] + hp['w_rec_init']].append(log['n_cluster'])

            if hp['l1_h'] == 0:
                initdict['l1_h_0'].append(log['n_cluster'])
            else:  # hp['l1_h'] == 1e-3 or 1e-4 or 1e-5:
                keyvalstr = 'l1_h_1emin' + str(int(abs(np.log10(hp['l1_h']))))
                initdict[keyvalstr].append(log['n_cluster'])

            # fix: l1 only good?
            if hp['l1_weight'] == 0:
                initdict['l1_weight_0'].append(log['n_cluster'])
            else:  # hp['l1_h'] == 1e-3 or 1e-4 or 1e-5:
                keyvalstr = 'l1_weight_1emin' + str(int(abs(np.log10(hp['l1_weight']))))
                initdict[keyvalstr].append(log['n_cluster'])

                # initdict[hp['l1_weight']].append(log['n_cluster'])

    # Check no of clusters under various conditions.
    f, axarr = plt.subplots(7, 1, figsize=(3, 12), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_' not in key:
            title = (key + ' ' + str(len(initdict[key])) +
                     ' mean: ' + str(round(np.mean(initdict[key]), 2)))
            axarr[u].set_title(title)
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_96nets.png')
    # plt.savefig('./figure/histforcases__pt9_192nets.pdf')
    # plt.savefig('./figure/histforcases___leakygrunotanh_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 8), sharex=True)
    u = 0
    for key in initdictother.keys():
        if 'l1_' not in key:
            axarr[u].set_title(
                key + ' ' + str(len(initdictother[key])) + ' mean: ' + str(round(np.mean(initdictother[key]), 2)))
            axarr[u].hist(initdictother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases__leakyrnngrurelusoftplus_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 6), sharex=True)
    u = 0
    for key in initdictotherother.keys():
        if 'l1_' not in key and 'diag' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictotherother[key])) + ' mean: ' + str(
                round(np.mean(initdictotherother[key]), 2)))
            axarr[u].hist(initdictotherother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_randortho_notanh_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 6), sharex=True)
    u = 0
    for key in initdictotherother.keys():
        if 'l1_' not in key and 'randortho' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictotherother[key])) + ' mean: ' + str(
                round(np.mean(initdictotherother[key]), 2)))
            axarr[u].hist(initdictotherother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_diag_notanh_pt9_192nets.pdf')

    # regu--
    f, axarr = plt.subplots(4, 1, figsize=(3, 8), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_h_' in key:
            axarr[u].set_title(key + ' ' + str(len(initdict[key])) + ' mean: ' + str(round(np.mean(initdict[key]), 2)))
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/noofclusters_pt9_l1_h_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 8), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_weight_' in key:
            axarr[u].set_title(key + ' ' + str(len(initdict[key])) + ' mean: ' + str(round(np.mean(initdict[key]), 2)))
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/noofclusters_pt9_l1_weight_192nets.pdf')

def _get_hp_ranges():
    """Get ranges of hp."""
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus', 'relu', 'tanh', 'linear']
    # hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU', 'MultiLayer']
    hp_ranges['n_rnn'] = [8, 16, 32, 64, 128, 156, 256, 512]
    hp_ranges['w_rec_init'] = ['randortho', 'randgauss', 'diag', 'brainStructure']
    hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_weight'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['learning_rate'] = [0.002, 0.0015, 0.001, 0.0005, 0.0001, 0.00005]
    hp_ranges['learning_rate_mode'] = ['constant', 'exp_range', 'triangular2']
    hp_ranges['batch_size'] = [40, 80]
    # hp_ranges['errorBalancingValue'] = [1., 5.]
    return hp_ranges

def general_hp_plot(n_clusters, silhouette_score, hp_list, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, directory, sort_variable, mode, batchPlot, model_dir_batches):
    hp_ranges = _get_hp_ranges()
    hp_plots = list(hp_ranges.keys())

    # get negative modularity scores to 0 as they occur in ~0 cases
    modularity_list_sparse = [max(0, m) for m in modularity_list_sparse]

    # Sort by descending number of sort_variable
    if sort_variable == 'performance' and mode == 'test':
        ind_sort = np.argsort(avg_perf_test_list)[::-1]
    elif sort_variable == 'performance' and mode == 'train':
        ind_sort = np.argsort(avg_perf_train_list)[::-1]
    # elif sort_variable == 'clustering':
    #     ind_sort = np.argsort(n_clusters)[::-1]
    # elif sort_variable == 'silhouette':
    #     ind_sort = np.argsort(silhouette_score)[::-1]
    # elif sort_variable == 'modularity':
    #     ind_sort = np.argsort(modularity_list_sparse)[::-1]

    n_clusters_sorted = [n_clusters[i] for i in ind_sort]
    silhouette_score_sorted = [silhouette_score[i] for i in ind_sort]
    hp_list_sorted = [hp_list[i] for i in ind_sort]
    avg_perf_train_list_sorted = [avg_perf_train_list[i] for i in ind_sort]
    avg_perf_test_list_sorted = [avg_perf_test_list[i] for i in ind_sort]
    if hp_list[0]['rnn_type'] != 'MultiLayer':
        avg_clustering_list_sorted = [avg_clustering_list[i] for i in ind_sort]
        modularity_list_sparse_sorted = [modularity_list_sparse[i] for i in ind_sort]
        participation_coefficient_list_sorted = [participation_coefficient_list[i] for i in ind_sort]
    successful_model_dirs_sorted = [successful_model_dirs[i] for i in ind_sort]

    # Prepare heatmap data
    hp_visualize = np.zeros((len(hp_plots), len(n_clusters)))
    color_indices_per_hp = {}
    for i, hp_name in enumerate(hp_plots):
        values = [hp[hp_name] for hp in hp_list_sorted]
        unique_vals = hp_ranges[hp_name]
        color_indices_per_hp[hp_name] = {v: j / (len(unique_vals) - 1) for j, v in enumerate(unique_vals)}
        for j, val in enumerate(values):
            hp_visualize[i, j] = color_indices_per_hp[hp_name][val]

    # === MAIN PLOTS ===
    fig, axs = plt.subplots(6, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1.5]})
    plt.subplots_adjust(hspace=0.5)

    if hp_list[0]['rnn_type'] != 'MultiLayer': # attention: You have to add the calculation for modularity for multiRNN
        axs[0].plot(avg_clustering_list_sorted, '-')
        axs[0].set_ylabel(f'Clustering ({mode})', fontsize=7)
        axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

        axs[1].plot(modularity_list_sparse_sorted, '-')
        axs[1].set_ylabel(f'Modularity ({mode})', fontsize=7)
        axs[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)

        axs[2].plot(participation_coefficient_list_sorted, '-')
        axs[2].set_ylabel(f'Participation ({mode})', fontsize=7)
        axs[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[2].spines["top"].set_visible(False)
        axs[2].spines["right"].set_visible(False)

        # Add light grey dashed lines at y=0.3, 0.5, 0.7
        # for y in [0.3, 0.5, 0.7]:
        #     axs[0].axhline(y=y, color='lightgrey', linestyle='--', linewidth=0.8, zorder=0)
    else:
        axs[0].plot(silhouette_score_sorted, '-')
        axs[0].set_ylabel(f'Silhouette score ({mode})', fontsize=7)
        axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

    # axs[1].plot(n_clusters_sorted, '-')
    # axs[1].set_ylabel(f'Num. clusters ({mode})', fontsize=7)
    # axs[1].set_yticks([0, 10, 20, 30])
    # axs[1].spines["top"].set_visible(False)
    # axs[1].spines["right"].set_visible(False)

    axs[3].plot(avg_perf_train_list_sorted, '-')
    axs[3].set_ylabel('Avg. perf. train', fontsize=7)
    axs[3].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)

    axs[4].plot(avg_perf_test_list_sorted, '-')
    axs[4].set_ylabel('Avg. perf. test', fontsize=7)
    axs[4].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[4].spines["top"].set_visible(False)
    axs[4].spines["right"].set_visible(False)

    im = axs[5].imshow(hp_visualize, aspect='auto', cmap='viridis')
    axs[5].set_yticks(range(len(hp_plots)))
    axs[5].set_yticklabels([HP_NAME[hp] for hp in hp_plots], fontsize=7)
    axs[5].set_xlabel('Networks')
    axs[5].tick_params(length=0)
    axs[5].spines["top"].set_visible(False)
    axs[5].spines["right"].set_visible(False)
    axs[5].set_xticks([0, len(n_clusters_sorted) - 1])
    axs[5].set_xticklabels([1, len(n_clusters_sorted)])

    # === Add best model_dir text ===
    best_model_dir = '||'.join(successful_model_dirs_sorted[0].split('\\')[-3:])
    print('Best Model:      ', successful_model_dirs_sorted[0])

    # Save ordered best model list
    if batchPlot == False:
        sortedModels_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f"bestModels_{sort_variable}_{mode}.txt")
    else:
        sortedModels_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', model_dir_batches[0], f"bestModels_{sort_variable}_{mode}_{model_dir_batches[0]}.txt")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(sortedModels_path), exist_ok=True)
    # Save best model path
    with open(sortedModels_path, "w") as f:
        # text_file.write(successful_model_dirs_sorted[0]) # fix Write all models in descending order into text file and make them callable like a list
        json.dump(successful_model_dirs_sorted, f, indent=2)

    fig.text(0.5, 0.95, f'highest {sort_variable} {mode} model: {best_model_dir}', ha='center', va='top', fontsize=8)

    # === Save figure ===
    if batchPlot == True:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', model_dir_batches[0], f"{model_dir_batches[0]}_{sort_variable}_{mode}_hp_plot.png")
    else:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f"general_{sort_variable}_{mode}_hp_plot.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def plot_vertical_hp_legend(hp_ranges, hp_plots, HP_NAME, directory):
    cmap = mpl.cm.get_cmap('viridis')
    entries_per_hp = [len(hp_ranges[hp]) + 1 for hp in hp_plots]  # +1 for title
    total_lines = sum(entries_per_hp)

    line_height_in = 0.25
    fig_height_in = total_lines * line_height_in

    fig, ax = plt.subplots(figsize=(3.2, fig_height_in))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_lines)
    ax.axis("off")

    bar_width = 0.1
    bar_height = 0.8
    y = total_lines - 1

    for hp_name in hp_plots:
        label = HP_NAME.get(hp_name, hp_name)
        values = hp_ranges[hp_name]
        n = len(values)
        # print(label, values)

        ax.text(0.05, y + 0.3, f"{label}:", fontsize=9, fontweight='bold', va='top')
        y -= 1.0

        for j, val in enumerate(values):
            color = cmap(j / (n - 1) if n > 1 else 0.5)
            ax.add_patch(plt.Rectangle((0.05, y - bar_height / 2), bar_width, bar_height,
                                       facecolor=color, edgecolor='black', linewidth=0.3))
            ax.text(0.05 + bar_width + 0.05, y, val, fontsize=8, va='center')
            y -= 0.9

    # === Save figure ===
    save_path = os.path.join(directory, 'visuals', "legend_general_hp_plot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
    # plt.close()

def _individual_hp_plot(hp_plot, sort_variable, mode, directory, batchPlot, model_dir_batches, density, n_clusters=None, silhouette_score=None, avg_perf_test_list=None, avg_perf_train_list=None, hp_list=None, avg_clustering_list=None, modularity_list_sparse=None, participation_coefficient_list=None):
    """Plot histogram for number of clusters, separating by an attribute.

    Args:
        hp_plot: str, the attribute to separate histogram by
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    if hp_list is None: # attention: Maybe wrong fix here
        n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, n_modules_list, global_efficiency_list, node_degree_list, betweenness_centrality_list, eigenvector_centrality_list = get_n_clusters(successful_model_dirs, density) # fix: variable still to deliver

    # Compare activation, ignore tanh that can not be trained with LeakyRNN
    # hp_plot = 'activation'
    # hp_plot = 'rnn_type'
    # hp_plot = 'w_rec_init'

    sort_variable_dict = OrderedDict()
    hp_ranges = _get_hp_ranges()
    for key in hp_ranges[hp_plot]:
        sort_variable_dict[key] = list()

    if sort_variable == 'performance':
        if mode == 'test':
            for hp, perf_test in zip(hp_list, avg_perf_test_list):
                sort_variable_dict[hp[hp_plot]].append(perf_test)
        elif mode == 'train':
            for hp, perf_train in zip(hp_list, avg_perf_train_list):
                sort_variable_dict[hp[hp_plot]].append(perf_train)

    # elif sort_variable == 'clustering':
    #     for hp, n_cluster in zip(hp_list, n_clusters):
    #         sort_variable_dict[hp[hp_plot]].append(n_cluster)
    #
    # elif sort_variable == 'silhouette':
    #     for hp, silhouette in zip(hp_list, silhouette_score):
    #         sort_variable_dict[hp[hp_plot]].append(silhouette)
    #
    # elif sort_variable == 'modularity':
    #     for hp, modu in zip(hp_list, modularity_list_sparse):
    #         sort_variable_dict[hp[hp_plot]].append(modu)

    label_map = {'softplus': 'Softplus',
                 'relu': 'ReLU',
                 'tanh': 'Tanh',
                 'LeakyGRU': 'GRU',
                 'LeakyRNN': 'RNN',
                 'MultiLayer': 'MultiRNN',
                 'randortho': 'Rand. Ortho.',
                 'diag': 'Diagonal',
                 'randgauss': 'Rand. Gaussian',
                 'brainStructure': 'Brain Struc.',
                 'constant': 'Constant',
                 'exp_range': 'Exp. Range',
                 'triangular2': 'Triangular'
                 }

    # hp_ranges = OrderedDict()
    # hp_ranges['activation'] = ['softplus', 'relu', 'tanh']
    # hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU', 'MultiLayer']
    # hp_ranges['n_rnn'] = [128, 256, 512]
    # hp_ranges['w_rec_init'] = ['randortho', 'randgauss', 'diag', 'brainStructure']
    # hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l2_h'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l2_weight'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['learning_rate'] = [0.0015, 0.001, 0.0005]
    # hp_ranges['learning_rate_mode'] = ['constant', 'exp_range', 'triangular2']

    # fig = plt.figure(figsize=(1.5, 1.2))
    # ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    f, axs = plt.subplots(len(sort_variable_dict), 1,
                          sharex=True, figsize=(1.2, 1.8))

    for i, (key, val) in enumerate(sort_variable_dict.items()):
        ax = axs[i]
        # hist, bin_edges = np.histogram(val, density=True, range=(0, 30),
        #                                bins=30)
        # plt.bar(bin_edges[:-1], hist, label=key)
        color_ind = i / (len(hp_ranges[hp_plot]) - 1.)
        color = mpl.cm.viridis(color_ind)
        if isinstance(key, float):
            label = '{:1.0e}'.format(key)
        else:
            label = label_map.get(key, str(key))

        if sort_variable == 'performance' or sort_variable == 'silhouette' or sort_variable == 'modularity':
            ax.hist(val, label=label, range=(0, 1),
                    density=True, bins=16, ec=color, facecolor=color,lw=1.5)
        elif sort_variable == 'clustering':
            ax.hist(val, label=label, range=(0, 30),
                    density=True, bins=16, ec=color, facecolor=color, lw=1.5)

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_yticks([])

        if sort_variable == 'performance' or sort_variable == 'silhouette' or sort_variable == 'modularity':
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xlim([0, 1.0])
            ax.text(0.7, 0.7, label, fontsize=7, transform=ax.transAxes)
        elif sort_variable == 'clustering':
            ax.set_xticks([0, 15, 30])
            ax.set_xlim([0, 30])
            ax.text(0.7, 0.7, label, fontsize=7, transform=ax.transAxes)

        if i == 0:
            ax.set_title(HP_NAME[hp_plot], fontsize=7)

    # ax.legend(loc=3, bbox_to_anchor=(1, 0), title=HP_NAME[hp_plot], frameon=False)
    if sort_variable == 'performance':
        ax.set_xlabel('Performance', fontsize=7)
    elif sort_variable == 'clustering':
        ax.set_xlabel('Number of clusters', fontsize=7)
    elif sort_variable == 'silhouette':
        ax.set_xlabel('Silhouette score', fontsize=7)
    elif sort_variable == 'modularity':
        ax.set_xlabel('Modularity score', fontsize=7)

    # plt.tight_layout()
    # plt.show()
    # figname = os.path.join(FIGPATH, 'NumClustersHist' + hp_plot + '.pdf')
    # plt.savefig(figname, transparent=True)

        # === Save figure ===
    if batchPlot == True:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', model_dir_batches[0], f"{model_dir_batches[0]}_{hp_plot}_{mode}_plot.png")
    else:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f"general_{hp_plot}_{mode}_plot.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    return sort_variable_dict

def individual_hp_plot(n_clusters, silhouette_score, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, directory, hp_list, sort_variable, mode, batchPlot, model_dir_batches, density):
    """Plot histogram of number of clusters.

    Args:
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    hp_plots = ['activation', 'n_rnn', 'w_rec_init', 'l1_h', 'l1_weight', 'l2_h', 'l2_weight', 'learning_rate', 'learning_rate_mode', 'batch_size']
    # hp_plots = ['activation', 'rnn_type', 'n_rnn', 'w_rec_init', 'l1_h', 'l1_weight', 'l2_h', 'l2_weight', 'learning_rate', 'learning_rate_mode', 'errorBalancingValue']

    for hp_plot in hp_plots:
        n_cluster_dict = _individual_hp_plot(hp_plot, sort_variable, mode, directory, batchPlot, model_dir_batches, density, n_clusters, silhouette_score, avg_perf_test_list, avg_perf_train_list, hp_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list)

# fix: Add network size here please
HP_NAME = {'activation': 'Activation fun.',
           # 'rnn_type': 'Network type',
           'w_rec_init': 'Initialization',
           'n_rnn': 'Num. hidden units',
           'l1_h': 'L1 rate',
           'l1_weight': 'L1 weight',
           'l2_h': 'L2 rate',
           'l2_weight': 'L2 weight',
           'target_perf': 'Target perf.',
           'learning_rate': 'Learning rate',
           'learning_rate_mode': 'Learning rate mode',
           'batch_size': 'Batch size'}
           # 'errorBalancingValue': 'Error balancing value'}

if __name__ == '__main__':

    evaluateBestHP = True

    folderList = ['show-8task_multi_beRNN_03_highDim_correctOnly_128']

    # evaluate metrics for finding best hp for paper
    mean_clustering_list = []
    mean_modularity_list = []
    mean_participation_list = []

    std_clustering_list = []
    std_modularity_list = []
    std_participation_list = []

    for folder in folderList:
        final_model_dirs = []

        participantList = ['beRNN_00', 'beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
        # participantList = ['beRNN_03']
        participant = [participant for participant in participantList if participant in folder][0]
        # participant = participantList[2]
        dataType = 'highDim_correctOnly' if 'highDim_correctOnly' in folder or 'highDimCorrects' in folder else 'highDim'
        # dataType = 'highDim_benchmark'

        mode = ['train', 'test'][1]
        sort_variable = ['clustering', 'performance', 'silhouette'][1]
        batchPlot = [True, False][0] # many batches of models or only one like in robustness tests
        lastMonth = '5'
        density = 0.1

        directory = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\{folder}\{dataType}\{participant}'

        model_dir_batches = os.listdir(directory)

        # Create list of models to integrate in one hp overview plot
        model_dir_batches = [batch for batch in model_dir_batches if batch != 'visuals']
        for model_dir_batch in model_dir_batches:
            model_dirs_ = os.listdir(os.path.join(directory, model_dir_batch))
            model_dirs = [model_dir for model_dir in model_dirs_ if model_dir != 'overviews' and not model_dir.endswith('.txt')]
            for model_dir in model_dirs:
                model_dir_lastMonth_ = os.listdir(os.path.join(directory, model_dir_batch, model_dir))
                model_dir_lastMonth = [model_dir for model_dir in model_dir_lastMonth_ if lastMonth in model_dir]
                # Concatenate all models in one list
                try:
                    if 'model' in model_dir_lastMonth[0]: # Be sure to add anything else but models
                        final_model_dirs.append(os.path.join(directory, model_dir_batch, model_dir, model_dir_lastMonth[0]))

                except Exception as e:
                    # if something goes wrong (e.g. index error), skip this model_dir
                    print(f"Skipping {model_dir} due to error: {e}")
                    continue

        # First compute n_clusters for each model then collect them in lists
        successful_model_dirs = compute_n_cluster(final_model_dirs, mode) # includes _analysis.clustering (w. n_cluster and rdm)
        n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, n_modules_list, global_efficiency_list, node_degree_list, betweenness_centrality_list, eigenvector_centrality_list = get_n_clusters(successful_model_dirs, density)

        # Create histogramms for each hyperparameter seperatly w.r.t. performance or clustering
        individual_hp_plot(n_clusters, silhouette_score, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, directory, hp_list, sort_variable, mode, batchPlot, model_dir_batches, density)

        # Create legend
        hp_ranges = _get_hp_ranges()
        hp_plots = list(hp_ranges.keys())
        plot_vertical_hp_legend(hp_ranges, hp_plots, HP_NAME, directory)

        # Create hp_plots sorted by performance or clustering
        general_hp_plot(n_clusters, silhouette_score, hp_list, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list, directory, sort_variable, mode, batchPlot, model_dir_batches)

        if evaluateBestHP:
            ########################################################################################################################
            # head. evaluate metrics for finding best hp for paper
            ########################################################################################################################
            mean_clustering = np.mean(avg_clustering_list)
            mean_clustering_list.append(mean_clustering)
            print(np.round(mean_clustering, 3))
            mean_modularity = np.mean(modularity_list_sparse)
            mean_modularity_list.append(mean_modularity)
            print(np.round(mean_modularity, 3))
            mean_participation = np.mean(participation_coefficient_list)
            mean_participation_list.append(mean_participation)
            print(np.round(mean_participation, 3))

            std_clustering = np.std(avg_clustering_list)
            std_clustering_list.append(std_clustering)
            print(np.round(std_clustering, 3))
            std_modularity = np.std(modularity_list_sparse)
            std_modularity_list.append(std_modularity)
            print(np.round(std_modularity, 3))
            std_participation = np.std(participation_coefficient_list)
            std_participation_list.append(std_participation)
            print(np.round(std_participation, 3))

    # Evaluate the hp sets with smalles topological marker variance
    average_topMarker_value_list = []
    for indice in range(0,len(std_clustering_list)):
        average_topMarker_value = (std_clustering_list[indice] + std_modularity_list[indice] + std_participation_list[indice]) / 3
        average_topMarker_value_list.append(average_topMarker_value)

    # Get the order from low to high variance
    sorted_indices = sorted(range(len(average_topMarker_value_list)), key=lambda k: average_topMarker_value_list[k])


