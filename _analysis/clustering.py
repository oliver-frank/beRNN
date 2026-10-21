from __future__ import division

import os
import numpy as np
# import pickle
# import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.cluster import KMeans  # AgglomerativeClustering
import tensorflow as tf

# from Network_Analysis import rule_name
from _network import Model
import _tools
from _tools import rule_name, createSplittedDatasets, create_cMask
from _network import get_perf
from _analysis import variance

'''
- Analyzes how units are involved in tasks
'''

# Colors used for clusters
kelly_colors = \
[np.array([ 0.94901961,  0.95294118,  0.95686275]),
 np.array([ 0.13333333,  0.13333333,  0.13333333]),
 np.array([ 0.95294118,  0.76470588,  0.        ]),
 np.array([ 0.52941176,  0.3372549 ,  0.57254902]),
 np.array([ 0.95294118,  0.51764706,  0.        ]),
 np.array([ 0.63137255,  0.79215686,  0.94509804]),
 np.array([ 0.74509804,  0.        ,  0.19607843]),
 np.array([ 0.76078431,  0.69803922,  0.50196078]),
 np.array([ 0.51764706,  0.51764706,  0.50980392]),
 np.array([ 0.        ,  0.53333333,  0.3372549 ]),
 np.array([ 0.90196078,  0.56078431,  0.6745098 ]),
 np.array([ 0.        ,  0.40392157,  0.64705882]),
 np.array([ 0.97647059,  0.57647059,  0.4745098 ]),
 np.array([ 0.37647059,  0.30588235,  0.59215686]),
 np.array([ 0.96470588,  0.65098039,  0.        ]),
 np.array([ 0.70196078,  0.26666667,  0.42352941]),
 np.array([ 0.8627451 ,  0.82745098,  0.        ]),
 np.array([ 0.53333333,  0.17647059,  0.09019608]),
 np.array([ 0.55294118,  0.71372549,  0.        ]),
 np.array([ 0.39607843,  0.27058824,  0.13333333]),
 np.array([ 0.88627451,  0.34509804,  0.13333333]),
 np.array([ 0.16862745,  0.23921569,  0.14901961]),
 np.array([0, 0.5, 0.5]),  # Teal
 np.array([0.5, 0, 0.5]),  # Purple
 np.array([0.5, 0.5, 0]),  # Olive
 np.array([0, 0, 1]),      # Blue
 np.array([0.5, 0, 0]),    # Maroon
 np.array([0, 0.5, 0]),    # Green
 np.array([1, 0, 0]),      # Red
 np.array([0, 1, 1]),      # Cyan
 np.array([1, 0, 1]),      # Magenta
 np.array([1, 1, 0])      # Yellow
 ]

save = True

def all_variance_files_exist(model_dir, numberOfLayers, mode, data_dir):
    for layer in range(numberOfLayers):
        for data_type in ['rule', 'epoch']:
            fname = os.path.join(model_dir, 'var_' + mode + '_lay' + str(layer) + '_' + data_type + '.pkl')

            if not os.path.exists(fname):
                return False
    return True

def compute_rdm(task_matrix, rdm_metric, normalize=False):
    # Incoming matrix should represent average activity over time and trials for each unit (h_mean_all) with form (n_units, n_tasks)
    # Final matrix must have form (n_tasks, n_tasks)
    # Normalization over tasks (unit_activity_task_A/ all_unit_activity_task_A) - actually meaningless if cosine is calculated
    # rdm_metric: correlation or cosine

    if normalize: # actually unnecessary as it won't change the rank
        h_normmean_all = (task_matrix.T / np.sum(task_matrix, axis=1)) # (n_tasks, n_units)
    else:
        h_normmean_all = task_matrix.T # objective shape: (n_tasks, n_units)

    rdm = squareform(pdist(h_normmean_all, metric=rdm_metric))
    rdm_vector = rdm[np.triu_indices_from(rdm, k=1)]

    rdm = np.nan_to_num(rdm)  # replaces NaNs with 0
    rdm_vector = np.nan_to_num(rdm_vector)  # replaces NaNs with 0

    return rdm, rdm_vector

class Analysis(object):
    def __init__(self, data_dir, model_dir, layer, rdm_metric, mode, monthsConsidered, data_type, networkAnalysis, normalization_method='sum'):

        hp = tools.load_hp(model_dir)

        # multiLayer *******************************************************************************************
        # # Do a task variance _analysis for each hidden layer
        # if hp.get('multiLayer') == True:
        #     numberOfLayers = len(hp['n_rnn_per_layer'])
        # else:
        #     numberOfLayers = 1
        #
        # if not all_variance_files_exist(model_dir, numberOfLayers, mode, data_dir):

        # compVarianceList = [compVariance for compVariance in os.listdir(model_dir) if compVariance.startswith('var')]
        # self.h_normvar_all_list = []
        # for compVariance in compVarianceList:
        # fname = os.path.join(model_dir, compVariance)
        # **************************************************************************************************************

        fname, fname2, fname3, fname4 = variance.compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, networkAnalysis)

        # fname = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_gridSearch_domainTask-EF_beRNN_03_highDim_64\highDim\beRNN_03\1\beRNN_03_AllTask_4-6_data_highDim_trainingBatch1_iteration3_LeakyRNN_64_softplus\model_month_6\var_test_lay1_rule_taskSubset.pkl'
        # res = tools.load_pickle(fname)

        res = tools.load_pickle(fname)
        # res3 = tools.load_pickle(fname3) # only necessary for rdm
        h_var_all_ = res['h_var_all']
        # h_mean_all = res3['h_mean_all']

        self.keys  = res['keys']

        # h_corr_all as representative for modularity analysis reflecting similar neuron behavior
        res2 = tools.load_pickle(fname2)
        h_corr_all_ = res2['h_corr_all']
        h_corr_all = h_corr_all_.mean(axis=2)  # average over all tasks

        # head: Clustering =========================================================================================
        # info. legacy variance filtering
        activityThreshold = 1e-3 # > 1e-3 - min > 0
        ind_active = np.where(h_var_all_.sum(axis=1) >= activityThreshold)[0]

        if ind_active.shape[0] < h_corr_all_.shape[0] and ind_active.shape[0] < h_corr_all_.shape[1] and ind_active.shape[0] > 1:
        #     h_corr_all_ = h_corr_all_[ind_active, :]
        #     h_corr_all = h_corr_all_[:, ind_active]
        #
            h_var_all = h_var_all_[ind_active, :]
        else:
        #     # Fallback
        #     h_corr_all = np.zeros_like(h_corr_all_)
        #
            h_var_all = np.zeros_like(h_var_all_)

        # print(f"Active units after thresholding: {len(ind_active)}")

        # fallback if total activity is too low
        if h_var_all.shape[0] <= 2 or np.all(h_var_all.sum(axis=1) <= 1e-2):
        # if h_var_all.shape[0] < 2 or np.where(h_var_all_.sum(axis=1) < activityThreshold): # legacy: Yang
            print(f"Creating dummy clustering and rdm for model {model_dir} — insufficient data or variance.")

            # Determine expected dimensions safely
            hp = tools.load_hp(model_dir)
            rules = [key for key in hp["rule_prob_map"].keys() if hp["rule_prob_map"][key] != 0]
            n_tasks = len(rules)
            n_units = h_var_all.shape[0]

            # Create minimal but consistent dummy data
            self.h_corr_all = h_corr_all
            self.h_var_all = h_var_all
            self.h_normvar_all = np.full((n_units, n_tasks), 0.1, dtype=float)

            self.labels = np.array([0, 1])
            self.ind_active = np.array([0, 1])
            # self.n_clusters = [2] # Yang legacy
            # self.scores = [0.0] # Yang legacy
            # self.n_cluster = 2 # Yang legacy
            self.unique_labels = np.array([0, 1])

            # Metadata
            self.normalization_method = normalization_method
            self.model_dir = model_dir
            self.hp = hp
            self.data_type = data_type
            self.rules = hp.get('rules', [f"task_{i}" for i in range(n_tasks)])


            # Create dummy RDM of proper size and symmetry - neutral, symmetric matrix: diagonal 0 (self-similarity), off-diagonals 0.5
            self.rdm = np.full((n_tasks, n_tasks), 0.5, dtype=float)
            np.fill_diagonal(self.rdm, 0.0)

            # Vectorized upper triangle (like in real RDMs)
            self.rdm_vector = self.rdm[np.triu_indices_from(self.rdm, k=1)]
            self.rdm_metric = rdm_metric

            # Ensure consistent attributes for downstream alignment
            self.coords_dummy = np.zeros((n_tasks, 2))  # optional: MDS-compatible placeholder
        # attention: fallback if activity across task is too low +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        else:
            # Normalize by the total variance across tasks
            if normalization_method == 'sum':
                h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T
            elif normalization_method == 'max':
                h_normvar_all = (h_var_all.T/np.max(h_var_all, axis=1)).T
            elif normalization_method == 'none':
                h_normvar_all = h_var_all
            else:
                raise NotImplementedError()

            # Sort clusters by its task preference (important for consistency across nets)
            # Task preference is the task vector with highest sum of activity over all its units (for that cluster assigned by kmeans)
            # The labeled clusters have no intrinsic meaning
            # By assining the most dominant predefined task vectors to the clusters we can derive meaning
            # if labels is None or labels.shape[0] != h_normvar_all.shape[0]:
            #     print("Warning: labels/taskVector shape mismatch. Skipping preference ordering.")
            #     # ind_sort = np.arange(len(labels))
            # else:
            #     if data_type == 'rule':
            #         label_prefs = [np.argmax(h_normvar_all[labels == l].sum(axis=0)) for l in set(labels)]
            #     elif data_type == 'epoch':
            #         label_prefs = [self.keys[np.argmax(h_normvar_all[labels == l].sum(axis=0))][0] for l in set(labels)]
            #
            #     ind_label_sort = np.argsort(label_prefs)
            #     # label_prefs = np.array(label_prefs)[ind_label_sort]
            #     # Relabel
            #     labels2 = np.zeros_like(labels)
            #     for i, ind in enumerate(ind_label_sort):
            #         labels2[labels == ind] = i
            #     labels = labels2
            #
            #     ind_sort = np.argsort(labels)
            #     labels = labels[ind_sort]

            # info. legacy variance filtering
            # self.h_normvar_all = h_normvar_all[ind_sort, :]
            self.h_normvar_all = h_normvar_all
            # self.ind_active = ind_active[ind_sort]

            # head: Compute Representational Dissimilarity Matrix (RSA-style) ==========================================
            # ind_active = np.where(h_mean_all_.sum(axis=1) >= activityThreshold)[0]
            # h_mean_all = h_mean_all_[ind_active, :]
            # rdm, rdm_vector = compute_rdm(h_mean_all, rdm_metric)

            self.rdm_metric = rdm_metric
            # self.rdm = rdm.tolist() # not ideal for saving as json
            # self.rdm_vector = rdm_vector # not ideal for saving as json

            self.h_corr_all = h_corr_all
            self.h_var_all = h_var_all # legacy: h_var_all
            self.normalization_method = normalization_method

            self.model_dir = model_dir
            self.hp = hp
            self.data_type = data_type
            self.rules = hp['rules']

    def plot_cluster_score(self, save_name=None):
        """Plot the score by the number of clusters."""
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.3, 0.3, 0.55, 0.55])
        ax.plot(self.n_clusters, self.scores, 'o-', ms=3)
        ax.set_xlabel('Number of clusters', fontsize=7)
        ax.set_ylabel('Silhouette score', fontsize=7)
        ax.set_title('Chosen number of clusters: {:d}'.format(self.n_cluster),fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_ylim([0, 0.32])
        if save:
            fig_name = 'cluster_score'
            if save_name is None:
                save_name = self.hp['activation']
            fig_name = fig_name + save_name
            plt.savefig('figure/'+fig_name+'.pdf', transparent=True)
        # plt.show()

    def plot_variance(self, model_dir, figurePath, mode_, save_name=None):
        labels = self.labels
        # Plotting Variance --------------------------------------------------------------------------------------------
        # Plot Normalized Variance
        if self.data_type == 'rule':
            figsize = (3.5,2.5)
            rect = [0.25, 0.2, 0.6, 0.7]
            rect_color = [0.25, 0.15, 0.6, 0.05]
            rect_cb = [0.87, 0.2, 0.03, 0.7]
            tick_names = [rule_name[r] for r in self.rules]
            fs = 6
            labelpad = 13
        elif self.data_type == 'epoch':
            figsize = (3.5,4.5)
            rect = [0.25, 0.1, 0.6, 0.85]
            rect_color = [0.25, 0.05, 0.6, 0.05]
            rect_cb = [0.87, 0.1, 0.03, 0.85]
            tick_names = [rule_name[key[0]]+' '+key[1] for key in self.keys]
            fs = 5
            labelpad = 18
        else:
            raise ValueError

        h_plot  = self.h_normvar_all.T
        vmin, vmax = 0, 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = ax.imshow(h_plot, cmap='hot', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        plt.yticks(range(len(tick_names)), tick_names,rotation=0, va='center', fontsize=fs)
        plt.xticks([])
        plt.title('Units', fontsize=7, y=1.0)
        plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        if self.normalization_method == 'sum':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'max':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'none':
            clabel = 'Variance'

        cb.set_label(clabel, fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)

        # Plot color bars indicating clustering
        if True:
            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
                ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                        color=kelly_colors[il+1])
                ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=kelly_colors[il+1])
            ax.set_xlim([0, len(labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

        taskVarianceFolder = os.path.join(figurePath, '_'.join(mode_.split('_')[1:3]), 'taskVariances')
        os.makedirs(name=taskVarianceFolder, exist_ok=True)
        plt.savefig(os.path.join(taskVarianceFolder, mode_ + '_' + 'batch_' + model_dir.split("\\")[-5] + '_' + model_dir.split("\\")[-3].split('_')[-4] + '_' + 'taskVariance.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

        # plt.show()
        # plt.close()

    def lesions(self, data_dir, mode_):
        labels = self.labels

        # The first will be the intact network
        lesion_units_list = [None]
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels == l)[0]
            # In original indices
            lesion_units_list += [self.ind_active[ind_l]]

        perfs_store_list = list()
        perfs_changes = list()
        cost_store_list = list()
        cost_changes = list()

        # # III: Split the data ##############################################################################################
        # # List of the subdirectories
        # subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if
        #            os.path.isdir(os.path.join(data_dir, d))]
        #
        # # Initialize dictionaries to store training and evaluation data
        # train_data = {}
        # eval_data = {}
        #
        # # Function to split the files
        # def split_files(files, split_ratio=0.8):
        #     random.seed(42)  # info: add seed to always shuffle similiar
        #     random.shuffle(files)
        #     split_index = int(len(files) * split_ratio)
        #     return files[:split_index], files[split_index:]
        #
        # # Create the training and test data
        # for subdir in subdirs:
        #     # Collect all file triplets in the current subdirectory
        #     file_quartett = []
        #     for file in os.listdir(subdir):
        #         if file.endswith('Input.npy'):
        #             # # III: Exclude files with specific substrings in their names
        #             # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
        #             #     continue
        #             # Include only files that contain any of the months in monthsConsidered
        #             if not any(month in file for month in ['month_4', 'month_5', 'month_6']): # fix: hp['monthsConsidered'] should be variable
        #                 continue
        #             base_name = file.split('Input')[0]
        #             input_file = os.path.join(subdir, base_name + 'Input.npy')
        #             yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
        #             output_file = os.path.join(subdir, base_name + 'Output.npy')
        #             response_file = os.path.join(subdir, base_name + 'Response.npy')
        #
        #             file_quartett.append((input_file, yloc_file, output_file, response_file))
        #
        #     # Split the file triplets
        #     train_files, eval_files = split_files(file_quartett)
        #
        #     # Store the results in the dictionaries
        #     train_data[subdir] = train_files
        #     eval_data[subdir] = eval_files
        #     # III: Split the data ##############################################################################################

        hp = self.hp
        month = '_'.join(self.model_dir.split('_')[-2:]) # only current model's month considered
        train_data, eval_data = createSplittedDatasets(hp, data_dir, month)

        for i, lesion_units in enumerate(lesion_units_list):
            model = Model(self.model_dir)
            hp = model.hp
            with tf.Session() as sess:
                model.restore()
                model.lesion_units(sess, lesion_units)

                perfs_store = list()
                cost_store = list()
                for task in self.rules:
                    n_rep = 16
                    # batch_size_test = 256
                    # batch_size_test_rep = int(batch_size_test / n_rep)
                    clsq_tmp = list()
                    perf_tmp = list()
                    for i_rep in range(n_rep):
                        # attention: #######################################################################################
                        # Start evaluation of this task
                        print('Evaluate task: ', task)
                        mode = mode_.split('_')[-1]
                        # if mode == 'test':
                        #     x, y, y_loc, response = tools.load_trials(task, mode, hp['batch_size'], eval_data, False)
                        # elif mode == 'train':
                        #     x, y, y_loc, response = tools.load_trials(task, mode, hp['batch_size'], train_data, False)

                        # Fallback for taskRepresentation.py - hp['rng'] is popped out before saving in training
                        if 'rng' not in hp:
                            hp['rng'] = np.random.default_rng()

                        if mode == 'test':
                            x, y, y_loc, response = tools.load_trials(hp['rng'], task, mode, hp['batch_size'], eval_data, False)
                        elif mode == 'train':
                            x, y, y_loc, response = tools.load_trials(hp['rng'], task, mode, hp['batch_size'], train_data, False)  # y_loc is participantResponse_perfEvalForm
                        # Create cMask
                        c_mask = create_cMask(y, response, hp, mode)

                        # if c_mask == None:
                        #     continue

                        # # Generating feed_dict.
                        # feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)
                        #
                        # h = sess.run(model.train_step,
                        #              feed_dict=feed_dict)  # info: Trainables are actualized - train_step should represent the step in _training.py and the global_step in network.py
                        #

                        # epochs = tools.find_epochs(x)
                        #
                        # # info: ################################################################################################
                        # fixation_steps = tools.getEpochSteps(y)
                        #
                        # # Create c_mask for current batch
                        # if hp['loss_type'] == 'lsq':
                        #     c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
                        #     for i in range(y.shape[1]):
                        #         # Fixation epoch
                        #         c_mask[:fixation_steps, i, :] = 1.
                        #         # Response epoch
                        #         c_mask[fixation_steps:, i, :] = 1.
                        #
                        #     # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
                        #     c_mask[:, :, 0] *= 2.  # Fixation is important
                        #     c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))
                        #     c_mask /= c_mask.mean()
                        #
                        # else:
                        #     c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
                        #     for i in range(y.shape[1]):
                        #         # Fixation epoch
                        #         c_mask[:fixation_steps, i, :] = 1.
                        #         # Response epoch
                        #         c_mask[fixation_steps:, i, :] = 1.
                        #
                        #     c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
                        #     c_mask /= c_mask.mean()
                        # # info: ################################################################################################

                        feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)
                        y_hat_test, c_lsq = sess.run([model.y_hat, model.cost_lsq], feed_dict=feed_dict)
                        # attention ####################################################################################

                        # Cost is first summed over time, and averaged across batch and units
                        # We did the averaging over time through c_mask

                        # IMPORTANT CHANGES: take overall mean
                        perf_test = np.mean(get_perf(y_hat_test, y_loc))

                        clsq_tmp.append(c_lsq)
                        perf_tmp.append(perf_test)

                    perfs_store.append(np.mean(perf_tmp))
                    cost_store.append(np.mean(clsq_tmp))

            perfs_store = np.array(perfs_store)
            cost_store = np.array(cost_store)

            perfs_store_list.append(perfs_store)
            cost_store_list.append(cost_store)

            if i > 0:
                perfs_changes.append(perfs_store - perfs_store_list[0])
                cost_changes.append(cost_store - cost_store_list[0])

        perfs_changes = np.array(perfs_changes)
        cost_changes = np.array(cost_changes)

        return perfs_changes, cost_changes

    def plot_lesions(self, data_dir, model_dir, figurePath, mode_):
        """Lesion individual cluster and show performance."""

        perfs_changes, cost_changes = self.lesions(data_dir, mode_)

        cb_labels = ['Performance change after lesioning',
                     'Cost change after lesioning']
        vmins = [-0.5, -0.5]
        vmaxs = [+0.5, +0.5]
        ticks = [[-0.5, 0.5], [-0.5, 0.5]]
        changes_plot = [perfs_changes, cost_changes]

        fs = 6
        figsize = (2.5, 2.5)
        rect = [0.3, 0.2, 0.5, 0.7]
        rect_cb = [0.87, 0.2, 0.03, 0.7]
        rect_color = [0.3, 0.15, 0.5, 0.05]

        i = 0  # only performance plot

        labels = self.labels
        cluster_edges = [0]
        for l in self.unique_labels:
            count = np.sum(labels == l)
            cluster_edges.append(cluster_edges[-1] + count)
        x_edges = np.array(cluster_edges)
        y_edges = np.arange(changes_plot[i].T.shape[0] + 1)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        pc = ax.pcolormesh(x_edges, y_edges, changes_plot[i].T,
                           cmap='coolwarm', vmin=vmins[i], vmax=vmaxs[i], shading='flat')
        ax.invert_yaxis()

        tick_names = [rule_name[r] for r in self.rules]
        ax.set_yticks(np.arange(len(tick_names)) + 0.5)
        ax.set_yticklabels(tick_names, fontsize=fs, va='center')
        ax.set_xticks([])
        ax.set_xlabel('Clusters', fontsize=7)
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(pc, cax=ax, ticks=ticks[i])
        cb.outline.set_linewidth(0.5)
        cb.ax.set_ylabel(cb_labels[i], fontsize=7, labelpad=0)
        cb.ax.tick_params(labelsize=6)

        ax = fig.add_axes(rect_color)
        for il, l in enumerate(self.unique_labels):
            inds = np.where(labels == l)[0]
            if len(inds) == 0:
                continue
            start, end = inds[0], inds[-1] + 1
            ax.plot([start, end], [0, 0], linewidth=4, solid_capstyle='butt',
                    color=kelly_colors[il + 1])
            ax.text((start + end) / 2, -0.5, str(il + 1), fontsize=6,
                    ha='center', va='top', color=kelly_colors[il + 1])
        ax.set_xlim([0, len(labels)])
        ax.set_ylim([-1, 1])
        ax.axis('off')

        lesionPlotsFolder = os.path.join(figurePath, '_'.join(mode_.split('_')[1:3]), 'lesionPlots')
        os.makedirs(name=lesionPlotsFolder, exist_ok=True)
        plt.savefig(os.path.join(lesionPlotsFolder, mode_ + '_' + 'batch_' + model_dir.split("\\")[-5] + '_' +
                                 model_dir.split("\\")[-3].split('_')[-4] + '_' + 'lesionPlot.png'),
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

        # plt.show()

        return plt

    def get_dotProductCorrelation(self): # legacy: task variance matrices only used for activity threshold - not topological analysis
        # Center and normalize the data
        data_centered = self.h_normvar_all - self.h_normvar_all.mean(axis=1, keepdims=True)

        norm = np.linalg.norm(data_centered, axis=1, keepdims=True)
        norm[norm == 0] = 1e-8  # Prevent division by zero

        data_normalized = data_centered / norm

        corr_coef = np.dot(data_normalized, data_normalized.T)

        return corr_coef

    def get_similarity(self):
        # labels = self.labels
        from sklearn.metrics.pairwise import cosine_similarity

        # # info: Standardize variance - This is not very reasonable as it will partly vanish the distribution representation of the variance values
        # # Compute the global mean and standard deviation
        # global_mean = self.h_var_all.mean()
        # global_std = self.h_var_all.std()
        # # Standardize the data over all values
        # h_var_all_standardized = (self.h_var_all - global_mean) / global_std
        # similarity = cosine_similarity(h_var_all_standardized)  # Compute similarity based on standardized data

        # info: Use normalized variance
        similarity = cosine_similarity(self.h_normvar_all)  # Compute similarity based on standardized data
        return similarity

    def plot_similarity_matrix(self, model_dir, mode):
        # labels = self.labels
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(self.h_normvar_all)  # Compute similarity

        # Set up the figure
        fig = plt.figure(figsize=(10, 10))

        # Create the main similarity matrix plot
        matrix_left = 0.1
        matrix_bottom = 0.3
        matrix_width = 0.6
        matrix_height = 0.6

        ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
        im = ax_matrix.imshow(similarity, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

        # Add title
        ax_matrix.set_title('Average Functional Cosine Similiarity - BeRNN', fontsize=22, pad=20)
        # Add x-axis and y-axis labels
        ax_matrix.set_xlabel('Hidden units', fontsize=16, labelpad=15)
        ax_matrix.set_ylabel('Hidden units', fontsize=16, labelpad=15)

        # Remove x and y ticks
        ax_matrix.set_xticks([])  # Disable x-ticks
        ax_matrix.set_yticks([])  # Disable y-ticks

        # # # Create the cluster bar directly below the matrix
        # bar_bottom = matrix_bottom - 0.07
        # bar_height = 0.05
        #
        # ax_color = fig.add_axes([matrix_left, bar_bottom, matrix_width, bar_height], sharex=ax_matrix)
        # for il, l in enumerate(self.unique_labels):
        #     ind_l = np.where(labels == l)[0][[0, -1]] + np.array([0, 1])
        #     ax_color.plot(ind_l, [0, 0], linewidth=4, solid_capstyle='butt',
        #                   color=kelly_colors[il + 1])
        #     ax_color.text(np.mean(ind_l), -0.5, str(il + 1), fontsize=8,
        #                   ha='center', va='top', color=kelly_colors[il + 1])

        # ax_color.set_xlim([0, len(labels)])
        # ax_color.set_ylim([-1, 1])
        # ax_color.axis('off')
        #
        # # Create the x-axis label directly below the cluster bar
        # label_bottom = bar_bottom - 0.07
        # label_height = 0.05
        #
        # ax_label = fig.add_axes([matrix_left, label_bottom, matrix_width, label_height], sharex=ax_matrix)
        # ax_label.axis('off')
        # ax_label.text(0.5, 0.5, 'Clusters', fontsize=18, ha='center', va='center', transform=ax_label.transAxes)

        # Create the colorbar on the right side, aligned with the matrix
        colorbar_left = matrix_left + matrix_width + 0.02
        colorbar_width = 0.03

        ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
        cb = plt.colorbar(im, cax=ax_cb)
        cb.set_ticks([-1, 1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Similarity', fontsize=18, labelpad=0)

        # # Set the title above the similarity matrix, centered
        # if mode == 'Training':
        #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
        # elif mode == 'Evaluation':
        #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'

        # ax_matrix.set_title(title, fontsize=14, pad=20)
        # Save the figure with a tight bounding box to ensure alignment
        save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNN_models\\Visuals\\Similiarity\\finalReport', model_dir.split("\\")[-1] + '_' + mode + '.png')
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_2Dvisualization(self, model_dir, mode, method='tSNE'):
        labels = self.labels
        # Plotting 2-D visualization of variance map -------------------------------------------------------------------
        from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
        from sklearn.decomposition import PCA

        # model = LocallyLinearEmbedding()
        if method == 'PCA':
            model = PCA(n_components=2, whiten=False)
        elif method == 'MDS':
            model = MDS(metric=True, n_components=2, n_init=10, max_iter=1000)
        elif method == 'tSNE':
            model = TSNE(n_components=2, random_state=0, init='pca',
                         verbose=1, method='exact',
                         learning_rate=100, perplexity=30)
        else:
            raise NotImplementedError

        Y = model.fit_transform(self.h_normvar_all)

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0]
            ax.scatter(Y[ind_l, 0], Y[ind_l, 1], color=kelly_colors[il+1], s=10)
        ax.axis('off')
        plt.title(method, fontsize=7)
        figname = 'figure/taskvar_visual_by'+method+self.data_type+'.pdf'
        # if save:
        #     plt.savefig(figname, transparent=True)

        plt.savefig(os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNN_models\\Visuals\\2D_Clustering',model_dir.split("\\")[-1] + '_' + mode + '.png'), \
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        # if show == True:
        # plt.show()
        # else:
        #     plt.close(fig)
        # else:
        #     plt.close(fig)

        # fig = plt.figure(figsize=(3.5, 3.5))
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.scatter(Y[:,0], Y[:,1], color='black')
        # ax.axis('off')

    def plot_example_unit(self, show):
        # Plotting Variance for example unit ---------------------------------------------------------------------------
        if self.data_type == 'rule':
            tick_names = [rule_name[r] for r in self.rules]

            ind = 2 # example unit
            fig = plt.figure(figsize=(1.2,1.0))
            ax = fig.add_axes([0.4,0.4,0.5,0.45])
            ax.plot(range(self.h_var_all.shape[1]), self.h_var_all[ind, :], 'o-', color='black', lw=1, ms=2)
            plt.xticks(range(len(tick_names)), [tick_names[0]] + ['.']*(len(tick_names)-2) + [tick_names[-1]],
                       rotation=90, fontsize=6, horizontalalignment='center')
            plt.xlabel('Task', fontsize=7, labelpad=-10)
            plt.ylabel('Task Variance', fontsize=7)
            plt.title('Unit {:d}'.format(self.ind_active[ind]), fontsize=7, y=0.85)
            plt.locator_params(axis='y', nbins=3)
            ax.tick_params(axis='both', which='major', labelsize=6, length=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if save:
                plt.savefig('figure/exampleunit_variance.pdf', transparent=True)
            # if show == True:
            # plt.show()
            # else:
            #     plt.close(fig)

    def plot_connectivity_byclusters(self, model_dir, mode):
        """Plot connectivity of the model"""

        ind_active = self.ind_active

        # Sort data by labels and by input connectivity
        model = Model(self.model_dir)
        hp = model.hp
        with tf.Session() as sess:
            model.restore()
            w_in = sess.run(model.w_in).T
            w_rec = sess.run(model.w_rec).T
            w_out = sess.run(model.w_out).T
            b_rec = sess.run(model.b_rec)
            b_out = sess.run(model.b_out)

        w_rec = w_rec[ind_active, :][:, ind_active]
        w_in = w_in[ind_active, :]
        w_out = w_out[:, ind_active]
        b_rec = b_rec[ind_active]

        # nx, nh, ny = hp['shape']
        nr = hp['n_eachring']

        sort_by = 'w_in'
        if sort_by == 'w_in':
            w_in_mod1 = w_in[:, 1:nr+1]
            w_in_mod2 = w_in[:, nr+1:2*nr+1]
            w_in_modboth = w_in_mod1 + w_in_mod2
            w_prefs = np.argmax(w_in_modboth, axis=1)
        elif sort_by == 'w_out':
            w_prefs = np.argmax(w_out[1:], axis=0)

        # sort by labels then by prefs
        ind_sort = np.lexsort((w_prefs, self.labels))

        # Plotting Connectivity ----------------------------------------------------------------------------------------
        nx = self.hp['n_input']
        ny = self.hp['n_output']
        nh = len(self.ind_active)
        nr = self.hp['n_eachring']
        nrule = len(self.hp['rules'])

        # Plot active units
        _w_rec  = w_rec[ind_sort,:][:,ind_sort]
        _w_in   = w_in[ind_sort,:]
        _w_out  = w_out[:,ind_sort]
        _b_rec  = b_rec[ind_sort, np.newaxis]
        _b_out  = b_out[:, np.newaxis]
        labels  = self.labels[ind_sort]

        l = 0.3
        l0 = (1-1.5*l)/nh

        plot_infos = [(_w_rec              , [l               ,l          ,nh*l0    ,nh*l0]),
                      (_w_in[:,[0]]        , [l-(nx+15)*l0    ,l          ,1*l0     ,nh*l0]), # Fixation input
                      (_w_in[:,1:nr+1]     , [l-(nx+11)*l0    ,l          ,nr*l0    ,nh*l0]), # Mod 1 stimulus
                      (_w_in[:,nr+1:2*nr+1], [l-(nx-nr+8)*l0  ,l          ,nr*l0    ,nh*l0]), # Mod 2 stimulus
                      (_w_in[:,2*nr+1:]    , [l-(nx-2*nr+5)*l0,l          ,nrule*l0 ,nh*l0]), # Rule inputs
                      (_w_out[[0],:]       , [l               ,l-(4)*l0   ,nh*l0    ,1*l0]),
                      (_w_out[1:,:]        , [l               ,l-(ny+6)*l0,nh*l0    ,(ny-1)*l0]),
                      (_b_rec              , [l+(nh+6)*l0     ,l          ,l0       ,nh*l0]),
                      (_b_out              , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0       ,ny*l0])]

        # cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
        cmap = 'coolwarm'
        fig = plt.figure(figsize=(6, 6))
        for plot_info in plot_infos:
            ax = fig.add_axes(plot_info[1])
            vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5,50,95])
            _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                          vmin=vmid-(vmax-vmin)/2, vmax=vmid+(vmax-vmin)/2)
            ax.axis('off')

        ax1 = fig.add_axes([l     , l+nh*l0, nh*l0, 6*l0])
        ax2 = fig.add_axes([l-6*l0, l      , 6*l0 , nh*l0])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
            ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        plt.savefig(os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNN_models\\Visuals\\connectivityByClusters',model_dir.split("\\")[-1] + '_' + mode + '.png'), \
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.show()

    def plot_connectivity_byclusters_WrecOnly(self, model_dir, mode):
        """Plot connectivity of the model's recurrent weights only"""

        ind_active = self.ind_active

        # Sort data by labels
        model = Model(self.model_dir)
        with tf.Session() as sess:
            model.restore()
            w_rec = sess.run(model.w_rec).T

        # Filter and sort the recurrent weights for active units
        w_rec = w_rec[ind_active, :][:, ind_active]

        # Sort by labels
        ind_sort = np.argsort(self.labels)

        # Plotting Connectivity for Recurrent Weights ---------------------------------------------------------------
        nh = len(self.ind_active)

        # Sort the recurrent weight matrix
        _w_rec = w_rec[ind_sort, :][:, ind_sort]
        labels = self.labels[ind_sort]

        # Plotting settings
        l = 0.3
        l0 = (1 - 1.5 * l) / nh

        plot_infos = [(_w_rec, [l, l, nh * l0, nh * l0])]

        cmap = 'coolwarm'
        fig = plt.figure(figsize=(6, 6))
        for plot_info in plot_infos:
            ax = fig.add_axes(plot_info[1])
            vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5, 50, 95])
            _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                          vmin=vmid - (vmax - vmin) / 2, vmax=vmid + (vmax - vmin) / 2)
            ax.axis('off')

        ax1 = fig.add_axes([l, l + nh * l0, nh * l0, 6 * l0])
        ax2 = fig.add_axes([l - 6 * l0, l, 6 * l0, nh * l0])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels == l)[0][[0, -1]] + np.array([0, 1])
            ax1.plot(ind_l, [0, 0], linewidth=2, solid_capstyle='butt',
                     color=kelly_colors[il + 1])
            ax2.plot([0, 0], len(labels) - ind_l, linewidth=2, solid_capstyle='butt',
                     color=kelly_colors[il + 1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        # plt.savefig(
        #     os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNN_models\\Visuals\\connectivityByClustersWrecOnly\\finalReport',
        #                  model_dir.split("\\")[-1] + '_' + mode + '.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1) # fix
        # plt.show()

    def easy_connectivity_plot(self, model_dir):
        """A simple plot of network connectivity."""

        model = Model(model_dir)
        with tf.Session() as sess:
            model.restore()
            # get all connection weights and biases as tensorflow variables
            var_list = model.var_list
            # evaluate the parameters after training
            params = [sess.run(var) for var in var_list]
            # get name of each variable
            names = [var.name for var in var_list]

        # Plot weights
        for param, name in zip(params, names):
            if len(param.shape) != 2:
                continue

            vmax = np.max(abs(param)) * 0.7
            plt.figure()
            # notice the transpose
            plt.imshow(param.T, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax,
                       interpolation='none', origin='lower')
            plt.title(name)
            plt.colorbar()
            plt.xlabel('From')
            plt.ylabel('To')
            # plt.show()

    def easy_connectivity_plot_recurrentWeightsOnly(self, model_dir):
        """A simple plot of network weight connectivity."""

        model = Model(model_dir)
        with tf.Session() as sess:
            model.restore()
            # get all connection weights and biases as tensorflow variables
            var_list = model.var_list
            # evaluate the parameters after training
            params = [sess.run(var) for var in var_list]
            # get name of each variable
            names = [var.name for var in var_list]

            name = names[2]
            params = params[2][77:,:]

            # params_centered = params - params.mean(axis=1, keepdims=True)
            # params_normalized = params_centered / np.linalg.norm(params_centered, axis=1, keepdims=True)

            corr_coef = np.dot(params, params.T)

            return corr_coef

    def easy_connectivity_plot_excitatoryGatedWeightsOnly(self, model_dir):
        """A simple plot of network weight connectivity."""

        model = Model(model_dir)
        with tf.Session() as sess:
            model.restore()
            # get all connection weights and biases as tensorflow variables
            var_list = model.var_list
            # evaluate the parameters after training
            params = [sess.run(var) for var in var_list]
            # get name of each variable
            names = [var.name for var in var_list]

            name = names[0]
            params = params[0][77:, 128:256] # excitatory gate fix: Add generic network size variable

            # params_centered = params - params.mean(axis=1, keepdims=True)
            # params_normalized = params_centered / np.linalg.norm(params_centered, axis=1, keepdims=True)

            corr_coef = np.dot(params, params.T)

            return corr_coef

    def easy_connectivity_plot_inhibitoryGatedWeightsOnly(self, model_dir):
        """A simple plot of network weight connectivity."""

        model = Model(model_dir)
        with tf.Session() as sess:
            model.restore()
            # get all connection weights and biases as tensorflow variables
            var_list = model.var_list
            # evaluate the parameters after training
            params = [sess.run(var) for var in var_list]
            # get name of each variable
            names = [var.name for var in var_list]

            name = names[0]
            params = params[0][77:, :128] # inhibitory gate fix: Add generic network size variable

            # params_centered = params - params.mean(axis=1, keepdims=True)
            # params_normalized = params_centered / np.linalg.norm(params_centered, axis=1, keepdims=True)

            corr_coef = np.dot(params, params.T)

            return corr_coef


if __name__ == '__main__':
    pass


