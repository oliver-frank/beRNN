from __future__ import division

import os
# import time
import numpy as np
import pickle
from collections import OrderedDict
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import tensorflow as tf

# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import mlflow
# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++

# import random
import errno

# from task import *
from _benchmark import generate_trials
from _network import Model
import _tools

save = True

'''
- Preprocesses functional hidden activity into various necessary representations for subsequent analysis 
'''

########################################################################################################################
# head. Define functions
########################################################################################################################
def _compute_variance_bymodel(data_dir, model_dir, layer, data_type, networkAnalysis, model: object, sess: object, mode: str, monthsConsidered: list, rules: object = None, random_rotation: object = False) -> object:
    """Compute variance for all tasks.

        Args:
            model: network.Model instance
            sess: tensorflow session
            rules: list of rules to compute variance, list of strings
            random_rotation: boolean. If True, rotate the neural activity.
        """
    h_complete = OrderedDict()
    h_all_byrule = OrderedDict()
    h_all_byepoch = OrderedDict()
    hp = model.hp

    # Fallback for some scenarios
    hp.setdefault('rng', np.random.default_rng())
    # model_dir = model.model_dir

    # attention: ###################################################################################################
    rules = [key for key in hp["rule_prob_map"].keys() if hp["rule_prob_map"][key] != 0]

    if len(rules) == 12:
        ruleset = hp['ruleset']
    elif len(rules) == 8:
        ruleset = '8task'
    elif len(rules) == 4:
        ruleset = '4task'
    else:
        ruleset = '1task'

    # attention: ###################################################################################################

    # print(rules)

    if hp.get('multiLayer') == True:
        # n_hidden = hp['n_rnn_per_layer'][layer-1]
        n_hidden = 0
        for n_rnn_layer in hp['n_rnn_per_layer']: # info: Accumulate number of hidden units
            n_hidden += n_rnn_layer
    else:
        n_hidden = hp['n_rnn']

    if random_rotation:
        # Generate random orthogonal matrix
        from scipy.stats import ortho_group
        random_ortho_matrix = ortho_group.rvs(dim=n_hidden)

    # data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\Data\\BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '\\PreprocessedData_wResp_ALL'
    month = '_'.join(model_dir.split('_')[-2:]) # only current model's month considered
    train_data, eval_data = _tools.createSplittedDatasets(hp, data_dir, month)

    # Skip taskVariance creation if already exists
    save_name = 'var_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'
    save_name2 = 'corr_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'
    save_name3 = 'mean_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'
    save_name4 = 'corr_epoch_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'

    if random_rotation:
        save_name += '_rr'

    fname = os.path.join(model_dir, save_name + '.pkl')
    fname2 = os.path.join(model_dir, save_name2 + '.pkl')
    fname3 = os.path.join(model_dir, save_name3 + '.pkl')
    fname4 = os.path.join(model_dir, save_name4 + '.pkl')

    if os.path.exists(fname) == True or os.path.exists(fname2) == False or os.path.exists(fname3) == False: # attention: Set all in False again
        try:
            for task in rules:
                # print(task)
                if mode == 'train':
                    data = train_data
                elif mode == 'test':
                    data = eval_data

                # hp['benchmark'] = False # attention. comment out in the future
                if hp['benchmark'] != True:
                    x, y, y_loc, response = _tools.load_trials(hp['rng'], task, mode, hp['batch_size'], eval_data,False)  # y_loc is participantResponse_perfEvalForm

                    epochs = _tools.find_epochs(x)
                    c_mask = _tools.create_cMask(y, response, hp, mode)

                    # attention: hard-coded bug fix ************************************************************************
                    if c_mask is None:
                        print(
                            f"Skipping {task} in _compute_variance_bymodel: invalid c_mask (random bug - mainly on RP)")
                        continue
                    # attention: hard-coded bug fix ************************************************************************

                else:
                    # benchmark training with simplified tasks (like Yang, Driscoll, Brenner)
                    rule_train_now = task
                    # Generate a random batch of trials
                    # Each batch has the same trial length
                    trial = generate_trials(rule_train_now, hp, 'random', batch_size=hp['batch_size'])

                    x, y, y_loc, c_mask, epochs = trial.x, trial.y, trial.y_loc, trial.c_mask, trial.epochs

                feed_dict = _tools.gen_feed_dict(model, x, y, c_mask, hp)

                if hp.get('multiLayer') == True: # info: Only concatenate layers for hpOverview & taskRepresentation
                    h = sess.run(model.h_all_layers, feed_dict=feed_dict)
                    h = np.concatenate([h[0], h[1], h[2]], axis=-1)
                elif hp.get('multiLayer') == True and hp.get('networkAnalysis') == True: # info: Visualization of performance and funcCorrelation for all layers individually
                    h = sess.run(model.h_all_layers, feed_dict=feed_dict)
                    h = h[layer-1] # info: choose the layer of current interest
                elif hp.get('multiLayer') == False or hp.get('multiLayer') == None:
                    h = sess.run(model.h, feed_dict=feed_dict)

                if random_rotation:
                    h = np.dot(h, random_ortho_matrix)  # randomly rotate

                for e_name, e_time in epochs.items():
                    if 'fix' not in e_name:  # Ignore fixation period
                        h_all_byepoch[(task, e_name)] = h[e_time[0]:e_time[1], :, :]

                # Ignore fixation period
                h_all_byrule[task] = h[epochs['fix1'][1]:, :, :]
                h_complete[task] = h[:, :, :]

                # Reorder h_all_byepoch by epoch-first
                keys = list(h_all_byepoch.keys())
                # ind_key_sort = np.lexsort(zip(*keys))
                # Using mergesort because it is stable
                ind_key_sort = np.argsort(list(zip(*keys))[1], kind='mergesort')
                h_all_byepoch = OrderedDict([(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

                if data_type == 'rule':
                    h_all = h_all_byrule # info: only task orientated over all epochs (fixation excluded)
                elif data_type == 'epoch':
                    h_all = h_all_byepoch # info: task and epoch orientated (fixation excluded), similar to h_all_byrule as there is only fixation and response epoch

                h_mean_all = np.zeros((n_hidden, len(h_all.keys())))
                h_var_all = np.zeros((n_hidden, len(h_all.keys())))
                h_corr_all_epoch = np.zeros((n_hidden, n_hidden, len(h_all.keys())))
                h_corr_all = np.zeros((n_hidden, n_hidden, len(h_all.keys())))
                # info: default shape: (timesteps, trials, units)
                for i, val in enumerate(h_all.values()):
                    # val is Time, Batch, Units for each task in h_all dict
                    # Variance across time and stimulus
                    # h_var_all[:, i] = val[t_start:].reshape((-1, n_hidden)).var(axis=0)
                    # Variance acros trial, then averaged across time
                    h_var_all[:, i] = val.var(axis=1).mean(axis=0) # info: yang - task variance vectors
                    h_var_all = np.nan_to_num(h_var_all)
                    # reorder tensor - flatten across timesteps and trials - correlate all hidden unit vectors over flattened dimension
                    h_corr_all_epoch[:, :, i] = np.corrcoef(val.transpose(2, 0, 1).reshape(val.shape[2], -1))  # info: frank - correlation matrices for topological markers - response epoch only
                    h_corr_all_epoch = np.nan_to_num(h_corr_all)  # replaces NaNs with 0
                    # Generate average activity representations for each hidden unit and each task individually
                    h_mean_all[:, i] = val.mean(axis=(0, 1)) # info: frank - mean matrices for RDA
                    h_mean_all = np.nan_to_num(h_mean_all) # replaces NaNs with 0

                # info: Take the original full sequence for each trial with fixation and response epoch - like in functional correlation matrices in fmriprep
                for i, val in enumerate(h_complete.values()):
                    # reorder tensor - flatten across timesteps and trials - correlate all hidden unit vectors over flattened dimension
                    h_corr_all[:, :, i] = np.corrcoef(val.transpose(2, 0, 1).reshape(val.shape[2], -1))  # info: frank - correlation matrices for topological markers - both epochs
                    h_corr_all = np.nan_to_num(h_corr_all)  # replaces NaNs with 0

                result = {'h_var_all': h_var_all, 'keys': list(h_all.keys())}
                result2 = {'h_corr_all': h_corr_all, 'keys': list(h_complete.keys())}
                result3 = {'h_mean_all': h_mean_all, 'keys': list(h_all.keys())}
                result4 = {'h_corr_all_epoch': h_corr_all_epoch, 'keys': list(h_all.keys())}

            save_name = 'var_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'
            save_name2 = 'corr_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'
            save_name3 = 'mean_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'
            save_name4 = 'corr_epoch_' + mode + '_lay' + str(layer) + '_' + data_type + f'_{ruleset}'

            if random_rotation:
                save_name += '_rr'

            fname = os.path.join(model_dir, save_name + '.pkl')
            fname2 = os.path.join(model_dir, save_name2 + '.pkl')
            fname3 = os.path.join(model_dir, save_name3 + '.pkl')
            fname4 = os.path.join(model_dir, save_name4 + '.pkl')
            # dir_path = os.path.dirname(fname)

            # C:\\Users\\oliver.frank\\Desktop\\PyProjects\\beRNNmodels\\2025_04_multiLayerHpGridSearch\\1\\beRNN_03_AllTask_3-5_data_highDim_correctOnly_3stimTC_iteration0_LeakyRNN_64-64-64_relu-relu-linear\\model_month_3\\variance_test_layer0_rule_data_highDim_correctOnly_3stimTC.pkl
            # C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\2025_04_multiLayerHpGridSearch\1\beRNN_03_AllTask_3-5_data_highDim_correctOnly_3stimTC_iteration0_LeakyRNN_64-64-64_relu-relu-linear\model_month_3

        except Exception as e:
            print(f"Error in _compute_variance_bymodel: {e}")

        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            os.makedirs(os.path.dirname(fname2), exist_ok=True)
            os.makedirs(os.path.dirname(fname3), exist_ok=True)
            os.makedirs(os.path.dirname(fname4), exist_ok=True)

            print("Saving to:", fname)
            print("Dir to create:", os.path.dirname(fname))
            print("Exists:", os.path.exists(os.path.dirname(fname)))

            print("model_dir =", model_dir)

            with open(fname, 'wb') as f:
                pickle.dump(result, f)

            with open(fname2, 'wb') as f:
                pickle.dump(result2, f)

            with open(fname3, 'wb') as f:
                pickle.dump(result3, f)

            with open(fname4, 'wb') as f:
                pickle.dump(result4, f)

            print(f"Variance, Correlation and Mean file saved: {fname}")

        except OSError as e:
            if e.errno == errno.EEXIST:
                print(f"Directory already exists: {fname}")
            else:
                print(f"OS Error while saving file: {e}")

        except Exception as e:
            print(f"Failed to save variance file {fname}: {e}")

    else:
        print(f"task variance file: {fname} already exists. Skipping repetition.")
        print(f"task correlation file: {fname2} already exists. Skipping repetition.")
        print(f"task mean file: {fname3} already exists. Skipping repetition.")
        print(f"task correlation epoch file: {fname4} already exists. Skipping repetition.")
    return fname, fname2, fname3, fname4

def _compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, networkAnalysis, rules=None, random_rotation=False, **kwargs):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    if networkAnalysis == True:
        model = Model(model_dir, sigma_rec=0)
        with tf.Session() as sess:
            model.restore()
            fname, fname2, fname3, fname4 = _compute_variance_bymodel(data_dir, model_dir, layer, data_type, networkAnalysis, model, sess, mode, monthsConsidered, rules, random_rotation)
            return fname, fname2, fname3, fname4
    else:
        # Get model and sess from kwargs
        model = kwargs.get('model')
        sess = kwargs.get('sess')

        fname, fname2, fname3, fname4 = _compute_variance_bymodel(data_dir, model_dir, layer, data_type, networkAnalysis, model, sess, mode, monthsConsidered, rules, random_rotation)
        return fname, fname2, fname3, fname4

def compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, networkAnalysis, rules=None, random_rotation=False, **kwargs):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    dirs = _tools.valid_model_dirs(model_dir)
    for d in dirs:
        fname, fname2, fname3, fname4 = _compute_variance(data_dir, d, layer, mode, monthsConsidered, data_type, networkAnalysis, rules, random_rotation, **kwargs)
        return fname, fname2, fname3, fname4


if __name__ == '__main__':
    pass


