from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import time
from pathlib import Path
from collections import defaultdict

import os
import json
import random
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
# import matplotlib.pyplot as plt

# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mlflow
# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

from _analysis import variance
from _network import Model, get_perf, get_perf_lowDIM
from _benchmark import generate_trials
import _tools

'''
Roles:
- Trains models with defined hyperparameters
'''

########################################################################################################################
# head. Predefine functions
########################################################################################################################
def apply_threshold(matrix, threshold):
    # info: added function second time from networkAnalysis for training on server with only training.py
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix, 0)
    return matrix_thresholded


def apply_density_threshold(matrix, density=0.1):
    n = matrix.shape[0]
    # Get all upper-triangle values
    triu_vals = matrix[np.triu_indices(n, k=1)]
    cutoff = np.quantile(triu_vals, 1 - density)  # cutoff represents the value that divides all values at 1-density
    # Zero out everything below cutoff
    thresholded = np.where(matrix >= cutoff, matrix, 0)
    return thresholded


def getAndSafeModValue(data_dir, model_dir, hp, model, sess, log, step):
    # info. saves modularity, but also clustering and participation
    fname, fname2, fname3, fname4 = variance.compute_variance(data_dir, model_dir, layer=1, mode='test',
                                                      monthsConsidered=hp['monthsConsidered'], data_type='rule',
                                                      networkAnalysis=False,
                                                      model=model, sess=sess)

    # info. Keep variance filtering here to ensure stability of marker calculation during training
    # h_mean_all as basis for thresholding dead neurons as h_corr_all can result in high values for dead neurons
    res = tools.load_pickle(fname)
    h_var_all_ = res['h_var_all']
    activityThreshold = 1e-3
    ind_active = np.where(h_var_all_.sum(axis=1) >= activityThreshold)[0]

    # h_corr_all as representative for modularity analysis reflecting similar neuron behavior
    res2 = tools.load_pickle(fname2)
    h_corr_all_ = res2['h_corr_all']
    h_corr_all_ = h_corr_all_.mean(axis=2)  # average over all tasks

    numberOfHiddenUnits = hp['n_rnn']

    if ind_active.shape[0] < h_corr_all_.shape[0] and ind_active.shape[0] < h_corr_all_.shape[1] and ind_active.shape[
        0] > 1:
        h_corr_all_ = h_corr_all_[ind_active, :]
        h_corr_all = h_corr_all_[:, ind_active]
        # Apply threshold
        functionalCorrelation_density = apply_density_threshold(h_corr_all, density=hp['threshold'])
    else:
        functionalCorrelation_density = np.zeros((numberOfHiddenUnits,
                                                  numberOfHiddenUnits))  # fix: Get individual number of hidden units # Create different dummy matrix, that leads to lower realtive count

    # Compute modularity
    np.fill_diagonal(functionalCorrelation_density, 0)  # prevent self-loops
    G_sparse = nx.from_numpy_array(functionalCorrelation_density)

    if G_sparse.number_of_edges() == 0 or G_sparse.number_of_nodes() < 2:
        print(f"Skipping modularity calculation for {model_dir} — graph has no edges.")
        avg_clustering = 0
        mod_value_sparse = 0
        avg_participation = 0
    else:
        try:
            # clustering
            clustering = nx.clustering(G_sparse)
            avg_clustering = np.mean(list(clustering.values()))
            # modularity
            communities_sparse = greedy_modularity_communities(G_sparse)
            mod_value_sparse = modularity(G_sparse, communities_sparse)
            # participation
            pc_dict = tools.participation_coefficient(G_sparse, communities_sparse)
            avg_participation = np.mean(list(pc_dict.values()))

        except Exception as e:
            print(f"Greedy modularity failed for {model_dir}. Setting mod_value=0. ({e})")
            avg_clustering = 0
            mod_value_sparse = 0
            avg_participation = 0

    # head. new mlflow logic ###########################################################################################
    mlflow.log_metric("avg_clustering", avg_clustering, step=step)
    mlflow.log_metric("modularity_sparse", mod_value_sparse, step=step)
    mlflow.log_metric("avg_participation", avg_participation, step=step)
    # head. new mlflow logic ###########################################################################################

    log[('averageg_clustering')].append(avg_clustering)
    log['modularity_sparse'].append(mod_value_sparse)
    log['average_participation'].append(avg_participation)

    tools.save_log(log)


def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    num_ring = tools.get_num_ring(ruleset)
    n_rule = tools.get_num_rule(ruleset)

    machine = 'local'  # 'local' 'pandora' 'hitkip'
    data = 'data_highDim'
    trainingBatch = '01'
    trainingYear_Month = 'test_XX'  # as short as possible to avoid too long paths for avoiding linux2windows transfer issues
    # trainingYear_Month = '_grid_bench_multi_beRNN_00_highDim_correctOnly_256'  # as short as possible to avoid too long paths for avoiding linux2windows transfer issues

    if 'highDim' in data:  # fix: lowDim_timeCompressed needs to be skipped here
        n_eachring = 32
        n_outputring = n_eachring
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_outputring + 1
    else:
        n_eachring = 10
        n_outputring = 2
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_outputring + 1

    hp = {
        'activation': 'relu',  # Type of activation runctions, relu, softplus, tanh, elu, linear
        'activations_per_layer': ['relu', 'tanh', 'linear'],
        'alpha': 0.2, # (redundant) discretization time step/time constant - dt/tau = alpha - ratio decides on how much previous states are taken into account for current state - low alpha more memory, high alpha more forgetting - alpha * h(t-1)
        'batch_size': 80,  # 20/40/80/120/160
        # 'batch_size_test': 640, # batch_size for testing
        'base_lr': [5e-4],
        'benchmark': False,
        # 'c_intsyn': 0, # intelligent synapses parameters, tuple (c, ksi) -> Yang et al. only apply these in sequential training
        'c_mask_responseValue': 5.,
        'data': data,
        'distanceOfEvaluationData': 0,
        'dt': 20,  # discretization time step (ms)
        'errorBalancingValue': 1.,  # multiplier of error costs and influence on backprop contribution
        'generalizationTest': False,  # month-wise distance between train and eval data for generalization testing
        'grad_clip': None,  # set None to disable
        'grad_clip_by': 'global_norm',  # or 'value'
        'in_type': 'normal',  # input type: normal, multi
        'l1_h': 1e-04,  # l1 lambda
        'l1_weight': 1e-03,  # l1 regularization on weight
        'l2_h': 0,  # l2 lambda
        'l2_weight': 1e-05,  # l2 regularization on weight
        'l2_weight_init': 0,  # l2 regularization on deviation from initialization
        'learning_rate': 0.001,  # learning rate
        'learning_rate_mode': 'triangular2',  # None - 'triangular', 'triangular2', 'exp_range', 'decay'
        'loss_type': 'lsq',  # # Type of loss functions - Cross-entropy loss
        # 'ksi_intsyn': 0,
        'machine': machine,
        # 'mask_threshold': .999,  # .999 or .975
        'max_lr': [15e-4],
        'monthsConsidered': ['month_3'],
        'monthsString': '3-5',  # monthsTaken
        'multiLayer': False,  # only applicaple with LeakyRNN
        'n_eachring': n_eachring,  # number of units each ring
        'n_input': n_input,  # number of input units
        'n_output': n_output,  # number of output units
        'num_ring': num_ring,  # number of rings
        'n_rule': n_rule,  # number of rules
        'n_rnn': 256,  # number of recurrent units for one hidden layer architecture
        'n_rnn_per_layer': [16, 16, 16],
        'optimizer': 'adam',  # 'adam', 'sgd'
        'participant': 'beRNN_00',  # Participant to take
        'p_weight_train': None,
        'rnn_type': 'LeakyRNN',  # Type of RNNs: NonRecurrent, LeakyRNN, LeakyGRU, EILeakyGRU | GRU, LSTM
        'rng': np.random.default_rng(),  # add seed here if you want to make it reproducible e.g. (42)
        'ruleset': 'all',  # all_benchmark
        'rule_probs': None,  # Rule probabilities to be drawn
        'rule_start': 1 + num_ring * n_eachring,  # first input index for rule units
        'save_name': 'test',  # name to save
        'sequenceMode': True,  # Decide if models are trained sequentially month-wise
        'sigma_rec': 0.01,  # recurrent noise - directly influencing the noise added to the network
        'sigma_x': 0,  # input noise
        's_mask': None,  # 'brain_256', None - info: only accesible on local machine
        'target_perf': 1.0,  # Stopping performance
        'tasksString': 'Alltask',  # tasks taken
        'tau': 100,  # # Time constant (ms)- default 100
        'threshold': 0.1,  # threshold applied to correlation matrix for adjacency matrix creation
        'trainingBatch': trainingBatch,
        'trainingYear_Month': trainingYear_Month,
        'use_separate_input': False,  # whether rule and stimulus inputs are represented separately
        'w_mask_value': 0.1,  # proportion of weights not to be regularized
        'w_rec_init': 'randgauss',
        # diag, randortho, randgauss, brainStructure (only accessible with LeakyRNN : 32-256)
    }

    if hp['ruleset'] == 'all':
        hp['rule_prob_map'] = dict({"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1,
                                    "RP_Ctx2": 1, "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1})
    elif hp['ruleset'] == 'all_benchmark':
        hp['rule_prob_map'] = dict(
            {"contextdm1": 1, "contextdm2": 1, "reactgo": 1, "reactanti": 1, "dmsgo": 1, "dmsnogo": 1,
             "dmcgo": 1, "dmcnogo": 1, "delaygo": 1, "delayanti": 1, "delaydm1": 1, "delaydm2": 1})
    else:
        raise ValueError('defined ruleset not recognized. Stopping process.')

    return hp


def do_eval(sess, model, log, rule_train, eval_data, step, data_dir):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    mode = 'test'

    weight_norms = []
    grad_norms = []

    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    for task in hp['rules']:
        n_rep = 20  # 20 * 40 or 20 * 20 trials per evaluation are taken, depending on batch_size
        # batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()

        for i_rep in range(n_rep):
            try:

                if hp['benchmark'] != True:
                    x, y, y_loc, response = tools.load_trials(hp['rng'], task, mode, hp['batch_size'], eval_data,
                                                              False)  # y_loc is participantResponse_perfEvalForm

                    c_mask = tools.create_cMask(y, response, hp, mode)

                    # fix: for incongruence between y and response dimension 1
                    if c_mask is None:
                        continue

                else:
                    # attention: CHANGES ####################################################
                    # benchmark training with simplified tasks (like Yang, Driscoll, Brenner)
                    rule_train_now = task
                    # Generate a random batch of trials
                    trial = generate_trials(
                        rule_train_now, hp, 'random',
                        batch_size=hp['batch_size'])

                    # # info. Load the preprocessed trials
                    # # Randomly draw batch in folder
                    # batchNumber = hp['rng'].choice(np.arange(40, 50))
                    # with open(os.path.join(data_dir, rule_train_now, f"{rule_train_now}_batch_{batchNumber}.pkl"),
                    #           "rb") as f:
                    #     trial = pickle.load(f)
                    #
                    # x, y, y_loc, c_mask = trial[0]['x'], trial[0]['y'], trial[0]['y_loc'], trial[0]['c_mask']
                    # # attention: CHANGES ####################################################
                    x, y, y_loc, c_mask = trial.x, trial.y, trial.y_loc, trial.c_mask

                feed_dict = tools.gen_feed_dict(model, x, y, c_mask,
                                                hp)  # y: participnt response, that gives the lable for what the network is trained for

                # head. new logic ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                fetches = [model.cost_lsq, model.cost_reg, model.y_hat, model.var_list]

                has_grads = hasattr(model, 'grads_and_vars') and model.grads_and_vars is not None
                if has_grads:
                    # model.grads_and_vars is usually a list of (gradient, variable) tuples
                    # We only want the gradient tensors (the first element [0])
                    grad_tensors = [gv[0] for gv in model.grads_and_vars if gv is not None and gv[0] is not None]
                    fetches.append(grad_tensors)

                results = sess.run(fetches, feed_dict=feed_dict)

                c_lsq, c_reg, y_hat_test, all_vars = results[0:4]

                # Calculate Global Weight Norm (L2) for this specific run
                current_w_norm = np.sqrt(np.sum([np.sum(np.square(v)) for v in all_vars]))
                weight_norms.append(current_w_norm)

                if has_grads:
                    all_grads = results[4]
                    # Calculate Global Gradient Norm (L2)
                    current_g_norm = np.sqrt(np.sum([np.sum(np.square(g)) for g in all_grads]))
                    grad_norms.append(current_g_norm)
                # head. new logic ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                if 'lowDim' in hp['data'] and hp['benchmark'] != True:
                    perf_test = np.mean(get_perf_lowDIM(y_hat_test, y_loc))

                elif 'RP_Anti' in task and hp['benchmark'] != True:
                    performanceList = []
                    perf_test = np.mean(
                        get_perf(y_hat_test, y_loc))  # info: y_loc is participant response as groundTruth
                    performanceList.append(perf_test)

                    pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)
                    # Check how many corrects are available per trial
                    for trial in range(np.shape(y_loc)[1]):
                        try:
                            if y_loc[-1][trial] != 0.05:  # exclude highDim noResponse responses
                                # Get number of correct responses
                                correctResponseDirection = np.where(pref == y_loc[-1][trial])[0][0]
                                correctIndicesArray_color = \
                                    np.where(x[-1, trial, 1:33] == x[-1, trial, correctResponseDirection + 1])[0]
                                correctIndicesArray_form = \
                                    np.where(x[-1, trial, 33:65] == x[-1, trial, correctResponseDirection + 33])[0]
                                # Compare both lists and keep only overlaps
                                correctIndicesArray_ = [x for x in correctIndicesArray_color if
                                                        x in correctIndicesArray_form]

                                # Keep all except the one you already used for perf_test calculation
                                correctIndicesArray = [x for x in correctIndicesArray_ if x != correctResponseDirection]

                                # Generate perf_test for each possible correct response
                                if len(correctIndicesArray) > 0:
                                    y_loc[35:, trial] = pref[correctIndicesArray[
                                        0]]  # store the second objectively correct response direction as y_loc - last one enough for evaluation

                        except Exception as e:  # Mainly issues with data_highDim occure
                            print(f"Skipping trial {trial} due to error: {e}")
                            continue

                    # Calculate performance one more time
                    perf_test = np.mean(
                        get_perf(y_hat_test, y_loc))  # info: y_loc is participant response as groundTruth
                    performanceList.append(perf_test)

                    # Compare both perf_test and take the most sucessful (giving the network the possibility to correctly respond to several correct directions)
                    perf_test = np.max(performanceList)

                elif 'RP_Ctx2' in task and hp['benchmark'] != True:
                    performanceList = []
                    perf_test = np.mean(
                        get_perf(y_hat_test, y_loc))  # info: y_loc is participant response as groundTruth
                    performanceList.append(perf_test)

                    pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)
                    # Check how many corrects are available per trial
                    for trial in range(np.shape(y_loc)[1]):
                        try:
                            if y_loc[-1][trial] != 0.05:  # exclude highDim noResponse responses
                                # Get number of correct responses
                                correctResponseDirection = np.where(pref == y_loc[-1][trial])[0][0]
                                # correctIndicesArray_color = np.where(x[-1, trial, 1:33] == x[-1, trial, correctResponseDirection + 1])[0]
                                correctIndicesArray_form = \
                                    np.where(x[-1, trial, 33:65] == x[-1, trial, correctResponseDirection + 33])[0]
                                # Compare both lists and keep only overlaps
                                # correctIndicesArray_ = [x for x in correctIndicesArray_color if x in correctIndicesArray_form]

                                # Keep all except the one you already used for perf_test calculation
                                correctIndicesArray = [x for x in correctIndicesArray_form if
                                                       x != correctResponseDirection]

                                # Generate perf_test for each possible correct response
                                if len(correctIndicesArray) > 0:
                                    y_loc[35:, trial] = pref[correctIndicesArray[
                                        0]]  # store the second objectively correct response direction as y_loc - last one enough for evaluation

                        except Exception as e:  # Mainly issues with data_highDim occure
                            print(f"Skipping trial {trial} due to error: {e}")
                            continue

                    # Calculate performance one more time
                    perf_test = np.mean(
                        get_perf(y_hat_test, y_loc))  # info: y_loc is participant response as groundTruth
                    performanceList.append(perf_test)

                    # Compare both perf_test and take the most sucessful (giving the network the possibility to correctly respond to several correct directions)
                    perf_test = np.max(performanceList)

                else:
                    perf_test = np.mean(
                        get_perf(y_hat_test, y_loc))  # info: y_loc is participant response as groundTruth

                clsq_tmp.append(c_lsq)
                creg_tmp.append(c_reg)
                perf_tmp.append(perf_test)

            except Exception as e:
                print(f"Error during evaluation of task {task}, iteration {i_rep}: {e}")
                continue  # Skip iteration on error

        # If no valid results were collected, continue with next task
        if not clsq_tmp or not perf_tmp:
            print(f"Skipping task {task} as no valid data was processed.")
            continue

        # head. new mlflow logic ###########################################################################################
        mlflow.log_metric(f"cost_leastSquareError_{task}", np.mean(clsq_tmp, dtype=np.float64), step=step)
        mlflow.log_metric(f"cost_regularization_{task}", np.mean(creg_tmp, dtype=np.float64), step=step)
        mlflow.log_metric(f"performance_test_{task}", np.mean(perf_tmp, dtype=np.float64), step=step)
        # head. new mlflow logic ###########################################################################################

        log['cost_' + task].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_' + task].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_' + task].append(np.mean(perf_tmp, dtype=np.float64))

        print('{:15s}'.format(task) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    # info: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]

    try:
        perf_tests_mean = np.mean([log['perf_' + r][-1] for r in rule_tmp])
        log['perf_avg'].append(perf_tests_mean)
        perf_tests_min = np.min([log['perf_' + r][-1] for r in rule_tmp])
        log['perf_min'].append(perf_tests_min)

        log['global_weight_norm_avg'].append(np.mean(weight_norms, dtype=np.float64))
        log['global_grad_norm_avg'].append(np.mean(grad_norms, dtype=np.float64))

        # head. new mlflow logic ###########################################################################################
        mlflow.log_metric(f"average_performance_test", perf_tests_mean, step=step)
        mlflow.log_metric(f"minimum_performance_test", perf_tests_min, step=step)
        mlflow.log_metric("global_weight_norm_avg", np.mean(weight_norms), step=step)
        if grad_norms: mlflow.log_metric("global_grad_norm_avg", np.mean(grad_norms), step=step)
        # head. new mlflow logic ###########################################################################################

    except KeyError as e:
        print(f"Warning: Could not compute final performance due to missing key {e}.")
        return log

    # Save the model and log
    try:
        model.save()
        tools.save_log(log)
    except Exception as e:
        print(f"Warning: Could not save model/log due to {e}.")

    return log


def train(data_dir, model_dir, train_data, eval_data, hp=None, max_steps=4800, display_step=500,
          rule_trains=None, rule_prob_map=None, seed=0,
          load_dir=None, trainables=None, robustnessTest=True):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

    # attention: standard hp ##########################################################################################
    # Network parameters
    ruleset = hp['ruleset']
    default_hp = get_default_hp(ruleset)
    # default_hp = get_default_hp('all')
    if hp is not None:
        default_hp.update(hp)
    # attention: standard hp ##########################################################################################

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = tools.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if hp['rule_prob_map'] is None:
        hp['rule_prob_map'] = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array([hp['rule_prob_map'].get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))
    # if robustnessTest != True:
    hp['seed'] = seed  # Plug in seed into rng creation for reproducability - default: 0
    tools.save_hp(hp, model_dir)

    # info: Add 'rng' here after it was pop out
    hp['rng'] = np.random.default_rng()  # set rng_seed here for reproducability - default: not set

    # Build the model
    model = Model(model_dir, hp=hp)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir
    log['trials'].append(0)

    # Record time
    t_start = time.time()
    # Count loaded trials/batches
    trialsLoaded = 0

    # head. new mlflow logic ###########################################################################################
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    experiment_name = hp['trainingYear_Month']

    if hp['machine'] == 'local':
        db_path = Path(rf"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\mlflow_{experiment_name}_{task_id}.db").absolute()
    else:
        db_path = Path(f"/zi/home/oliver.frank/Desktop/beRNNmodels/mlflow_{experiment_name}_{task_id}.db")

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    # mlflow.set_experiment(experiment_name)

    # Protect database initialization for several simultaneous accessing
    # if hp['machine'] != 'local':
    #     time.sleep(1)  # wait 10 seconds between each job start .info for the current setup with task individual db unnecessary

    # Experiment-Handling with MlflowClient
    client = mlflow.tracking.MlflowClient()
    try:
        # check if existent
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            client.create_experiment(experiment_name)
    except Exception as e:
        print(f"MLflow Error: {e}")

    mlflow.set_experiment(experiment_name)

    # # Restart if database is currently blocked
    # max_retries = 5
    # for attempt in range(max_retries):
    with mlflow.start_run(run_name=os.path.basename(model_dir)) as run:
        # Nur Strings/Zahlen loggen (keine Dicts/Listen)
        clean_hp = {k: str(v) for k, v in hp.items() if not isinstance(v, (dict, list))}
        mlflow.log_params(clean_hp)
        # head. new mlflow logic ###########################################################################################

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
                print('model restored')
            else:
                # Initialize variables from scratch
                sess.run(tf.global_variables_initializer())

            # Set trainable parameters
            if trainables is None or trainables == 'all':
                var_list = model.var_list  # train everything
            elif trainables == 'input':
                # train all nputs
                var_list = [v for v in model.var_list if ('input' in v.name) and ('rnn' not in v.name)]
            elif trainables == 'rule':
                # train rule inputs only
                var_list = [v for v in model.var_list if 'rule_input' in v.name]
            else:
                raise ValueError('Unknown trainables')

            # Define variables to optimize
            model.set_optimizer(var_list=var_list)

            # penalty on deviation from initial weight
            if hp['l2_weight_init'] > 0:
                anchor_ws = sess.run(model.weight_list)
                for w, w_val in zip(model.weight_list, anchor_ws):
                    model.cost_reg += (hp['l2_weight_init'] * tf.nn.l2_loss(w - w_val))

                model.set_optimizer(var_list=var_list)

            # partial weight training
            # Explanation: In summary, this code introduces a form of partial weight training by applying L2 regularization
            # only to a subset of the weights. The subset is determined by random masking, controlled by the hyperparameter
            # 'p_weight_train'. All weights below the p_weight_train threshold won't be trained in this iteration.
            if ('p_weight_train' in hp and
                    (hp['p_weight_train'] is not None) and
                    hp['p_weight_train'] < 1.0):
                for w in model.weight_list:
                    w_val = sess.run(w)
                    w_size = sess.run(tf.size(w))
                    w_mask_tmp = np.linspace(0, 1, w_size)
                    hp['rng'].shuffle(w_mask_tmp)
                    ind_fix = w_mask_tmp > hp['p_weight_train']
                    w_mask = np.zeros(w_size, dtype=np.float32)
                    w_mask[ind_fix] = hp['w_mask_value']  # 1e-1  # will be squared in l2_loss
                    w_mask = tf.constant(w_mask)
                    w_mask = tf.reshape(w_mask, w.shape)
                    model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
                model.set_optimizer(var_list=var_list)

            step = 0
            while log['trials'][-1] <= 1e6:
            # while step <= max_steps:
                try:
                    # Validation
                    if step % display_step == 0:  # III: Every 500 steps (20000 trials) do the evaluation
                        log['trials'].append(step * hp['batch_size'])
                        log['times'].append(time.time() - t_start)
                        log = do_eval(sess, model, log, hp['rule_trains'], train_data, step, data_dir)
                        elapsed_time = time.time() - t_start  # Calculate elapsed time
                        print(f"Elapsed time after batch number {trialsLoaded}: {elapsed_time:.2f} seconds")
                        # After training
                        total_time = time.time() - t_start
                        print(f"Total training time: {total_time:.2f} seconds")
                        # if log['perf_avg'][-1] > model.hp['target_perf']:
                        # check if minimum performance is above target
                        if log['perf_min'][-1] > model.hp['target_perf']:
                            print('Perf reached the target: {:0.2f}'.format(
                                hp['target_perf']))
                            break

                        # info: Add modularity value once each evaluation ##################################################
                        if hp['multiLayer'] == False:
                            getAndSafeModValue(data_dir, model_dir, hp, model, sess, log, step)

                        # if rich_output:
                        #     display_rich_output(model, sess, step, log, model_dir)

                    # Training
                    task = hp['rng'].choice(hp['rule_trains'], p=hp['rule_probs'])
                    # Generate a random batch of trials; each batch has the same trial length
                    mode = 'train'

                    if hp['benchmark'] != True:
                        x, y, y_loc, response = tools.load_trials(hp['rng'], task, mode, hp['batch_size'],
                                                                  train_data,
                                                                  False)  # y_loc is participantResponse_perfEvalForm

                        c_mask = tools.create_cMask(y, response, hp, mode)

                        # fix: for inconcruence between y and response dimension 1
                        if c_mask is None:
                            continue

                    else:
                        # benchmark training with simplified tasks (like Yang, Driscoll, Brenner)
                        rule_train_now = task

                        # Generate a random batch of trials
                        trial = generate_trials(
                            rule_train_now, hp, 'random',
                            batch_size=hp['batch_size'])

                        # info. Load the preprocessed trials
                        # Randomly draw batch in folder
                        # batchNumber = hp['rng'].choice(np.arange(0, 40))
                        # with open(os.path.join(data_dir, rule_train_now, f"{rule_train_now}_batch_{batchNumber}.pkl"), "rb") as f:
                        #     trial = pickle.load(f)

                        # x, y, y_loc, c_mask = trial[0]['x'], trial[0]['y'], trial[0]['y_loc'], trial[0]['c_mask']

                        x, y, y_loc, c_mask = trial.x, trial.y, trial.y_loc, trial.c_mask

                    # fix: for inconcruence between y and response on dimension 1 - probably preprocessing related
                    # fix: for inconcruence between y and response on dimension 1 - probably preprocessing related
                    if (c_mask is None or (isinstance(c_mask, np.ndarray) and (
                            c_mask.size == 0 or np.all(c_mask == None) or np.any(c_mask == None)))):
                        continue

                    trialsLoaded += 1

                    # Generating feed_dict.
                    feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)
                    # print('passed feed_dict Training')
                    # print(feed_dict)

                    sess.run(model.train_step,
                             feed_dict=feed_dict)  # info: Trainables are actualized - train_step should represent the step in training.py and the global_step in network.py

                    # Get Training performance in a similiar fashion as in do_eval
                    clsq_train_tmp = list()
                    creg_train_tmp = list()
                    perf_train_tmp = list()
                    c_lsq_train, c_reg_train, y_hat_train = sess.run([model.cost_lsq, model.cost_reg, model.y_hat],
                                                                     feed_dict=feed_dict)  # info: lsq+reg = total_loss - updates the network parameters

                    if 'lowDim' in hp['data']:
                        perf_train = np.round(np.mean(get_perf_lowDIM(y_hat_train, y_loc)),
                                              3)  # info: y_loc is participant response as groundTruth
                    else:
                        perf_train = np.round(np.mean(get_perf(y_hat_train, y_loc)),
                                              3)  # info: y_loc is participant response as groundTruth

                    clsq_train_tmp.append(c_lsq_train)
                    creg_train_tmp.append(c_reg_train)
                    perf_train_tmp.append(perf_train)

                    log['cost_train_' + task].append(np.mean(clsq_train_tmp, dtype=np.float64))
                    log['creg_train_' + task].append(np.mean(creg_train_tmp, dtype=np.float64))
                    log['perf_train_' + task].append(np.mean(perf_train_tmp, dtype=np.float64))

                    # head. new mlflow logic ###########################################################################################
                    mlflow.log_metric(f"cost_leastSquareError_train_{task}",
                                      np.mean(clsq_train_tmp, dtype=np.float64), step=step)
                    mlflow.log_metric(f"cost_regularization_train_{task}",
                                      np.mean(creg_train_tmp, dtype=np.float64), step=step)
                    mlflow.log_metric(f"performance_train_{task}", np.mean(perf_train_tmp, dtype=np.float64),
                                      step=step)

                    # Create avg perf train value over all tasks for each step
                    current_tasks_perf = [log['perf_train_' + t][-1] for t in hp['rule_trains'] if
                                          log['perf_train_' + t]]
                    if current_tasks_perf:
                        train_perf_avg = np.mean(current_tasks_perf)
                        mlflow.log_metric("performance_train_avg", train_perf_avg, step=step)
                    # head. new mlflow logic ###########################################################################################

                    print('{:15s}'.format(task) +
                          '| train cost {:0.6f}'.format(np.mean(clsq_train_tmp)) +
                          '| train c_reg {:0.6f}'.format(np.mean(c_reg_train)) +
                          '  | train perf {:0.2f}'.format(np.mean(perf_train)))

                    step += 1

                except KeyboardInterrupt:
                    print("Optimization interrupted by user")
                    break

            print("Optimization finished!")



        ########################################################################################################################

########################################################################################################################
# head. Train model
########################################################################################################################
if __name__ == '__main__':
    # Initialize list for all training times for each model
    trainingTimeList = []

    # if hp['machine'] == 'local':
    # attention: local #############################################################################################
    # info: if used, comment out standard hp in train()
    hp = get_default_hp('all')  # 'all_benchmark' - not important at this point as both sets have 12 tasks
    # attention: local #############################################################################################

    # Define number of iterations for docker image on remote VM
    if hp['machine'] == 'hitkip':  # docker image should always be build with hp['machine'] == 'hitkip'
        config_path = os.getenv("CONFIG_PATH", "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        n_iteration = config.get("n_iteration", 2)  # Default if not specified in config

    # Define number of iterations for docker image on remote VM
    else:
        n_iteration = 2

    for modelNumber in range(1, n_iteration):  # Define number of iterations and models to be created for every month, respectively

        # Measure time for every model, respectively
        trainingTimeTotal_hours = 0
        # Start it
        start_time = time.perf_counter()
        print(f'START TRAINING MODEL: {modelNumber}')

        if hp['machine'] == 'hitkip':
            # attention: cluster ############################################################################################
            # Convert the JSON string to a Python dictionary
            robustnessTest_model = config.get("robustnessTest_model", 'hp_1') # Default if not specified in config

            with open(
                    f"/zi/home/oliver.frank/Desktop/RNN/multitask_BeRNN-main/_bestModels_scripts/best10_gridSearch_multiTask_beRNN_03_highDimCorrects_256/{robustnessTest_model}.json",
                    "r") as f:
                hp = json.load(f)

            hp['participant'] = config.get("participant", 'beRNN_03') # Default if not specified in config
            participant = hp['participant']

            hp['data'] = config.get("data", 'highDim_correctOnly') # Default if not specified in config
            data = hp['data']

            hp['n_rnn'] = config.get("n_rnn", 256) # Default if not specified in config
            n_rnn = hp['n_rnn']

            hp['trainingYear_Month'] = config.get("trainingYear_Month", f'_robust_multi_{participant}_{data}_{n_rnn}') # Default if not specified in config

            # legacy low dimensional encoding of input layers - not important at this point
            # hp['n_eachring'] = 10
            # hp['n_outputring'] = 2
            # hp['n_input'], hp['n_output'] = 1 + hp['num_ring'] * hp['n_eachring'] + hp['n_rule'], hp['n_outputring'] + 1
            # attention: cluster #############################################################################################

        load_dir = None

        # Define main path
        if hp['machine'] == 'local':
            path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'
        elif hp['machine'] == 'hitkip':
            path = '/zi/home/oliver.frank/Desktop'
        elif hp['machine'] == 'pandora':
            path = '/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main'

        # Define data path
        # Define data path
        preprocessedData_path = os.path.join(path, 'Data', hp['participant'], hp['data'])

        for month in hp['monthsConsidered']:  # info: You have to delete this if cascade training should be set OFF
            # Adjust variables manually as needed
            model_name = f'model_{month}'

            # Define model_dir for different servers
            if hp['machine'] == 'local':
                if hp['multiLayer'] == True:
                    hp['rnn_type'] = 'LeakyRNN'  # info: force rnn_type to always be LeakyRNN for multiLayer case
                    numberOfLayers = len(hp['n_rnn_per_layer'])
                    if numberOfLayers == 2:
                        model_dir = os.path.join(
                            f"{path}\\beRNNmodels\\{hp['trainingYear_Month']}\\{hp['data'].split('data_')[-1]}\\{hp['participant']}\\{hp['trainingBatch']}\\iter{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}_{hp['activations_per_layer'][0][0]}-{hp['activations_per_layer'][1][0]}",
                            model_name)
                    else:
                        model_dir = os.path.join(
                            f"{path}\\beRNNmodels\\{hp['trainingYear_Month']}\\{hp['data'].split('data_')[-1]}\\{hp['participant']}\\{hp['trainingBatch']}\\iter{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}-{hp['n_rnn_per_layer'][2]}_{hp['activations_per_layer'][0][0]}-{hp['activations_per_layer'][1][0]}-{hp['activations_per_layer'][2][0]}",
                            model_name)
                else:
                    model_dir = os.path.join(
                        f"{path}\\beRNNmodels\\{hp['trainingYear_Month']}\\{hp['data'].split('data_')[-1]}\\{hp['participant']}\\{hp['trainingBatch']}\\iter{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn']}_{hp['activation']}",
                        model_name)

            elif hp['machine'] == 'hitkip' or hp['machine'] == 'pandora':
                if hp['multiLayer'] == True:
                    hp['rnn_type'] = 'LeakyRNN'  # info: force rnn_type to always be LeakyRNN for multiLayer case
                    numberOfLayers = len(hp['n_rnn_per_layer'])
                    if numberOfLayers == 2:
                        model_dir = os.path.join(
                            f"{path}/beRNNmodels/{hp['trainingYear_Month']}/{hp['data'].split('data_')[-1]}/{hp['participant']}/{hp['trainingBatch']}/iter{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}_{hp['activations_per_layer'][0][0]}-{hp['activations_per_layer'][1][0]}",
                            model_name)
                    else:
                        model_dir = os.path.join(
                            f"{path}/beRNNmodels/{hp['trainingYear_Month']}/{hp['data'].split('data_')[-1]}/{hp['participant']}/{hp['trainingBatch']}/iter{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}-{hp['n_rnn_per_layer'][2]}_{hp['activations_per_layer'][0][0]}-{hp['activations_per_layer'][1][0]}-{hp['activations_per_layer'][2][0]}",
                            model_name)
                else:
                    model_dir = os.path.join(
                        f"{path}/beRNNmodels/{hp['trainingYear_Month']}/{hp['data'].split('data_')[-1]}/{hp['participant']}/{hp['trainingBatch']}/iter{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn']}_{hp['activation']}",
                        model_name)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            if hp['generalizationTest'] == True:
                # Create train and eval data
                train_data, eval_data = tools.createSplittedDatasets_generalizationTest(hp, preprocessedData_path,
                                                                                        month, hp['distanceOfEvaluationData'])
            elif hp['participant'] == 'beRNN_00':
                train_data, eval_data = tools.createSplittedDatasets_beRNN_00(hp, preprocessedData_path, month)

            else:
                # Create train and eval data
                train_data, eval_data = tools.createSplittedDatasets(hp, preprocessedData_path, month)

            # info: If you want to initialize the new model with an old one
            # load_dir = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\beRNNmodels\\2025_03\\sc_mask_final\\beRNN_03_All_3-5_data_highDim_correctOnly_iteration1_LeakyRNN_1000_relu\\model_month_3'
            # Start Training ---------------------------------------------------------------------------------------------------
            train(preprocessedData_path, model_dir=model_dir, train_data=train_data, eval_data=eval_data, hp=hp,
                  load_dir=load_dir)

            # info: If True previous model parameters will be taken to initialize consecutive model, creating sequential training
            if hp['sequenceMode'] == True:
                load_dir = model_dir

        end_time = time.perf_counter()
        # Training time taken into account
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        elapsed_time_hours = elapsed_time_minutes / 60

        print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_seconds:.2f} seconds")
        print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_minutes:.2f} minutes")
        print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_hours:.2f} hours")

        # Accumulate trainingTime
        trainingTimeList.append(elapsed_time_hours)
        trainingTimeTotal_hours += elapsed_time_hours

    # Save training time total and list to folder as a text file
    file_path = os.path.join(path, 'beRNNmodels', hp['trainingYear_Month'], hp['data'].split('data_')[-1],
                             hp['participant'], hp['trainingBatch'], 'times.txt')

    with open(file_path, 'w') as f:
        f.write(f"training time total (hours): {trainingTimeTotal_hours}\n")
        f.write("training time each individual model (hours):\n")
        for time in trainingTimeList:
            f.write(f"{time}\n")

    print(f"Training times saved to {file_path}")


