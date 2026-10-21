import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import errno
import six
import json
import random
import pickle
# import shutil
# from glob import glob
import numpy as np

'''
Roles:
- Defines helper functions for the whole repository
'''

rules_dict = {'all' : ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2',
              'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2'], # All tasks
              'DMtasks' : ['DM', 'DM_Anti'], # DM tasks isolated
              'EFtasks' : ['EF', 'EF_Anti'], # EF tasks isolated
              'DM&EFtasks' : ['DM', 'DM_Anti', 'EF', 'EF_Anti'],
              'RPtasks' : ['RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2'], # RP tasks isolated
              'WMtasks' : ['WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2'], # WM tasks isolated
              'RP&WMCTXtasks' : ['RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM_Ctx1', 'WM_Ctx2'],
              'WM&WMANTItasks' : ['WM', 'WM_Anti'],
              'fundamentals' : ['DM', 'EF', 'RP', 'WM'],
              'all_benchmark': [
                  'contextdm1', 'contextdm2',  # s_DM_Ctx1, s_DM_Ctx2
                  'reactgo', 'reactanti',  # s_EF_Pro, s_EF_Anti
                  'dmsgo', 'dmsnogo',  # s_RP_Pro1, s_RP_Anti1
                  'dmcgo', 'dmcnogo',  # s_RP_Pro2, s_RP_Anti2
                  'delaygo', 'delayanti',  # s_WM_Pro, s_WM_Anti
                  'delaydm1', 'delaydm2']  # s_WM_Ctx1, s_WM_Ctx2
              }

rule_name = {
            'DM': 'Decison Making Pro (DM Pro)',
            'DM_Anti': 'Decision Making Anti (DM Anti)',
            'EF': 'Executive Functioning Pro (EF Pro)',
            'EF_Anti': 'Executive Functioning Anti (EF Anti)',
            'RP': 'Relational Processing Pro (RP Pro)',
            'RP_Anti': 'Relational Processing Anti (RP Anti)',
            'RP_Ctx1': 'Relational Processing Context 1 (RP Ctx1)',
            'RP_Ctx2': 'Relational Processing Context 2 (RP Ctx2)',
            'WM': 'Working Memory Pro (WM Pro)',
            'WM_Anti': 'Working Memory Anti (WM Anti)',
            'WM_Ctx1': 'Working Memory Context 1 (WM Ctx1)',
            'WM_Ctx2': 'Working Memory Context 2 (WM Ctx2)'
            }

rule_name_benchmark = {
                'contextdm1': 'Simple Decison Making Context 1 (s_DM_Ctx1)',
                'contextdm2': 'Simple Decison Making Context 2 (s_DM_Ctx2)',
                'reactgo': 'Simple Executive Functioning Pro (s_EF_Pro)',
                'reactanti': 'Simple Executive Functioning Anti (s_EF_Anti)',
                'dmsgo': 'Simple Relational Processing Pro 1 (s_RP_Pro1)',
                'dmsnogo': 'Simple Relational processing Anti 1 (s_RP_Anti1)',
                'dmcgo': 'Simple Relational Processing Pro 2 (s_RP_Pro2)',
                'dmcnogo': 'Simple Relational Processing Anti 2 (s_RP_Anti2)',
                'delaygo': 'Simple Working Memory Pro (s_WM_Pro)',
                'delayanti': 'Simple Working Memory Anti (s_WM_Anti)',
                'delaydm1': 'Simple Working Memory Context 1 (s_WM_Ctx1)',
                'delaydm2': 'Simple Working Memory Context 2 (s_WM_Ctx2)'
                }

# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind

def participation_coefficient(G_sparse, communities_sparse):
    # Guimerà & Amaral (2005): Measures involvement of each node of a graph in all communities.
    # 0 if only involved in own assigned community - converging 1 if involved in all other communities similarily
    # info: Has to be calculated in combination with networkx modularity

    # Map each node to its community ID for easy lookup
    node_to_comm = {}
    for i, comm in enumerate(communities_sparse):
        for node in comm:
            node_to_comm[node] = i

    pc_dict = {}
    for node in G_sparse.nodes():
        degree = G_sparse.degree(node) # total number of connections for node i
        if degree == 0:
            pc_dict[node] = 0
            continue

        # Count edges from this node to each community
        community_edge_counts = {}
        for neighbor in G_sparse.neighbors(node):
            comm_id = node_to_comm[neighbor]
            community_edge_counts[comm_id] = community_edge_counts.get(comm_id, 0) + 1

        # Apply the formula: 1 - sum((edges_to_comm / total_degree)^2)
        sum_sq_ratios = sum((count / degree) ** 2 for count in community_edge_counts.values())
        pc_dict[node] = 1 - sum_sq_ratios

    return pc_dict

def get_num_ring(ruleset):
    '''get number of stimulus rings'''
    return 2 if ruleset=='all' else 2 # leave it felxible for potential future rulesets

def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])

def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule]+config['rule_start']

def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

def truncate_to_smallest(arrays):
    """Truncate each array in the list to the smallest first dimension size."""
    # Find the smallest first dimension size
    min_size = min(arr.shape[0] for arr in arrays)

    # Truncate each array to the min_size
    if arrays[0].ndim == 2:
        truncated_arrays = [arr[arr.shape[0] - min_size:, :] for arr in arrays]
    elif arrays[0].ndim == 3:
        truncated_arrays = [arr[arr.shape[0]-min_size:, :, :] for arr in arrays]

    return truncated_arrays

def load_trials(rng,task,mode,batchSize,data,errorComparison):
    '''Load trials from pickle file'''
    # Build-in mechanism to prevent interruption of code as for many .npy files there errors are raised
    max_attempts = 30
    attempt = 0
    while attempt < max_attempts:
        # if mode == 'Training':
        #     # random choose one of the preprocessed files according to the current chosen task
        #     file_splits = random.choice(os.listdir(os.path.join(trial_dir,task))).split('-')
        #     while file_splits[1].split('_')[1] not in monthsConsidered:
        #         # randomly choose another file until the one for the right considered month is found
        #         file_splits = random.choice(os.listdir(os.path.join(trial_dir, task))).split('-')
        # elif mode == 'Evaluation':
        #     # random choose one of the preprocessed files according to the current chosen task
        #     file_splits = random.choice(os.listdir(os.path.join(trial_dir, '_Evaluation_Data', task))).split('-')
        #     while file_splits[1].split('_')[1] not in monthsConsidered:
        #         # randomly choose another file until the one for the right considered month is found
        #         file_splits = random.choice(os.listdir(os.path.join(trial_dir, '_Evaluation_Data', task))).split('-')
        # file_stem = '-'.join(file_splits[:-1]) # '-'.join(...)
        try:
            numberOfBatches = int(batchSize / 40) # Choose numberOfBatches for one training step
            if numberOfBatches < 1: numberOfBatches = 1  # Be sure to load at least one batch and then sample it down if defined by batchSize

            if mode == 'train':
                # Choose the triplet from the splitted data
                currenTask_values = []
                for key, values in data.items():
                    if key.endswith(task):
                        currenTask_values.extend(values)
                # Select number of batches according to defined batchSize
                currentQuartett = rng.choice(currenTask_values, size=numberOfBatches, replace=False).tolist()

                # Load the files
                if numberOfBatches <= 1:
                    x = np.load(currentQuartett[0][0]) # Input
                    y = np.load(currentQuartett[0][2]) # Participant Response
                    y_loc = np.load(currentQuartett[0][1]) # Human Ground Truth
                    response = np.load(currentQuartett[0][3], allow_pickle=True) # Objective Ground Truth - only needed for training if error balancing is applied

                    if batchSize < 40:
                        # randomly choose ratio for part of batch to take
                        choice = rng.choice(['first', 'last', 'middle'])
                        if choice == 'first':
                            # Select rows for either training
                            x = x[:, :batchSize, :]
                            y = y[:, :batchSize, :]
                            y_loc = y_loc[:, :batchSize]
                            response = response[:, :batchSize]
                        elif choice == 'last':
                            # Select rows for either training
                            x = x[:, 40-batchSize:, :]
                            y = y[:, 40-batchSize:, :]
                            y_loc = y_loc[:, 40-batchSize:]
                            response = response[:, 40-batchSize:]
                        elif choice == 'middle':
                            # Select the middle batchSize rows
                            mid_start = (x.shape[1] - batchSize) // 2
                            mid_end = mid_start + batchSize
                            x = x[:, mid_start:mid_end, :]
                            y = y[:, mid_start:mid_end, :]
                            y_loc = y_loc[:, mid_start:mid_end]
                            response = response[:, mid_start:mid_end]
                elif numberOfBatches == 2:
                    x_0 = np.load(currentQuartett[0][0])  # Input
                    y_0 = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc_0 = np.load(currentQuartett[0][1])  # Ground Truth # yLoc
                    response_0 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    x_1 = np.load(currentQuartett[1][0])  # Input
                    y_1 = np.load(currentQuartett[1][2])  # Participant Response
                    y_loc_1 = np.load(currentQuartett[1][1])  # Ground Truth # yLoc
                    response_1 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    # Decrease first dimension size of batches to the size of the smallest by truncating first fixation epoch rows
                    truncated_arrays_x = truncate_to_smallest([x_0, x_1])
                    truncated_arrays_y = truncate_to_smallest([y_0, y_1])
                    truncated_arrays_y_loc = truncate_to_smallest([y_loc_0, y_loc_1])
                    truncated_arrays_response = truncate_to_smallest([response_0, response_1])
                    # Concatenate the trauncated batches
                    x = np.concatenate((truncated_arrays_x[0], truncated_arrays_x[1]), axis=1)
                    y = np.concatenate((truncated_arrays_y[0], truncated_arrays_y[1]), axis=1)
                    y_loc = np.concatenate((truncated_arrays_y_loc[0], truncated_arrays_y_loc[1]), axis=1)
                    response = np.concatenate((truncated_arrays_response[0], truncated_arrays_response[1]), axis=1)

                elif numberOfBatches == 3:
                    x_0 = np.load(currentQuartett[0][0])  # Input
                    y_0 = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc_0 = np.load(currentQuartett[0][1])  # Ground Truth # yLoc
                    response_0 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    x_1 = np.load(currentQuartett[1][0])  # Input
                    y_1 = np.load(currentQuartett[1][2])  # Participant Response
                    y_loc_1 = np.load(currentQuartett[1][1])  # Ground Truth # yLoc
                    response_1 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    x_2 = np.load(currentQuartett[2][0])  # Input
                    y_2 = np.load(currentQuartett[2][2])  # Participant Response
                    y_loc_2 = np.load(currentQuartett[2][1])  # Ground Truth # yLoc
                    response_2 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    # Decrease first dimension size of batches to the size of the smallest by truncating first fixation epoch rows
                    truncated_arrays_x = truncate_to_smallest([x_0, x_1, x_2])
                    truncated_arrays_y = truncate_to_smallest([y_0, y_1, y_2])
                    truncated_arrays_y_loc = truncate_to_smallest([y_loc_0, y_loc_1, y_loc_2])
                    truncated_arrays_response = truncate_to_smallest([response_0, response_1, response_2])
                    # Concatenate the trauncated batches
                    x = np.concatenate((truncated_arrays_x[0], truncated_arrays_x[1], truncated_arrays_x[2]), axis=1)
                    y = np.concatenate((truncated_arrays_y[0], truncated_arrays_y[1], truncated_arrays_y[2]), axis=1)
                    y_loc = np.concatenate((truncated_arrays_y_loc[0], truncated_arrays_y_loc[1], truncated_arrays_y_loc[2]), axis=1)
                    response = np.concatenate((truncated_arrays_response[0], truncated_arrays_response[1], truncated_arrays_response[2]), axis=1)

                elif numberOfBatches == 4:
                    x_0 = np.load(currentQuartett[0][0])  # Input
                    y_0 = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc_0 = np.load(currentQuartett[0][1])  # Ground Truth # yLoc
                    response_0 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    x_1 = np.load(currentQuartett[1][0])  # Input
                    y_1 = np.load(currentQuartett[1][2])  # Participant Response
                    y_loc_1 = np.load(currentQuartett[1][1])  # Ground Truth # yLoc
                    response_1 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    x_2 = np.load(currentQuartett[2][0])  # Input
                    y_2 = np.load(currentQuartett[2][2])  # Participant Response
                    y_loc_2 = np.load(currentQuartett[2][1])  # Ground Truth # yLoc
                    response_2 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    x_3 = np.load(currentQuartett[3][0])  # Input
                    y_3 = np.load(currentQuartett[3][2])  # Participant Response
                    y_loc_3 = np.load(currentQuartett[3][1])  # Ground Truth # yLoc
                    response_3 = np.load(currentQuartett[0][3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

                    # Decrease first dimension size of batches to the size of the smallest by truncating first fixation epoch rows
                    truncated_arrays_x = truncate_to_smallest([x_0, x_1, x_2, x_3])
                    truncated_arrays_y = truncate_to_smallest([y_0, y_1, y_2, y_3])
                    truncated_arrays_y_loc = truncate_to_smallest([y_loc_0, y_loc_1, y_loc_2, y_loc_3])
                    truncated_arrays_response = truncate_to_smallest([response_0, response_1, response_2, response_3])
                    # Concatenate the trauncated batches
                    x = np.concatenate((truncated_arrays_x[0], truncated_arrays_x[1], truncated_arrays_x[2], truncated_arrays_x[3]), axis=1)
                    y = np.concatenate((truncated_arrays_y[0], truncated_arrays_y[1], truncated_arrays_y[2], truncated_arrays_y[3]), axis=1)
                    y_loc = np.concatenate((truncated_arrays_y_loc[0], truncated_arrays_y_loc[1], truncated_arrays_y_loc[2], truncated_arrays_y_loc[3]), axis=1)
                    response = np.concatenate((truncated_arrays_response[0], truncated_arrays_response[1], truncated_arrays_response[2], truncated_arrays_response[3]), axis=1)
                else:
                    raise ValueError(f"batchSize {batchSize} is not valid")

            elif mode == 'test':
                response = [] # only needed for training if error balancing is applied
                # Choose the triplet from the splitted data
                if errorComparison == False:
                    currenTask_values = []
                    for key, values in data.items():
                        if key.endswith(task):
                            currenTask_values.extend(values)
                    if len(currenTask_values) < numberOfBatches: # info: In case there is not enough data to create batches with trials > 40
                        currentQuartett = rng.choice(currenTask_values, size=numberOfBatches, replace=True).tolist()

                    else:
                        currentQuartett = rng.choice(currenTask_values, size=numberOfBatches, replace=False).tolist()

                elif errorComparison == True:
                    currentQuartett = rng.choice(data, size=numberOfBatches, replace=False).tolist() # info: for errorComparison.py
                    base_name = currentQuartett[0][0].split('Input')[0]
                    # print('chosenFile:  ', base_name)
                # Load the files
                if numberOfBatches <= 1:
                    x = np.load(currentQuartett[0][0])  # Input
                    y = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc = np.load(currentQuartett[0][1])  # Ground Truth # yLoc
                    if batchSize < 40:
                        # randomly choose ratio for part of batch to take
                        choice = rng.choice(['first', 'last', 'middle'])
                        if choice == 'first':
                            # Select rows for either training
                            x = x[:, :batchSize, :]
                            y = y[:, :batchSize, :]
                            y_loc = y_loc[:, :batchSize]
                        elif choice == 'last':
                            # Select rows for either training
                            x = x[:, 40 - batchSize:, :]
                            y = y[:, 40 - batchSize:, :]
                            y_loc = y_loc[:, 40 - batchSize:]
                        elif choice == 'middle':
                            # Select the middle batchSize rows
                            mid_start = (x.shape[1] - batchSize) // 2
                            mid_end = mid_start + batchSize
                            x = x[:, mid_start:mid_end, :]
                            y = y[:, mid_start:mid_end, :]
                            y_loc = y_loc[:, mid_start:mid_end]
                elif numberOfBatches == 2:
                    x_0 = np.load(currentQuartett[0][0])  # Input
                    y_0 = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc_0 = np.load(currentQuartett[0][1])  # Ground Truth # yLoc

                    x_1 = np.load(currentQuartett[1][0])  # Input
                    y_1 = np.load(currentQuartett[1][2])  # Participant Response
                    y_loc_1 = np.load(currentQuartett[1][1])  # Ground Truth # yLoc

                    # Decrease first dimension size of batches to the size of the smallest by truncating first fixation epoch rows
                    truncated_arrays_x = truncate_to_smallest([x_0, x_1])
                    truncated_arrays_y = truncate_to_smallest([y_0, y_1])
                    truncated_arrays_y_loc = truncate_to_smallest([y_loc_0, y_loc_1])
                    # Concatenate the trauncated batches
                    x = np.concatenate((truncated_arrays_x[0], truncated_arrays_x[1]), axis=1)
                    y = np.concatenate((truncated_arrays_y[0], truncated_arrays_y[1]), axis=1)
                    y_loc = np.concatenate((truncated_arrays_y_loc[0], truncated_arrays_y_loc[1]), axis=1)

                elif numberOfBatches == 3:
                    x_0 = np.load(currentQuartett[0][0])  # Input
                    y_0 = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc_0 = np.load(currentQuartett[0][1])  # Ground Truth # yLoc

                    x_1 = np.load(currentQuartett[1][0])  # Input
                    y_1 = np.load(currentQuartett[1][2])  # Participant Response
                    y_loc_1 = np.load(currentQuartett[1][1])  # Ground Truth # yLoc

                    x_2 = np.load(currentQuartett[2][0])  # Input
                    y_2 = np.load(currentQuartett[2][2])  # Participant Response
                    y_loc_2 = np.load(currentQuartett[2][1])  # Ground Truth # yLoc

                    # Decrease first dimension size of batches to the size of the smallest by truncating first fixation epoch rows
                    truncated_arrays_x = truncate_to_smallest([x_0, x_1, x_2])
                    truncated_arrays_y = truncate_to_smallest([y_0, y_1, y_2])
                    truncated_arrays_y_loc = truncate_to_smallest([y_loc_0, y_loc_1, y_loc_2])
                    # Concatenate the trauncated batches
                    x = np.concatenate((truncated_arrays_x[0], truncated_arrays_x[1], truncated_arrays_x[2]), axis=1)
                    y = np.concatenate((truncated_arrays_y[0], truncated_arrays_y[1], truncated_arrays_y[2]), axis=1)
                    y_loc = np.concatenate(
                        (truncated_arrays_y_loc[0], truncated_arrays_y_loc[1], truncated_arrays_y_loc[2]), axis=1)

                elif numberOfBatches == 4:
                    x_0 = np.load(currentQuartett[0][0])  # Input
                    y_0 = np.load(currentQuartett[0][2])  # Participant Response
                    y_loc_0 = np.load(currentQuartett[0][1])  # Ground Truth # yLoc

                    x_1 = np.load(currentQuartett[1][0])  # Input
                    y_1 = np.load(currentQuartett[1][2])  # Participant Response
                    y_loc_1 = np.load(currentQuartett[1][1])  # Ground Truth # yLoc

                    x_2 = np.load(currentQuartett[2][0])  # Input
                    y_2 = np.load(currentQuartett[2][2])  # Participant Response
                    y_loc_2 = np.load(currentQuartett[2][1])  # Ground Truth # yLoc

                    x_3 = np.load(currentQuartett[3][0])  # Input
                    y_3 = np.load(currentQuartett[3][2])  # Participant Response
                    y_loc_3 = np.load(currentQuartett[3][1])  # Ground Truth # yLoc

                    # Decrease first dimension size of batches to the size of the smallest by truncating first fixation epoch rows
                    truncated_arrays_x = truncate_to_smallest([x_0, x_1, x_2, x_3])
                    truncated_arrays_y = truncate_to_smallest([y_0, y_1, y_2, y_3])
                    truncated_arrays_y_loc = truncate_to_smallest([y_loc_0, y_loc_1, y_loc_2, y_loc_3])
                    # Concatenate the trauncated batches
                    x = np.concatenate((truncated_arrays_x[0], truncated_arrays_x[1], truncated_arrays_x[2],
                                        truncated_arrays_x[3]), axis=1)
                    y = np.concatenate((truncated_arrays_y[0], truncated_arrays_y[1], truncated_arrays_y[2],
                                        truncated_arrays_y[3]), axis=1)
                    y_loc = np.concatenate((truncated_arrays_y_loc[0], truncated_arrays_y_loc[1], truncated_arrays_y_loc[2], truncated_arrays_y_loc[3]), axis=1)
                else:
                    raise ValueError(f"batchSize {batchSize} is not valid")
            if errorComparison == True:
                return x,y,y_loc,base_name
            else:
                if 'lowDim' in key:
                    # Increase scale of input representation by factor x to help network differentiate better between stimuli
                    for timeStep in range(0,x.shape[0]):
                        for trial in range(0,x.shape[1]):
                            x[timeStep,trial,:] = x[timeStep,trial,:] * 2

                # fix: Hardcode debug for batches having arbitrary single nan encodings
                # print("NaNs in x:", np.isnan(x).any())
                if np.isnan(x).any():
                    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                # print("Infs in x:", np.isinf(x).any())
                # print("NaNs in y:", np.isnan(y).any())
                # print("Infs in y:", np.isinf(y).any())
                # print("NaNs in y_loc:", np.isnan(y_loc).any())
                # print("Infs in y_loc:", np.isinf(y_loc).any())

                return x,y,y_loc,response
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            attempt += 1
    if attempt == max_attempts:
        print("Maximum attempts reached. The function failed to execute successfully.")

def find_epochs(array):
    for i in range(0,np.shape(array)[0]):
        row = array[i, 0, :]
        # Checking each "row" in the first dimension
        if (row > 0).sum() > 2:
            epochs = {'fix1':(None,i), 'go1':(i,None)}
            return epochs

def getEpochSteps(y):
    previous_value = None
    fixation_steps = None
    response_steps = None
    for i in range(y.shape[0]):
        if y.shape[1] != 0:
            current_value = y[i, 0, 0]
            if previous_value == np.float32(0.8) and current_value == np.float32(0.05):
                # print('Length of fixation epoch: ', i)
                fixation_steps = i
                response_steps = y.shape[0] - i

                # fixation = y[:fixation_steps,:,:]
                # response = y[fixation_steps:,:,:]

            previous_value = current_value

    # Fallback if steps were not created
    if fixation_steps == None and response_steps == None:
        fixation_steps, response_steps = 35, 35
        # continue

    return fixation_steps, response_steps

def createSplittedDatasets(hp, preprocessedData_path, month):
    # Split the data into training and test data -----------------------------------------------------------------------
    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if
               os.path.isdir(os.path.join(preprocessedData_path, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}

    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_quartett = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # # III: Exclude files with specific substrings in their names
                # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                #     continue
                # Include only files that contain any of the months in monthsConsidered
                # Merge files for participants if committment was too low in specific months
                # beRNN_05
                if hp['participant'] == 'beRNN_05' and month == 'month_8':
                    if 'month_8' not in file and 'month_7' not in file:
                        continue
                # beRNN_04
                elif hp['participant'] == 'beRNN_04' and month == 'month_3':
                    if 'month_2' not in file and 'month_3' not in file:
                        continue
                elif hp['participant'] == 'beRNN_04' and month == 'month_6':
                    if 'month_6' not in file and 'month_5' not in file:
                        continue
                elif hp['participant'] == 'beRNN_04' and month == 'month_7':
                    if 'month_6' not in file and 'month_7' not in file:
                        continue
                # beRNN_02
                elif hp['participant'] == 'beRNN_02' and month == 'month_12':
                    if 'month_12' not in file and 'month_11' not in file and 'month_10' not in file:
                        continue
                elif hp['participant'] == 'beRNN_02' and month == 'month_11':
                    if 'month_11' not in file and 'month_10' not in file:
                        continue
                elif hp['participant'] == 'beRNN_02' and month == 'month_8':
                    if 'month_8' not in file and 'month_7' not in file:
                        continue
                # beRNN_01
                elif hp['participant'] == 'beRNN_01' and month == 'month_12':
                    if 'month_12' not in file and 'month_11' not in file:
                        continue
                elif month not in file:  # Sort out months which should not be considered
                    continue
                # Add all necessary files to triplets
                base_name = file.split('Input')[0]
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                response_file = os.path.join(subdir, base_name + 'Response.npy')

                file_quartett.append((input_file, yloc_file, output_file, response_file))

        # Split the file triplets
        train_files, eval_files = split_files(hp, file_quartett)

        # Store the results in the dictionaries
        train_data[subdir] = train_files
        eval_data[subdir] = eval_files

    return train_data, eval_data

def createSplittedDatasets_beRNN_00(hp, preprocessedData_path, month):
    # Split the data into training and test data -----------------------------------------------------------------------
    participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']

    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if
               os.path.isdir(os.path.join(preprocessedData_path, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}

    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_quartett = []

        for participant in participantList:
            subdir_ = subdir.replace('beRNN_00', participant)
            # Reinitialize count - save 5 files per participant and task
            count = 0

            for file in os.listdir(subdir_):
                if file.endswith('Input.npy'):
                    # If 5 were added for this task and participant - continue
                    if count == 6:
                        break
                    # # III: Exclude files with specific substrings in their names
                    # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                    #     continue
                    # Include only files that contain any of the months in monthsConsidered
                    if month not in file:  # Sort out months which should not be considered
                        continue
                    else:
                        count += 1

                    # Add all necessary files to triplets
                    base_name = file.split('Input')[0]
                    input_file = os.path.join(subdir_, base_name + 'Input.npy')
                    yloc_file = os.path.join(subdir_, base_name + 'yLoc.npy')
                    output_file = os.path.join(subdir_, base_name + 'Output.npy')
                    response_file = os.path.join(subdir_, base_name + 'Response.npy')

                    file_quartett.append((input_file, yloc_file, output_file, response_file))


            # Split the file triplets
            train_files, eval_files = split_files(hp, file_quartett)

            # Store the results in the dictionaries
            train_data[subdir] = train_files
            eval_data[subdir] = eval_files

    return train_data, eval_data

def createSplittedDatasets_generalizationTest(hp, preprocessedData_path, month, distanceOfEvaluationData):
    month1 = month
    month2 = '_'.join(['month', str(int(month.split('_')[1])+distanceOfEvaluationData)])

    # Split the data into training and test data -----------------------------------------------------------------------
    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if
               os.path.isdir(os.path.join(preprocessedData_path, d))]


    # Initialize dictionaries to store training and evaluation data
    train_data1 = {}
    eval_data1 = {}

    # info: Get training data
    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_quartett = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # # III: Exclude files with specific substrings in their names
                # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                #     continue
                # Include only files that contain any of the months in monthsConsidered
                if month1 not in file:  # Sort out months which should not be considered
                    continue
                # Add all necessary files to triplets
                base_name = file.split('Input')[0]
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                response_file = os.path.join(subdir, base_name + 'Response.npy')

                file_quartett.append((input_file, yloc_file, output_file, response_file))

        # Split the file triplets
        train_files1, eval_files1 = split_files(hp, file_quartett)

        # Store the results in the dictionaries
        train_data1[subdir] = train_files1
        eval_data1[subdir] = eval_files1


    # Initialize dictionaries to store training and evaluation data
    train_data2 = {}
    eval_data2 = {}

    # info: Get eval data
    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_quartett = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # # III: Exclude files with specific substrings in their names
                # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                #     continue
                # Include only files that contain any of the months in monthsConsidered
                if month2 not in file:  # Sort out months which should not be considered
                    continue
                # Add all necessary files to triplets
                base_name = file.split('Input')[0]
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                response_file = os.path.join(subdir, base_name + 'Response.npy')

                file_quartett.append((input_file, yloc_file, output_file, response_file))

        # Split the file triplets
        train_files2, eval_files2 = split_files(hp, file_quartett)

        # Store the results in the dictionaries
        train_data2[subdir] = train_files2
        eval_data2[subdir] = eval_files2


    return train_data1, eval_data2

def split_files(hp, files, split_ratio=0.8):
    if 'rng' not in hp:
        hp['rng'] = np.random.default_rng()
    hp['rng'].shuffle(files)
    split_index = int(len(files) * split_ratio)
    return files[:split_index], files[split_index:]

def create_cMask(y, response, hp, mode):
    fixation_steps, response_steps = getEpochSteps(y)

    if fixation_steps == None or response_steps == None:  # hardcoded bug fix: if no fixation_steps could be found
        return None

    # Creat c_mask for current batch
    if hp['loss_type'] == 'lsq':

        c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')

        if mode == 'train':
            # fix: random bug: inconcruence between y and response dimension 1
            if response.shape[1] != y.shape[1]:
                return None

            # info: Create a c_mask that emphasizes errors by multiplying the error contribution for backProp by 5. and corrects by 1.
            errorBalancingVector = np.zeros(response.shape[1], dtype='float32')

            for j in range(response.shape[1]):
                if response[0][j] == response[1][j]:
                    errorBalancingVector[j] = 1.  # weight value for corrects
                else:
                    errorBalancingVector[j] = hp['errorBalancingValue']

            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = hp['c_mask_responseValue'] * errorBalancingVector[i]  # info: 1 or 5

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important

            c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))
            c_mask /= c_mask.mean()

        c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))

    else:
        c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
        for i in range(y.shape[1]):
            # Fixation epoch
            c_mask[:fixation_steps, i] = 1.
            # Response epoch
            c_mask[fixation_steps:, i] = hp['c_mask_responseValue']  # info: 1 or 5

        c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
        c_mask /= c_mask.mean()

    return c_mask

def adjust_ndarray_size(arr):
    if arr.size == 4:
        arr_list = arr.tolist()
        arr_list.insert(2, None)  # Insert None at position 3 (index 2)
        arr_list.append(None)     # Insert None at position 6 (end of the list)
        return np.array(arr_list, dtype=object)
    return arr

def gen_feed_dict(model, x, y, c_mask, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {model.x: x,
                     model.y: y,
                     model.c_mask: c_mask}
    elif hp['in_type'] == 'multi':
        n_time, batch_size = x.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(x[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = \
                x[:, i, :hp['rule_start']]

        feed_dict = {model.x: x,
                     model.y: y,
                     model.c_mask: c_mask}
    else:
        raise ValueError()

    return feed_dict

def _contain_model_file(model_dir):
    """Check if the directory contains model files."""
    for f in os.listdir(model_dir):
        if 'model.ckpt' in f:
            return True
    return False

def valid_model_dirs2(root_dir):
    """Get valid model directories given a root directory."""
    model_dirs = list()
    for model_dir in root_dir:
        if _contain_model_file(model_dir):
            model_dirs.append(model_dir)
    return model_dirs

def _valid_model_dirs(root_dir):
    """Get valid model directories given a root directory."""
    return [x[0] for x in os.walk(root_dir) if _contain_model_file(x[0])]

def valid_model_dirs(root_dir):
    """Get valid model directories given a root directory(s).

    Args:
        root_dir: str or list of strings
    """
    if isinstance(root_dir, six.string_types):
        return _valid_model_dirs(root_dir)
    else:
        model_dirs = list()
        for d in root_dir:
            model_dirs.extend(_valid_model_dirs(d))
        return model_dirs

def load_log(model_dir):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, 'log.json')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log

def save_log(log):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)

def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, 'hparams.json')  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, 'r') as f:
        hp = json.load(f)
    return hp

def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    if 'rng' in hp_copy:
        hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data

def find_all_models(root_dir, hp_target):
    """Find all models that satisfy hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters

    Returns:
        model_dirs: list of model directories
    """
    dirs = valid_model_dirs(root_dir)

    model_dirs = list()
    for d in dirs:
        hp = load_hp(d)
        if all(hp[key] == val for key, val in hp_target.items()):
            model_dirs.append(d)

    return model_dirs

def find_model(root_dir, hp_target, perf_min=None):
    """Find one model that satisfies hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters
        perf_min: float or None. If not None, minimum performance to be chosen

    Returns:
        d: model directory
    """
    model_dirs = find_all_models(root_dir, hp_target)
    if perf_min is not None:
        model_dirs = select_by_perf(model_dirs, perf_min)

    if not model_dirs:
        # If list empty
        print('Model not found')
        return None, None

    d = model_dirs[0]
    hp = load_hp(d)

    log = load_log(d)
    # check if performance exceeds target
    if log['perf_min'][-1] < hp['target_perf']:
        print("""Warning: this network perform {:0.2f}, not reaching target
              performance {:0.2f}.""".format(
              log['perf_min'][-1], hp['target_perf']))

    return d

def select_by_perf(model_dirs, perf_min):
    """Select a list of models by a performance threshold."""
    new_model_dirs = list()
    for model_dir in model_dirs:
        log = load_log(model_dir)
        # check if performance exceeds target
        if log['perf_min'][-1] > perf_min:
            new_model_dirs.append(model_dir)
    return new_model_dirs

def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H


