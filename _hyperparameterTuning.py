from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
# from sklearn.model_selection import ParameterGrid
import numpy as np
import itertools
import argparse
import json

import _training
import _tools

'''
- Random grid search of different hyperparameter sets 
- Operates locally, on cluster (default) or on VM (hitkip_local)
'''

########################################################################################################################
# Create HP combinations and randomly choose a selection
########################################################################################################################
def sample_param_combinations(param_grid, sample_size):
    keys = list(param_grid.keys())
    sampled_combinations = []
    for _ in range(sample_size):
        combo = {k: random.choice(param_grid[k]) for k in keys}
        sampled_combinations.append(combo)
    return sampled_combinations

def create_repeated_param_combinations(param_grid, sample_size):
    # Create the single combination of parameters
    keys, values = zip(*param_grid.items())
    single_combination = dict(zip(keys, [v[0] for v in values]))

    # Return the same combination 'sample_size' times
    repeated_combinations = [single_combination for _ in range(sample_size)]

    return repeated_combinations

def extract_hpSets_4robustnessTest(model_directory):
    # --- Settings ---
    output_base_dir = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1'
    output_folder_name = 'hpSets'  # legacy: 'test'
    numberOfModelsToTest = 10

    # Create output folder if it doesn't exist
    output_folder = os.path.join(output_base_dir, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Read first 10 directories from the text file
    txt_path = os.path.join(model_directory, 'bestModels_performance_test.txt')
    with open(txt_path, 'r', encoding='utf-8') as f:
        listOfModelsToTest_ = []
        for line in f:
            clean_line = line.strip()  # remove whitespace and newlines
            clean_line = clean_line.strip(',')  # remove trailing commas
            clean_line = clean_line.strip('"')  # remove double quotes at both ends
            clean_line = clean_line.strip("'")  # remove single quotes if any
            listOfModelsToTest_.append(clean_line)

    listOfModelsToTest = listOfModelsToTest_[1:numberOfModelsToTest + 1]

    # Loop over directories
    for idx, model_dir in enumerate(listOfModelsToTest, start=1):
        hp_file = os.path.join(model_dir, "hp.json")

        if os.path.exists(hp_file):
            # Load JSON
            with open(hp_file, 'r') as f:
                hp_data = json.load(f)

            # Prepare new filename with index
            new_hp_filename = f'hp_{idx}.json'
            new_hp_path = os.path.join(output_folder, new_hp_filename)

            # Save JSON to new location
            with open(new_hp_path, 'w') as f:
                json.dump(hp_data, f, indent=4)

            print(f"Saved: {new_hp_filename}")
        else:
            print(f"WARNING: hp.json not found in {model_dir}")

# info. Choose local or cluster training
cluster, hitkip_local = True, False

if cluster:
    # Load config file with environment variables defined for docker container
    config_path = os.getenv("CONFIG_PATH", "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Load hp defined within sbatch in sh. file
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjParams", type=str, required=True)
    args = parser.parse_args()

    # Datei öffnen und laden statt String parsen
    with open(args.adjParams, 'r') as f:
        sampled_combinations = json.load(f)

elif hitkip_local:
    batchDirectory = '/zi/home/oliver.frank/Desktop/RNN/multitask_BeRNN-main/paramCombinations_highDim_hitkip/sampled_combinations_beRNN_03.json'
    with open(batchDirectory, 'r') as f:
        sampled_combinations = json.load(f)

else:
    # attention: local ##############################################################################################
    # Get input and output dimension for network, depending on higDim and lowDim data and ruleset (standard: 'all')
    num_ring = _tools.get_num_ring('all')
    n_rule = _tools.get_num_rule('all')
    # Choose right dataset
    data = ['data_highDim_correctOnly']
    # data = ['data_highDim', 'data_highDim_correctOnly', 'data_highDim_lowCognition', 'data_lowDim', 'data_lowDim_correctOnly', 'data_lowDim_lowCognition']

    if 'highDim' in data[0]:
        n_eachring = 32
        n_outputring = n_eachring
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_outputring + 1
    else:
        n_eachring = 10
        n_outputring = 2
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_outputring + 1

    adjParams = {
        'activation': ['relu', 'softplus', 'tanh'],  # 'relu', 'elu', 'tanh', 'softplus'
        'activations_per_layer': [['relu', 'relu', 'relu'], ['softplus', 'softplus', 'softplus'], ['tanh', 'tanh', 'tanh']],
        'base_lr': [0.0005],
        'batch_size': [40, 80],
        'benchmark': [False],
        'c_mask_responseValue': [5.],
        'data': data,
        'dt': [20],
        'errorBalancingValue': [1.],
        'in_type': ['normal'],
        'l1_h': [0, 0.00001, 0.0001, 0.001],
        'l1_weight': [0, 0.00001, 0.0001, 0.001],
        'l2_h': [0, 0.00001, 0.0001, 0.001],
        'l2_weight': [0, 0.00001, 0.0001, 0.001],
        'l2_weight_init': [0],
        'learning_rate': [0.0015, 0.001, 0.0005, 0.0001],
        'learning_rate_mode': [None, None, 'exp_range', 'triangular2'],
        # Will overwrite learning_rate if it is not None - 'triangular', 'triangular2', 'exp_range'
        'loss_type': ['lsq'],  # 'Cross-entropy'
        'machine': ['hitkip'],  # 'local' 'hitkip'
        'max_lr': [0.0015],
        'monthsConsidered': [['month_3', 'month_4', 'month_5']],  # list of lists
        'monthsString': ['3-5'],
        'multiLayer': [False],
        'n_input': [n_input],  # number of input units
        'n_eachring': [n_eachring],
        'n_output': [n_output],  # number of output units
        'n_rnn': [16],
        'n_rnn_per_layer': [[32, 32, 32]],
        'optimizer': ['adam'],  # 'sgd'
        'p_weight_train': [None],
        'participant': ['beRNN_03'],  # Participant to take
        'rnn_type': ['LeakyRNN'],  # 'LeakyGRU'
        'ruleset': ['all'],  # all_benchmark
        'rule_start': [1 + num_ring * n_eachring],  # first input index for rule units
        's_mask': [None],  # 'brain_256', None
        'sequenceMode': [True],  # Decide if models are trained sequentially month-wise
        'sigma_rec': [0, 0.01],
        'sigma_x': [0, 0.01],
        'target_perf': [1.0],
        'tasksString': ['AllTask'],  # tasksTaken
        'tau': [100],  # Decides how fast previous information decays to calculate current state activity
        'threshold': [0.1],  # threshold applied to correlation matrix for adjacency matrix creation
        'trainingBatch': ['1'],
        'trainingYear_Month': ['grid_multi_beRNN_03_highDimCorrects_16'],
        'use_separate_input': [False],
        'w_mask_value': [0.1],
        # default .1 - value that will be multiplied with L2 regularization (combined with p_weight_train), <1 will decrease it
        'w_rec_init': ['randortho', 'randgauss', 'diag'],  # , 'brainStructure'
    }

    if adjParams['ruleset'][0] == 'all':
        adjParams['rule_prob_map'] = [
            dict({"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1,
                  "RP_Ctx2": 1, "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1})]
    elif adjParams['ruleset'][0] == 'all_benchmark':
        adjParams['rule_prob_map'] = [
            dict({"contextdm1": 1, "contextdm2": 1, "reactgo": 1, "reactanti": 1, "dmsgo": 1, "dmsnogo": 1,
                  "dmcgo": 1, "dmcnogo": 1, "delaygo": 1, "delayanti": 1, "delaydm1": 1, "delaydm2": 1})]
    else:
        raise ValueError('defined ruleset not recognized. Stopping process.')
    # attention: local #####################################################################################################

    # Randomly sample combinations
    sampled_combinations = sample_param_combinations(adjParams,
                                                     10000)  # info. random hp set sampling

    # # Create one combination and repeat it according to defined number
    # sampled_combinations = create_repeated_param_combinations(adjParams, 5)

# # info: Save paramCombinations locally for cluster training ####################################################################
# dir = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\paramCombinations_{adjParams["trainingYear_Month"][0]}'
# os.makedirs(dir, exist_ok=True)
# os.chdir(dir)
#
# for paramBatch in range(1,17):
#     # Randomly sample combinations
#     sampled_combinations = sample_param_combinations(adjParams, 8)
#
#     with open(f'sampled_combinations_beRNN_03_{paramBatch}.json', 'w') as f:
#         json.dump(sampled_combinations, f, indent=4)
# # info: Save paramCombinations locally for cluster training ####################################################################
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # info: Adjust paramCombinations locally for cluster training ##################################################################
# participantList = ['beRNN_03']
# for participant in participantList:
#     for paramBatch in range(1, 17):
#         with open(f'sampled_combinations_{participant}_{paramBatch}.json', 'r') as f:
#             sampled_combinations = json.load(f)
#             for i in range(len(sampled_combinations)):
#                 sampled_combinations[i]['trainingBatch'] = str(paramBatch)
#         # info: Overwrite previous file
#         with open(f'sampled_combinations_{participant}_{paramBatch}.json', 'w') as f:
#             json.dump(sampled_combinations, f, indent=4)
# # info: Adjust paramCombinations locally for cluster training ##################################################################

# Training #############################################################################################################
# Initialize list for all training times for each model
trainingTimeList = []
# Measure time for every model, respectively
trainingTimeTotal_hours = 0
# Example iteration through the grid
for modelNumber, params in enumerate(sampled_combinations):

    # attention. Add post hoc variables for cluster training ************************************************************
    # info. obligatory
    params['participant'] = config.get("participant", 'beRNN_03')
    participant = params['participant']
    params['data'] = config.get("data", 'data_highDim_correctOnly')
    data = params['data']
    params['n_rnn'] = config.get("n_rnn", 256)
    n_rnn = params['n_rnn']

    params['trainingYear_Month'] = config.get("trainingYear_Month", f'_grid_multi_{participant}_{data}_{n_rnn}')

    # info. optional
    # params["activation"] = config.get("activation", "relu")
    # params["rnn_type"] = config.get("rnn_type", "LeakyRNN")
    # params["benchmark"] = config.get("rnn_type", "LeakyRNN")
    # params['ruleset'] = config.get("ruleset", "all")
    # params["tasksString"] = config.get("tasksString", "Alltask")
    # params['rule_prob_map'] = config.get("rule_prob_map", dict({"contextdm1": 1, "contextdm2": 1, "reactgo": 1, "reactanti": 1, "dmsgo": 1, "dmsnogo": 1,
    #                                                        "dmcgo": 1, "dmcnogo": 1, "delaygo": 1, "delayanti": 1, "delaydm1": 1, "delaydm2": 1}))
    # params['rule_prob_map'] = config.get("rule_prob_map", dict({"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1,
    #                             "RP_Ctx2": 1, "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1}))
    # attention. Add post hoc variables for cluster training ************************************************************

    # Start
    start_time = time.perf_counter()
    print(f'START TRAINING MODEL: {modelNumber}')

    print(params)
    print(modelNumber)

    print('START TRAINING FOR NEW MODEL')

    load_dir = None

    # Define main path
    if params['machine'] == 'local':
        path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'
    elif params['machine'] == 'hitkip':
        path = '/zi/home/oliver.frank/Desktop'
    elif params['machine'] == 'pandora':
        path = '/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main'

    # Define data path
    preprocessedData_path = os.path.join(path, 'Data', params['participant'], params['data'])

    for month in params['monthsConsidered']:  # attention: You have to delete this if cascade training should be set OFF
        # Adjust variables manually as needed
        model_name = f'model_{month}'

        # Define model_dir for different servers
        if params['machine'] == 'local':

            if params['multiLayer'] == True:
                numberOfLayers = len(params['n_rnn_per_layer'])
                params['rnn_type'] = 'LeakyRNN'  # info: force rnn_type to always be LeakyRNN for multiLayer case
                if numberOfLayers == 2:
                    model_dir = os.path.join(
                        f"{path}\\beRNNmodels\\{params['trainingYear_Month']}\\{params['data'].split('data_')[-1]}\\{params['participant']}\\{params['trainingBatch']}\\{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_tB{params['trainingBatch']}_iter{modelNumber}_{params['rnn_type']}_{params['n_rnn_per_layer'][0]}-{params['n_rnn_per_layer'][1]}_{params['activations_per_layer'][0]}-{params['activations_per_layer'][1]}",
                        model_name)
                else:
                    model_dir = os.path.join(
                        f"{path}\\beRNNmodels\\{params['trainingYear_Month']}\\{params['data'].split('data_')[-1]}\\{params['participant']}\\{params['trainingBatch']}\\{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_tB{params['trainingBatch']}_iter{modelNumber}_{params['rnn_type']}_{params['n_rnn_per_layer'][0]}-{params['n_rnn_per_layer'][1]}-{params['n_rnn_per_layer'][2]}_{params['activations_per_layer'][0]}-{params['activations_per_layer'][1]}-{params['activations_per_layer'][2]}",
                        model_name)
            else:
                model_dir = os.path.join(
                    f"{path}\\beRNNmodels\\{params['trainingYear_Month']}\\{params['data'].split('data_')[-1]}\\{params['participant']}\\{params['trainingBatch']}\\{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_tB{params['trainingBatch']}_iter{modelNumber}_{params['rnn_type']}_{params['n_rnn']}_{params['activation']}",
                    model_name)

        elif params['machine'] == 'hitkip' or params['machine'] == 'pandora':

            if params['multiLayer'] == True:
                params['rnn_type'] = 'LeakyRNN'  # info: force rnn_type to always be LeakyRNN for multiLayer case
                numberOfLayers = len(params['n_rnn_per_layer'])
                if numberOfLayers == 2:
                    model_dir = os.path.join(
                        f"{path}/beRNNmodels/{params['trainingYear_Month']}/{params['data'].split('data_')[-1]}/{params['participant']}/{params['trainingBatch']}/{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_tB{params['trainingBatch']}_iter{modelNumber}_{params['rnn_type']}_{params['n_rnn_per_layer'][0]}-{params['n_rnn_per_layer'][1]}_{params['activations_per_layer'][0]}-{params['activations_per_layer'][1]}",
                        model_name)
                else:
                    model_dir = os.path.join(
                        f"{path}/beRNNmodels/{params['trainingYear_Month']}/{params['data'].split('data_')[-1]}/{params['participant']}/{params['trainingBatch']}/{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_tB{params['trainingBatch']}_iter{modelNumber}_{params['rnn_type']}_{params['n_rnn_per_layer'][0]}-{params['n_rnn_per_layer'][1]}-{params['n_rnn_per_layer'][2]}_{params['activations_per_layer'][0]}-{params['activations_per_layer'][1]}-{params['activations_per_layer'][2]}",
                        model_name)
            else:
                model_dir = os.path.join(
                    f"{path}/beRNNmodels/{params['trainingYear_Month']}/{params['data'].split('data_')[-1]}/{params['participant']}/{params['trainingBatch']}/{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_tB{params['trainingBatch']}_iter{modelNumber}_{params['rnn_type']}_{params['n_rnn']}_{params['activation']}",
                    model_name)

        print('MODELDIR: ', model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Split the data ---------------------------------------------------------------------------------------------------
        # List of the subdirectories
        subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if
                   os.path.isdir(os.path.join(preprocessedData_path, d))]

        # Initialize dictionaries to store training and evaluation data
        train_data = {}
        eval_data = {}

        # Function to split the files
        for subdir in subdirs:
            # Collect all file triplets in the current subdirectory
            file_quartett = []
            for file in os.listdir(subdir):
                if file.endswith('Input.npy'):
                    # III: Exclude files with specific substrings in their names
                    # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                    #     continue
                    if month not in file:  # Sort out months which should not be considered
                        continue
                    # Add all necessary files to triplets
                    base_name = file.split('Input')[0]
                    input_file = os.path.join(subdir, base_name + 'Input.npy')
                    yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                    output_file = os.path.join(subdir, base_name + 'Output.npy')
                    response_file = os.path.join(subdir, base_name + 'Response.npy')

                    file_quartett.append((input_file, yloc_file, output_file, response_file))

            # Split the file triplets
            train_files, eval_files = _tools.split_files(params, file_quartett)

            # Store the results in the dictionaries
            train_data[subdir] = train_files
            eval_data[subdir] = eval_files

        try:
            # Start Training ---------------------------------------------------------------------------------------------------
            _training.train(preprocessedData_path, model_dir=model_dir, train_data=train_data, eval_data=eval_data,
                            hp=params, load_dir=load_dir)

        except Exception as e:
            print("An exception occurred with model number:", modelNumber)
            print("Error message:", str(e))

        # info: If True previous model parameters will be taken to initialize consecutive model, creating sequential training
        if params['sequenceMode'] == True:
            load_dir = model_dir

    end_time = time.perf_counter()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    elapsed_time_hours = elapsed_time_minutes / 60

    print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_seconds:.2f} seconds")
    print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_minutes:.2f} minutes")
    print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_hours:.2f} hours")

    # Accumulate trainingTime
    trainingTimeList.append(elapsed_time_hours)
    trainingTimeTotal_hours += elapsed_time_hours

# Save training time for each model to current batch folder as a text file
file_path = os.path.join(path, 'beRNNmodels', params['trainingYear_Month'], params['data'].split('data_')[-1],
                         params['participant'], params['trainingBatch'], 'times.txt')

with open(file_path, 'w') as f:
    f.write(f"training time total (hours): {trainingTimeTotal_hours}\n")
    f.write("training time each individual model (hours):\n")
    for time in trainingTimeList:
        f.write(f"{time}\n")

print(f"Training times saved to {file_path}")

# info: optional extraction and saving of hp sets for robustness testing
# extract_hpSets_4robustnessTest(r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_gridSearch_multiTask_beRNN_03_highDimCorrects_256\highDim_correctOnly\beRNN_03\visuals\performance_test')


