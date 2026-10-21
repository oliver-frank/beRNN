import numpy as np
import pandas as pd
import _tools
import glob
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
- Preprocess the cogntive-behavioral data collected into the form that can be used to train the models
'''

# fix: Average reaction times for error to correct trials - see preprocessing_highDim_correctOnly.py

########################################################################################################################
# Predefine functions
########################################################################################################################

# Helper function to calculate the distance for circular activation ####################################################
def get_dist(original_dist):
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))

# Helper function to calculate the circular activation
def add_x_loc(x_loc, pref):
    dist = get_dist(x_loc - pref)
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2)

# Find the right questionare for the current session to extract the right meta information afterwards
def find_questionaire(target, list_allSessions, questionnare_files):
    for list in list_allSessions:
        for string in list:
            if string == target:
                for questionaire in questionnare_files:
                    if questionaire.split('-')[2] == list[0]:
                        return string, questionaire # , drugVector, sleepingQuality
    return None

# Function giving us the sleepingQuality and drugVector for the current session
def find_sleepingQuality_drugVector(opened_questionare, date):
    for index, row in opened_questionare.iterrows():
        if str(row['UTC Date and Time']).split(' ')[0] == date:
            sleepingQ = opened_questionare.at[index + 1, 'Question']
            if opened_questionare.at[index + 1, 'Key'] == 'value':
                sleepingR = opened_questionare.at[index + 1, 'Response']
            else:
                sleepingR = opened_questionare.at[index + 2, 'Response']
            drugQ = opened_questionare.loc[index + 2:index + 18, 'Key'].to_list()
            drugR = opened_questionare.loc[index + 2:index + 18, 'Response'].to_list()
            # print(sleepingQ, sleepingR, drugQ, drugR)
            return sleepingQ, sleepingR, drugQ, drugR
    return None, None, None, None  # Return None if no match is found

def safe_isnan(value):
    """
    Return True if `value` is a NaN float, otherwise False.
    """
    try:
        return np.isnan(value)  # works if value is float or array of floats
    except TypeError:
        # value is likely not a float (e.g., int, string)
        return False

########################################################################################################################
# info: DM tasks
########################################################################################################################
def preprocess_DM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1], list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl', dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[
        ['Spreadsheet', 'TimeLimit', 'Onset Time', 'Response', 'Spreadsheet: CorrectAnswer', 'Correct',
         'Component Name', 'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
         'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
         'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
         'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
         'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
         'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
         'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
         'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
         'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
         'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
         'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
         'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
         'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
         'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
         'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
         'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
         'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM', 'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti',
         'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'UTC Date and Time']]

    # Count sequences, save particpant response and ground truth for later error _analysis
    incrementList = []
    responseParticipantEntries = []
    responseGroundTruthEntries = []

    for i, name in enumerate(opened_xlsxFile_selection['Component Name']):
        if name == 'Fixation Timing':
            incrementList.append(i + 1)
            # Check if the next row exists to avoid IndexError
            if i + 1 < len(opened_xlsxFile_selection):
                responseParticipantEntry = opened_xlsxFile_selection['Response'].iloc[i + 1]
                responseParticipantEntries.append(responseParticipantEntry)
                responseGroundTruthEntry = opened_xlsxFile_selection['Spreadsheet: CorrectAnswer'].iloc[i + 1]
                responseGroundTruthEntries.append(responseGroundTruthEntry)
            else:
                # Append None or some placeholder if there is no next row
                responseParticipantEntries.append(None)
                responseGroundTruthEntries.append(None)

    import numpy as np # attention: Only added dueto pycharm bug
    concatResponseEntries = np.array((responseParticipantEntries, responseGroundTruthEntries))
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        # Prepare response array for this batch
        currentConcatResponseEntries = concatResponseEntries[:,batchOn:batchOff]
        # info: Compress time and set both epochs to 1
        numFixStepsAverage = 1
        numRespStepsAverage = 1
        totalStepsAverage = numFixStepsAverage + numRespStepsAverage

        finalSequenceList = []
        concatedValuesAndOccurrencesList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            # ----------------------------------------------------------------------------------------------------------
            # Get the rest of the trial information for later error _analysis
            png_strings = np.array([s for s in currentTrial.loc[0] if isinstance(s, str) and s.endswith('.png')])
            unique_values, occurrence_counts = np.unique(png_strings, return_counts=True)
            unique_values, occurrence_counts = unique_values[1:], occurrence_counts[1:] # exclude 000_000.png
            # Check if values and counts are below 2 and zeropad them, so that all columns have the same length
            if len(unique_values) == 1:
                unique_values, occurrence_counts = np.concatenate((unique_values, ['None']),axis=0), np.concatenate((occurrence_counts, [0]),axis=0)
            elif len(unique_values) == 0:
                currentTrialNumber = [i for i, value in enumerate(incrementList) if value == j][0]-batchOn
                currentConcatResponseEntries = np.delete(currentConcatResponseEntries, currentTrialNumber, axis=1)
                continue
            concatedValuesAndOccurrences = np.concatenate([unique_values, occurrence_counts], axis=0)
            concatedValuesAndOccurrencesList.append(concatedValuesAndOccurrences)
            # ----------------------------------------------------------------------------------------------------------
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)
        # --------------------------------------------------------------------------------------------------------------
        # Concatenate trial information for error anaylsis to response entries
        currentConcatResponseEntriesFinal = np.concatenate([currentConcatResponseEntries,np.array(concatedValuesAndOccurrencesList, dtype=object).T], axis=0)
        # --------------------------------------------------------------------------------------------------------------


        # fix: Create final df for INPUT and OUPUT #####################################################################
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # info: Create lowDim form #####################################################################################
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 86))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0, 0, 85]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0, 0, 0],
                     'timeLimit': finalTrialsList_array[0, 0, 1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8,85], axis=2) # fix: Delete all except for 33 (epoch: 1, mod1: 10, mode2: 10, taskV:12)
        correctAnswer = Output[...,4]
        Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,85], axis=2) # fix: Delete all except for 3 (epoch: 1, direction: 2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]
        responseEntries = currentConcatResponseEntriesFinal[:,sequence_on:sequence_off]

        # INPUT --------------------------------------------------------------------------------------------------------
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all epoch units to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(0)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns fix: 21-33
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # fix: Get all fields with not 000_000.png
        stimListList = [] # Will be taken for color and form
        positionListList = [] # Will be taken for angle
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                stimList = []
                positionList = []
                for k in range(1, 33): # Positions for one ring are enough as they are similar on both
                    if Input[i][j][k] != '000_000.png' and Input[0][0][k].endswith('.png'):
                        stimList.append(Input[i][j][k])
                        positionList.append(k)
                stimListList.append(stimList)
                positionListList.append(positionList)

        stimNumber = int(finalTrialsList_array[0][0][0].split('stim')[0].split('_')[-1])
        indices2remove = []
        for i, list in enumerate(stimListList):
            if safe_isnan(list[0]) == False:
                if len(list) < stimNumber or [j for j in list if '000' in j]:  # remove wrong trials
                    indices2remove.append(i)
            else:
                indices2remove.append(i)
        # fix: ADD zero padding if number of trials < 5, so that you always have the same input structure, randomize the location of the stimuli on these 5 units
        # remove lists of lists with corresponding indices
        stimListList_filtered = [sublist for i, sublist in enumerate(stimListList) if i not in indices2remove]
        positionListList_filtered = [sublist for i, sublist in enumerate(positionListList) if i not in indices2remove]

        # Define modulation dictionaries for specific columns
        mod1Dict = {'lowest': float(0.25), 'low': float(0.5), 'strong': float(0.75), 'strongest': float(1.0)}
        # Define modulation dictionaries for specific columns fix: Eventuell mit sin/cos value austauschen
        mod2Dict = {'right': float(0.25), 'down': float(0.5), 'left': float(0.75), 'up': float(1.0)}

        # Transform stim into color and form encoding
        colorListList = []
        formListList = []
        for stimList in stimListList_filtered:
            colorList = []
            formList = []
            for stim in stimList:
                colorList.append(mod1Dict[stim.split('.png')[0].split('_')[0]])
                formList.append(mod2Dict[stim.split('.png')[0].split('_')[1]])
            colorListList.append(colorList)
            formListList.append(formList)

        # fix: Embed the lists information into the Input structure
        import numpy as np
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant : polar
        for i, positionList in enumerate(positionListList_filtered):
            # print(positionList, i)
            # Exchange values is lists
            for j, position in enumerate(positionList):
                positionList[j] = pref[position-1] # prefs are defined between 0 and 31, positions between 1 and 32, therefore -1
            # Exchange list in list of lists
            positionListList_filtered[i] = positionList


        # Only take the first number of lists according to number of filtered trials in that batch len(listList)/totalStepsAverage
        numberOfTrials = len(positionListList_filtered)/totalStepsAverage
        positionListList_filtered_compressed = positionListList_filtered[0:int(numberOfTrials)]
        colorListList_compressed = colorListList[0:int(numberOfTrials)]
        formListList_compressed = formListList[0:int(numberOfTrials)]
        # fix: colorListList
        # fix: formListList
        # fix: positionListList

        # info: i is batch size here
        fullTrial_list = []
        for i, positionList in enumerate(positionListList_filtered_compressed):
            trialMod1vectors = []
            trialMod2vectors = []
            for j, position in enumerate(positionList):
                trialMod1vectors.append(np.array((np.sin(position), np.cos(position))) * colorListList_compressed[i][j])
                trialMod2vectors.append(np.array((np.sin(position), np.cos(position))) * formListList_compressed[i][j])
            # Zero pad missing stim vectors, so that every trial, every task and every spreadsheet is encoded with the same input structure
            for i in range(0, 5-len(trialMod1vectors)):
                trialMod1vectors.append(np.array([0,0]))
                trialMod2vectors.append(np.array([0,0]))

            # Ensure exact same permutation for both vectors
            permutation_indices = np.random.permutation(len(trialMod1vectors))
            # Apply it
            trialMod1vectors_randomized = [trialMod1vectors[i] for i in permutation_indices]
            trialMod2vectors_randomized = [trialMod2vectors[i] for i in permutation_indices]

            trialMod1vectors_concat = [array for vector in trialMod1vectors_randomized for array in vector]
            trialMod2vectors_concat = [array for vector in trialMod2vectors_randomized for array in vector]
            fullTrial = [1] + trialMod1vectors_concat + trialMod2vectors_concat + Input[0][0][65:77].tolist() # add epoch, two mod lists and task vector together

            fullTrial_list.append(fullTrial) # Add task vector before appending


        # Create the whole Input with new encoding
        newInput = np.zeros((totalStepsAverage, int(numberOfTrials), 33))

        for i in range(0, Input.shape[0]):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0:33] = np.array(fullTrial_list[j])

        # fix: Set epoch information unit to 1 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                    newInput[i][j][0] = float(1)
        # fix: Set epoch information unit to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(0)
        # fix: Set all modality untis to 0 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 21): #
                    newInput[i][j][k] = float(0)

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newInput.shape[0]):
            for j in range(0, newInput.shape[1]):
                for k in range(0, newInput.shape[2]):
                    newInput[i][j][k] = np.float32(newInput[i][j][k])
        # Also change dtype for entire array
        newInput = newInput.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                         xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Input'
        np.save(input_filename, newInput)
        # Save response information for later error class detection
        response_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                            xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Response'
        np.save(response_filename, responseEntries)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',opened_xlsxFile_selection['TimeLimit'][0])
        print('Response solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',opened_xlsxFile_selection['TimeLimit'][0])



        # OUTPUT -------------------------------------------------------------------------------------------------------
        # Create the whole Input with new encoding
        newOutput = np.zeros((totalStepsAverage, int(numberOfTrials), 3))

        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 3):
                    newOutput[i][j][k] = float(0.05)

        # float all epoch unit values to .8 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.8)
        # float all epoch unit values to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.05)

        # Define an output dictionary for specific response values
        outputDict = {'U': 31, 'R': 7, 'L': 23, 'D': 15} # info: Substracted in lowDim encoding by 1 to avoid indice error

        indices2remove_filtered = [i for i in indices2remove if i < 40]
        Output = np.delete(Output, indices2remove_filtered, axis=1)

        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant/degree = polar
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                if isinstance(correctAnswer[i][j], str) and correctAnswer[i][j] != 'noResponse' and correctAnswer[i][j] != 'NoResponse':
                    # Translate field into radiant
                    position = pref[outputDict[correctAnswer[i][j]]-1]
                    # Translate radiant into sin/cos vector indicating the target response direction for the network
                    newOutput[i][j][1] = np.sin(position)
                    newOutput[i][j][2] = np.cos(position)
                else:
                    newOutput[i][j][1] = np.sin(0.05)  # info: yang et al.: -1
                    newOutput[i][j][2] = np.sin(0.05)  # info: yang et al.: -1

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newOutput.shape[0]):
            for j in range(0, newOutput.shape[1]):
                for k in range(0, newOutput.shape[2]):
                    newOutput[i][j][k] = np.float32(newOutput[i][j][k])
        # Also change dtype for entire array
        newOutput = newOutput.astype('float32')


        # info: Create y_loc
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((newOutput.shape[0], newOutput.shape[1]))

        for k in range(0, numFixStepsAverage):
            for j in range (0, newOutput.shape[1]):
                y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Complete y_loc matrix
        for k in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, newOutput.shape[1]):
                if isinstance(correctAnswer[k][j], str) and correctAnswer[k][j] != 'noResponse' and correctAnswer[k][j] != 'NoResponse':
                    y_loc[k][j] = pref[outputDict[correctAnswer[k][j]]-1] # radiant form direction
                else:
                    y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                              1] + '-' + 'Output'
        np.save(output_filename, newOutput)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])

        # print('Stop')

########################################################################################################################
# info: EF tasks
########################################################################################################################
def preprocess_EF(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[
                      ['Spreadsheet', 'TimeLimit', 'Onset Time', 'Response', 'Spreadsheet: CorrectAnswer', 'Correct',
                       'Component Name', 'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
                       'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
                       'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
                       'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
                       'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
                       'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
                       'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
                       'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
                       'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
                       'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
                       'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
                       'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
                       'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
                       'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
                       'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM', 'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti',
                       'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'UTC Date and Time']]

    # Count sequences, save particpant response and ground truth for later error _analysis
    incrementList = []
    responseParticipantEntries = []
    responseGroundTruthEntries = []

    for i, name in enumerate(opened_xlsxFile_selection['Component Name']):
        if name == 'Fixation Timing':
            incrementList.append(i + 1)
            # Check if the next row exists to avoid IndexError
            if i + 1 < len(opened_xlsxFile_selection):
                responseParticipantEntry = opened_xlsxFile_selection['Response'].iloc[i + 1]
                responseParticipantEntries.append(responseParticipantEntry)
                responseGroundTruthEntry = opened_xlsxFile_selection['Spreadsheet: CorrectAnswer'].iloc[i + 1]
                responseGroundTruthEntries.append(responseGroundTruthEntry)
            else:
                # Append None or some placeholder if there is no next row
                responseParticipantEntries.append(None)
                responseGroundTruthEntries.append(None)

    import numpy as np  # attention: Only added dueto pycharm bug
    concatResponseEntries = np.array((responseParticipantEntries, responseGroundTruthEntries))
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        # Prepare response array for this batch
        currentConcatResponseEntries = concatResponseEntries[:, batchOn:batchOff]
        # info: Compress time and set both epochs to 1
        numFixStepsAverage = 1
        numRespStepsAverage = 1
        totalStepsAverage = numFixStepsAverage + numRespStepsAverage

        finalSequenceList = []
        concatedValuesAndOccurrencesList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            # ----------------------------------------------------------------------------------------------------------
            # Get the rest of the trial information for later error _analysis
            png_strings = np.array([s for s in currentTrial.loc[0] if isinstance(s, str) and s.endswith('.png')])
            unique_values, occurrence_counts = np.unique(png_strings, return_counts=True)
            unique_values, occurrence_counts = unique_values[1:], occurrence_counts[1:]  # exclude 000_000.png
            # Check if values and counts are below 2 and zeropad them, so that all columns have the same length
            if len(unique_values) == 1:
                unique_values, occurrence_counts = np.concatenate((unique_values, ['None']), axis=0), np.concatenate((occurrence_counts, [0]), axis=0)
            elif len(unique_values) == 0:
                currentTrialNumber = [i for i, value in enumerate(incrementList) if value == j][0]-batchOn
                currentConcatResponseEntries = np.delete(currentConcatResponseEntries, currentTrialNumber, axis=1)
                continue
            concatedValuesAndOccurrences = np.concatenate([unique_values, occurrence_counts], axis=0)
            concatedValuesAndOccurrencesList.append(concatedValuesAndOccurrences)
            # ----------------------------------------------------------------------------------------------------------
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)
        # --------------------------------------------------------------------------------------------------------------
        # Concatenate trial information for error anaylsis to response entries
        currentConcatResponseEntriesFinal = np.concatenate([currentConcatResponseEntries, np.array(concatedValuesAndOccurrencesList, dtype=object).T], axis=0)
        # --------------------------------------------------------------------------------------------------------------

        # Create final df for INPUT and OUPUT
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # info: Create lowDim form #####################################################################################
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 86))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0, 0, 85]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0, 0, 0],
                     'timeLimit': finalTrialsList_array[0, 0, 1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8,85], axis=2)
        correctAnswer = Output[...,4]
        Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,85], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]
        responseEntries = currentConcatResponseEntriesFinal[:, sequence_on:sequence_off]

        # INPUT --------------------------------------------------------------------------------------------------------
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all epoch units to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(0)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns fix: 21-33
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' +
                                         opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + \
                                 opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # fix: Get all fields with not 000_000.png
        stimListList = []  # Will be taken for color and form
        positionListList = []  # Will be taken for angle
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                stimList = []
                positionList = []
                for k in range(1, 33):  # Positions for one ring are enough as they are similar on both
                    if Input[i][j][k] != '000_000.png' and Input[0][0][k].endswith('.png'):
                        stimList.append(Input[i][j][k])
                        positionList.append(k)
                stimListList.append(stimList)
                positionListList.append(positionList)

        stimNumber = 5 # info: always 5
        indices2remove = []
        for i, list in enumerate(stimListList):
            if safe_isnan(list[0]) == False:
                if len(list) < stimNumber or [j for j in list if '000' in j]:  # remove wrong trials
                    indices2remove.append(i)
            else:
                indices2remove.append(i)
        # fix: ADD zero padding if number of trials < 5, so that you always have the same input structure, randomize the location of the stimuli on these 5 units
        # remove lists of lists with corresponding indices
        stimListList_filtered = [sublist for i, sublist in enumerate(stimListList) if i not in indices2remove]
        positionListList_filtered = [sublist for i, sublist in enumerate(positionListList) if i not in indices2remove]

        # Define modulation dictionaries for specific columns
        mod1Dict = {'green': float(0.5), 'red': float(1.0)}
        # mod1Dict = {'green': float(1.0), 'red': float(2.0)}
        # Define modulation dictionaries for specific columns
        mod2Dict = {'right': float(0.2), 'down': float(0.4), 'left': float(0.6), 'up': float(0.8), 'X': float(1.0)}
        # mod2Dict = {'right.png': float(0.4), 'down.png': float(0.8), 'left.png': float(1.2), 'up.png': float(1.6), 'X.png': float(2.0)}

        # Transform stim into color and form encoding
        colorListList = []
        formListList = []
        for stimList in stimListList_filtered:
            colorList = []
            formList = []
            for stim in stimList:
                colorList.append(mod1Dict[stim.split('.png')[0].split('_')[0]])
                formList.append(mod2Dict[stim.split('.png')[0].split('_')[1]])
            colorListList.append(colorList)
            formListList.append(formList)

        # fix: Embed the lists information into the Input structure
        import numpy as np
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant/degree = polar
        for i, positionList in enumerate(positionListList_filtered):
            # print(positionList, i)
            # Exchange values is lists
            for j, position in enumerate(positionList):
                positionList[j] = pref[position-1]
            # Exchange list in list of lists
            positionListList_filtered[i] = positionList

        # Only take the first number of lists according to number of filtered trials in that batch len(listList)/totalStepsAverage
        numberOfTrials = len(positionListList_filtered) / totalStepsAverage
        positionListList_filtered_compressed = positionListList_filtered[0:int(numberOfTrials)]
        colorListList_compressed = colorListList[0:int(numberOfTrials)]
        formListList_compressed = formListList[0:int(numberOfTrials)]
        # attention: for different input encoding style you can just put these compressed lists together as fullTrial

        # info: i is batch size here
        fullTrial_list = []
        for i, positionList in enumerate(positionListList_filtered_compressed):
            trialMod1vectors = []
            trialMod2vectors = []
            for j, position in enumerate(positionList):
                trialMod1vectors.append(np.array((np.sin(position), np.cos(position))) * colorListList_compressed[i][j])
                trialMod2vectors.append(np.array((np.sin(position), np.cos(position))) * formListList_compressed[i][j])
            # Zero pad missing stim vectors, so that every trial, every task and every spreadsheet is encoded with the same input structure
            for i in range(0, 5 - len(trialMod1vectors)):
                trialMod1vectors.append(np.array([0, 0]))
                trialMod2vectors.append(np.array([0, 0]))

            # Ensure exact same permutation for both vectors
            permutation_indices = np.random.permutation(len(trialMod1vectors))
            # Apply it
            trialMod1vectors_randomized = [trialMod1vectors[i] for i in permutation_indices]
            trialMod2vectors_randomized = [trialMod2vectors[i] for i in permutation_indices]

            trialMod1vectors_concat = [array for vector in trialMod1vectors_randomized for array in vector]
            trialMod2vectors_concat = [array for vector in trialMod2vectors_randomized for array in vector]
            fullTrial = [1] + trialMod1vectors_concat + trialMod2vectors_concat + Input[0][0][
                                                                                  65:77].tolist()  # add epoch, two mod lists and task vector together

            fullTrial_list.append(fullTrial)  # Add task vector before appending

        # Create the whole Input with new encoding
        newInput = np.zeros((totalStepsAverage, int(numberOfTrials), 33))

        for i in range(0, Input.shape[0]):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0:33] = np.array(fullTrial_list[j])

        # fix: Set epoch information unit to 1 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(1)
        # fix: Set epoch information unit to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(0)
        # fix: Set all modality untis to 0 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 21):  #
                    newInput[i][j][k] = float(0)

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newInput.shape[0]):
            for j in range(0, newInput.shape[1]):
                for k in range(0, newInput.shape[2]):
                    newInput[i][j][k] = np.float32(newInput[i][j][k])
        # Also change dtype for entire array
        newInput = newInput.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                         xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                             1] + '-' + 'Input'
        np.save(input_filename, newInput)
        # Save response information for later error class detection
        response_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                            xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                                1] + '-' + 'Response'
        np.save(response_filename, responseEntries)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])
        print('Response solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])


        # OUTPUT -------------------------------------------------------------------------------------------------------
        # Create the whole Input with new encoding
        newOutput = np.zeros((totalStepsAverage, int(numberOfTrials), 3))

        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 3):
                    newOutput[i][j][k] = float(0.05)

        # float all epoch unit values to .8 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.8)
        # float all epoch unit values to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.05)

        # Define an output dictionary for specific response values
        outputDict = {'U': 31, 'R': 7, 'L': 23, 'D': 15}  # info: Substracted in lowDim encoding by 1 to avoid indice error

        # indices2remove_filtered = [i for i in indices2remove if i < 40]
        # # Output = np.delete(Output, indices2remove_filtered, axis=1)

        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant/degree = polar
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                if isinstance(correctAnswer[i][j], str) and correctAnswer[i][j] != 'noResponse' and correctAnswer[i][j] != 'NoResponse':
                    # Translate field into radiant
                    position = pref[outputDict[correctAnswer[i][j]]-1]
                    # Translate radiant into sin/cos vector indicating the target response direction for the network
                    newOutput[i][j][1] = np.sin(position)
                    newOutput[i][j][2] = np.cos(position)
                else:
                    newOutput[i][j][1] = np.sin(0.05)  # info: yang et al.: -1
                    newOutput[i][j][2] = np.sin(0.05)  # info: yang et al.: -1

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newOutput.shape[0]):
            for j in range(0, newOutput.shape[1]):
                for k in range(0, newOutput.shape[2]):
                    newOutput[i][j][k] = np.float32(newOutput[i][j][k])
        # Also change dtype for entire array
        newOutput = newOutput.astype('float32')

        # info: Create y_loc
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((newOutput.shape[0], newOutput.shape[1]))

        for k in range(0, numFixStepsAverage):
            for j in range(0, newOutput.shape[1]):
                y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Complete y_loc matrix
        for k in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, newOutput.shape[1]):
                if isinstance(correctAnswer[k][j], str) and correctAnswer[k][j] != 'noResponse' and correctAnswer[k][j] != 'NoResponse':
                    y_loc[k][j] = pref[outputDict[correctAnswer[k][j]]-1]  # radiant form direction
                else:
                    y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                              1] + '-' + 'Output'
        np.save(output_filename, newOutput)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])

########################################################################################################################
# info: RP tasks
########################################################################################################################
def preprocess_RP(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Spreadsheet: CorrectAnswer1', 'Correct', 'Component Name', \
                           'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                           'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                           'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                           'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                           'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                           'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                           'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                           'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                           'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                           'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                           'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                           'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                           'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                           'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                           'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display', 'UTC Date and Time', 'Response']]

    # Count sequences, save particpant response and ground truth for later error _analysis
    incrementList = []
    responseParticipantEntries = []
    responseGroundTruthEntries = []

    for i, name in enumerate(opened_xlsxFile_selection['Component Name']):
        if name == 'Fixation Timing':
            incrementList.append(i + 1)
            # Check if the next row exists to avoid IndexError
            if i + 1 < len(opened_xlsxFile_selection):
                responseParticipantEntry = opened_xlsxFile_selection['Response'].iloc[i + 1]
                responseParticipantEntries.append(responseParticipantEntry)
                responseGroundTruthEntry = opened_xlsxFile_selection['Spreadsheet: CorrectAnswer1'].iloc[i + 1]
                responseGroundTruthEntries.append(responseGroundTruthEntry)
            else:
                # Append None or some placeholder if there is no next row
                responseParticipantEntries.append(None)
                responseGroundTruthEntries.append(None)

    concatResponseEntries = np.array((responseParticipantEntries, responseGroundTruthEntries))
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        # Prepare response array for this batch
        currentConcatResponseEntries = concatResponseEntries[:, batchOn:batchOff]
        # info: Compress time and set both epochs to 1
        numFixStepsAverage = 1
        numRespStepsAverage = 1
        totalStepsAverage = numFixStepsAverage + numRespStepsAverage

        finalSequenceList = []
        concatedValuesAndOccurrencesList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            # ----------------------------------------------------------------------------------------------------------
            # Get the rest of the trial information for later error _analysis
            png_strings = np.array([s for s in currentTrial.loc[0][9:73] if isinstance(s, str) and s.endswith('.png')])
            unique_values, occurrence_counts = np.unique(png_strings, return_counts=True)
            unique_values, occurrence_counts = unique_values[1:], occurrence_counts[1:]  # exclude 000_000.png
            # Check if values and counts are equal/below 4 and zeropad them, so that all columns have the same length
            if len(unique_values) <= 4:
                for i in range(5-len(unique_values)):
                    unique_values, occurrence_counts = np.concatenate((unique_values,['None']),axis=0), np.concatenate((occurrence_counts,[0]),axis=0)
            concatedValuesAndOccurrences = np.concatenate([unique_values, occurrence_counts], axis=0)
            concatedValuesAndOccurrencesList.append(concatedValuesAndOccurrences)
            # ----------------------------------------------------------------------------------------------------------
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)
        # --------------------------------------------------------------------------------------------------------------
        # Concatenate trial information for error anaylsis to response entries
        currentConcatResponseEntriesFinal = np.concatenate([currentConcatResponseEntries, np.array(concatedValuesAndOccurrencesList, dtype=object).T],axis=0)
        # --------------------------------------------------------------------------------------------------------------

        # fix: Create final df for INPUT and OUPUT #####################################################################
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # info: Create lowDim form #####################################################################################
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 88))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0,0,86]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0,0,0],
                     'timeLimit': finalTrialsList_array[0,0,1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8, 85,86,87], axis=2) # info: 77 statt 85 ????
        correctAnswer = Output[..., 4]
        Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,86,87], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]
        responseEntries = currentConcatResponseEntriesFinal[:, sequence_on:sequence_off]

        # INPUT --------------------------------------------------------------------------------------------------------
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all epoch units to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(0)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' +
                                         opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + \
                                 opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # fix: Get all fields with not 000_000.png
        stimListList = []  # Will be taken for color and form
        positionListList = []  # Will be taken for angle
        correctPositionListList = []
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                stimList = []
                positionList = []
                for k in range(1, 33):  # Positions for one ring are enough as they are similar on both
                    if Input[i][j][k] != '000_000.png' and Input[0][0][k].endswith('.png'):
                        stimList.append(Input[i][j][k])
                        positionList.append(k)
                    if Input[i][j][k] == correctAnswer[i][j]:
                        correctPositionList = [k] # if a stim corresponds to the objective correct answer, save the position defined by k
                    if correctAnswer[i][j] == 'noResponse':
                        correctPositionList = [0.05]

                stimListList.append(stimList)
                positionListList.append(positionList)
                correctPositionListList.append(correctPositionList)

        stimNumber = int(finalTrialsList_array[0][0][0].split('stim')[0].split('_')[-1])
        indices2remove = []
        for i, list in enumerate(stimListList):
            if safe_isnan(list[0]) == False:
                if len(list) < stimNumber or [j for j in list if '000' in j]:  # remove wrong trials
                    indices2remove.append(i)
            else:
                indices2remove.append(i)

        # fix: ADD zero padding if number of trials < 5, so that you always have the same input structure, randomize the location of the stimuli on these 5 units
        # remove lists of lists with corresponding indices
        stimListList_filtered = [sublist for i, sublist in enumerate(stimListList) if i not in indices2remove]
        positionListList_filtered = [sublist for i, sublist in enumerate(positionListList) if i not in indices2remove]

        # Define modulation dictionaries for specific columns
        mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5, 'green': 0.58, \
                    'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}

        # Define modulation dictionaries for specific columns
        mod2Dict = {'triangle': float(0.2), 'pentagon': float(0.4), 'heptagon': float(0.6),
                    'nonagon': float(0.8), 'circle': float(1.0)}

        # Transform stim into color and form encoding
        colorListList = []
        formListList = []
        for stimList in stimListList_filtered:
            colorList = []
            formList = []
            for stim in stimList:
                colorList.append(mod1Dict[stim.split('.png')[0].split('_')[0]])
                formList.append(mod2Dict[stim.split('.png')[0].split('_')[1]])
            colorListList.append(colorList)
            formListList.append(formList)

        # fix: Embed the lists information into the Input structure
        # import numpy as np
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant/degree = polar
        for i, positionList in enumerate(positionListList_filtered):
            # print(positionList, i)
            # Exchange values is lists
            for j, position in enumerate(positionList):
                positionList[j] = pref[position-1]
            # Exchange list in list of lists
            positionListList_filtered[i] = positionList

        # Only take the first number of lists according to number of filtered trials in that batch len(listList)/totalStepsAverage
        numberOfTrials = len(positionListList_filtered) / totalStepsAverage
        positionListList_filtered_compressed = positionListList_filtered[0:int(numberOfTrials)]
        colorListList_compressed = colorListList[0:int(numberOfTrials)]
        formListList_compressed = formListList[0:int(numberOfTrials)]
        # fix: colorListList
        # fix: formListList
        # fix: positionListList

        # info: i is batch size here
        fullTrial_list = []
        for i, positionList in enumerate(positionListList_filtered_compressed):
            trialMod1vectors = []
            trialMod2vectors = []
            for j, position in enumerate(positionList):
                trialMod1vectors.append(np.array((np.sin(position), np.cos(position))) * colorListList_compressed[i][j])
                trialMod2vectors.append(np.array((np.sin(position), np.cos(position))) * formListList_compressed[i][j])
            # Zero pad missing stim vectors, so that every trial, every task and every spreadsheet is encoded with the same input structure
            for i in range(0, 5 - len(trialMod1vectors)):
                trialMod1vectors.append(np.array([0, 0]))
                trialMod2vectors.append(np.array([0, 0]))

            # Ensure exact same permutation for both vectors
            permutation_indices = np.random.permutation(len(trialMod1vectors))
            # Apply it
            trialMod1vectors_randomized = [trialMod1vectors[i] for i in permutation_indices]
            trialMod2vectors_randomized = [trialMod2vectors[i] for i in permutation_indices]

            trialMod1vectors_concat = [array for vector in trialMod1vectors_randomized for array in vector]
            trialMod2vectors_concat = [array for vector in trialMod2vectors_randomized for array in vector]
            fullTrial = [1] + trialMod1vectors_concat + trialMod2vectors_concat + Input[0][0][65:77].tolist()  # add epoch, two mod lists and task vector together

            fullTrial_list.append(fullTrial)  # Add task vector before appending

        # Create the whole Input with new encoding
        newInput = np.zeros((totalStepsAverage, int(numberOfTrials), 33))

        for i in range(0, Input.shape[0]):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0:33] = np.array(fullTrial_list[j])

        # fix: Set epoch information unit to 1 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(1)
        # fix: Set epoch information unit to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(0)
        # fix: Set all modality untis to 0 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 21):  #
                    newInput[i][j][k] = float(0)

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newInput.shape[0]):
            for j in range(0, newInput.shape[1]):
                for k in range(0, newInput.shape[2]):
                    newInput[i][j][k] = np.float32(newInput[i][j][k])
        # Also change dtype for entire array
        newInput = newInput.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                         xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                             1] + '-' + 'Input'
        np.save(input_filename, newInput)
        # Save response information for later error class detection
        response_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                            xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                                1] + '-' + 'Response'
        np.save(response_filename, responseEntries)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])
        print('Response solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])

        # OUTPUT -------------------------------------------------------------------------------------------------------
        # Create the whole Input with new encoding
        newOutput = np.zeros((totalStepsAverage, int(numberOfTrials), 3))

        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 3):
                    newOutput[i][j][k] = float(0.05)
        # float all epoch unit values to .8 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.8)
        # float all epoch unit values to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.05)

        # outputDict = {'Image 2': 2, 'Image 4': 4, 'Image 6': 6, 'Image 8': 8, 'Image 10': 10, 'Image 12': 12, 'Image 14': 14,\
        #         'Image 16': 16, 'Image 18': 18, 'Image 20': 20, 'Image 22': 22, 'Image 24': 24, 'Image 26': 26, 'Image 28': 28, 'Image 30': 30, 'Image 32': 32, \
        #         'Image 1': 1, 'Image 3': 3, 'Image 5': 5, 'Image 7': 7, 'Image 9': 9, 'Image 11': 11,'Image 13': 13, 'Image 15': 15, 'Image 17': 17, 'Image 19': 19, 'Image 21': 21,
        #         'Image 23': 23, 'Image 25': 25, 'Image 27': 27, 'Image 29': 29, 'Image 31': 31}

        indices2remove_filtered = [i for i in indices2remove if i < 40]
        Output = np.delete(Output, indices2remove_filtered, axis=1)

        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant/degree = polar
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                if isinstance(correctAnswer[i][j], str) and correctAnswer[i][j] != 'screen' and correctAnswer[i][j] != 'noResponse' and \
                        correctAnswer[i][j] != 'NoResponse' and correctAnswer[i][j] != 'Fixation Cross':
                    # Translate field into radiant
                    if correctPositionListList[j][0] == 0.05:
                        position = 0.05
                    else:
                        position = pref[correctPositionListList[j][0]-1]
                    # Translate radiant into sin/cos vector indicating the target response direction for the network
                    newOutput[i][j][1] = np.sin(position)
                    newOutput[i][j][2] = np.cos(position)
                else:
                    newOutput[i][j][1] = np.sin(0.05)  # info: yang et al.: -1
                    newOutput[i][j][2] = np.sin(0.05)  # info: yang et al.: -1

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newOutput.shape[0]):
            for j in range(0, newOutput.shape[1]):
                for k in range(0, newOutput.shape[2]):
                    newOutput[i][j][k] = np.float32(newOutput[i][j][k])

        # Also change dtype for entire array
        newOutput = newOutput.astype('float32')
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((Output.shape[0], Output.shape[1]))

        for k in range(0, numFixStepsAverage):
            for j in range(0, newOutput.shape[1]):
                y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Complete y_loc matrix
        for k in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, newOutput.shape[1]):
                if isinstance(correctAnswer[k][j], str) and correctAnswer[k][j] != 'screen' and correctAnswer[k][j] != 'noResponse' and \
                        correctAnswer[k][j] != 'NoResponse' and correctAnswer[k][j] != 'Fixation Cross':
                    y_loc[k][j] = pref[correctPositionListList[j][0]-1]  # radiant form direction
                else:
                    y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                              1] + '-' + 'Output'
        np.save(output_filename, newOutput)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])

########################################################################################################################
# info: WM tasks
########################################################################################################################
def preprocess_WM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Object ID', 'Spreadsheet: CorrectAnswer', 'Correct', 'Component Name', \
                       'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                       'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                       'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                       'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                       'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                       'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                       'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display', 'UTC Date and Time', 'Response']]

    # Count sequences, save particpant response and ground truth for later error _analysis
    incrementList = []
    responseParticipantEntries = []
    responseGroundTruthEntries = []

    for i, name in enumerate(opened_xlsxFile_selection['Component Name']):
        if name == 'Fixation Timing':
            incrementList.append(i + 1)
            # Check if the next row exists to avoid IndexError
            if i + 1 < len(opened_xlsxFile_selection):
                responseParticipantEntry = opened_xlsxFile_selection['Response'].iloc[i + 1]
                responseParticipantEntries.append(responseParticipantEntry)
                responseGroundTruthEntry = opened_xlsxFile_selection['Spreadsheet: CorrectAnswer'].iloc[i + 1]
                responseGroundTruthEntries.append(responseGroundTruthEntry)
            else:
                # Append None or some placeholder if there is no next row
                responseParticipantEntries.append(None)
                responseGroundTruthEntries.append(None)

    concatResponseEntries = np.array((responseParticipantEntries, responseGroundTruthEntries))
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches): # info: here moregen weitermachen mit debugging
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        # Prepare response array for this batch
        currentConcatResponseEntries = concatResponseEntries[:, batchOn:batchOff]
        # info: Compress time and set both epochs to 1
        numFixStepsAverage = 1
        numRespStepsAverage = 1
        totalStepsAverage = numFixStepsAverage + numRespStepsAverage

        finalSequenceList = []
        concatedValuesAndOccurrencesList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            # ----------------------------------------------------------------------------------------------------------
            # Get the rest of the trial information for later error _analysis
            png_strings = np.array([s for s in currentTrial.loc[0][10:74] if isinstance(s, str) and s.endswith('.png')])
            unique_values, occurrence_counts = np.unique(png_strings, return_counts=True)
            unique_values, occurrence_counts = unique_values[1:], occurrence_counts[1:]  # exclude 000_000.png
            # Check if values and counts are below 2 and zeropad them, so that all columns have the same length
            if len(unique_values) == 1:
                unique_values, occurrence_counts = np.concatenate((unique_values, ['None']), axis=0), np.concatenate((occurrence_counts, [0]), axis=0)
            elif len(unique_values) == 0:
                currentTrialNumber = [i for i, value in enumerate(incrementList) if value == j][0]-batchOn
                currentConcatResponseEntries = np.delete(currentConcatResponseEntries, currentTrialNumber, axis=1)
                continue
            concatedValuesAndOccurrences = np.concatenate([unique_values, occurrence_counts], axis=0)
            concatedValuesAndOccurrencesList.append(concatedValuesAndOccurrences)
            # ----------------------------------------------------------------------------------------------------------
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)
        # --------------------------------------------------------------------------------------------------------------
        # Concatenate trial information for error anaylsis to response entries
        # currentConcatResponseEntriesFinal = np.concatenate([currentConcatResponseEntries, np.array(concatedValuesAndOccurrencesList, dtype=object).T.reshape((40, 1))],axis=0)
        adjusted_concatedValuesAndOccurrencesList = [tools.adjust_ndarray_size(arr) for arr in concatedValuesAndOccurrencesList]
        currentConcatResponseEntriesFinal = np.concatenate([currentConcatResponseEntries, np.array(adjusted_concatedValuesAndOccurrencesList, dtype=object).reshape((len(adjusted_concatedValuesAndOccurrencesList),6)).T],axis=0)
        # --------------------------------------------------------------------------------------------------------------

        # fix: Create final df for INPUT and OUPUT #####################################################################
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # Create Yang form
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 89))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0, 0, 87]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        # print('# DEBUGGING ###########################################################################################')
        # print('>>>>>>>>>>>>>>>>>>>>>>>>     ', taskString, final_questionaire, date)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>     ', 'sleepingQualityValue:', sleepingQualityValue, 'drugVector:',drugVectorValue)
        # print('# DEBUGGING ###########################################################################################')
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0, 0, 0],
                     'timeLimit': finalTrialsList_array[0, 0, 1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0,1,2,3,4,5,6,7,8,86,87,88], axis=2)
        correctAnswer = Output[...,5]
        Output = np.delete(Output, np.s_[0,1,2,5,6,7,8,87,88], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]
        responseEntries = currentConcatResponseEntriesFinal[:, sequence_on:sequence_off]

        # INPUT ############################################################################################################
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all epoch units to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(0)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        outputDict_WM_Ctx = {'Match': 8, 'Mismatch': 24, 'noResponse': -1}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4 or len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 5 and \
                        opened_xlsxFile_selection['Spreadsheet'][0].split('_')[2] == '3stim': # That is actually for WM, but on 3stim spreadsheets it will result in 5 entities after splitting
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # fix: Get all fields with not 000_000.png
        stimListList = []  # Will be taken for color and form
        positionListList = []  # Will be taken for angle
        correctPositionListList = []
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                stimList = []
                positionList = []
                correctPositionList = []
                for k in range(1, 33):  # Positions for one ring are enough as they are similar on both
                    if Input[i][j][k] != '000_000.png' and Input[0][0][k].endswith('.png'):
                        stimList.append(Input[i][j][k])
                        positionList.append(k)
                    if Input[i][j][k] == correctAnswer[i][j]:
                        correctPositionList = [k] # if a stim corresponds to the objective correct answer, save the position defined by k
                    if 'Ctx' in opened_xlsxFile_selection['Spreadsheet'][1]:
                        correctPositionList = [outputDict_WM_Ctx[correctAnswer[i][j]]]

                stimListList.append(stimList)
                positionListList.append(positionList)
                if len(correctPositionList) == 0: # e.g. noResponse for WM and WM_Anti tasks
                    correctPositionList = [0.05]
                correctPositionListList.append(correctPositionList)

        if '3stim' in finalTrialsList_array[0][0][0]: stimNumber = 3
        else: stimNumber = 2

        indices2remove = []
        for i, list in enumerate(stimListList):
            if safe_isnan(list[0]) == False:
                if len(list) < stimNumber or [j for j in list if '000' in j]:  # remove wrong trials
                    indices2remove.append(i)
            else:
                indices2remove.append(i)
        # fix: ADD zero padding if number of trials < 5, so that you always have the same input structure, randomize the location of the stimuli on these 5 units
        # remove lists of lists with corresponding indices
        stimListList_filtered = [sublist for i, sublist in enumerate(stimListList) if i not in indices2remove]
        positionListList_filtered = [sublist for i, sublist in enumerate(positionListList) if i not in indices2remove]

        # Define modulation dictionaries for specific columns
        # mod1Dict = {'red': 0.16, 'rust': 0.34, 'orange': 0.5, 'amber': 0.66, 'yellow': 0.84, 'lime': 1.0, 'green': 1.16, \
        #             'moss': 1.34, 'blue': 1.5, 'violet': 1.66, 'magenta': 1.84, 'purple': 2.0}
        mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5, 'green': 0.58, \
                    'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}
        # Define modulation dictionaries for specific columns
        mod2Dict = {'triangle': float(0.2), 'pentagon': float(0.4), 'heptagon': float(0.6), 'nonagon': float(0.8), 'circle': float(1.0)}
        # mod2Dict = {'triangle.png': float(0.4), 'pentagon.png': float(0.8), 'heptagon.png': float(1.2), 'nonagon.png': float(1.6), 'circle.png': float(2.0)}

        # Transform stim into color and form encoding
        colorListList = []
        formListList = []
        for stimList in stimListList_filtered:
            colorList = []
            formList = []
            for stim in stimList:
                colorList.append(mod1Dict[stim.split('.png')[0].split('_')[0]])
                formList.append(mod2Dict[stim.split('.png')[0].split('_')[1]])
            colorListList.append(colorList)
            formListList.append(formList)

        # fix: Embed the lists information into the Input structure
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)  # fix: pref[positionList[0]] = radiant/degree = polar
        for i, positionList in enumerate(positionListList_filtered):
            # print(positionList, i)
            # Exchange values is lists
            for j, position in enumerate(positionList):
                positionList[j] = pref[position-1]
            # Exchange list in list of lists
            positionListList_filtered[i] = positionList

        # Only take the first number of lists according to number of filtered trials in that batch len(listList)/totalStepsAverage
        numberOfTrials = len(positionListList_filtered) / totalStepsAverage
        positionListList_filtered_compressed = positionListList_filtered[0:int(numberOfTrials)]
        colorListList_compressed = colorListList[0:int(numberOfTrials)]
        formListList_compressed = formListList[0:int(numberOfTrials)]
        # fix: colorListList
        # fix: formListList
        # fix: positionListList

        # info: i is batch size here
        fullTrial_list = []
        for i, positionList in enumerate(positionListList_filtered_compressed):
            trialMod1vectors = []
            trialMod2vectors = []
            for j, position in enumerate(positionList):
                trialMod1vectors.append(
                    np.array((np.sin(position), np.cos(position))) * colorListList_compressed[i][j])
                trialMod2vectors.append(
                    np.array((np.sin(position), np.cos(position))) * formListList_compressed[i][j])
            # Zero pad missing stim vectors, so that every trial, every task and every spreadsheet is encoded with the same input structure
            for i in range(0, 5 - len(trialMod1vectors)):
                trialMod1vectors.append(np.array([0, 0]))
                trialMod2vectors.append(np.array([0, 0]))

            # Ensure exact same permutation for both vectors
            permutation_indices = np.random.permutation(len(trialMod1vectors))
            # Apply it
            trialMod1vectors_randomized = [trialMod1vectors[i] for i in permutation_indices]
            trialMod2vectors_randomized = [trialMod2vectors[i] for i in permutation_indices]

            trialMod1vectors_concat = [array for vector in trialMod1vectors_randomized for array in vector]
            trialMod2vectors_concat = [array for vector in trialMod2vectors_randomized for array in vector]
            fullTrial = [1] + trialMod1vectors_concat + trialMod2vectors_concat + Input[0][0][65:77].tolist()  # add epoch, two mod lists and task vector together
            fullTrial_list.append(fullTrial)  # Add task vector before appending

        # Create the whole Input with new encoding
        newInput = np.zeros((totalStepsAverage, int(numberOfTrials), 33))

        for i in range(0, Input.shape[0]):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0:33] = np.array(fullTrial_list[j])

        # fix: Set epoch information unit to 1 during fixation (already done)
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(1)
        # fix: Set epoch information unit to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newInput[i][j][0] = float(0)
        # fix: Set all modality untis to 0 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 21):  #
                    newInput[i][j][k] = float(0)

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newInput.shape[0]):
            for j in range(0, newInput.shape[1]):
                for k in range(0, newInput.shape[2]):
                    newInput[i][j][k] = np.float32(newInput[i][j][k])

        # Also change dtype for entire array
        newInput = newInput.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                         xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                             1] + '-' + 'Input'
        np.save(input_filename, newInput)
        # Save response information for later error class detection
        response_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                            xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                                1] + '-' + 'Response'
        np.save(response_filename, responseEntries)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])
        print('Response solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])

        # OUTPUT ###########################################################################################################
        # Create the whole Input with new encoding
        newOutput = np.zeros((totalStepsAverage, int(numberOfTrials), 3))

        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                for k in range(1, 3):
                    newOutput[i][j][k] = float(0.05)

        # float all epoch unit values to .8 during fixation
        for i in range(0, numFixStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.8)
        # float all epoch unit values to 0 during response
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, int(numberOfTrials)):
                newOutput[i][j][0] = float(0.05)

        # # Assign field units to their according participant response value after fixation period
        # outputDict_WM = {'Image 1': 1, 'Image 2': 2, 'Image 3': 3, 'Image 4': 4, 'Image 5': 5, 'Image 6': 6, 'Image 7': 7,\
        #     'Image 8': 8, 'Image 9': 9, 'Image 10': 10, 'Image 11': 11, 'Image 12': 12, 'Image 13': 13, 'Image 14': 14,\
        #     'Image 15': 15, 'Image 16': 16, 'Image 17': 17, 'Image 18': 18, 'Image 19': 19, 'Image 20': 20, 'Image 21': 21,\
        #     'Image 22': 22, 'Image 23': 23, 'Image 24': 24, 'Image 25': 25, 'Image 26': 26, 'Image 27': 27, 'Image 28': 28,\
        #     'Image 29': 29, 'Image 30': 30, 'Image 31': 31, 'Image 32': 32}
        #
        # outputDict_WM_Ctx = {'object-1591': 8, 'object-1593': 8, 'object-1595': 8, 'object-1597': 8, 'object-2365': 8, 'object-2313': 8, 'object-2391': 8, 'object-2339': 8,
        #                      'object-1592': 24, 'object-1594': 24, 'object-1596': 24, 'object-1598': 24, 'object-2366': 24, 'object-2314': 24, 'object-2392': 24, 'object-2340': 24}

        indices2remove_filtered = [i for i in indices2remove if i < 40]
        Output = np.delete(Output, indices2remove_filtered, axis=1)

        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                if isinstance(correctAnswer[i][j], str):
                    # # Get the right dictionary
                    # if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4 or opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1] == 'Anti' or \
                    #         len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 5 and opened_xlsxFile_selection['Spreadsheet'][0].split('_')[2] == '3stim':
                    #     outputDict = outputDict_WM
                    #     chosenColumn = 0
                    # else:
                    #     outputDict = outputDict_WM_Ctx
                    #     chosenColumn = 1

                    if correctAnswer[i][j] != 'screen' and correctAnswer[i][j] != 'noResponse' and correctAnswer[i][j] != 'NoResponse'\
                            and correctAnswer[i][j] != 'Fixation Cross' and correctAnswer[i][j] != 'Response':
                        # Translate field into radiant
                        position = pref[correctPositionListList[j][0]-1]
                        # Translate radiant into sin/cos vector indicating the target response direction for the network
                        newOutput[i][j][1] = np.sin(position)
                        newOutput[i][j][2] = np.cos(position)
                    else:
                        for k in range(1, 3):  # if noResponse was given
                            newOutput[i][j][1] = np.sin(0.05)  # info: yang et al.: -1
                            newOutput[i][j][2] = np.sin(0.05)  # info: yang et al.: -1
                else:
                    for k in range(1, 3):  # if noResponse was given
                        newOutput[i][j][1] = np.sin(0.05)  # info: yang et al.: -1
                        newOutput[i][j][2] = np.sin(0.05)  # info: yang et al.: -1

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, newOutput.shape[0]):
            for j in range(0, newOutput.shape[1]):
                for k in range(0, newOutput.shape[2]):
                    newOutput[i][j][k] = np.float32(newOutput[i][j][k])
        # Also change dtype for entire array
        newOutput = newOutput.astype('float32')

        # Drop unnecessary columns
        # Output = np.delete(Output, [0,1,35], axis=2)
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((Output.shape[0], Output.shape[1]))

        for k in range(0, numFixStepsAverage):
            for j in range(0, newOutput.shape[1]):
                y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Complete y_loc matrix
        for k in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, newOutput.shape[1]):
                if isinstance(correctAnswer[k][j], str) and correctAnswer[k][j] != 'noResponse' and correctAnswer[k][j] != 'NoResponse' and correctAnswer[k][j] != 'screen':
                    y_loc[k][j] = pref[correctPositionListList[j][0]-1]  # radiant form direction
                else:
                    y_loc[k][j] = np.float(0.05)  # info: yang et al.: -1

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[
                              1] + '-' + 'Output'
        np.save(output_filename, newOutput)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(
            batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ',
              opened_xlsxFile_selection['TimeLimit'][0])

########################################################################################################################
# info: Execute Preprocessing
########################################################################################################################
def check_permissions(file_path):
    permissions = {
        'read': os.access(file_path, os.R_OK),
        'write': os.access(file_path, os.W_OK),
        'execute': os.access(file_path, os.X_OK)
    }
    return permissions

# Preallocation of variables
dataFolder = "Data"
subfolders = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
preprocessing_folder = 'data_lowDim_correctOnly_timeCompressed'
participants = ['beRNN_03']
# participants = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'BeRNN_04', 'beRNN_05']
months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'] # info: debugging '13'

for participant in participants:
    # attention: change to right path
    path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'  # local
    # path = 'W:\\group_csp\\analyses\\oliver.frank'  # Fl storage
    # path = '/data' # hitkip cluster
    # path = '/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main' # pandora server

    # Create current main_path
    main_path = os.path.join(path, dataFolder, participant, preprocessing_folder)

    # Create Folder Structure
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    for folder in subfolders:
        subpath = os.path.join(main_path, folder)
        if not os.path.exists(subpath):
            os.makedirs(subpath)

    # Processing path allocation
    processing_path = os.path.join(path,dataFolder, participant)

    # list of task names in each session; very first name is the one of the associated questionare
    list_d4fh = ['d4fh', 'b4ya', 'bert', '7py6', 'bx2n', '2p6f', '9ivx', 'fhgh', 'k1jg', '6x4x', 'aiut', 'ar8p', '627g']
    list_2353 = ['2353', '6vnh', '113d', 'pbq6', 'q4ei', 'u7od', '9qcv', '4lrw', 'u31n', 'jr36', 'hia1', 'odic', 'qkw4']
    list_sdov = ['sdov', 'ohf2', '7y8y', 'p3op', '715w', 'hbck', '8dc4', 'pfww', 'kid1', 'z84v', 'qfff', 'o9l4', 'fiv9']
    list_h3ph = ['h3ph', 'kvmz', 'gpb4', 'x1mk', 'qxae', '4cnx', '9wpb', 'ujcn', 'o3t1', 'qf8s', 't271', 'lypz', '7l94']
    list_jwd5 = ['jwd5', 'zner', 'fvox', 'qqvk', 'qilu', 'xqal', 'q9wz', 'p7mk', '2kln', 'ifgy', '6at8', 'zolj', '9utw']
    list_allSessions = [list_d4fh, list_2353, list_sdov, list_h3ph, list_jwd5]
    # Go through all .xlsx files in the defined months for one participant and save them in PreprocessedData folder's subfolders
    for month in months:
        processing_path_month = os.path.join(processing_path, month)
        if os.path.exists(processing_path_month):
            pattern = os.path.join(processing_path_month, '*.xlsx')
            xlsx_files = glob.glob(pattern)

            # remove session_completion and monthly_feedback files
            for file in xlsx_files:
                try:
                    file_path = os.path.join(processing_path_month, file)
                    opened_xlsxFile = pd.read_excel(file_path, engine='openpyxl')
                    if opened_xlsxFile['Task Name'].iloc[0] == '000_session_completion' or opened_xlsxFile['Task Name'].iloc[0] == '000_monthly_feedback':
                        xlsx_files.remove(file)
                except Exception as e:
                    xlsx_files.remove(file)
                    print(f"An error occurred with file {file}: {e}. File was removed")

            task_files = [os.path.basename(file) for file in xlsx_files if 'questionnaire' not in os.path.basename(file).lower()] # remove questionaire files
            questionnare_files = [os.path.basename(file) for file in xlsx_files if 'task' not in os.path.basename(file).lower()]
            # Iterate through all .xlsx files in current month folder
            for xlsxFile in task_files:
                file_path = os.path.join(processing_path_month, xlsxFile)
                print(' ')
                print(' NEW FILE ---------------------------------------------------------------------------------------')
                print(' ')
                permissions = check_permissions(file_path)

                print(f"Read: {'Yes' if permissions['read'] else 'No'}")
                print(f"Write: {'Yes' if permissions['write'] else 'No'}")
                print(f"Execute: {'Yes' if permissions['execute'] else 'No'}")
                if permissions['read']:
                    if os.path.isfile(file_path):
                        try:
                            opened_xlsxFile = pd.read_excel(file_path, engine='openpyxl')
                            file = file_path.split('\\')[-1]
                            print(f"Processing file: {file}")
                            print(' ')
                            sequence_on, sequence_off, batchLength = 0, 40, 40
                            try:
                                # Preprocess the xlsxFile according to its task type and directly save it to the right directory
                                if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'DM':
                                    preprocess_DM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                                if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'EF':
                                    preprocess_EF(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                                if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'RP':
                                    preprocess_RP(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                                if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'WM':
                                    preprocess_WM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                            except Exception as e:
                                print(f"An error occurred with file {xlsxFile}: {e}")
                        except Exception as e:
                            print(f"An error occurred with file {xlsxFile}: {e}")
                    else:
                        print(f"File not found: {file_path}")
                else:
                    print(f"Read permission denied for file: {file_path}")
        else:
            print(f"Month directory not found: {processing_path_month}")


