from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mpl_toolkits.axes_grid1 import Divider, Size
from collections import defaultdict

# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import mlflow
# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++

from mpl_toolkits.axes_grid1 import make_axes_locatable
from _network import Model, get_perf
from _tools import split_files
import _tools

import numpy as np
import pandas as pd
import json
import glob
import pickle
import itertools

import os
import matplotlib.pyplot as plt
plt.ioff() # prevents windows to pop up when figs and plots are created
import seaborn as sns
# from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde

'''
- Creates errorDistribution for each task individually (fine-grained is outdated)
- Creates contingencyTable for each task individually
- Creates task complexity x relative count plots
- Create csv saving all counted error categories for each subject
- Creates top5 error heatmaps for each task individually
'''

errorDistributions, contingencyTables, taskComplexity, saveCSVerrors, top5errors = True, False, False, False, False


if errorDistributions:
    #################################################################################################################
    # head: create error distributions for each task individually
    #################################################################################################################
    def convert_numpy(obj):
        """Konvertiert NumPy-Objekte in Standard-Python-Objekte."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Konvertiert Array zu Liste
        if isinstance(
            obj, (np.int64, np.int32, np.int16, np.int8)
        ):  # Behebt auch Integer-Fehler
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def plot_errorDistribution_relative(errors_dict, directory, participant, task, granularity, titleAdd, ax=None):
        # Prepare data for plotting
        categories = list(errors_dict.keys())
        total_occurrences = sum(len(values) for values in errors_dict.values())

        # Calculate relative occurrences (percentages)
        occurrences = [(len(values) / total_occurrences) * 100 for values in errors_dict.values()]

        # Reverse the order for a nicer top-down view
        categories = categories[::-1]
        occurrences = occurrences[::-1]

        palette = sns.color_palette("coolwarm", len(categories))
        colors = palette[::-1]

        # Enforce exact figure size
        is_standalone = ax is None
        if is_standalone:
            fig, ax = plt.subplots(figsize=(2, 2))

        if granularity == 'rough' and 'WM_Ctx' in task:
            bar_thickness = 8.0
        elif granularity == 'rough' and 'WM' in task:
            bar_thickness = 4.0
        elif granularity == 'rough':
            bar_thickness = 1.0
        elif granularity == 'fine':
            bar_thickness = 20.0

        # Generate explicit numeric coordinates for the center of each bar
        y_positions = np.arange(len(categories))

        # Draw horizontal bars manually (added alpha for line contrast)
        for i, (cat, val, color) in enumerate(zip(categories, occurrences, colors)):
            ax.barh(i, val, color=color, height=bar_thickness, alpha=0.5)

        # --- NEW: Smooth, Comparable Density Trend Curve ---
        # Recreate the underlying vertical data frequency to compute the true density
        y_data_points = []
        for i, cat in enumerate(categories):
            y_data_points.extend([i] * len(errors_dict[cat]))

        if len(y_data_points) > 1:
            # 'bw_method' controls smoothing.
            # Increase this number (e.g., 0.6 to 1.0) for a broader, smoother general trend.
            kde = gaussian_kde(y_data_points, bw_method=0.7)

            # Evaluate density smoothly across the vertical category axis
            y_smooth = np.linspace(0, len(categories) - 1, 300)
            density_values = kde(y_smooth)

            # IMPORTANT FOR COMPARISON: Scale the raw density values to match your percentage axis.
            # This keeps the curve height aligned and proportional to the max bar magnitude.
            scaling_factor = max(occurrences) / max(density_values)
            x_smooth = density_values * scaling_factor

            # Plot directly on the primary percentage axis
            ax.plot(x_smooth, y_smooth, color='black', linewidth=1.5, alpha=0.8)

        # Completely remove category text labels and ticks
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Gridlines and X-axis limits
        ax.set_xticks([0, 25, 50])
        ax.tick_params(axis='x', labelsize=15)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.set_xlim([0, 50])

        if is_standalone:
            save_path = os.path.join(directory.split('\\data')[0], 'ErrorGraphics_monthWise', task,
                                     f'errorDistribution_{participant}_{task}_{titleAdd}_relative.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

        # Return top categories for rough granularity
        if granularity == 'rough':
            category_counts = {cat: len(values) for cat, values in errors_dict.items()}
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            top_four_categories = [cat for cat, count in sorted_categories[:4]]
            return top_four_categories

    def sort_rows_descending(array): # Higher value on 4th
        for col in range(array.shape[1]):
            if array[4, col] < array[5, col]:  # If value in 5th row is higher than in 4th row
                # Swap values in 4th and 5th rows
                array[4, col], array[5, col] = array[5, col], array[4, col]
                # Swap corresponding values in 2nd and 3rd rows
                array[2, col], array[3, col] = array[3, col], array[2, col]
        return array

    def sort_rows_ascending(array): # Higher value on 5th
        for col in range(array.shape[1]):
            if array[4, col] > array[5, col]:  # If value in 5th row is higher than in 4th row
                # Swap values in 4th and 5th rows
                array[4, col], array[5, col] = array[5, col], array[4, col]
                # Swap corresponding values in 2nd and 3rd rows
                array[2, col], array[3, col] = array[3, col], array[2, col]
        return array

    def sort_rows_correctOn2(array): # Correct stim must be on 2nd row
        for col in range(array.shape[1]):
            if array[1, col] != array[2, col]:
                # Swap values in 2th and 3th rows
                array[2, col], array[3, col] = array[3, col], array[2, col]
                # Swap corresponding values in 4th and 5th rows is not necessary
        return array

    def get_fine_grained_error(sortedResponse, errors_dict_fineGrained, task):
        for i in range(sortedResponse.shape[1]):
            if task == 'DM' or task == 'DM_Anti':
                # Distraction
                errorComponent_1 = 'distract' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize()
                errorComponent_2 = sortedResponse[3, i].split('_')[0].capitalize()
                # Missed correct stimulus
                errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
                errorComponent_4 = sortedResponse[2, i].split('_')[0].capitalize()
                # Incorrect answer
                errorComponent_5 = 'response' + sortedResponse[0, i].capitalize()
                # Concatenate error components
                currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}_{errorComponent_5}'
                errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

            elif task == 'EF' or task == 'EF_Anti':
                # Distraction
                errorComponent_1 = 'distract' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize()
                errorComponent_2 = sortedResponse[3, i].split('_')[0].capitalize()
                # Missed correct stimulus
                errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
                errorComponent_4 = sortedResponse[2, i].split('_')[0].capitalize()
                # Incorrect answer
                errorComponent_5 = 'response' + sortedResponse[0, i]
                # Concatenate error components
                currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}_{errorComponent_5}'
                errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

            elif task == 'RP' or task == 'RP_Anti' or task == 'RP_Ctx1' or task == 'RP_Ctx2':
                # Wrongly chosen distraction
                if sortedResponse[0, i] == '000_000.png' or sortedResponse[0, i] == 'BEGIN' or sortedResponse[0, i] == 'nan' or sortedResponse[0, i] == None: # errors where the wrong stim was hitten
                    continue
                elif sortedResponse[0, i] == 'NoResponse':
                    errorComponent_1 = 'distract' + 'NoResponse'
                    errorComponent_2 = sortedResponse[1, i].split('_')[0].capitalize()
                else:
                    errorComponent_1 = 'distract' + sortedResponse[0, i].split('_')[1].split('.')[0].capitalize()
                    errorComponent_2 = sortedResponse[0, i].split('_')[0].capitalize()
                # Missed correct stimulus
                errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
                if sortedResponse[0, i].split('_')[0] != '000' and sortedResponse[0, i].split('_')[0] != 'NoResponse':
                    distractClass = next((cls for cls, colors in colorDict.items() if sortedResponse[0, i].split('_')[0] in colors), None)
                    correctClass = next((cls for cls, colors in colorDict.items() if sortedResponse[2, i].split('_')[0] in colors), None)
                    if distractClass == correctClass:
                        errorComponent_4 = 'Similiar'
                    else:
                        errorComponent_4 = 'NonSimiliar'
                else:
                    errorComponent_4 = 'NonSimiliar'

                # Concatenate error components
                currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'
                errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

            elif task == 'WM' or task == 'WM_Anti':
                # Wrongly chosen distraction OR distraction that led to noResponse/empty field
                if sortedResponse[0, i] == '000_000.png':  # errors where the empty field was hitten
                    continue
                elif sortedResponse[0, i] == 'noResponse' or sortedResponse[0, i] == 'NoResponse':
                    errorComponent_1 = 'noResponse' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize() # Form
                    if sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                        errorComponent_2 = 'ClassYellow'
                    elif sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue']:
                        errorComponent_2 = 'ClassBlue'
                    elif sortedResponse[3, i].split('_')[0] in colorDict['ClassRed']:
                        errorComponent_2 = 'ClassRed'
                    elif sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                        errorComponent_2 = 'ClassGreen'
                else:
                    errorComponent_1 = 'distract' + sortedResponse[0, i].split('_')[1].split('.')[0].capitalize() # Form
                    if sortedResponse[0, i].split('_')[0] in colorDict['ClassYellow']:
                        errorComponent_2 = 'ClassYellow'
                    elif sortedResponse[0, i].split('_')[0] in colorDict['ClassBlue']:
                        errorComponent_2 = 'ClassBlue'
                    elif sortedResponse[0, i].split('_')[0] in colorDict['ClassRed']:
                        errorComponent_2 = 'ClassRed'
                    elif sortedResponse[0, i].split('_')[0] in colorDict['ClassGreen']:
                        errorComponent_2 = 'ClassGreen'

                # Missed correct stimulus
                correctColorClass = next((cls for cls, colors in colorDict.items() if sortedResponse[2, i].split('_')[0] in colors), None)
                distractColorClass = next((cls for cls, colors in colorDict.items() if sortedResponse[3, i].split('_')[0] in colors), None)
                if correctColorClass == distractColorClass:
                    errorComponent_3 = 'simColor'
                else:
                    errorComponent_3 = 'diffColor'

                correctFormClass = next((cls for cls, forms in formDict.items() if sortedResponse[2, i].split('_')[1].split('.')[0] in forms), None)
                distractFormClass = next((cls for cls, forms in formDict.items() if sortedResponse[3, i].split('_')[1].split('.')[0] in forms), None)
                if correctFormClass == distractFormClass:
                    errorComponent_4 = 'simForm'
                else:
                    errorComponent_4 = 'diffForm'

                # errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials_')[1] # info: Don't count 3stim extra as there would be too many error classes

                # Concatenate error components
                currentChosenList = f'{errorComponent_1}{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'
                errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

            elif task == 'WM_Ctx1' or task == 'WM_Ctx2':
                if sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassCircle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'formClassCombi_' + 'CircleCircle'
                elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'formClassCombi_' + 'PolygonPolygon'
                elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                    errorComponent_1 = 'formClassCombi_' + 'TriangleTriangle'
                elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassCircle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] or\
                        sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'formClassCombi_' + 'CirclePolygon'
                elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassCircle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] or\
                        sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'formClassCombi_' + 'CircleTriangle'
                elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] or\
                        sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'formClassCombi_' + 'PolygonTriangle'
                else:
                    continue

                if sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                    errorComponent_2 = 'colorClassCombi_' + 'YellowYellow'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                    errorComponent_2 = 'colorClassCombi_' + 'GreenGreen'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue']:
                    errorComponent_2 = 'colorClassCombi_' + 'BlueBlue'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed']:
                    errorComponent_2 = 'colorClassCombi_' + 'RedRed'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen'] or\
                        sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                    errorComponent_2 = 'colorClassCombi_' + 'YellowGreen'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue'] or\
                        sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                    errorComponent_2 = 'colorClassCombi_' + 'YellowBlue'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed'] or\
                        sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                    errorComponent_2 = 'colorClassCombi_' + 'YellowRed'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue'] or\
                        sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                    errorComponent_2 = 'colorClassCombi_' + 'GreenBlue'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed'] or\
                        sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                    errorComponent_2 = 'colorClassCombi_' + 'GreenRed'
                elif sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed'] or\
                        sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue']:
                    errorComponent_2 = 'colorClassCombi_' + 'BlueRed'
                else:
                    continue

                # Missed correct stimulus
                correctColorClass = next(
                    (cls for cls, colors in colorDict.items() if sortedResponse[2, i].split('_')[0] in colors), None)
                distractColorClass = next(
                    (cls for cls, colors in colorDict.items() if sortedResponse[3, i].split('_')[0] in colors), None)
                if correctColorClass == distractColorClass:
                    errorComponent_3 = 'simColor'
                else:
                    errorComponent_3 = 'diffColor'

                correctFormClass = next((cls for cls, forms in formDict.items() if
                                         sortedResponse[2, i].split('_')[1].split('.')[0] in forms), None)
                distractFormClass = next((cls for cls, forms in formDict.items() if
                                          sortedResponse[3, i].split('_')[1].split('.')[0] in forms), None)
                if correctFormClass == distractFormClass:
                    errorComponent_4 = 'simForm'
                else:
                    errorComponent_4 = 'diffForm'

                # errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials_')[1] # info: Don't count 3stim extra as there would be too many error classes
                if sortedResponse[0, i] == 'noResponse':
                    errorComponent_5 = 'response' + 'NoResponse'
                else:
                    errorComponent_5 = 'response' + sortedResponse[0, i]

                # Concatenate error components
                currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}_{errorComponent_5}'
                errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])
        return errors_dict_fineGrained

    participant = 'beRNN_05'
    focusedMonths = ['month_1','month_2','month_3','month_4','month_5','month_6','month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']
    directory = f'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\{participant}\\data_highDim' # info: whole script made for original dataset only
    dict_directory = rf'C:\Users\oliver.frank\Desktop\PyProjects\Data\{participant}\ErrorGraphics_monthWise'

    for current_month in focusedMonths:

        fig, axs = plt.subplots(12, 1, figsize=(2, 24))
        # info: ################################################################################################################
        # info: Decision Making
        # info: ################################################################################################################
        def get_errors_DM(Response, errors_dict, distract_dict, opposite_dict, strength_dict):

            for i in range(Response.shape[1]):
                # Choose task
                participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
                # Evaluate errors
                if participantResponse != correctResponse:
                    errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(correctResponse) \
                        else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
                        else 'distractOrtho'

                    errorComponent_2 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
                        else 'responseNone' if participantResponse == 'NoResponse' \
                        else 'responseOrtho'

                    strengthDiff =  abs(strength_dict.get(distractStim.split('_')[0], 0) - strength_dict.get(correctStim.split('_')[0], 0))
                    errorComponent_3 = f'strengthDiff{int(strengthDiff * 100)}'

                    # Concatenate error components
                    currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

                    errors_dict[currentChosenList].append(Response[:, i])

            return errors_dict

        # Define dicts
        distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R'}
        opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}
        strength_dict = {'lowest':0.25, 'low':0.5, 'strong':0.75, 'strongest':1.0}

        # Create categorical names
        list1 = ['distractOpposite', 'distractSame', 'distractOrtho']
        list2 = ['responseOpposite', 'responseNone', 'responseOrtho']
        list3 = ['strengthDiff0', 'strengthDiff25', 'strengthDiff5', 'strengthDiff75']
        # Generating all combinations of categorical names
        categorical_names = ['_'.join(combination) for combination in iter_tools.product(list1, list2, list3)]

        ########################################################################################################################
        # DM -------------------------------------------------------------------------------------------------------------------
        errors_dict_DM = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'DM')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        selected_months_files = [file for file in npy_files if current_month in file]
        # selected_months_files = [file for file in npy_files if any(current_month in file for month in focusedMonths)]

        for npy_file in selected_months_files:
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            # Sort the 4th and 5th row, so that HIGHER value is on 4th row. Sort 2nd and 3rd accordingly
            sortedResponse = sort_rows_descending(Response)
            errors_dict_DM = get_errors_DM(sortedResponse, errors_dict_DM, distract_dict, opposite_dict, strength_dict)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_DM_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_DM, f)
        with open(os.path.join(dict_directory, f'errors_dict_DM_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_DM, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_DM, participantDirectory, participant, 'DM', 'rough', titleAdd = 'all', ax=axs[0])

        ########################################################################################################################
        # DM Anti --------------------------------------------------------------------------------------------------------------
        errors_dict_DM_Anti = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'DM_Anti')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

        for npy_file in selected_months_files:
            Response = np.load(npy_file, allow_pickle=True)
            # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
            sortedResponse = sort_rows_ascending(Response)
            errors_dict_DM_Anti = get_errors_DM(sortedResponse, errors_dict_DM_Anti, distract_dict, opposite_dict, strength_dict)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_DM_Anti_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_DM_Anti, f)
        with open(os.path.join(dict_directory, f'errors_dict_DM_Anti_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_DM_Anti, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_DM_Anti,participantDirectory, participant, 'DM_Anti', 'rough', titleAdd = 'all', ax=axs[1])

        # info: ################################################################################################################
        # info: Executive Functioning
        # info: ################################################################################################################
        # Define dicts
        distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R', 'X.png':'X'}
        opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}

        # Create categorical names
        list1 = ['distractOpposite', 'distractSame', 'distractOrtho', 'distractX']
        list2 = ['colorsDiff', 'colorsSame']
        list3 = ['responseOpposite', 'responseNone', 'responseOrtho']

        # Generating all combinations of categorical names
        categorical_names = ['_'.join(combination) for combination in iter_tools.product(list1, list2, list3)]

        def get_errors_EF(Response, errors_dict, distract_dict, opposite_dict):

            for i in range(Response.shape[1]):
                if Response[5,i] == '10':
                    Response[2,i] = Response[3,i]
                participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
                # Evaluate errors
                if participantResponse != correctResponse:
                    errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(
                        correctResponse) \
                        else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
                        else 'distractX' if distract_dict.get(distractStim.split('_')[1]) == 'X' \
                        else 'distractOrtho'

                    if distractStim.split('_')[0] == correctStim.split('_')[0]:
                        errorComponent_2 = 'colorsSame'
                    else:
                        errorComponent_2 = 'colorsDiff'

                    errorComponent_3 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
                        else 'responseNone' if participantResponse == 'NoResponse' \
                        else 'responseOrtho'

                    # Concatenate error components
                    currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

                    errors_dict[currentChosenList].append(Response[:, i])

            return errors_dict

        ########################################################################################################################
        # EF -------------------------------------------------------------------------------------------------------------------
        errors_dict_EF = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'EF')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

        for npy_file in selected_months_files:
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
            sortedResponse = sort_rows_ascending(Response)
            errors_dict_EF = get_errors_EF(sortedResponse, errors_dict_EF, distract_dict, opposite_dict)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_EF_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_EF, f)
        with open(os.path.join(dict_directory, f'errors_dict_EF_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_EF, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_EF, participantDirectory, participant, 'EF', 'rough', titleAdd = 'all', ax=axs[2])

        ########################################################################################################################
        # EF Anti --------------------------------------------------------------------------------------------------------------
        errors_dict_EF_Anti = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'EF_Anti')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

        for npy_file in selected_months_files:
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
            sortedResponse = sort_rows_ascending(Response)
            errors_dict_EF_Anti = get_errors_EF(Response, errors_dict_EF_Anti, distract_dict, opposite_dict)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_EF_Anti_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_EF_Anti, f)
        with open(os.path.join(dict_directory, f'errors_dict_EF_Anti_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_EF_Anti, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_EF_Anti, participantDirectory, participant, 'EF_Anti', 'rough', titleAdd = 'all', ax=axs[3])

        # info: ################################################################################################################
        # info: Relational Processing
        # info: ################################################################################################################
        # Define dicts
        colorDict = {'ClassYellow': ['yellow', 'amber', 'orange'],
                     'ClassGreen' : ['green', 'lime', 'moss'],
                     'ClassBlue': ['purple', 'violet', 'blue'],
                     'ClassRed': ['rust', 'red', 'magenta']}

        # Create categorical names
        list1 = ['distractClassYellowCircle', 'distractClassYellowNonagon', 'distractClassYellowHeptagon', 'distractClassYellowPentagon', 'distractClassYellowTriangle',\
                 'distractClassBlueCircle', 'distractClassBlueNonagon', 'distractClassBlueHeptagon', 'distractClassBluePentagon', 'distractClassBlueTriangle',\
                 'distractClassRedCircle', 'distractClassRedNonagon', 'distractClassRedHeptagon', 'distractClassRedPentagon', 'distractClassRedTriangle',\
                 'distractClassGreenCircle', 'distractClassGreenNonagon', 'distractClassGreenHeptagon', 'distractClassGreenPentagon', 'distractClassGreenTriangle',\
                 'noResponse']

        # Generating all combinations of categorical names
        categorical_names = ['_'.join(combination) for combination in iter_tools.product(list1)]

        def get_errors_RP(Response, errors_dict, opened_meta_file):

            for i in range(Response.shape[1]):
                participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
                # Evaluate errors
                if participantResponse != correctResponse:
                    # Chosen wrong distraction belonging class
                    if Response[0, i].split('_')[0] in colorDict['ClassYellow']:
                        errorComponent_1 = 'distract' + 'ClassYellow' + participantResponse.split('_')[1].split('.')[0].capitalize()
                    elif Response[0, i].split('_')[0] in colorDict['ClassBlue']:
                        errorComponent_1 = 'distract' + 'ClassBlue' + participantResponse.split('_')[1].split('.')[0].capitalize()
                    elif Response[0, i].split('_')[0] in colorDict['ClassRed']:
                        errorComponent_1 = 'distract' + 'ClassRed' + participantResponse.split('_')[1].split('.')[0].capitalize()
                    elif Response[0, i].split('_')[0] in colorDict['ClassGreen']:
                        errorComponent_1 = 'distract' + 'ClassGreen' + participantResponse.split('_')[1].split('.')[0].capitalize()
                    else:
                        errorComponent_1 = 'noResponse'

                    # Concatenate error components
                    currentChosenList = f'{errorComponent_1}'

                    errors_dict[currentChosenList].append(Response[:, i])

            return errors_dict

        ########################################################################################################################
        # RP -------------------------------------------------------------------------------------------------------------------
        errors_dict_RP = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'RP')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            errors_dict_RP = get_errors_RP(Response, errors_dict_RP, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_RP_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_RP, f)
        with open(os.path.join(dict_directory, f'errors_dict_RP_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_RP, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_RP,participantDirectory, participant, 'RP', 'rough', titleAdd = 'all', ax=axs[4])

        ########################################################################################################################
        # RP Anti --------------------------------------------------------------------------------------------------------------
        errors_dict_RP_Anti = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'RP_Anti')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            errors_dict_RP_Anti = get_errors_RP(Response, errors_dict_RP_Anti, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_RP_Anti_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_RP_Anti, f)
        with open(os.path.join(dict_directory, f'errors_dict_RP_Anti_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_RP_Anti, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_RP_Anti,participantDirectory, participant, 'RP_Anti', 'rough', titleAdd = 'all', ax=axs[5])

        ########################################################################################################################
        # RP Ctx1 --------------------------------------------------------------------------------------------------------------
        errors_dict_RP_Ctx1 = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'RP_Ctx1')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            errors_dict_RP_Ctx1 = get_errors_RP(Response, errors_dict_RP_Ctx1, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_RP_Ctx1_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_RP_Ctx1, f)
        with open(os.path.join(dict_directory, f'errors_dict_RP_Ctx1_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_RP_Ctx1, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_RP_Ctx1,participantDirectory, participant, 'RP_Ctx1', 'rough', titleAdd = 'all', ax=axs[6])

        ########################################################################################################################
        # RP Ctx2 --------------------------------------------------------------------------------------------------------------
        errors_dict_RP_Ctx2 = {name: [] for name in categorical_names}
        # Get list of necessary files in directory
        participantDirectory = os.path.join(directory, 'RP_Ctx2')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            errors_dict_RP_Ctx2 = get_errors_RP(Response, errors_dict_RP_Ctx2, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_RP_Ctx2_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_RP_Ctx2, f)
        with open(os.path.join(dict_directory, f'errors_dict_RP_Ctx2_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_RP_Ctx2, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_RP_Ctx2,participantDirectory, participant, 'RP_Ctx2', 'rough', titleAdd = 'all', ax=axs[7])

        # info: ################################################################################################################
        # info: Working Memory
        # info: ################################################################################################################
        colorDict = {'ClassYellow': ['yellow', 'amber', 'orange'],
                     'ClassGreen' : ['green', 'lime', 'moss'],
                     'ClassBlue': ['purple', 'violet', 'blue'],
                     'ClassRed': ['rust', 'red', 'magenta']}
        formDict = {'ClassCircle': ['circle', 'nonagon'],
                    'ClassPolygon': ['heptagon', 'pentagon'],
                    'ClassTriangle': ['triangle']}

        # Create categorical names for WM tasks
        list1 = ['distractClassYellowCircle', 'distractClassYellowPolygon', 'distractClassYellowTriangle',\
                 'distractClassBlueCircle', 'distractClassBluePolygon', 'distractClassBlueTriangle',\
                 'distractClassRedCircle', 'distractClassRedPolygon', 'distractClassRedTriangle',\
                 'distractClassGreenCircle', 'distractClassGreenPolygon', 'distractClassGreenTriangle',\
                 'noResponse']
        list2 = ['3stim']
        list3 = ['diffColor_diffForm', 'simColor_diffForm', 'simColor_simForm']

        categorical_names_WM_1 = ['_'.join(combination) for combination in iter_tools.product(list1, list2, list3)]
        categorical_names_WM_2 = ['_'.join(combination) for combination in iter_tools.product(list1, list3)]

        categorical_names_WM = categorical_names_WM_1 + categorical_names_WM_2

        # Create categorical names for WM_Ctx tasks
        list1 = ['formClassCombi_CircleCircle', 'formClassCombi_CirclePolygon', 'formClassCombi_CircleTriangle',\
                 'formClassCombi_PolygonPolygon', 'formClassCombi_PolygonTriangle', 'formClassCombi_TriangleTriangle']
        list2 = ['3stim'] # info: naming for ctx tasks accidentally different than above
        list3 = ['diffColor_diffForm', 'simColor_diffForm','diffColor_simForm', 'simColor_simForm']
        list4 = ['responseMatch', 'responseMismatch', 'responseNoResponse']

        categorical_names_WM_Ctx_1 = ['_'.join(combination) for combination in iter_tools.product(list1, list2, list3, list4)]
        categorical_names_WM_Ctx_2 = ['_'.join(combination) for combination in iter_tools.product(list1, list3, list4)]

        categorical_names_WM_Ctx = categorical_names_WM_Ctx_2 + categorical_names_WM_Ctx_1

        # shows current trial in detail and similiarity in form and color to previous trials
        def get_errors_WM(Response, errors_dict, opened_meta_file):

            for i in range(Response.shape[1]):
                participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
                # Evaluate errors
                if participantResponse != correctResponse:
                    # Chosen wrong distraction belonging class
                    if participantResponse.split('_')[0] in colorDict['ClassYellow']:
                        if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'distract' + 'ClassYellow' + 'Circle'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                            errorComponent_1 = 'distract' + 'ClassYellow' + 'Polygon'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                            errorComponent_1 = 'distract' + 'ClassYellow' + 'Triangle'
                    elif participantResponse.split('_')[0] in colorDict['ClassGreen']:
                        if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'distract' + 'ClassGreen' + 'Circle'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                            errorComponent_1 = 'distract' + 'ClassGreen' + 'Polygon'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                            errorComponent_1 = 'distract' + 'ClassGreen' + 'Triangle'
                    elif participantResponse.split('_')[0] in colorDict['ClassBlue']:
                        if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'distract' + 'ClassBlue' + 'Circle'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                            errorComponent_1 = 'distract' + 'ClassBlue' + 'Polygon'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                            errorComponent_1 = 'distract' + 'ClassBlue' + 'Triangle'
                    elif participantResponse.split('_')[0] in colorDict['ClassRed']:
                        if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'distract' + 'ClassRed' + 'Circle'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                            errorComponent_1 = 'distract' + 'ClassRed' + 'Polygon'
                        elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                            errorComponent_1 = 'distract' + 'ClassRed' + 'Triangle'
                    elif participantResponse.split('_')[0] == 'noResponse' or participantResponse.split('_')[0] == 'NoResponse':
                        errorComponent_1 = 'noResponse'
                    else:
                        continue

                    errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1]
                    # Concatenate error components
                    currentChosenList = f'{errorComponent_1}_{errorComponent_2}'

                    errors_dict[currentChosenList].append(Response[:, i])

            return errors_dict

        def get_errors_WM_Ctx(Response, errors_dict, opened_meta_file):

            for i in range(Response.shape[1]):
                participantResponse, correctResponse, Stim1, Stim2 = Response[0:4, i]
                # Evaluate errors
                if participantResponse != correctResponse:
                    # Find this error's belonging to color class
                    try:
                        if Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'formClassCombi_CircleCircle'
                        elif Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                            errorComponent_1 = 'formClassCombi_PolygonPolygon'
                        elif Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                            errorComponent_1 = 'formClassCombi_TriangleTriangle'
                        elif Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon'] or\
                                Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'formClassCombi_CirclePolygon'
                        elif Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle'] or \
                             Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                            errorComponent_1 = 'formClassCombi_CircleTriangle'
                        elif Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle'] or \
                             Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                            errorComponent_1 = 'formClassCombi_PolygonTriangle'
                    except Exception as e:
                        print('Error occured: ', e) # info: Happens rarely if accidentally one of two stims is None
                        continue

                    errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1]

                    if participantResponse == 'Match' or participantResponse == 'Mismatch':
                        errorComponent_3 = 'response' + participantResponse
                    else:
                        errorComponent_3 = 'responseNoResponse'

                    # Concatenate error components
                    currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

                    errors_dict[currentChosenList].append(Response[:, i])

            return errors_dict

        ########################################################################################################################
        # WM -------------------------------------------------------------------------------------------------------------------
        errors_dict_WM = {name: [] for name in categorical_names_WM}
        participantDirectory = os.path.join(directory, 'WM')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            # Sort the correct stim on the 2nd row
            sortedResponse = sort_rows_correctOn2(Response)
            errors_dict_WM = get_errors_WM(sortedResponse, errors_dict_WM, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_WM_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_WM, f)
        with open(os.path.join(dict_directory, f'errors_dict_WM_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_WM, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_WM,participantDirectory, participant, 'WM', 'rough', titleAdd = 'all', ax=axs[8])

        ########################################################################################################################
        # WM Anti --------------------------------------------------------------------------------------------------------------
        errors_dict_WM_Anti = {name: [] for name in categorical_names_WM}
        participantDirectory = os.path.join(directory, 'WM_Anti')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            # Sort the correct stim on the 2nd row
            sortedResponse = sort_rows_correctOn2(Response)
            errors_dict_WM_Anti = get_errors_WM(sortedResponse, errors_dict_WM_Anti, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_WM_Anti_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_WM_Anti, f)
        with open(os.path.join(dict_directory, f'errors_dict_WM_Anti_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_WM_Anti, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_WM_Anti,participantDirectory, participant, 'WM_Anti', 'rough', titleAdd = 'all', ax=axs[9])

        #######################################################################################################################
        # WM Ctx1 --------------------------------------------------------------------------------------------------------------
        errors_dict_WM_Ctx1 = {name: [] for name in categorical_names_WM_Ctx}
        participantDirectory = os.path.join(directory, 'WM_Ctx1')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            errors_dict_WM_Ctx1 = get_errors_WM_Ctx(Response, errors_dict_WM_Ctx1, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_WM_Ctx1_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_WM_Ctx1, f)
        with open(os.path.join(dict_directory, f'errors_dict_WM_Ctx1_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_WM_Ctx1, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_WM_Ctx1,participantDirectory, participant, 'WM_Ctx1', 'rough', titleAdd = 'all', ax=axs[10])

        ########################################################################################################################
        # WM Ctx2 --------------------------------------------------------------------------------------------------------------
        errors_dict_WM_Ctx2 = {name: [] for name in categorical_names_WM_Ctx}
        participantDirectory = os.path.join(directory, 'WM_Ctx2')
        npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
        meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
        selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
        selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

        for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
            # Load the JSON content from the file
            with open(meta_file, 'r') as file:
                opened_meta_file = json.load(file)
            # Use the function
            Response = np.load(npy_file, allow_pickle=True)
            errors_dict_WM_Ctx2 = get_errors_WM_Ctx(Response, errors_dict_WM_Ctx2, opened_meta_file)

        # Save dict for other analysis
        with open(os.path.join(dict_directory, f'errors_dict_WM_Ctx2_{participant}_{current_month}.pkl'), 'wb') as f:
            pickle.dump(errors_dict_WM_Ctx2, f)
        with open(os.path.join(dict_directory, f'errors_dict_WM_Ctx2_{participant}_{current_month}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict_WM_Ctx2, f, ensure_ascii=False, indent=4, default=convert_numpy)
        # Visualize results
        top_four_categories = plot_errorDistribution_relative(errors_dict_WM_Ctx2,participantDirectory, participant, 'WM_Ctx2', 'rough', titleAdd = 'all', ax=axs[11])

        # Once all 12 are assigned, save the single master stacked image sheet
        plt.tight_layout()
        master_save_path = os.path.join(directory.split('\\data')[0], 'ErrorGraphics_monthWise', f'all_12_tasks_{participant}_{current_month}.pdf')
        os.makedirs(os.path.dirname(master_save_path), exist_ok=True)
        plt.savefig(master_save_path, dpi=100, bbox_inches='tight')
        plt.close()


if contingencyTables:
    #################################################################################################################
    # head: contingencyTable for each task individually
    #################################################################################################################
    def evaluate_model_responses(directory, dataDirectory, models, tasks):
        accumulated_data_percentage = 0
        task_count = 0
        for i, task in enumerate(tasks):
            # modelDirectories
            figurePath = os.path.join(directory, 'contingencyTable')
            # Create directory for saving figures if it doesn't exist
            if not os.path.exists(figurePath):
                os.makedirs(figurePath)

            # mode = 'Average' for the future
            mode = 'Average'

            for model_dir in models:
                correct_match, error_match, correct_mismatch, error_mismatch, total_trials = evaluate_task(model_dir, task, i)

                # Calculate percentages for the contingency table
                data_percentage = np.round(
                    np.array([[correct_match / total_trials * 100, correct_mismatch / total_trials * 100],
                              [error_match / total_trials * 100, error_mismatch / total_trials * 100]]), 2)
                # data_percentage = np.round(
                #     np.array([[(correct_match / total_trials * 100) - 5, (correct_mismatch / total_trials * 100) + 5],
                #               [(error_match / total_trials * 100) - 5, (error_mismatch / total_trials * 100) + 5]]), 2)

                # Accumulate the percentages
                if accumulated_data_percentage is None:
                    accumulated_data_percentage = data_percentage
                else:
                    accumulated_data_percentage += data_percentage

            task_count += 1

        # Average the accumulated percentages
        average_data_percentage = accumulated_data_percentage / task_count

        ratioCorrect = average_data_percentage[0][0] / average_data_percentage[0][1]
        ratioError = average_data_percentage[1][0] / average_data_percentage[1][1]

        print(ratioCorrect)
        print(ratioError)

        # Visualize the averaged contingency table
        visualize_contingency_table(average_data_percentage, task, figurePath, mode, model_dir=model_dir, ratio_correct=ratioCorrect, ratio_error=ratioError)

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
            train_files, eval_files = split_files(hp, file_quartett)

            # Store the results in the dictionaries
            train_data[subdir] = train_files
            eval_data[subdir] = eval_files

        return train_data, eval_data

    def evaluate_task(model_dir, task, indice):
        error_match = 0
        error_mismatch = 0
        correct_match = 0
        correct_mismatch = 0
        total_trials = 0

        # Prepare model restore
        hp = _tools.load_hp(model_dir)

        hp['rng'] = np.random.default_rng(42) # as for training

        file_quartett = []

        for month in hp['monthsConsidered']:
            train_data, eval_data = createSplittedDatasets(hp, dataDirectory, month)
            file_quartett_ = eval_data[list(eval_data)[indice]]

            file_quartett.extend(file_quartett_)

        hp['rng'] = np.random.default_rng() # as for training
        model = Model(model_dir, hp=hp)

        with tf.Session() as sess:
            model.restore(model_dir)

            for i in range(0, 20): # Info: Each iteration represents one batch - after the spliting most often you only have 4 batches for one month

                x, y, y_loc, base_name = _tools.load_trials(hp['rng'], task, 'test', 40, file_quartett, True)
                ground_truth = np.load(os.path.join(base_name + 'Response.npy'), allow_pickle=True)

                print(base_name)

                # Sort response data
                c_mask = np.zeros((y.shape[0] * y.shape[1], y.shape[2]), dtype='float32')
                # Generate model response for the current batch
                feed_dict = _tools.gen_feed_dict(model, x, y, c_mask, hp)
                c_lsq, c_reg, modelResponse_machineForm = sess.run([model.cost_lsq, model.cost_reg, model.y_hat],feed_dict=feed_dict)

                perf = get_perf(modelResponse_machineForm, y_loc)

                for i in range(0,len(perf)):
                    if perf[i] == 1:  # Model response matches participant response
                        if ground_truth[0, i] == ground_truth[1, i]:
                            correct_match += 1
                        else:
                            error_match += 1
                    elif perf[i] == 0:  # Model response does not match participant response
                        if ground_truth[0, i] == ground_truth[1, i]:
                            correct_mismatch += 1
                        else:
                            error_mismatch += 1
                total_trials += len(perf)

        return correct_match, error_match, correct_mismatch, error_mismatch, total_trials

    def visualize_contingency_table(data, task, figure_path, mode, model_dir, ratio_correct, ratio_error):
        # Define labels for the table
        row_labels = ['Participant Response Correct', 'Participant Response Incorrect']
        col_labels = ['Model Response Correct', 'Model Response Incorrect']

        # Increase the figure size for better alignment and spacing
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased size for the table

        # Create a divider to manage the color bar size relative to the heatmap
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Plot the contingency table with clear annotations
        sns.heatmap(data, annot=True, fmt='.0f', cmap='Greys', cbar=True,
                    xticklabels=col_labels, yticklabels=row_labels, ax=ax,
                    linewidths=0, linecolor='black', annot_kws={"fontsize": 30, "color": "black"}, square=True,
                    cbar_ax=cax,
                    vmin=0, vmax=100)

        # Set axis labels with adequate spacing and no rotation
        ax.set_xticklabels(col_labels, fontsize=16)
        ax.set_yticklabels(row_labels, fontsize=16, rotation=90)

        # Configure the color bar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

        # Add ratio text to the right of the heatmap, making it closer but avoiding overlap
        ax.text(2.3, 0.5, f'Ratio Correct: {ratio_correct:.2f}', va='center', fontsize=18, color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        ax.text(2.3, 1.5, f'Ratio Error: {ratio_error:.2f}', va='center', fontsize=18, color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        # Add title directly above the heatmap in the center
        subject = '_'.join(model_dir.split('\\')[-2].split('_')[0:2])
        plt.suptitle(f'{subject}', fontsize=18, y=.95, x=.4)

        # Adjust layout to fit all elements within the figure bounds
        plt.tight_layout(rect=[0, 0, .9, .95])  # Adjusted the layout to accommodate the title

        # Save the plot ensuring all elements are included

        save_path = os.path.join(figure_path, f'{mode}_Contingency_{subject}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)

        # Display the plot
        plt.show()

    # info. Model evaluation for chosen tasks
    # Directory has to point exactly on the model folder of interest
    participant = 'beRNN_03'
    directory = rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\show-grid_multi_beRNN_03_highDim_256\highDim\beRNN_03\8\beRNN_03_AllTask_3-5_data_highDim_tB8_iter5_LeakyRNN_256_relu'
    dataDirectory = f'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\{participant}\\data_highDim' # add right participant and dataset
    modelsList = [os.path.join(directory, item) for item in os.listdir(directory) if 'model_month_5' in item]

    # Tasks to evaluate
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
    evaluate_model_responses(directory, dataDirectory, modelsList, tasks)


if taskComplexity:
    #################################################################################################################
    # head: task complexity x relative count plots
    #################################################################################################################
    fig, axs = plt.subplots(12, 1, figsize=(3, 24))

    participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti',
             'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']

    for iteration, task in enumerate(tasks):
        ax = axs[iteration]

        combined_errors = defaultdict(int)

        for participant in participantList:
            dict_directory = rf'C:\Users\oliver.frank\Desktop\PyProjects\Data\{participant}\ErrorGraphics'

            file_path = os.path.join(dict_directory, f'errors_dict_{task}_{participant}.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    loaded_dict = pickle.load(f)

                for cat, values in loaded_dict.items():
                    combined_errors[cat] += len(values)

        total_occurrences = sum(combined_errors.values())

        easy_pct = 0
        medium_pct = 0
        difficult_pct = 0

        for cat, count in combined_errors.items():
            pct = (count / total_occurrences) * 100 if total_occurrences > 0 else 0

            if 'DM' in task:
                if 'strengthDiff0' in cat:
                    difficult_pct += pct
                elif 'distractSame' in cat:
                    easy_pct += pct
                else:
                    medium_pct += pct
            elif 'EF' in task:
                if 'colorsDiff' in cat:
                    difficult_pct += pct
                elif 'distractSame' in cat and 'colorSame' in cat:
                    easy_pct += pct
                else:
                    medium_pct += pct
            elif 'RP' in task:
                if 'Nonagon' in cat or 'Heptagon' in cat or 'Pentagon' in cat:
                    difficult_pct += pct
                elif 'Triangle' in cat:
                    easy_pct += pct
                elif 'Circle' in cat:
                    medium_pct += pct
            elif 'WM' in task:
                if 'simColor_simForm' in cat:
                    difficult_pct += pct
                else:
                    medium_pct += pct

        difficulty_groups = ['easy', 'medium','difficult']
        percentage_values = [easy_pct, medium_pct, difficult_pct]

        sns.barplot(x=difficulty_groups, y=percentage_values, palette='coolwarm', ax=ax)

        # Styling-Vorgaben für die Achsen anwenden
        ax.set_ylim([0, 100])
        ax.set_yticks([0, 50, 100])
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    master_save_path = r'C:\Users\oliver.frank\Desktop\PyProjects\Data\combined_tasks_difficulty.png'
    os.makedirs(os.path.dirname(master_save_path), exist_ok=True)
    plt.savefig(master_save_path, dpi=100, bbox_inches='tight')
    plt.close()


if saveCSVerrors:
    #################################################################################################################
    # head: Create csv saving all counted error categories for each subject and month
    #################################################################################################################
    # Creates a dictionary that automatically puts a new dict inside missing keys
    completedict = defaultdict(lambda: defaultdict(dict))
    participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
    tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']

    for task in tasks:
        for participant in participantList:
            if participant == 'beRNN_04':
                focusedMonths = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7']
            else:
                focusedMonths = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
                                 'month_9', 'month_10', 'month_11', 'month_12']

            path = rf'C:\Users\oliver.frank\Desktop\PyProjects\Data\{participant}\ErrorGraphics_monthWise'
            for current_month in focusedMonths:
                with open(os.path.join(path, f'errors_dict_{task}_{participant}_{current_month}.json'), 'r') as file:
                    currentDictonary = json.load(file)

                completedict[task][participant][current_month] = currentDictonary

    rows = []

    # Iterate through tasks, participants, months, and error categories
    for task_name, participants in completedict.items():
        for participant_id, months_dict in participants.items():
            for month_name, categories in months_dict.items():
                for category_name, trials in categories.items():

                    if isinstance(trials, (list, tuple)):
                        error_count = len(trials)
                    elif hasattr(trials, "shape"):
                        error_count = trials.shape[0]
                    else:
                        error_count = 0

                    rows.append(
                        {
                            "task": task_name,
                            "subject": participant_id,
                            "month": month_name,
                            "error_category": category_name,
                            "count": error_count,
                        }
                    )

    df_flat = pd.DataFrame(rows)

    output_path = (
        r"C:\Users\oliver.frank\Desktop\PyProjects\Data\all_tasks_flat_counts_monthWise.csv"
    )
    df_flat.to_csv(output_path, index=False, encoding="utf-8")


if top5errors:
    #################################################################################################################
    # head: Create top5 error heatmaps for each task individually
    #################################################################################################################
    df = pd.read_csv(r'C:\Users\oliver.frank\Desktop\PyProjects\Data\all_tasks_flat_counts.csv')

    participant_list = ['beRNN_03', 'beRNN_04', 'beRNN_01', 'beRNN_02', 'beRNN_05']
    paper_nomenclatur_dict = ['HC1', 'HC2', 'MDD', 'ASD', 'SCZ']
    mapping_dict = dict(zip(participant_list, paper_nomenclatur_dict))

    # Realtive amount of error category X in task Y for each subject individually
    subject_task_totals = (
        df.groupby(['task', 'subject'])['count']
        .transform('sum')
    )
    df['relative_amount_pct'] = (df['count'] / subject_task_totals) * 100
    df['subject'] = df['subject'].map(mapping_dict)

    output_folder = r'C:\Users\oliver.frank\Desktop\PyProjects\Data\finalHeatmaps'
    os.makedirs(output_folder, exist_ok=True)

    unique_tasks = df['task'].unique()

    for task in unique_tasks:
        task_all_data = df[df['task'] == task]

        # Evaluate top 5 highest sums of relative amounts for each task across subjects
        top5_errors_for_task = (
            task_all_data.groupby('error_category')['relative_amount_pct']
            .sum()
            .nlargest(5)
            .index
            .tolist()
        )

        # Take top 5
        task_df = task_all_data[task_all_data['error_category'].isin(top5_errors_for_task)].copy()

        heatmap_data = task_df.pivot(
            index='error_category',
            columns='subject',
            values='relative_amount_pct'
        ).reindex(index=top5_errors_for_task, columns=paper_nomenclatur_dict).fillna(0)

        # Relative amount of each subjects contribution to total relative sum for each category and task
        row_sums = heatmap_data.sum(axis=1)
        heatmap_data_normalized = heatmap_data.div(row_sums, axis=0).fillna(0) * 100

        fig = plt.figure(figsize=(11, 8))
        horiz = [Size.Fixed(4.0), Size.Fixed(5.0), Size.Fixed(0.3), Size.Fixed(0.2)]
        vert = [Size.Fixed(1.0), Size.Fixed(5.0)]

        divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), horiz, vert, aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))
        cbar_ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=3, ny=1))

        sns.heatmap(
            heatmap_data_normalized,
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
            vmin=0,
            vmax=50,
            cbar_ax=cbar_ax,
            cbar_kws={'label': '% Share among Subjects'},
            linewidths=5.0,
            linecolor='white',
            ax=ax
        )

        ax.set_title(f'Task: {task} (Top 5 Errors by Relative Sum)', fontsize=12, pad=15, weight='bold')
        ax.set_xlabel('Study Groups (Paper Nomenclature)', fontsize=11)
        ax.set_ylabel('Error Category', fontsize=11)

        safe_task_name = "".join([c if c.isalnum() else "_" for c in str(task)])
        pdf_filename = f"heatmap_task_{safe_task_name}_v2.pdf"
        pdf_path = os.path.join(output_folder, pdf_filename)

        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Gespeichert: {pdf_path}")
        plt.close(fig)


