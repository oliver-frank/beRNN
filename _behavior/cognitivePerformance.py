import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''
- Evaluates monthly task performance for individual participant
- Plots training effect showing change of pefromance over the whole data collection period
- Plots training effect as boxplots
'''

evaluateMonthlyTaskPerf, plotTrainingEffect, plotTrainingEffectAsBoxPlots = True, False, False

if evaluateMonthlyTaskPerf:
    participant_dir = 'W:\\group_csp\\analyses\\oliver.frank\\Data\\'
    participantList = os.listdir(participant_dir)

    participant = participantList[2] # choose which particpant to analyze
    month = '11' # choose which month to analyze

    percentCorrect_DM, count_DM = 0, 0
    percentCorrect_DM_Anti, count_DM_Anti = 0, 0
    percentCorrect_EF, count_EF = 0, 0
    percentCorrect_EF_Anti, count_EF_Anti = 0, 0
    percentCorrect_RP, count_RP = 0, 0
    percentCorrect_RP_Anti, count_RP_Anti = 0, 0
    percentCorrect_RP_Ctx1, count_RP_Ctx1 = 0, 0
    percentCorrect_RP_Ctx2, count_RP_Ctx2 = 0, 0
    percentCorrect_WM, count_WM = 0, 0
    percentCorrect_WM_Anti, count_WM_Anti = 0, 0
    percentCorrect_WM_Ctx1, count_WM_Ctx1 = 0, 0
    percentCorrect_WM_Ctx2, count_WM_Ctx2 = 0, 0

    # co: Download data as .xlsx long format
    list_testParticipant_month = os.listdir(os.path.join(participant_dir,participant,month))
    for i in list_testParticipant_month:
        if i.split('.')[1] != 'png':
            currentFile = pd.read_excel(os.path.join(participant_dir,participant,month,i), engine='openpyxl')
            # print(currentFile['UTC Date and Time'])
            if currentFile['Task Name'][0] != '000_state_questions' and currentFile['Task Name'][0] != '000_session_completion': # avoid files with state questions and session completion
                # print(currentFile.iloc[0,28].split('_trials_')[0])
                # print('W:/AG_CSP/Projekte/beRNN_v1/02_Daten/BeRNN_main/' + participant + month + i)
                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'DM':
                    # percentCorrect_DM += currentFile['Store: PercentCorrectDM'][len(currentFile['Store: PercentCorrectDM'])-3]
                    # count_DM += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy() # info: every now and then the 125th event is missing
                    percentCorrect_DM += sum(filtered_rows['Store: PercentCorrectDM'])
                    count_DM += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'DM_Anti':
                    # percentCorrect_DM_Anti += currentFile['Store: PercentCorrectDMAnti'][len(currentFile['Store: PercentCorrectDMAnti'])-3]
                    # count_DM_Anti += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_DM_Anti += sum(filtered_rows['Store: PercentCorrectDMAnti'])
                    count_DM_Anti += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'EF':
                    # percentCorrect_EF += currentFile['Store: PercentCorrectEF'][len(currentFile['Store: PercentCorrectEF'])-3]
                    # count_EF += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_EF += sum(filtered_rows['Store: PercentCorrectEF'])
                    count_EF += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'EF_Anti':
                    # percentCorrect_EF_Anti += currentFile['Store: PercentCorrectEFAnti'][len(currentFile['Store: PercentCorrectEFAnti'])-3] # no extra displays for Anti were made
                    # count_EF_Anti += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_EF_Anti += sum(filtered_rows['Store: PercentCorrectEFAnti'])
                    count_EF_Anti += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'RP':
                    # percentCorrect_RP += currentFile['Store: PercentCorrectRP'][len(currentFile['Store: PercentCorrectRP'])-3]
                    # count_RP += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_RP += sum(filtered_rows['Store: PercentCorrectRP'])
                    count_RP += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'RP_Anti':
                    # percentCorrect_RP_Anti += currentFile['Store: PercentCorrectRPAnti'][len(currentFile['Store: PercentCorrectRPAnti'])-3]
                    # count_RP_Anti += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_RP_Anti += sum(filtered_rows['Store: PercentCorrectRPAnti'])
                    count_RP_Anti += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'RP_Ctx1':
                    # percentCorrect_RP_Ctx1 += currentFile['Store: PercentCorrectRPCtx1'][len(currentFile['Store: PercentCorrectRPCtx1'])-3]
                    # count_RP_Ctx1 += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                    percentCorrect_RP_Ctx1 += sum(filtered_rows['Store: PercentCorrectRPCtx1'])
                    count_RP_Ctx1 += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'RP_Ctx2':
                    # percentCorrect_RP_Ctx2 += currentFile['Store: PercentCorrectRPCtx2'][len(currentFile['Store: PercentCorrectRPCtx2'])-3]
                    # count_RP_Ctx2 += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_RP_Ctx2 += sum(filtered_rows['Store: PercentCorrectRPCtx2'])
                    count_RP_Ctx2 += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'WM':
                    # percentCorrect_WM += currentFile['Store: PercentCorrectWM'][len(currentFile['Store: PercentCorrectWM'])-3]
                    # count_WM += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_WM += sum(filtered_rows['Store: PercentCorrectWM'])
                    count_WM += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'WM_Anti':
                    # percentCorrect_WM_Anti += currentFile['Store: PercentCorrectWMAnti'][len(currentFile['Store: PercentCorrectWMAnti'])-3]
                    # count_WM_Anti += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_WM_Anti += sum(filtered_rows['Store: PercentCorrectWMAnti'])
                    count_WM_Anti += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0]  == 'WM_Ctx1':
                    # percentCorrect_WM_Ctx1 += currentFile['Store: PercentCorrectWMCtx1'][len(currentFile['Store: PercentCorrectWMCtx1'])-3]
                    # count_WM_Ctx1 += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_WM_Ctx1 += sum(filtered_rows['Store: PercentCorrectWMCtx1'])
                    count_WM_Ctx1 += len(filtered_rows)
                    print('currentFile processed')

                if currentFile['Spreadsheet'][0].split('_trials_')[0]  == 'WM_Ctx2': # info: Change to _3stim_trials_ from 8th month again
                    # percentCorrect_WM_Ctx2 += currentFile['Store: PercentCorrectWMCtx2'][len(currentFile['Store: PercentCorrectWMCtx2'])-3]
                    # count_WM_Ctx2 += 1

                    filtered_rows = currentFile[currentFile['Event Index'] == 124].copy()
                    percentCorrect_WM_Ctx2 += sum(filtered_rows['Store: PercentCorrectWMCtx2'])
                    count_WM_Ctx2 += len(filtered_rows)
                    print('currentFile processed')

    acc_DM = percentCorrect_DM/count_DM
    acc_DM_Anti = percentCorrect_DM_Anti/count_DM_Anti
    acc_EF = percentCorrect_EF/count_EF
    acc_EF_Anti = percentCorrect_EF_Anti/count_EF_Anti
    acc_WM = percentCorrect_WM/count_WM
    acc_WM_Anti = percentCorrect_WM_Anti/count_WM_Anti
    acc_WM_Ctx1 = percentCorrect_WM_Ctx1/count_WM_Ctx1
    acc_WM_Ctx2 = percentCorrect_WM_Ctx2/count_WM_Ctx2
    acc_RP = percentCorrect_RP/count_RP
    acc_RP_Anti = percentCorrect_RP_Anti/count_RP_Anti
    acc_RP_Ctx1 = percentCorrect_RP_Ctx1/count_RP_Ctx1
    acc_RP_Ctx2 = percentCorrect_RP_Ctx2/count_RP_Ctx2


if plotTrainingEffect:
    # Participant list
    participant_dir = r'W:\group_csp\analyses\oliver.frank\Data'
    # participant_dir = r'C:\Users\oliver.frank\Desktop\PyProjects\Data'
    months = [str(i) for i in range(1, 13)]
    strToSave = f"{months[0]}-{months[-1]}"
    newParticpantList = ['beRNN_01']

    filename_color_dict = {
        'DM': '#0d0a29', 'DM_Anti': '#271258',
        'EF': '#491078', 'EF_Anti': '#671b80',
        'RP': '#862781', 'RP_Anti': '#a6317d', 'RP_Ctx1': '#c53c74', 'RP_Ctx2': '#e34e65',
        'WM': '#f66c5c', 'WM_Anti': '#fc9065', 'WM_Ctx1': '#feb67c', 'WM_Ctx2': '#fdda9c'
    }

    for participant in newParticpantList:
        folder_paths = [os.path.join(participant_dir, participant, m) for m in months]
        all_files = []
        for folder in folder_paths:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".xlsx"):
                        all_files.append(os.path.join(root, file))

        fig, ax = plt.subplots(figsize=(11, 6))

        # Listen für globale X-Grenzen
        global_x_values = []

        for task, color in filename_color_dict.items():
            ycolumn = 'Store: PercentCorrect' + ''.join(task.split('_'))
            task_x = []
            task_y = []

            for filename in all_files:
                try:
                    df = pd.read_excel(filename, engine='openpyxl')
                    if not isinstance(df.iloc[0, 28], float) and df.iloc[0, 28].split('_trials_')[0] == task:
                        filtered_rows = df[df['Event Index'] == 125].copy()
                        if filtered_rows.empty:
                            continue

                        filtered_rows['Local Date and Time'] = pd.to_datetime(filtered_rows['Local Date and Time'],
                                                                              errors='coerce')
                        dates = pd.to_datetime(filtered_rows['Local Date and Time'].dt.date)

                        task_x.extend(dates)
                        task_y.extend(filtered_rows[ycolumn].tolist())
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

            if not task_x:
                continue

            sorted_pairs = sorted(zip(task_x, task_y))
            task_x, task_y = zip(*sorted_pairs)
            task_x = list(task_x)
            task_y = list(task_y)

            global_x_values.extend(task_x)

            ax.plot(task_x, task_y, color=color, alpha=0.4, linewidth=1.5, zorder=1)

            ax.scatter(task_x, task_y, color=color, alpha=0.8, edgecolors='none', s=35, zorder=2)

            avg_perf = np.mean(task_y)
            ax.plot([], [], color=color, label=f'{task} (Ø {avg_perf:.1f}%)', linestyle='-', marker='o')

        if not global_x_values:
            print(f"No data found for participant {participant}")
            continue

        ax.set_ylim(-5, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)], fontsize=10)
        ax.grid(axis='y', color='#e0e0e0', linestyle='--', linewidth=0.7)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        min_date = min(global_x_values)
        max_date = max(global_x_values)
        ax.set_xlim(min_date - pd.Timedelta(days=5), max_date + pd.Timedelta(days=5))

        ax.set_title(f"Performance Over Time — Participant: {participant}", fontsize=14, pad=15, weight='bold')
        ax.set_xlabel('Timeline', fontsize=11, labelpad=10)
        ax.set_ylabel('Accuracy', fontsize=11, labelpad=10)

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False, ncol=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')

        figure_path = os.path.join(participant_dir, participant, f"{participant}_{strToSave}_PerformanceOverTime.png")
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        plt.show()


if plotTrainingEffectAsBoxPlots:
    participant_dir = r'W:\group_csp\analyses\oliver.frank\Data'
    months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    strToSave = months[0] + '-' + months[-1]
    newParticpantList = ['beRNN_04']
    task_keys = [
        'DM', 'DM_Anti',
        'EF', 'EF_Anti',
        'RP', 'RP_Anti',
        'RP_Ctx1', 'RP_Ctx2',
        'WM', 'WM_Anti',
        'WM_Ctx1', 'WM_Ctx2'
    ]

    participants_dict = {
        participant: {month: {task: {} for task in task_keys} for month in months}
        for participant in newParticpantList
    }
    # with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_accuracy.pkl', 'rb') as f:
    #     participants_dict = pickle.load(f)

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(task_keys)))

    filename_color_dict = {
        task: mcolors.to_hex(color)
        for task, color in zip(task_keys, colors)
    }

    for participant in newParticpantList:
        fig, ax = plt.subplots(figsize=(10, 5))  # Wider figure to accommodate many side-by-side boxes

        # Layout parameters
        month_centers = np.arange(1, len(months) + 1)
        task_keys = list(filename_color_dict.keys())
        n_tasks = len(task_keys)
        box_width = 0.06  # Narrower width to fit 12 tasks
        # Calculate offsets for 12 tasks to be centered around the month tick
        offsets = np.linspace(-0.4, 0.4, n_tasks)

        legend_handles = []

        for m_idx, month_str in enumerate(months):
            month_center = month_centers[m_idx]
            folder = os.path.join(participant_dir, participant, month_str)

            # Collect all files for the month
            month_files = []
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".xlsx"):
                            month_files.append(os.path.join(root, file))

            for t_idx, task in enumerate(task_keys):
                color = filename_color_dict[task]
                ycolumn = 'Store: PercentCorrect' + ''.join(task.split('_'))
                task_data = []

                for filename in month_files:
                    try:
                        df = pd.read_excel(filename, engine='openpyxl')
                        # info. Adjustments made: Check this next week for months 7-12 *************************************
                        # info. Upload the already existing dicts for month 1-6 before *************************************
                        # Using provided validation logic
                        # if not isinstance(df.iloc[0, 28], float) and df['Spreadsheet'][0].split('_trials_')[0] == task:
                        if not isinstance(df.loc[0, 'Spreadsheet'], float) and df.loc[0, 'Spreadsheet'].split('_trials_')[0] == task:
                            filtered = df[df['Event Index'] == 125]
                            task_data.extend(filtered[ycolumn].dropna().tolist())
                    except:
                        continue

                # info. Save the accuracies for each participant, month and task, respectively
                participants_dict[participant][month_str][task] = task_data


                if task_data:
                    x_pos = month_center + offsets[t_idx]

                    # Plot side-by-side boxplot
                    bp = ax.boxplot(task_data, positions=[x_pos], widths=box_width,
                                    patch_artist=True, showmeans=True,
                                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": color,
                                               "markersize": 4},
                                    boxprops=dict(facecolor=color, color=color, alpha=0.5),
                                    medianprops={"color": "white"},
                                    showfliers=False)

                    # Add transparent jittered points next to/on each box
                    jitter = np.random.uniform(-box_width / 4, box_width / 4, size=len(task_data))
                    ax.scatter([x_pos] * len(task_data) + jitter, task_data,
                               color=color, alpha=0.3, s=8, zorder=3)

                    # Build legend handle once
                    if m_idx == 0:
                        legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=task))

        # Formatting
        ax.set_ylim(40, 105)
        ax.set_yticks(range(40, 101, 20))
        ax.set_yticklabels([f'{i}%' for i in range(40, 101, 20)])
        ax.grid(axis='y', color='lightgrey', linestyle='--', alpha=0.7)

        ax.set_xticks(month_centers)
        ax.set_xticklabels([f'Month {m}' for m in months], fontsize=12)
        ax.set_title(f"Task Performance Comparison - Participant: {participant}", fontsize=14)
        ax.set_ylabel("Accuracy (%)", fontsize=12)

        # Legend outside
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), title="Tasks")

        ax.set_xlim(month_centers[0] - 0.5, month_centers[-1] + 0.5)

        plt.tight_layout()
        figure_path = os.path.join(participant_dir, participant, f"{participant}_{strToSave}_SideBySide.pdf")
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        plt.show()

    # info. Save dict for LMM analysis
    import pickle
    # Save
    with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_accuracy.pkl', 'wb') as f:
        pickle.dump(participants_dict, f)


