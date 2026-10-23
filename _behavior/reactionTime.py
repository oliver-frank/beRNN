import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu
import matplotlib.colors as mcolors
import json

'''
- Visualizes reaction time of cognitive-behavioral tasks over months
- Create reaction time dicts for LMM analysis
- Visualize task complexity radar plots 
- Create task complexity dicts for LMM analysis
'''

# Configuration
participant_dir = r'W:\group_csp\analyses\oliver.frank\Data'
months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
strToSave = months[0] + '-' + months[-1]
newParticpantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_05'] # 'beRNN_04' : only first 7 months
reactionTime_comparison, plot_radars, wholePeriodPlots = True, True, False

paper_nomenclatur_dict = {
    'beRNN_03': 'HC1',
    'beRNN_04': 'HC2',
    'beRNN_01': 'MDD',
    'beRNN_02': 'ASD',
    'beRNN_05': 'SCZ',
    'beRNN_06': 'XX'}

# Task color dictionary
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
# with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_reactionTime.pkl', 'rb') as f:
#     participants_dict = pickle.load(f)

cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(task_keys)))

filename_color_dict = {
    task: mcolors.to_hex(color)
    for task, color in zip(task_keys, colors)
}



# info: ################################################################################################################
# info: Create reaction time box plots for seperated errors and corrects or all trials
# info: ################################################################################################################
if reactionTime_comparison == True:
    for participant in newParticpantList:

        fig_correct, ax_correct = plt.subplots(figsize=(10, 5))
        fig_error, ax_error = plt.subplots(figsize=(10, 5))
        fig_all, ax_all = plt.subplots(figsize=(10, 5))

        month_centers = np.arange(1, len(months) + 1)
        task_keys = list(filename_color_dict.keys())
        offsets = np.linspace(-0.4, 0.4, len(task_keys))
        box_width = 0.06

        legend_handles_correct = []
        legend_handles_error = []
        legend_handles_all = []
        seen_correct_tasks = set()
        seen_error_tasks = set()
        seen_all_tasks = set()

        stats_data = {task: {"correct": [], "error": [], "all": []} for task in task_keys}

        for m_idx, month_str in enumerate(months):

            # initialization of empirical average response epoch lengths with heuristic value
            averageResponseEpoch_corrects_dict = {
                'DM': 35,
                'DM_Anti': 35,
                'EF': 35,
                'EF_Anti': 35,
                'RP': 35,
                'RP_Anti': 35,
                'RP_Ctx1': 35,
                'RP_Ctx2': 35,
                'WM': 35,
                'WM_Anti': 35,
                'WM_Ctx1': 35,
                'WM_Ctx2': 35
            }

            month_center = month_centers[m_idx]
            folder = os.path.join(participant_dir, participant, month_str)

            month_files = []
            if os.path.exists(folder):
                for root, _, files in os.walk(folder):
                    month_files.extend(
                        os.path.join(root, f) for f in files if f.endswith(".xlsx")
                    )

            for t_idx, task in enumerate(task_keys):
                color = filename_color_dict[task]
                task_data_correct = []
                task_data_error = []
                task_data_all = []

                task_average_list = []

                for filename in month_files:

                    try:
                        df = pd.read_excel(filename, engine='openpyxl')

                        if (
                            isinstance(df.loc[0, 'Spreadsheet'], str)
                            and df.loc[0, 'Spreadsheet'].split('_trials_')[0] == task
                        ):
                            corr = (
                                df.loc[
                                    (df['Component Name'] == 'Click Response') &
                                    (df['Correct'] == 1),
                                    'Reaction Time'
                                ]
                                .dropna()
                                .tolist()
                            )

                            err = (
                                df.loc[
                                    (df['Component Name'] == 'Click Response') &
                                    (df['Correct'] == 0),
                                    'Reaction Time'
                                ]
                                .dropna()
                                .tolist()
                            )

                            all = (
                                df.loc[
                                    (df['Component Name'] == 'Click Response'),
                                    'Reaction Time'
                                ]
                                .dropna()
                                .tolist()
                            )

                            # Clean by min and max thresholds for defined reaction times - gorilla df saving errors added single values outside of range
                            if task == 'DM' or task == 'DM_Anti':
                                corr = [rt for rt in corr if rt <= 1700 and rt >= 300]
                                err = [rt for rt in err if rt <= 1700 and rt >= 300]
                                all = [rt for rt in all if rt <= 1700 and rt >= 300]
                            if task == 'EF' or task == 'EF_Anti':
                                corr = [rt for rt in corr if rt <= 900 and rt >= 300]
                                err = [rt for rt in err if rt <= 900 and rt >= 300]
                                all = [rt for rt in all if rt <= 900 and rt >= 300]
                            if task == 'RP' or task == 'RP_Anti' or task == 'RP_Ctx1' or task == 'RP_Ctx2':
                                corr = [rt for rt in corr if rt <= 1500 and rt >= 300]
                                err = [rt for rt in err if rt <= 1500 and rt >= 300]
                                all = [rt for rt in all if rt <= 1500 and rt >= 300]
                            if task == 'WM' or task == 'WM_Anti' or task == 'WM_Ctx1' or task == 'WM_Ctx2':
                                corr = [rt for rt in corr if rt <= 1500 and rt >= 300]
                                err = [rt for rt in err if rt <= 1500 and rt >= 300]
                                all = [rt for rt in all if rt <= 1500 and rt >= 300]

                            task_data_correct.extend(corr)
                            task_data_error.extend(err)
                            task_data_all.extend(all)

                            stats_data[task]["correct"].extend(corr)
                            stats_data[task]["error"].extend(err)
                            stats_data[task]["all"].extend(all)

                            # info. Divide all into batches and create average RT value for each batch to add it to the list
                            chunk_size = 40
                            num_chunks = len(all) // chunk_size

                            for i in range(num_chunks):
                                # Berechne Start- und Endindex für das 40er-Paket
                                start = i * chunk_size
                                end = (i + 1) * chunk_size

                                # Slice ziehen und Durchschnitt berechnen
                                chunk = all[start:end]
                                task_average_list.append(sum(chunk) / len(chunk))

                    except Exception as e:
                        print(f"Error in {filename}: {e}")

                # info. Save the accuracies for each participant, month and task, respectively
                participants_dict[participant][month_str][task] = task_average_list

                # # ---- Correct plots ----
                # if task_data_correct:
                #     x_pos = month_center + offsets[t_idx]
                #
                #     ax_correct.boxplot(
                #         task_data_correct,
                #         positions=[x_pos],
                #         widths=box_width,
                #         patch_artist=True,
                #         showmeans=True,
                #         showfliers=False,
                #         meanprops=dict(marker="D", markerfacecolor="white",
                #                        markeredgecolor=color, markersize=4),
                #         boxprops=dict(facecolor=color, color=color, alpha=0.5),
                #         medianprops=dict(color="white")
                #     )
                #
                #     jitter = np.random.uniform(-box_width / 4, box_width / 4,
                #                                size=len(task_data_correct))
                #     ax_correct.scatter(
                #         np.full(len(task_data_correct), x_pos) + jitter,
                #         task_data_correct,
                #         color=color, alpha=0.3, s=8
                #     )
                #
                #     if task not in seen_correct_tasks:
                #         legend_handles_correct.append(
                #             plt.Line2D([0], [0], color=color, lw=4, label=task)
                #         )
                #         seen_correct_tasks.add(task)
                #
                #
                # # ---- Error plots ----
                # if task_data_error:
                #     x_pos = month_center + offsets[t_idx]
                #
                #     ax_error.boxplot(
                #         task_data_error,
                #         positions=[x_pos],
                #         widths=box_width,
                #         patch_artist=True,
                #         showmeans=True,
                #         showfliers=False,
                #         meanprops=dict(marker="D", markerfacecolor="white",
                #                        markeredgecolor=color, markersize=4),
                #         boxprops=dict(facecolor=color, color=color, alpha=0.5),
                #         medianprops=dict(color="white")
                #     )
                #
                #     jitter = np.random.uniform(-box_width / 4, box_width / 4,
                #                                size=len(task_data_error))
                #     ax_error.scatter(
                #         np.full(len(task_data_error), x_pos) + jitter,
                #         task_data_error,
                #         color=color, alpha=0.3, s=8
                #     )
                #
                #     # ---- All plots ----
                #     if task not in seen_error_tasks:
                #         legend_handles_error.append(
                #             plt.Line2D([0], [0], color=color, lw=4, label=task)
                #         )
                #         seen_error_tasks.add(task)
                #
                #
                # # ---- Correct + Error plots ----
                # if task_data_all: # info. currently plotting batch averaged reaction times w. task_average_list - plot task_data_all else
                #     x_pos = month_center + offsets[t_idx]
                #
                #     ax_all.boxplot(
                #         task_average_list,
                #         positions=[x_pos],
                #         widths=box_width,
                #         patch_artist=True,
                #         showmeans=True,
                #         showfliers=False,
                #         meanprops=dict(marker="D", markerfacecolor="white",
                #                        markeredgecolor=color, markersize=4),
                #         boxprops=dict(facecolor=color, color=color, alpha=0.5),
                #         medianprops=dict(color="white")
                #     )
                #
                #     jitter = np.random.uniform(-box_width / 4, box_width / 4,
                #                                size=len(task_average_list))
                #     ax_all.scatter(
                #         np.full(len(task_average_list), x_pos) + jitter,
                #         task_average_list,
                #         color=color, alpha=0.3, s=8
                #     )
                #
                #     if task not in seen_all_tasks:
                #         legend_handles_all.append(
                #             plt.Line2D([0], [0], color=color, lw=4, label=task)
                #         )
                #         seen_all_tasks.add(task)
                #
                # # Log the current average reaction time of all correct trials for one particular month and participant
                # if task_data_correct:
                #     averageResponseEpoch_corrects_dict[task] = np.round(np.average(task_data_correct)/20) # neuronal time constant 20ms


            # # Save the RT dict for each month and participant, respectively
            # dir_correct_dicts = os.path.join(participant_dir, participant, 'averageResponseEpoch_corrects_dicts')
            # os.makedirs(dir_correct_dicts, exist_ok=True)
            # with open(os.path.join(dir_correct_dicts, f'averageResponseEpoch_corrects_dict_{participant}_{month_str}.pkl'), 'wb') as f:
            #     pickle.dump(averageResponseEpoch_corrects_dict, f)


        # # ---------- Formatting ----------
        # for ax in (ax_correct, ax_error, ax_all):
        #     ax.set_ylim(400, 1200)
        #     ax.set_yticks(range(400, 1001, 200))
        #     ax.grid(axis='y', linestyle='--', alpha=0.7)
        #     ax.set_xticks(month_centers)
        #     ax.set_xticklabels([f"Month {m}" for m in months])
        #     ax.set_xlim(month_centers[0] - 0.5, month_centers[-1] + 0.5)
        #     ax.set_ylabel("Reaction Time (ms)")
        #
        # ax_correct.set_title(f"Reaction Time – Correct ({participant})")
        # ax_error.set_title(f"Reaction Time – Errors ({participant})")
        # ax_all.set_title(f"Reaction Time – All ({participant})")
        #
        # ax_correct.legend(handles=legend_handles_correct,
        #                   loc='upper left', bbox_to_anchor=(1.02, 1))
        # ax_error.legend(handles=legend_handles_error,
        #                 loc='upper left', bbox_to_anchor=(1.02, 1))
        # ax_all.legend(handles=legend_handles_all,
        #                 loc='upper left', bbox_to_anchor=(1.02, 1))
        #
        # fig_correct.subplots_adjust(right=0.80)
        # fig_error.subplots_adjust(right=0.80)
        # fig_all.subplots_adjust(right=0.80)
        #
        # fig_correct.savefig(
        #     os.path.join(participant_dir, participant,
        #                  f"{participant}_{strToSave}_reactionTime_Corrects.pdf"),
        #     dpi=300
        # )
        # fig_error.savefig(
        #     os.path.join(participant_dir, participant,
        #                  f"{participant}_{strToSave}_reactionTime_Errors.pdf"),
        #     dpi=300
        # )
        # fig_all.savefig(
        #     os.path.join(participant_dir, participant,
        #                  f"{participant}_{strToSave}_reactionTime_All.pdf"),
        #     dpi=300
        # )
        #
        # plt.close(fig_correct)
        # plt.close(fig_error)
        # plt.close(fig_all)
        #
        # # ---------- Statistics ----------
        # print(f"\n{'=' * 95}")
        # print(f"STATISTICAL ANALYSIS: Correct vs. Error Reaction Times ({participant})")
        # print(f"{'=' * 95}")
        #
        # # Define thresholds
        # alpha_standard = 0.05
        # n_tests = len([t for t in stats_data if len(stats_data[t]["correct"]) >= 10 and len(stats_data[t]["error"]) >= 10])
        # # Protect against division by zero if no tasks have enough data
        # alpha_bonf = alpha_standard / max(n_tests, 1)
        #
        # print(
        #     f"Significance Thresholds: Standard Alpha = {alpha_standard:.2f} | Bonferroni Alpha (n={n_tests}) = {alpha_bonf:.4f}")
        # print(f"{'-' * 95}")
        # print(
        #     f"{'Task':<10} | {'n_corr':<7} | {'n_err':<7} | {'p-value':<10} | {'Effect (r)':<10} | {'Sig (0.05)':<10} | {'Sig (Bonf)':<10}")
        # print(f"{'-' * 95}")
        #
        # for task in task_keys:
        #     d_corr = stats_data[task]["correct"]
        #     d_err = stats_data[task]["error"]
        #     n1, n2 = len(d_corr), len(d_err)
        #
        #     if n1 >= 10 and n2 >= 10:
        #         # Perform Mann-Whitney U
        #         u_stat, p = mannwhitneyu(d_corr, d_err, alternative="two-sided")
        #
        #         # Calculate Effect Size r = |Z| / sqrt(N)
        #         # Standardizing U to Z
        #         mu_u = (n1 * n2) / 2
        #         sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        #         # Avoid division by zero in edge cases
        #         z = (u_stat - mu_u) / sigma_u if sigma_u != 0 else 0
        #         effect_size_r = abs(z) / np.sqrt(n1 + n2)
        #
        #         # Determine significance strings
        #         sig_std = "YES" if p < alpha_standard else "no"
        #         sig_bonf = "YES" if p < alpha_bonf else "no"
        #
        #         print(
        #             f"{task:<10} | {n1:<7} | {n2:<7} | {p:<10.2e} | {effect_size_r:<10.3f} | {sig_std:<10} | {sig_bonf:<10}")
        #     else:
        #         # Handle cases with insufficient data
        #         reason = f"n={n1}/{n2}"
        #         print(f"{task:<10} | {reason:<30} | Insufficient data for reliable test")
        #
        # print(f"{'=' * 95}\n")
        #
        #
        # # ---------- Statistics Visualization ----------
        # import matplotlib.pyplot as plt
        #
        # stats_rows = []
        #
        # for task in task_keys:
        #     d_corr = stats_data[task]["correct"]
        #     d_err = stats_data[task]["error"]
        #     n1, n2 = len(d_corr), len(d_err)
        #
        #     if n1 >= 10 and n2 >= 10:
        #         u_stat, p = mannwhitneyu(d_corr, d_err, alternative="two-sided")
        #
        #         mu_u = (n1 * n2) / 2
        #         sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        #         z = (u_stat - mu_u) / sigma_u if sigma_u != 0 else 0
        #         effect_r = abs(z) / np.sqrt(n1 + n2)
        #
        #         sig_std = "YES" if p < alpha_standard else "no"
        #         sig_bonf = "YES" if p < alpha_bonf else "no"
        #
        #         stats_rows.append([
        #             task,
        #             n1,
        #             n2,
        #             f"{p:.2e}",
        #             f"{effect_r:.3f}",
        #             sig_std,
        #             sig_bonf
        #         ])
        #     else:
        #         stats_rows.append([
        #             task,
        #             n1,
        #             n2,
        #             "n/a",
        #             "n/a",
        #             "n/a",
        #             "n/a"
        #         ])
        #
        # # Create figure
        # fig_stats, ax_stats = plt.subplots(figsize=(12, 0.6 * len(stats_rows) + 2))
        # ax_stats.axis("off")
        #
        # column_labels = [
        #     "Task",
        #     "n Correct",
        #     "n Error",
        #     "p-value",
        #     "Effect size (r)",
        #     "Sig (0.05)",
        #     "Sig (Bonf)"
        # ]
        #
        # table = ax_stats.table(
        #     cellText=stats_rows,
        #     colLabels=column_labels,
        #     loc="center",
        #     cellLoc="center"
        # )
        #
        # table.auto_set_font_size(False)
        # table.set_fontsize(9)
        # table.scale(1, 1.5)
        #
        # # Color significant cells
        # for row_idx, row in enumerate(stats_rows, start=1):
        #     if row[5] == "YES":
        #         table[row_idx, 5].set_facecolor("#c7e9c0")  # green
        #     if row[6] == "YES":
        #         table[row_idx, 6].set_facecolor("#a1d99b")  # darker green
        #
        # ax_stats.set_title(
        #     f"Reaction Time Statistics: Correct vs Error ({participant})\n"
        #     f"α = {alpha_standard} | Bonferroni α = {alpha_bonf:.4f}",
        #     fontsize=12,
        #     pad=20
        # )
        #
        # # Save alongside other PDFs
        # fig_stats.savefig(
        #     os.path.join(
        #         participant_dir,
        #         participant,
        #         f"{participant}_{strToSave}_reactionTime_Statistics.pdf"
        #     ),
        #     dpi=300,
        #     bbox_inches="tight"
        # )
        #
        # plt.close(fig_stats)
        #
        # # Create & Save dict
        # averageResponseEpoch_corrects_dict['DM'] = np.round(np.average(stats_data['DM']['correct'])/20)
        # averageResponseEpoch_corrects_dict['DM'] = np.round(np.average(stats_data['DM']['correct'])/20)

    # info. Save dict for python
    import pickle

    # Save
    with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_reactionTime.pkl', 'wb') as f:
        pickle.dump(participants_dict, f)

    # info. Save dict for python
    # Load
    with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_reactionTime.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    # info. Save dict for python
    import json
    # import numpy as np

    def default_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError

    with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_reactionTime.json', 'w') as f:
        json.dump(participants_dict, f, default=default_converter)

    from scipy.io import savemat

    savemat(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_reactionTime.mat', {"data": participants_dict})



# info: ################################################################################################################
# info: Create task complexity radar plots & complexities dict
# info: ################################################################################################################
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_beRNN_radar_months(beRNN_name, data, filename_color_dict, months):
    tasks = list(filename_color_dict.keys())
    n_tasks = len(tasks)

    # Compute angles for radar
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # close the polygon

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), subplot_kw=dict(polar=True))

    for ax, month in zip(axes.flatten(), months):
        values = np.array(data[month])
        values = np.concatenate([values, [values[0]]])  # close polygon

        # Plot polygon
        ax.plot(angles, values, linewidth=2, color="black")
        ax.fill(angles, values, color="gray", alpha=0.2)

        # Plot individual task points in viridis colors
        for angle, val, task in zip(angles[:-1], values[:-1], tasks):
            ax.plot(
                [angle, angle],
                [0, val],
                color=filename_color_dict[task],
                linewidth=4,
            )

        # Formatting
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, tasks, fontsize=9)
        ax.set_ylim(0, 6)
        # ax.set_yticks(range(1, 7))
        ax.set_yticklabels([])  # remove y-ticks
        ax.set_xticklabels([])  # remove y-ticks
        # ax.set_yticklabels([str(i) for i in range(1, 7)], fontsize=8)
        # ax.set_title(f"Month {month}", fontsize=36, y=-0.1)

    if len(months) < 12:
        for ax in axes.flatten()[len(months) :]:
            ax.axis("off")

    # fig.suptitle(paper_nomenclatur_dict[beRNN_name], fontsize=24)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            r"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__taskComplexities",
            f"{beRNN_name}_taskComplexity_month_{strToSave}.pdf",
        ),
        dpi=300,
    )


participants_dict_complexities = {
    participant: {month: {task: {} for task in task_keys} for month in months}
    for participant in newParticpantList
}

if plot_radars == True:
    # Manually standardized task complexity level for 12 tasks each month - fix: entry all months
    beRNN_01_month_1 = [4, 1, 2, 2, 2, 2 ,2 ,1 ,1, 1, 1, 1]
    beRNN_01_month_2 = [4, 1, 5, 5, 5, 5, 5, 1, 4 ,5, 1 ,4]
    beRNN_01_month_3 = [4, 1, 5, 5, 5, 5, 5, 1, 4, 5, 1, 4]
    beRNN_01_month_4 = [4,1,2,2,4,5,1,4,5,5,5,1]
    beRNN_01_month_5 = [4,1,4,4,4,5,1,4,5,5,5,1]
    beRNN_01_month_6 = [4,1,4,4,4,5,1,4,5,5,5,1]
    beRNN_01_month_7 = [4, 2, 6, 6, 5, 4, 5, 5, 4, 5, 1, 4]
    beRNN_01_month_8 = [4, 2, 6, 6, 4, 4, 5, 5, 5, 5, 1, 4]
    beRNN_01_month_9 = [4, 2, 6, 6, 4, 4, 5, 5, 4, 5, 2, 4]
    beRNN_01_month_10 = [4, 2, 6, 6, 4, 4, 4, 5, 4, 5, 2, 5]
    beRNN_01_month_11 = [4, 2, 6, 6, 4, 4, 5, 5, 4, 5, 2, 5]
    beRNN_01_month_12 = [4, 2, 6, 6, 5, 4, 5, 5, 4, 5, 2, 5]

    beRNN_02_month_1 = [4, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1]
    beRNN_02_month_2 = [4, 1, 2, 5, 5, 5, 5, 2, 2, 2, 1, 1]
    beRNN_02_month_3 = [4, 4, 2, 6, 5, 5, 5, 2, 2, 2, 1, 1]
    beRNN_02_month_4 = [4,4,2,2,1,2,1,1,5,5,5,2]
    beRNN_02_month_5 = [4,4,4,4,2,2,1,1,5,5,5,2]
    beRNN_02_month_6 = [4,4,4,4,4,4,1,1,5,5,5,2]
    beRNN_02_month_7 = [4, 4, 6, 6, 4, 4, 5, 4, 4, 4, 4, 5]
    beRNN_02_month_8 = [4, 4, 6, 6, 4, 4, 6, 4, 4, 5, 5, 5]
    beRNN_02_month_9 = [5, 4, 6, 6, 4, 4, 6, 5, 4, 5, 5, 5]
    beRNN_02_month_10 = [5, 4, 6, 6, 4, 4, 6, 5, 4, 5, 5, 5]
    beRNN_02_month_11 = [5, 4, 6, 6, 4, 4, 6, 5, 4, 5, 5, 5]
    beRNN_02_month_12 = [5, 4, 6, 6, 4, 4, 6, 5, 4, 5, 5, 5]

    beRNN_03_month_1 = [4, 1, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    beRNN_03_month_2 = [4, 1, 5, 5, 5, 5, 5, 5, 5, 4, 1, 2]
    beRNN_03_month_3 = [4, 1, 6, 6, 5, 6, 5, 5, 4, 4, 1, 2]
    beRNN_03_month_4 = [4,1,2,2,4,5,1,2,5,5,5,5]
    beRNN_03_month_5 = [4,1,4,4,4,5,1,2,5,5,5,5]
    beRNN_03_month_6 = [4,1,4,4,4,5,1,2,5,5,5,5]
    beRNN_03_month_7 = [4, 2, 6, 6, 4, 4, 5, 4, 4, 5, 5, 4]
    beRNN_03_month_8 = [4, 2, 6, 6, 4, 4, 5, 5, 4, 5, 5, 5]
    beRNN_03_month_9 = [4, 2, 6, 6, 4, 4, 5, 5, 4, 5, 4, 5]
    beRNN_03_month_10 = [4, 2, 6, 6, 4, 4, 5, 5, 4, 5, 4, 5]
    beRNN_03_month_11 = [4, 2, 6, 6, 4, 4, 6, 5, 4, 5, 4, 5]
    beRNN_03_month_12 = [4, 2, 6, 6, 4, 2, 6, 5, 4, 5, 4, 5]

    beRNN_04_month_1 = [4, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1]
    beRNN_04_month_2 = [4, 1, 5, 2, 5, 4, 5, 5, 4, 4, 1, 2]
    beRNN_04_month_3 = [4, 1, 5, 2, 6, 4, 5, 5, 4, 4, 1, 2]
    beRNN_04_month_4 = [4,1,5,2,4,5,1,2,5,5,5,5]
    beRNN_04_month_5 = [4,1,5,2,4,5,1,2,5,5,5,5]
    beRNN_04_month_6 = [4,1,5,5,4,5,1,2,5,5,5,5]
    beRNN_04_month_7 = [4, 6, 5, 5, 5, 5, 5, 5, 4, 5, 1, 2]
    beRNN_04_month_8 = []
    beRNN_04_month_9 = []
    beRNN_04_month_10 = []
    beRNN_04_month_11 = []
    beRNN_04_month_12 = []

    beRNN_05_month_1 = [4, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1]
    beRNN_05_month_2 = [4, 1, 5, 2, 5, 2, 2, 2, 2, 2, 1, 1]
    beRNN_05_month_3 = [4, 1, 5, 2, 6, 2, 2, 2, 2, 2, 1, 2]
    beRNN_05_month_4 = [4,1,2,2,4,4,1,4,6,5,2,2]
    beRNN_05_month_5 = [4,2,4,4,4,4,1,4,5,5,2,2]
    beRNN_05_month_6 = [4,2,4,4,4,4,1,4,5,5,2,4]
    beRNN_05_month_7 = [5, 2, 6, 6, 5, 5, 5, 5, 4, 5, 1, 4]
    beRNN_05_month_8 = [5, 2, 6, 6, 4, 4, 5, 5, 5, 5, 4, 5]
    beRNN_05_month_9 = [5, 2, 6, 6, 4, 4, 6, 6, 5, 5, 4, 5]
    beRNN_05_month_10 = [5, 2, 6, 6, 4, 4, 6, 6, 5, 5, 5, 5]
    beRNN_05_month_11 = [5, 2, 6, 6, 4, 4, 6, 6, 5, 5, 5, 5]
    beRNN_05_month_12 = [5, 2, 6, 6, 4, 4, 6, 6, 5, 5, 5, 5]

    beRNN_06_month_3 = [1,1,1,1,1,1,1,1,1,1,1,1] # for legend visualization
    beRNN_06_month_4 = [6,6,6,6,6,6,6,6,6,6,6,6] # for legend visualization

    beRNN_data = {
        "beRNN_01": {
            "1": beRNN_01_month_1,
            "2": beRNN_01_month_2,
            "3": beRNN_01_month_3,
            "4": beRNN_01_month_4,
            "5": beRNN_01_month_5,
            "6": beRNN_01_month_6,
            "7": beRNN_01_month_7,
            "8": beRNN_01_month_8,
            "9": beRNN_01_month_9,
            "10": beRNN_01_month_10,
            "11": beRNN_01_month_11,
            "12": beRNN_01_month_12,
        },
        "beRNN_02": {
            "1": beRNN_02_month_1,
            "2": beRNN_02_month_2,
            "3": beRNN_02_month_3,
            "4": beRNN_02_month_4,
            "5": beRNN_02_month_5,
            "6": beRNN_02_month_6,
            "7": beRNN_02_month_7,
            "8": beRNN_02_month_8,
            "9": beRNN_02_month_9,
            "10": beRNN_02_month_10,
            "11": beRNN_02_month_11,
            "12": beRNN_02_month_12,
        },
        "beRNN_03": {
            "1": beRNN_03_month_1,
            "2": beRNN_03_month_2,
            "3": beRNN_03_month_3,
            "4": beRNN_03_month_4,
            "5": beRNN_03_month_5,
            "6": beRNN_03_month_6,
            "7": beRNN_03_month_7,
            "8": beRNN_03_month_8,
            "9": beRNN_03_month_9,
            "10": beRNN_03_month_10,
            "11": beRNN_03_month_11,
            "12": beRNN_03_month_12,
        },
        "beRNN_04": {
            "1": beRNN_04_month_1,
            "2": beRNN_04_month_2,
            "3": beRNN_04_month_3,
            "4": beRNN_04_month_4,
            "5": beRNN_04_month_5,
            "6": beRNN_04_month_6,
            "7": beRNN_04_month_7,
            "8": beRNN_04_month_8,
            "9": beRNN_04_month_9,
            "10": beRNN_04_month_10,
            "11": beRNN_04_month_11,
            "12": beRNN_04_month_12,
        },
        "beRNN_05": {
            "1": beRNN_05_month_1,
            "2": beRNN_05_month_2,
            "3": beRNN_05_month_3,
            "4": beRNN_05_month_4,
            "5": beRNN_05_month_5,
            "6": beRNN_05_month_6,
            "7": beRNN_05_month_7,
            "8": beRNN_05_month_8,
            "9": beRNN_05_month_9,
            "10": beRNN_05_month_10,
            "11": beRNN_05_month_11,
            "12": beRNN_05_month_12,
        }
        ,
        "beRNN_06": {
            "1": beRNN_01_month_1,
            "2": beRNN_01_month_2,
            "3": beRNN_06_month_3,
            "4": beRNN_06_month_4,
            "5": beRNN_01_month_5,
            "6": beRNN_01_month_6,
            "7": beRNN_01_month_7,
            "8": beRNN_01_month_8,
            "9": beRNN_01_month_9,
            "10": beRNN_01_month_10,
            "11": beRNN_01_month_11,
            "12": beRNN_01_month_12,
        },
    }

    for participant in newParticpantList:
        plot_beRNN_radar_months(participant, beRNN_data[participant], filename_color_dict, months)

    # structured dict of all lists
    complexity_data = {
        "beRNN_01": [beRNN_01_month_1, beRNN_01_month_2, beRNN_01_month_3, beRNN_01_month_4, beRNN_01_month_5,
                     beRNN_01_month_6, beRNN_01_month_7, beRNN_01_month_8, beRNN_01_month_9, beRNN_01_month_10,
                     beRNN_01_month_11, beRNN_01_month_12],
        "beRNN_02": [beRNN_02_month_1, beRNN_02_month_2, beRNN_02_month_3, beRNN_02_month_4, beRNN_02_month_5,
                     beRNN_02_month_6, beRNN_02_month_7, beRNN_02_month_8, beRNN_02_month_9, beRNN_02_month_10,
                     beRNN_02_month_11, beRNN_02_month_12],
        "beRNN_03": [beRNN_03_month_1, beRNN_03_month_2, beRNN_03_month_3, beRNN_03_month_4, beRNN_03_month_5,
                     beRNN_03_month_6, beRNN_03_month_7, beRNN_03_month_8, beRNN_03_month_9, beRNN_03_month_10,
                     beRNN_03_month_11, beRNN_03_month_12],
        "beRNN_04": [beRNN_04_month_1, beRNN_04_month_2, beRNN_04_month_3, beRNN_04_month_4, beRNN_04_month_5,
                     beRNN_04_month_6, beRNN_04_month_7, beRNN_04_month_8, beRNN_04_month_9, beRNN_04_month_10,
                     beRNN_04_month_11, beRNN_04_month_12],
        "beRNN_05": [beRNN_05_month_1, beRNN_05_month_2, beRNN_05_month_3, beRNN_05_month_4, beRNN_05_month_5,
                     beRNN_05_month_6, beRNN_05_month_7, beRNN_05_month_8, beRNN_05_month_9, beRNN_05_month_10,
                     beRNN_05_month_11, beRNN_05_month_12],
    }

    # Befüllen des Haupt-Dicts
    for p_id, month_lists in complexity_data.items():
        if p_id in participants_dict_complexities:
            for m_idx, month_values in enumerate(month_lists):
                # Monat bestimmen (vorausgesetzt months ist z.B. [1, 2, 3...])
                month_key = months[m_idx]

                # Check, ob die Liste für den Monat leer ist (wie bei beRNN_04)
                if not month_values:
                    continue

                # Werte den Tasks zuordnen
                for t_idx, task_key in enumerate(task_keys):
                    if t_idx < len(month_values):
                        participants_dict_complexities[p_id][month_key][task_key] = month_values[t_idx]

    # Save
    with open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_complexities.pkl', 'wb') as f:
        pickle.dump(participants_dict_complexities, f)



# info: ################################################################################################################
# info: Create reaction time box plots for seperated errors and corrects or all trials
# info: ################################################################################################################

# if reactionTime_comparison == True:
#
#     data = json.load(open(r'W:\group_csp\analyses\oliver.frank\Data\participants_dict_reactionTime.json'))
#
#     first_p = list(data.keys())[0]
#     first_m = list(data[first_p].keys())[0]
#     task_keys = sorted(data[first_p][first_m].keys())
#
#     # color mapping
#     cmap = plt.cm.viridis
#     colors = cmap(np.linspace(0, 1, len(task_keys)))
#     filename_color_dict = {
#         task: mcolors.to_hex(color)
#         for task, color in zip(task_keys, colors)
#     }
#
#     participants = list(data.keys())
#
#     for participant, months_data in data.items():
#         print()
#         plt.figure(figsize=(12, 7))
#
#         sorted_months_str = sorted(months_data.keys(), key=int)
#         months_numeric = [int(m) for m in sorted_months_str]
#
#         for task in task_keys:
#             means = []
#             stds = []
#             valid_months = []
#
#             for m_str in sorted_months_str:
#                 if task in months_data[m_str]:
#                     trials = np.array(months_data[m_str][task])
#                     means.append(np.mean(trials))
#                     stds.append(np.std(trials))
#                     valid_months.append(int(m_str))
#
#             means = np.array(means)
#             stds = np.array(stds)
#             color = filename_color_dict[task]
#
#             plt.plot(valid_months, means, label=task, color=color,
#                      marker='o', markersize=6, linewidth=3, zorder=3)
#
#             # info: Margin for variances - RT and performance only
#             plt.fill_between(valid_months, means - stds, means + stds,
#                              color=color, alpha=0.1, zorder=2)
#
#         plt.ylim(400, 1000,1200)
#         plt.yticks([400, 600, 800, 1000, 1200])
#         plt.xlim(min(months_numeric), max(months_numeric))
#         plt.xticks(months_numeric)
#         # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) # info: for task complexity plots
#
#         # Graue gepunktete Linien entlang der Ticks
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#
#         plt.title(f"Reaction Time: {participant}", fontsize=14, pad=15)
#         plt.xlabel("Month", fontsize=12)
#         plt.ylabel("Reaction Time", fontsize=12)
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
#
#         plt.tight_layout()
#         plt.savefig(rf'W:\group_csp\analyses\oliver.frank\Data\reactionTime_{participant}_1-12.png', dpi=300, bbox_inches='tight')
#
#         plt.show()


