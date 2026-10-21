import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pandas as pd
import os

from _training import apply_density_threshold
from _tools import load_hp, load_pickle

'''
- Compares functional correlation matrices on RNN/RNN or Brain/Brain or RNN/Brain
- Executes fingerprints and/or procrusting comparison 
- Preprocess average brain matrices over tasks 
'''

fingerprinting, procrusting = True, True

setup = {
    'participants': ['sub-6IECX', 'sub-DKHPB', 'sub-KPB84', 'sub-YL4AS', 'sub-96WID'], # order for paper
    'folder_brain': r'W:\group_csp\analyses\oliver.frank\_brainModels',

    'participants_beRNN': ['beRNN_03', 'beRNN_04', 'beRNN_01', 'beRNN_02', 'beRNN_05'], # order for paper
    'folder_beRNN': fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_comparison_multiTask_beRNN_01_highDim_256_hp_9_month__1-12',
    # 'folder_beRNN': fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_comparison_multiTask_beRNN_01_highDimCorrects_256_hp_9_month__1-12',

    'subjects': ['HC1', 'HC2', 'MDD', 'ASS', 'SCZ'],
    'block_sizes': [5, 3, 5, 5, 5],
    'dataType': 'highDim',
    # 'dataType': 'highDim_correctOnly',
    # 'threshold': 1.0,
    # 'thresholds': [1.0]
    'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'significanceThreshold': 0.005,
    'permutateANNs': True,
    'createBrainAverages': True
}

beRNN_brain_dict = {
    'sub-KPB84': 'beRNN_01',
    'sub-YL4AS': 'beRNN_02',
    'sub-6IECX': 'beRNN_03',
    'sub-DKHPB': 'beRNN_04',
    'sub-96WID': 'beRNN_05'
}

########################################################################################################################
# head: Preprocessing
########################################################################################################################
# info: Brain correlation matrix upload
brain_correlationMatrix_list = []

for brain in setup['participants']:

    if beRNN_brain_dict[brain] == 'beRNN_04':
        numberOfModels = 3
    else:
        numberOfModels = 5

    for model in range(1, numberOfModels + 1):

        brain_directory = fr'{setup["folder_brain"]}\functional_matrices\{brain}_ses-0{model}-avg.npy' # -avg vs. _avg
        matrix_brain = np.load(brain_directory)

        brain_correlationMatrix_list.append(matrix_brain)

# info: beRNN correlation matrix upload
beRNN_correlationMatrix_list = []

for participant in setup['participants_beRNN']:

    if participant == 'beRNN_04':
        numberOfModels = 3
    else:
        numberOfModels = 5

    folder_beRNN = setup["folder_beRNN"].replace("beRNN_01", participant, 1) # fix: for general you will need 5 times the same folder (beRNN_03 only)
    batch = '1'
    _model_folder = rf'{folder_beRNN}\{setup["dataType"]}\{participant}\{batch}'
    folder = os.listdir(_model_folder)[0]
    model_folder = rf'{folder_beRNN}\{setup["dataType"]}\{participant}\{batch}\{folder}'

    for model in os.listdir(model_folder)[:numberOfModels]:

        # elif 'multiTask' in model_folder or 'AllTask' in model_folder:
        # pkl_beRNN3 = rf'{model_folder}\{model}\var_test_lay1_rule_all.pkl'
        pkl_beRNN2 = rf'{model_folder}\{model}\corr_test_lay1_rule_all.pkl'

        # info. legacy variance filtering - can still be applied for top. Marker analysis
        # h_var_all as basis for thresholding dead neurons as h_corr_all can result in high values for dead neurons
        # res3 = load_pickle(pkl_beRNN3)
        # h_var_all_ = res3['h_var_all']
        # ind_dead = np.where(h_var_all_.sum(axis=1) < 0.01)[0] # heuristic threshold

        # h_corr_all as representative for modularity _analysis reflecting similar neuron behavior
        res2 = load_pickle(pkl_beRNN2)
        h_corr_all_ = res2['h_corr_all']
        h_corr_all = h_corr_all_.mean(axis=2)  # average over all tasks

        hp = load_hp(rf'{model_folder}\{model}')
        numberOfHiddenUnits = hp['n_rnn']

        # info. legacy variance filtering - can still be applied for top. Marker analysis
        # if ind_dead.shape[0] < h_corr_all_.shape[0] and ind_dead.shape[0] > 1:
        #     # set the dead neurons to 0 for denoising correlation matrix of meaningless high correlations
        #     h_corr_all_[ind_dead, :] = 0
        #     h_corr_all_[:, ind_dead] = 0
        #
        #   # # Apply threshold
        #   # functionalCorrelation_density = apply_density_threshold(h_corr_all_, density=setup['threshold'])
        # else:
        #     functionalCorrelation_density = np.zeros((numberOfHiddenUnits, numberOfHiddenUnits))  # fix: Get individual number of hidden units # Create different dummy matrix, that leads to lower realtive count

        # Compute modularity
        # np.fill_diagonal(functionalCorrelation_density, 0)  # prevent self-loops

        # functionalCorrelation_density = apply_density_threshold(h_corr_all, density=setup['threshold'])

        beRNN_correlationMatrix_list.append(h_corr_all)

diagonal_off = True
# attention. comparing same network class to itself
# brain_correlationMatrix_list = beRNN_correlationMatrix_list
beRNN_correlationMatrix_list = brain_correlationMatrix_list

if setup['permutateANNs']:
    # info. Apply permutation on ANNs
    def get_permutation_template(all_ann_matrices, all_bnn_matrices):
        """
        all_ann_matrices: Liste von 5 Matrizen (256x256)
        all_bnn_matrices: Liste von 5 Matrizen (256x256)
        """

        # Fisher-Z Transformation vor der Mittelung
        ann_z = [np.arctanh(m) for m in all_ann_matrices]
        bnn_z = [np.arctanh(m) for m in all_bnn_matrices]

        # Durchschnitt bilden
        ann_avg = np.tanh(np.mean(ann_z, axis=0))
        bnn_avg = np.tanh(np.mean(bnn_z, axis=0))

        # Kostenmatrix berechnen:
        # Wie unähnlich ist das Konnektivitätsprofil von ANN-Unit i
        # zum Profil von BNN-Parcel j?
        cost_matrix = cdist(ann_avg, bnn_avg, metric='correlation')

        # import hashlib
        # print(f"Cost Matrix Hash: {hashlib.sha256(cost_matrix.tobytes()).hexdigest()}")

        # Optimales Assignment (Ungarischer Algorithmus)
        # ann_idx wird einfach [0, 1, 2... 255] sein
        # bnn_idx enthält die zugeordnete Reihenfolge
        ann_idx, bnn_permuted_indices = linear_sum_assignment(cost_matrix)

        return bnn_permuted_indices

    new_order = get_permutation_template(beRNN_correlationMatrix_list, brain_correlationMatrix_list)

    ann_aligned_list = []
    for ann_mat in beRNN_correlationMatrix_list:
        aligned_mat = ann_mat[new_order, :][:, new_order]
        ann_aligned_list.append(aligned_mat)

    beRNN_correlationMatrix_list = ann_aligned_list

# info. Apply thresholding
network = 'Complete'
methods = ['Original', 'Procrustes']
p_values_results = np.zeros((len(setup['thresholds']), 2))

for t_idx, threshold in enumerate(setup['thresholds']):
    brain_correlationMatrix_list_thresholded = []
    beRNN_correlationMatrix_list_thresholded = []

    setup['threshold'] = threshold

    for brain_corr_matrix in brain_correlationMatrix_list:
        averaged_brain_matrix_thresholded = apply_density_threshold(brain_corr_matrix, setup['threshold'])
        np.fill_diagonal(averaged_brain_matrix_thresholded, 0)  # prevent self-loops
        brain_correlationMatrix_list_thresholded.append(averaged_brain_matrix_thresholded)

    for beRNN_corr_matrix in beRNN_correlationMatrix_list:
        averaged_beRNN_matrix_thresholded = apply_density_threshold(beRNN_corr_matrix, density=setup['threshold'])
        np.fill_diagonal(averaged_beRNN_matrix_thresholded, 0)  # prevent self-loops
        beRNN_correlationMatrix_list_thresholded.append(averaged_beRNN_matrix_thresholded)

    ####################################################################################################################
    # head: Apply fingerptinting methods
    ####################################################################################################################
    if fingerprinting == True:
        # info: analysis Fingerprinting ********************************************************************************
        corr_score_list = []
        def apply_original_fingerprinting(m1, m2):
            triu_indices = np.triu_indices(m1.shape[0], k=1)

            # Vektorisieren
            vec1 = m1[triu_indices]
            vec2 = m2[triu_indices]

            # Korrelieren (Pearson)
            correlation, p_value = pearsonr(vec1, vec2)

            return correlation

        num_subjects = len(beRNN_correlationMatrix_list_thresholded)
        ident_matrix = np.zeros((num_subjects, num_subjects))

        for i in range(num_subjects):
            for j in range(num_subjects):
                ident_matrix[i, j] = apply_original_fingerprinting(
                    brain_correlationMatrix_list_thresholded[i],
                    beRNN_correlationMatrix_list_thresholded[j]
                )

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(ident_matrix,
                         annot=False,       # correlation within cells
                         fmt=".2f",
                         cmap="Greys",
                         xticklabels=False,
                         yticklabels=False,
                         vmin=0,
                         vmax=0.75)

        current_idx = 0

        for i, size in enumerate(setup['block_sizes']):
            # Rectangle around subject specific models
            ax.add_patch(plt.Rectangle((current_idx, current_idx), size, size,
                                       fill=False,
                                       edgecolor='black',
                                       lw=4,
                                       ls='-'))

            plt.text(current_idx + size / 2, current_idx + size / 2, setup['subjects'][i],
                     color='black', ha='center', va='center', weight='bold', fontsize=12)

            current_idx += size

        plt.title(f"Fingerprinting Original ({threshold})", fontsize=15)
        plt.xlabel("Biological brain", fontsize=12)
        plt.ylabel("Artificial Neural Network", fontsize=12)


        within_mask = np.zeros((num_subjects, num_subjects), dtype=bool)
        diag_mask = np.eye(num_subjects, dtype=bool)
        start_idx = 0

        for size in setup['block_sizes']:
            end_idx = start_idx + size
            within_mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

        if diagonal_off:
            within_mask = within_mask & ~diag_mask
        within_mean = np.mean(ident_matrix[within_mask])

        between_mean = np.mean(ident_matrix[~within_mask])

        i_diff = (within_mean - between_mean) * 100

        # permutationtest
        n_permutations = 1000
        perm_i_diffs = []

        for _ in range(n_permutations):
            perm_indices = np.random.permutation(num_subjects)
            perm_matrix = ident_matrix[:, perm_indices]

            p_within = np.mean(perm_matrix[within_mask])
            p_between = np.mean(perm_matrix[~within_mask])
            perm_i_diffs.append((p_within - p_between) * 100)

        p_val_perm = np.sum(np.array(perm_i_diffs) >= i_diff) / n_permutations


        sig_status = "SIGNIFIKANT" if p_val_perm < setup['significanceThreshold'] else "NICHT SIGNIFIKANT" # info. bonferroni corrected: 0.05/10 = 0.005 - alternatively holm

        text_str = (f"$I_{{diff}}$: {i_diff:.2f}%\n"
                    f"$p$-Wert: {p_val_perm:.4f}\n"
                    f"{sig_status}")

        plt.text(0.4, -0.05, text_str, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        p_values_results[t_idx, 0] = p_val_perm
        plt.show()

    if procrusting == True:
        # info: analysis Procrustes ************************************************************************************
        def apply_procrustes_fingerprinting(m1, m2):
            try:
                m1_norm = (m1 - np.mean(m1)) / np.std(m1) if np.std(m1) > 0 else m1
                m2_norm = (m2 - np.mean(m2)) / np.std(m2) if np.std(m2) > 0 else m2
                mt1, mt2, disparity = procrustes(m1_norm, m2_norm)
                similarity = 1 - disparity
            except ValueError:
                similarity = 0
            return similarity

        num_subjects_total = len(beRNN_correlationMatrix_list_thresholded)
        ident_matrix_proc = np.zeros((num_subjects_total, num_subjects_total))

        for i in range(num_subjects_total):
            for j in range(num_subjects_total):
                ident_matrix_proc[i, j] = apply_procrustes_fingerprinting(
                    brain_correlationMatrix_list_thresholded[i],
                    beRNN_correlationMatrix_list_thresholded[j]
                )


        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(ident_matrix_proc,
                         annot=False,
                         fmt=".2f",
                         cmap="Greys",
                         xticklabels=False,
                         yticklabels=False,
                         vmin=0,
                         vmax=0.75)

        current_idx = 0
        for i, size in enumerate(setup['block_sizes']):
            ax.add_patch(plt.Rectangle((current_idx, current_idx), size, size,
                                       fill=False,
                                       edgecolor='black',
                                       lw=4,
                                       ls='-'))
            plt.text(current_idx + size / 2, current_idx + size / 2, setup['subjects'][i],
                     color='black', ha='center', va='center', weight='bold', fontsize=12)
            current_idx += size


        within_mask = np.zeros((num_subjects_total, num_subjects_total), dtype=bool)
        diag_mask = np.eye(num_subjects, dtype=bool)
        start_idx = 0

        for size in setup['block_sizes']:
            end_idx = start_idx + size
            within_mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

        if diagonal_off:
            within_mask = within_mask & ~diag_mask
        within_mean = np.mean(ident_matrix_proc[within_mask])

        between_mean = np.mean(ident_matrix_proc[~within_mask])

        i_diff = (within_mean - between_mean) * 100

        # permutation test
        n_permutations = 1000
        perm_i_diffs = []

        for _ in range(n_permutations):
            perm_indices = np.random.permutation(num_subjects_total)
            perm_matrix = ident_matrix_proc[:, perm_indices]

            p_within = np.mean(perm_matrix[within_mask])
            p_between = np.mean(perm_matrix[~within_mask])
            perm_i_diffs.append((p_within - p_between) * 100)

        p_val_perm = np.sum(np.array(perm_i_diffs) >= i_diff) / n_permutations


        sig_status = "SIGNIFIKANT" if p_val_perm < setup['significanceThreshold'] else "NICHT SIGNIFIKANT" # info. bonferroni corrected: 0.05/10 = 0.005 - alternatively holm

        text_str = (f"$I_{{diff}}$: {i_diff:.2f}%\n"
                    f"$p$-Wert: {p_val_perm:.4f}\n"
                    f"{sig_status}")

        plt.text(0.4, -0.05, text_str, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.title(f"Fingerprinting Procrustes ({threshold})", fontsize=15)
        plt.xlabel("Biological brain", fontsize=12)
        plt.ylabel("Artificial Neural Network", fontsize=12)

        p_values_results[t_idx, 1] = p_val_perm
        plt.show()

####################################################################################################################
# head: Visualize results
####################################################################################################################
plt.figure(figsize=(8, 6))
# (p < 0.05)
sns.heatmap(p_values_results, annot=True, fmt=".4f",
            xticklabels=methods, yticklabels=setup['thresholds'],
            vmin=0, vmax=1,
            cmap="viridis_r", cbar_kws={'label': 'p-value'})

# highlight significant cells
for y in range(p_values_results.shape[0]):
    for x in range(p_values_results.shape[1]):
        if p_values_results[y, x] < setup['significanceThreshold']:
            plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2))

plt.xlabel("Fingerprint Method")
plt.ylabel("Density Threshold")

plt.savefig(os.path.join(setup['folder_brain'], 'ANN_BNN_comparisons', f'ANN_BNN_comparisons_{network}_{setup["dataType"]}.png'))

plt.show()

if setup['createBrainAverages']:
    # ###################################################################################################################
    # head: Preprocess brain averages over tasks
    # ###################################################################################################################
    directory = r'W:\group_csp\analyses\oliver.frank\_brainModels\functional_matrices'
    atlas_path = r'W:\group_csp\in_house_datasets\bernn\mri\derivatives\xcpd_0.11.1\atlases\atlas-4S456Parcels/atlas-4S456Parcels_dseg.tsv'
    subjects = ['sub-YL4AS', 'sub-DKHPB', 'sub-KPB84', 'sub-96WID', 'sub-6IECX']
    subnetwork = ['Logic', 'HigherCognition', 'External', 'IndividualStability', 'LH', 'RH'][3]

    for subject in subjects:
        if subject == 'sub-DKHPB':
            sessions = ['01','02','03']
        else:
            sessions = ['01','02','03', '04', '05']

        # ---------------------------------------------------------
        atlas_info = pd.read_csv(atlas_path, sep='\t')

        # info. 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'LH', 'RH', 'Cerebellar'
        # excluded_networks = ['Vis', 'Default'] # 'Logical' - 48 last
        # excluded_networks = ['Vis', 'SomMot', 'DorsAttn'] # 'Higher Cognition' - 16 last
        # excluded_networks = ['SalVentAttn', 'Default'] # 'External' - 62 last
        excluded_networks = ['DorsAttn', 'SalVentAttn', 'Limbic', 'Default'] # 'Individual Stability' + 10 missing
        # excluded_networks = ['RH'] # 'LH' + 23 missing
        # excluded_networks = ['LH'] # 'RH' + 23 missing
        mask_exclude = atlas_info['label'].str.contains('|'.join(excluded_networks))
        cognitive_indices = atlas_info[~mask_exclude].index.tolist()

        # print(f"Anzahl der Regionen nach gezieltem Ausschluss: {len(cognitive_indices)}")

        if len(cognitive_indices) > 256:
            diff = len(cognitive_indices) - 256
            print(f"Entferne die letzten {diff} Regionen (meist unspezifische subkortikale Gebiete)...")
            cognitive_indices = cognitive_indices[:256]
        if len(cognitive_indices) < 256:
            diff = 256 - len(cognitive_indices)
            print(f"Auffüllen: Es fehlen {diff} Regionen für die Zielgröße 256.")

            remaining_pool = [i for i in atlas_info.index if i not in cognitive_indices]
            extra_indices = remaining_pool[:diff]
            cognitive_indices.extend(extra_indices)

            cognitive_indices.sort()

        # --------------------------------------------------------

        for session in sessions:
            faces = np.load(os.path.join(directory, f'{subject}_ses-{session}-task-faces-atlas-4S456Parcels.npy'))
            flanker = np.load(os.path.join(directory, f'{subject}_ses-{session}-task-flanker-atlas-4S456Parcels.npy'))
            nback = np.load(os.path.join(directory, f'{subject}_ses-{session}-task-nback-atlas-4S456Parcels.npy'))
            reward = np.load(os.path.join(directory, f'{subject}_ses-{session}-task-reward-atlas-4S456Parcels.npy'))

            np.nan_to_num(faces, copy=False, nan=0)
            np.nan_to_num(flanker, copy=False, nan=0)
            np.nan_to_num(nback, copy=False, nan=0)
            np.nan_to_num(reward, copy=False, nan=0)

            matrices = [faces, flanker, nback, reward]
            brain_matrices_stacked = np.stack(matrices, axis=2)
            brain_correlation_averaged = brain_matrices_stacked.mean(axis=2)

            brain_correlation_256 = brain_correlation_averaged[np.ix_(cognitive_indices, cognitive_indices)]

            np.save(os.path.join(directory, f'{subject}_ses-{session}_avg_sub256_{subnetwork}.npy'), brain_correlation_256)


