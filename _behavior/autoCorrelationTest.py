import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, rankdata

import _tools

'''
- Create autocorrelation plot for collected data of defined participant and month
'''

# List of the subdirectories
path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'
preprocessedData_path = os.path.join(path, 'Data', 'beRNN_03', 'data_highDim')
month = 'month_5'

subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]

data = {}

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

    # Store the results in the dictionaries
    data[subdir] = file_quartett

dict_x = {}
dict_y = {}
for taskList in data.keys():
    task = taskList.split('\\')[-1]
    print(task)

    # x_list = []
    y_list = []
    for npyFileQuartett in data[taskList]:

        # Load the files
        x = np.load(npyFileQuartett[0])  # Input
        y = np.load(npyFileQuartett[2])  # Participant Response
        y_loc = np.load(npyFileQuartett[1])  # Human Ground Truth
        response = np.load(npyFileQuartett[3], allow_pickle=True)  # Objective Ground Truth - only needed for training if error balancing is applied

        # attention: create autocorrelation plot for either input or output information
        # x_list.append(x[-1,:,:])
        y_list.append(y[-1,:,:])

        # dict_x[task] = x_list
        dict_y[task] = y_list

def seq_autocorr_matrix(sequence, max_lag=None, method="pearson"):
    """
    Compute sequence-level autocorrelations for a sequence of d-dim vectors.
    sequence: array shape (N, d) where N = 760, d = 33

    method: "pearson" (default) or "spearman"

    Returns:
      lags: array([0..max_lag])
      corr_by_lag: list of numpy arrays; corr_by_lag[k] has shape (N-k,)
    """
    seq = np.asarray(sequence)
    N, d = seq.shape
    if max_lag is None:
        max_lag = N - 1
    max_lag = min(max_lag, N - 1)

    if method == "spearman":
        # Rank transform each row (vector) independently
        seq = np.apply_along_axis(rankdata, 1, seq)

    # Pre-demean across features for each vector (row)
    seq_centered = seq - seq.mean(axis=1, keepdims=True)  # shape (N, d)

    # Precompute row-wise sum of squares (for denominator)
    ss = np.sum(seq_centered ** 2, axis=1)  # shape (N,)

    corr_by_lag = []
    lags = np.arange(0, max_lag + 1)
    for k in lags:
        a = seq_centered[:N - k]  # shape (N-k, d)
        b = seq_centered[k:N]  # shape (N-k, d)

        # numerator = sum over features of a * b
        num = np.sum(a * b, axis=1)  # shape (N-k,)

        # denominator = sqrt( sum(a^2) * sum(b^2) )
        den = np.sqrt(ss[:N - k] * ss[k:N])  # shape (N-k,)

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(den > 0, num / den, 0.0)

        corr_by_lag.append(corr)

    return lags, corr_by_lag

def plot_sequence_autocorr_grid(data_dict, month, max_lag=100, figsize=(15, 12),
                                percentile_lower=10, percentile_upper=90,
                                cmap='tab10', method="pearson"):
    """
    data_dict: dict where each key -> list of arrays shape (40,33)
    method: "pearson" or "spearman"
    """
    keys = list(data_dict.keys())
    n_keys = len(keys)
    assert n_keys <= 12, "This plotting function expects up to 12 keys."

    fig, axes = plt.subplots(4, 3, figsize=figsize)
    axes = axes.ravel()

    mean_corr_dict = {}
    for i, key in enumerate(keys):
        arrs = data_dict[key]
        seq = np.vstack(arrs)  # (760, 33)

        N = seq.shape[0]
        L = min(max_lag, N - 1)

        lags, corr_by_lag = seq_autocorr_matrix(seq, max_lag=L, method=method)

        means = np.array([c.mean() if c.size > 0 else np.nan for c in corr_by_lag])

        mean_corr_dict[key] = means

        p_low = np.array([np.percentile(c, percentile_lower) if c.size > 0 else np.nan for c in corr_by_lag])
        p_high = np.array([np.percentile(c, percentile_upper) if c.size > 0 else np.nan for c in corr_by_lag])

        ax = axes[i]
        for k, c in zip(lags, corr_by_lag):
            if c.size > 0:
                x = np.full_like(c, k, dtype=float)
                ax.scatter(x, c, s=2, alpha=0.03, marker='o')

        ax.fill_between(lags, p_low, p_high, alpha=0.2)
        ax.plot(lags, means, lw=2, color='red', label='mean corr')

        ax.axhline(0.0, color='k', lw=0.6, linestyle='--')
        ax.axvline(0.0, color='k', lw=0.5)
        ax.set_xlim(0, L)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{key}", fontsize=18)
        ax.set_xlabel('Lag (vectors)', fontsize=16)
        ax.set_ylabel(f'{method.capitalize()}', fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=12)

    for j in range(n_keys, len(axes)):
        axes[j].axis('off')

    np.save(rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\pngs\means_dict_{month}.npy' ,mean_corr_dict, allow_pickle=True)

    plt.tight_layout()
    plt.savefig(rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\pngs\autoCorrelationTest_x_{month}_{method}.png')

    # plt.show()

plot_sequence_autocorr_grid(dict_y, month, method="pearson")  # default


#################################################################################################################
# head. train/test-gap difference X autocorrelation-gap difference
#################################################################################################################
# Calculate difference between train perf of trained model on month 4 and test perf on month 6 ~ 50 data points - task-wise
model_dir = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\autoCorrelationTest_2\highDim_correctOnly_3stimTC\beRNN_03\01\beRNN_03_Alltask_4-6_data_highDim_correctOnly_3stimTC_iteration1_LeakyRNN_diag_256_softplus'
months = ['model_month_4', 'model_month_5', 'model_month_6']

month_dict_of_test_perf_dict = {}
month_dict_of_train_perf_dict = {}

tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']

for month in months:
    test_perf_dict = {}
    model = os.path.join(model_dir, month)

    log = _tools.load_log(model)

    # info: perf_train already has form of 26 data points (one for each evaluation)
    for task in tasks:
        test_perf_dict[task] = log['perf_' + task]

    month_dict_of_test_perf_dict[month] = test_perf_dict

# Give length of very last task in last month
numberOfDataPoints = len(month_dict_of_test_perf_dict[month][task])
# info: train
for month in months:
    train_perf_dict = {}
    model = os.path.join(model_dir, month)

    log = _tools.load_log(model)

    # info: perf_train already has form of 26 data points (one for each evaluation)
    for task in tasks:
        stepsToTake = round(len(log['perf_train_' + task]) // numberOfDataPoints)
        train_perf_dict[task] = []

        for i in range(0, numberOfDataPoints): # Starting with 0 as evaluation perf is saved before first trained trials
            train_perf_dict[task].append(log['perf_train_' + task][i*stepsToTake]) # one evaluation after every 40k trials

    month_dict_of_train_perf_dict[month] = train_perf_dict

# Create difference dict for performance
month_dict_of_difference_perf_dict = {}
# difference
for month in months:
    month_dict_of_difference_perf_dict[month] = {}

    for task in tasks:
        month_dict_of_difference_perf_dict[month][task] = []

        for j in range(0, numberOfDataPoints):
            difference = month_dict_of_train_perf_dict[month][task][j] - month_dict_of_test_perf_dict[month][task][j]

            month_dict_of_difference_perf_dict[month][task].append(difference)

# Calculate difference between autocorrelation mean of month 4 and month 6 for first 50 lags - task-wise
means_month_4 = np.load(rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\pngs\means_dict_month_4.npy', allow_pickle=True).item()
means_month_6 = np.load(rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\pngs\means_dict_month_6.npy', allow_pickle=True).item()

means_month_differences = {}
# differences
for key in means_month_4:
    means_month_differences[key] = []
    for i in range(0,26):
        means_month_differences[key].append(means_month_4[key][i] - means_month_6[key][i])

# Correlate data sets with each other
perf_month_4_differences = month_dict_of_difference_perf_dict['model_month_4']

def plot_dict_correlations(dict_a, dict_b, method="pearson", ncols=4, figsize=(16, 12)):
    """
    Compare two dicts of lists by correlation (Pearson or Spearman).

    dict_a, dict_b: dicts with identical keys. Each value is a list/array of same length.
    method: "pearson" or "spearman"
    ncols: number of columns in subplot grid
    """
    dict_a = perf_month_4_differences
    dict_b = means_month_differences

    keys = list(dict_a.keys())
    n_keys = len(keys)
    nrows = int(np.ceil(n_keys / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()

    for i, key in enumerate(keys):
        a = np.array(dict_a[key])
        b = np.array(dict_b[key])

        if method == "pearson":
            r, p = pearsonr(a, b)
        elif method == "spearman":
            r, p = spearmanr(a, b)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        ax = axes[i]
        ax.scatter(a, b, alpha=0.7)
        ax.set_title(key, fontsize=12)
        ax.set_xlabel("means_month_differences")
        ax.set_ylabel("perf_month_4_differences")

        # Show correlation in the plot
        ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{method.capitalize()} correlations per key", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    fig.savefig(r"C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\pngs\correlation_perfANDautocorr_gap_difference.png")
    return fig

plot_dict_correlations(means_month_differences, perf_month_4_differences)


