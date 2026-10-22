from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import _tools
from _network import Model

'''
- Analyzes structural connectivity in ANNs
'''

########################################################################################################################
# Define functions
########################################################################################################################
def easy_connectivity_plot(model_dir):
    """A simple plot of network connectivity."""

    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get name of each variable
        names  = [var.name for var in var_list]

    # Plot weights
    for param, name in zip(params, names):
        if len(param.shape) != 2:
            continue

        vmax = np.max(abs(param))*0.7
        plt.figure()
        # notice the transpose
        plt.imshow(param.T, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax,
                   interpolation='none', origin='lower')
        plt.title(name)
        plt.colorbar()
        plt.xlabel('From')
        plt.ylabel('To')
        plt.show()

def schematic_plot(model_dir, monthsConsidered, rule, mode):
    fontsize = 6

    model = Model(model_dir, dt=1)
    hp = model.hp
    trial_dir = 'W://group_csp//analyses//oliver.frank//Data//BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '//PreprocessedData_wResp_ALL'

    with tf.Session() as sess:
        model.restore()
        x, y, y_loc, file_stem = _tools.load_trials(trial_dir, monthsConsidered, rule, mode)

        # info: ################################################################################################
        fixation_steps = _tools.getEpochSteps(y, file_stem)

        # Creat c_mask for current batch
        if hp['loss_type'] == 'lsq':
            c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 5.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important
            c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))

        else:
            c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 5.

            c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
            c_mask /= c_mask.mean()

        # info: ################################################################################################

        feed_dict = _tools.gen_feed_dict(model, x, y, c_mask, hp)
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)


    n_eachring = hp['n_eachring']
    n_hidden = hp['n_rnn']

    # Plot Stimulus
    fig = plt.figure(figsize=(1.0,1.2))
    heights = np.array([0.06,0.25,0.25])
    for i in range(3):
        ax = fig.add_axes([0.2,sum(heights[i+1:]+0.1)+0.05,0.7,heights[i]])
        cmap = 'Purples'
        plt.xticks([])

        # Fixed style for these plots
        ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if i == 0:
            plt.plot(x[:,0,0], color='xkcd:blue')
            plt.yticks([0, 1], ['', ''],rotation='vertical')
            plt.ylim([-0.1, 1.5])
            plt.title('Fixation input', fontsize=fontsize, y=0.9)
        elif i == 1:
            plt.imshow(x[:, 0, 1:1+n_eachring].T, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='none',origin='lower')
            plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
                       [r'0$\degree$', '', r'360$\degree$'],
                       rotation='vertical')
            plt.title('Stimulus mod 1', fontsize=fontsize, y=0.9)
        elif i == 2:
            plt.imshow(x[:, 0, 1+n_eachring:1+2*n_eachring].T, aspect='auto',
                       cmap=cmap, vmin=0, vmax=1,
                       interpolation='none', origin='lower')
            plt.yticks([0, (n_eachring-1)/2, n_eachring-1], ['', '', ''],
                       rotation='vertical')
            plt.title('Stimulus mod 2', fontsize=fontsize, y=0.9)
        ax.get_yaxis().set_label_coords(-0.12,0.5)
    # plt.savefig('figure/schematic_input.pdf',transparent=True)
    plt.show()

    # Plot Rule Inputs
    fig = plt.figure(figsize=(1.0, 0.5))
    ax = fig.add_axes([0.2,0.3,0.7,0.45])
    cmap = 'Purples'
    X = x[:, 0, 1+2*n_eachring:]
    plt.imshow(X.T, aspect='auto', vmin=0, vmax=1, cmap=cmap,
               interpolation='none', origin='lower')

    plt.xticks([0, X.shape[0]])
    ax.set_xlabel('Time (ms)', fontsize=fontsize, labelpad=-5)

    # Fixed style for these plots
    ax.tick_params(axis='both', which='major', labelsize=fontsize,
                   width=0.5, length=2, pad=3)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.yticks([0, X.shape[-1]-1], ['1',str(X.shape[-1])], rotation='vertical')
    plt.title('Rule inputs', fontsize=fontsize, y=0.9)
    ax.get_yaxis().set_label_coords(-0.12,0.5)

    # plt.savefig('figure/schematic_rule.pdf',transparent=True)
    plt.show()


    # Plot Units
    fig = plt.figure(figsize=(1.0, 0.8))
    ax = fig.add_axes([0.2,0.1,0.7,0.75])
    cmap = 'Purples'
    plt.xticks([])
    # Fixed style for these plots
    ax.tick_params(axis='both', which='major', labelsize=fontsize,
                   width=0.5, length=2, pad=3)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.imshow(h[:, 0, :].T, aspect='auto', cmap=cmap, vmin=0, vmax=1,
               interpolation='none',origin='lower')
    plt.yticks([0,n_hidden-1],['1',str(n_hidden)],rotation='vertical')
    plt.title('Recurrent units', fontsize=fontsize, y=0.95)
    ax.get_yaxis().set_label_coords(-0.12,0.5)
    # plt.savefig('figure/schematic_units.pdf',transparent=True)
    plt.show()


    # Plot Outputs
    fig = plt.figure(figsize=(1.0,0.8))
    heights = np.array([0.1,0.45])+0.01
    for i in range(2):
        ax = fig.add_axes([0.2, sum(heights[i+1:]+0.15)+0.1, 0.7, heights[i]])
        cmap = 'Purples'
        plt.xticks([])

        # Fixed style for these plots
        ax.tick_params(axis='both', which='major', labelsize=fontsize,
                       width=0.5, length=2, pad=3)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if i == 0:
            plt.plot(y_hat[:,0,0],color='xkcd:blue')
            plt.yticks([0.05,0.8],['',''],rotation='vertical')
            plt.ylim([-0.1,1.1])
            plt.title('Fixation output', fontsize=fontsize, y=0.9)

        elif i == 1:
            plt.imshow(y_hat[:,0,1:].T, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='none', origin='lower')
            plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
                       [r'0$\degree$', '', r'360$\degree$'],
                       rotation='vertical')
            plt.xticks([])
            plt.title('Response', fontsize=fontsize, y=0.9)

        ax.get_yaxis().set_label_coords(-0.12,0.5)

    # plt.savefig('figure/schematic_outputs.pdf',transparent=True)
    plt.show()

########################################################################################################################
# Execute
########################################################################################################################
if __name__ == '__main__':
    # Pre-allocate variables
    model_dir = 'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_112_BeRNN_01_Month_2-4'
    monthsConsidered = ['2','3','4']
    rule = 'DM'
    mode = 'Evaluation'
    # Execute functions
    easy_connectivity_plot(model_dir)
    schematic_plot(model_dir, monthsConsidered, rule, mode)


