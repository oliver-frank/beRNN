from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# import os

import _tools
from _network import Model

'''
- Analyzes activity of all involved untis in network
'''

########################################################################################################################
# head. Define functions
########################################################################################################################
def easy_activity_plot(model_dir, rule, monthsConsidered, mode):
    """A simple plot of neural activity from one task.

    Args:
        model_dir: directory where model file is saved
        rule: string, the rule to plot
    """

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore()

        trial_dir = 'W://group_csp//analyses//oliver.frank//Data//BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '//PreprocessedData_wResp_ALL'

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

        # trial = generate_trials(rule, hp, mode='test')
        # feed_dict = _tools.gen_feed_dict(model, trial, hp)
        # h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # # All matrices have shape (n_time, n_condition, n_neuron)

    # Take only the one example trial
    i_trial = 0

    for activity, title in zip([x, h, y_hat],['input', 'recurrent', 'output']):
        plt.figure()
        plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot', interpolation='none', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()

def pretty_inputoutput_plot(model_dir, rule, monthsConsidered, save=False, plot_ylabel=False):
    """Plot the input and output activity for a sample trial from one task.

    Args:
        model_dir: model directory
        rule: string, the rule
        save: bool, whether to save plots
        plot_ylabel: bool, whether to plot ylable
    """


    fs = 7

    model = Model(model_dir)
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

        t_plot = np.arange(x.shape[0])*hp['dt']/1000

        assert hp['num_ring'] == 2

        n_eachring = hp['n_eachring']

        fig = plt.figure(figsize=(1.3,2))
        ylabels = ['fix. in', 'stim. mod1', 'stim. mod2','fix. out', 'out']
        heights = np.array([0.03,0.2,0.2,0.03,0.2])+0.01
        for i in range(5):
            ax = fig.add_axes([0.15,sum(heights[i+1:]+0.02)+0.1,0.8,heights[i]])
            cmap = 'Purples'
            plt.xticks([])
            ax.tick_params(axis='both', which='major', labelsize=fs,
                           width=0.5, length=2, pad=3)

            if plot_ylabel:
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('none')

            if i == 0:
                plt.plot(t_plot, x[:,0,0], color='xkcd:blue')
                if plot_ylabel:
                    plt.yticks([0,1],['',''],rotation='vertical')
                plt.ylim([-0.1,1.5])
                plt.title(rule,fontsize=fs)
            elif i == 1:
                plt.imshow(x[:,0,1:1+n_eachring].T, aspect='auto', cmap=cmap,
                           vmin=0, vmax=1, interpolation='none',origin='lower')
                if plot_ylabel:
                    plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
                               [r'0$\degree$',r'180$\degree$',r'360$\degree$'],
                               rotation='vertical')
            elif i == 2:
                plt.imshow(x[:, 0, 1+n_eachring:1+2*n_eachring].T,
                           aspect='auto', cmap=cmap, vmin=0, vmax=1,
                           interpolation='none',origin='lower')

                if plot_ylabel:
                    plt.yticks(
                        [0, (n_eachring-1)/2, n_eachring-1],
                        [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
                        rotation='vertical')
            elif i == 3:
                plt.plot(t_plot, y[:,0,0],color='xkcd:green')
                plt.plot(t_plot, y_hat[:,0,0],color='xkcd:blue')
                if plot_ylabel:
                    plt.yticks([0.05,0.8],['',''],rotation='vertical')
                plt.ylim([-0.1,1.1])
            elif i == 4:
                plt.imshow(y_hat[:, 0, 1:].T, aspect='auto', cmap=cmap,
                           vmin=0, vmax=1, interpolation='none', origin='lower')
                if plot_ylabel:
                    plt.yticks(
                        [0, (n_eachring-1)/2, n_eachring-1],
                        [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
                        rotation='vertical')
                plt.xticks([0,y_hat.shape[0]], ['0', '2'])
                plt.xlabel('Time (s)',fontsize=fs, labelpad=-3)
                ax.spines["bottom"].set_visible(True)

            if plot_ylabel:
               plt.ylabel(ylabels[i],fontsize=fs)
            else:
                plt.yticks([])
            ax.get_yaxis().set_label_coords(-0.12,0.5)

        if save:
            save_name = 'figure/sample_'+ rule +'.pdf'
            plt.savefig(save_name, transparent=True)
        plt.show()

        # plt.figure()
        # _ = plt.plot(h_sample[:,0,:20])
        # plt.show()
        #
        # plt.figure()
        # _ = plt.plot(y_sample[:,0,:])
        # plt.show()


########################################################################################################################
# head. Execute
########################################################################################################################
if __name__ == '__main__':
    # Pre-allocate necessary variables
    model_dir = 'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_112_BeRNN_01_Month_2-4'
    rule = 'DM'
    monthsConsidered = ['2', '3', '4']
    mode = 'Evaluation'
    # Execute funtions
    easy_activity_plot(model_dir, rule, monthsConsidered, mode)
    pretty_inputoutput_plot(model_dir, rule, monthsConsidered, save=False, plot_ylabel=False)


