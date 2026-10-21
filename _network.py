from __future__ import division
import os
import numpy as np
# import matplotlib.pyplot as plt
# import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import tensorflow as tf

# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# attention. csp_frank_oliver_3 version ++++++++++++++++++++++++++++++++++++++++++++++

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import _tools

'''
Roles:
- Defines network architectures for _training.py
'''

########################################################################################################################
# head. helper functions
########################################################################################################################
def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)
^^
    Returns:
        Readout locations: Numpy array (Batch,)
    """

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)  # Sum of activity at last time point of response epoch for every trial in the batch, respectively
    temp_cos = np.sum(y * np.cos(pref),axis=-1) / temp_sum  # Multiplies each unit's activation by the cosine of its preferred direction.
    temp_sin = np.sum(y * np.sin(pref),axis=-1) / temp_sum  # Multiplies each unit's activation by the sine of its preferred direction.
    loc = np.arctan2(temp_sin,temp_cos)  # This line computes the arctangent of the normalized sine and cosine components, resulting in the angle (location) in radians for each trial in batch

    return np.mod(loc, 2 * np.pi)

def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = y.get_shape().as_list()[-1]
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / num_units)  # preferences
    cos_pref = np.cos(pref)
    sin_pref = np.sin(pref)
    temp_sum = tf.reduce_sum(y, axis=-1)
    temp_cos = tf.reduce_sum(y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.mod(loc, 2 * np.pi)

def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]  # info: Only at the last time point, but response epoch schould be < 0.5

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]  # Get only first fixation unit
    y_hat_loc = popvec(y_hat[..., 1:])  # y_hat[..., 1:] Get all units except for first epoch unit

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    corr_loc = dist < 0.2 * np.pi  # 35 degreee margin around exact correct respond - default .2

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    perf_rounded = np.round(perf, decimals=3)
    return perf_rounded

def cyclic_learning_rate(global_step, mode, base_lr=1e-5, max_lr=1e-3, step_size=2000, decay_rate=0.999,
                         decay_steps=50000):
    """
    Implements a cyclic learning rate schedule. Decay rate can be added or defined independent from cycling rate.

    Parameters:
    - global_step: Current training step (tf.Variable)
    - base_lr: Lower bound of learning rate
    - max_lr: Upper bound of learning rate
    - step_size: Half cycle length (in iterations)
    - mode: 'triangular', 'triangular2', or 'exp_range'

    Returns:
    - learning_rate: Cyclically adjusted learning rate
    """
    # Ensure step_size is float32
    step_size = tf.cast(step_size, tf.float32)

    # Compute the cycle
    cycle = tf.floor(1 + tf.cast(global_step, tf.float32) / (2 * step_size))
    x = tf.abs(tf.cast(global_step, tf.float32) / step_size - 2 * cycle + 1)

    # Ensure base_lr and max_lr are float32
    base_lr = tf.cast(base_lr, tf.float32)
    max_lr = tf.cast(max_lr, tf.float32)

    if mode == 'triangular': # info: Simple periodic exploration
        clr = base_lr + (max_lr - base_lr) * tf.maximum(tf.constant(0., dtype=tf.float32), (1 - x))
    elif mode == 'triangular2': # info: Gradual convergence
        clr = base_lr + (max_lr - base_lr) * tf.maximum(tf.constant(0., dtype=tf.float32), (1 - x)) / (2 ** (cycle - 1))
    elif mode == 'exp_range': # info: Long-term decay (gamma<1 term) + local flexibility
        gamma = tf.constant(0.99994, dtype=tf.float32)  # Ensure gamma is float32
        clr = base_lr + (max_lr - base_lr) * tf.maximum(tf.constant(0., dtype=tf.float32), (1 - x)) * (
                    gamma ** tf.cast(global_step, tf.float32))

    # Apply independent exponential decay if enabled
    elif mode == 'decay':
        global_step_f = tf.cast(global_step, tf.float32)
        decay_factor = tf.pow(decay_rate, global_step_f / decay_steps)
        clr = base_lr + (max_lr - base_lr) * tf.constant(1.0)  # optional: or just max_lr
        clr *= decay_factor

    return clr

def random_orthogonal(n, rng=None):
    """
    Draw a random orthogonal matrix distributed uniformly (Haar) over O(n).
    """
    if rng is None:
        rng = np.random.default_rng()          # or np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(H)
    # Make Q uniformly distributed by flipping signs so diag(R) > 0
    Q *= np.sign(np.diag(R))
    return Q

def popvec_lowDIM(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """

    loc = np.arctan2(y[:, 0], y[:, 1])
    return np.mod(loc, 2 * np.pi)  # check this? January 22 2019

def tf_popvec_lowDIM(y):
    """Population vector read-out in tensorflow."""

    loc = tf.atan2(y[:, 0], y[:, 1])
    return tf.mod(loc + np.pi, 2 * np.pi)  # check this? January 22 2019

def get_perf_lowDIM(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec_lowDIM(y_hat[..., 1:])  # info: Here for y_hat:networkOutput only unit 1 and 2 are taken

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    corr_loc = dist < 0.2 * np.pi  # fix: potential hyperparameter to play around with, should maybe set more liberal

    # Should fixate?
    should_fix = y_loc < 0  # attention: the reason for encoding y_loc -1 on fixation epoch - not as important because y_loc[-1] at this point anyway

    # performance
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    perf_rounded = np.round(perf, decimals=3)
    return perf_rounded


########################################################################################################################
# head. Network architectures
########################################################################################################################
class LeakyRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None,
                 mask=None,
                 participant=None,
                 machine=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self.mask = mask
        self.participant = participant

        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'elu':  # Adding the ELU activation function
            self._activation = tf.nn.elu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'linear':  # Adding the linear activation option
            self._activation = tf.identity
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units
        w_in0 = (self.rng.randn(n_input, n_hidden) /
                 np.sqrt(n_input) * self._w_in_start)

        # testMatrix = rng.randn(77, 256) / np.sqrt(77)

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start * np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start * tools.gen_ortho_matrix(n_hidden, rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start * self.rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden))
        elif self._w_rec_init == 'brainStructure':
            # Define main path
            # attention: Currently set up for local machine _analysis of remotely trained models ############################
            print("attention @ network.py (322): Currently set up for local machine _analysis of remotely trained models")
            if machine == 'local' or machine == 'hitkip':
                connectomePath = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\masks'
            elif machine == 'notAvailable':
                connectomePath = f'/zi/home/oliver.frank/Desktop/RNN/multitask_BeRNN-main/masks'
            # elif machine == 'pandora':
            #     connectomePath = f'/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/masks'
            # attention: Currently set up for local machine _analysis of remotely trained models ############################
            maskName = f'{participant}_ses-03_atlas-4S{num_units}Parcels_mapping-standardized.npy'
            structuralMask = np.load(os.path.join(connectomePath, maskName))

            # maskSize = int(mask.split('_')[1]) # fix: extract size from mask name string
            if num_units == 156:
                maskSize = 156
            elif num_units == 256:
                maskSize = 256
            else:
                print('Error: Mask size not recognized. Please check the mask file.')

            mean_activity = np.mean(structuralMask)
            structuralMask_binary = structuralMask.copy()
            counter1 = 0
            threshold = mean_activity - mean_activity * .999  # # fix hard code .975/.999 threshold defined for all values > 5%/.1% area around mean - after first paper
            for i in range(0, maskSize):
                for j in range(0, maskSize):
                    if np.abs(structuralMask[i, j]) > np.abs(threshold):
                        structuralMask_binary[i, j] = 1
                        counter1 += 1
                    else:
                        structuralMask_binary[i, j] = 0

            self.mask = structuralMask_binary

            # Load right weight matrix & rotated for equivalent randomness factor while preserving the spectral characteristics
            w_rec0_ = structuralMask

            # Ensure symmetry and remove NaNs
            w_rec0_ = np.nan_to_num((w_rec0_ + w_rec0_.T) / 2, nan=0.0, posinf=0.0, neginf=0.0)

            # Draw a new random rotation each run
            Q = random_orthogonal(w_rec0_.shape[0])
            w_rec0 = Q @ w_rec0_ @ Q.T

            # Normalize to controlled spectral radius (~1)
            eigvals = np.linalg.eigvals(w_rec0)
            max_eig = np.max(np.abs(eigvals))
            if max_eig > 0:
                w_rec0 = w_rec0 / max_eig * self._w_rec_start


        print('>>>>>>>>>>>>>> w_rec0.shape', w_rec0.shape)
        print('>>>>>>>>>>>>>> w_in0.shape', w_in0.shape)

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0
        self._initializer = tf.constant_initializer(matrix0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            'kernel',
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._initializer)

        # Split the candidate weights into input and recurrent parts
        w_in, w_rec = tf.split(self._kernel, [input_depth, self._num_units], axis=0)
        # info: Apply structural mask only to the recurrent part (w_rec)
        if isinstance(self.mask, np.ndarray):
            # Convert mask to tensor (ensure dtype matches)
            mask_tensor = tf.convert_to_tensor(self.mask, dtype=w_rec.dtype)
            # Apply mask (do not overwrite the variable!)
            w_rec_masked = w_rec * mask_tensor
            # Recombine masked recurrent weights with input weights
            self._kernel_masked = tf.concat([w_in, w_rec_masked], axis=0)
        else:
            # Concatenate the input and masked recurrent weights back together
            self._kernel = tf.concat([w_in, w_rec], axis=0)

        self._bias = self.add_variable(
            'bias',
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        # Use masked kernel if available (mask applied only to recurrent weights)
        kernel = getattr(self, "_kernel_masked", self._kernel)

        # Combine input and previous state
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # Add Gaussian noise to the gate inputs
        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        # Apply activation
        output = self._activation(gate_inputs)

        # Leaky integration
        output = (1 - self._alpha) * state + self._alpha * output

        return output, output

class FlexibleLeakyStackedRNNCell(tf.nn.rnn_cell.RNNCell):
    """
    Stack of LeakyRNNCell layers with individually specified units and activations per layer.
    """

    def __init__(self,
                 n_rnn_per_layer,         # List[int]
                 activations_per_layer,   # List[str], same length as n_rnn_per_layer
                 n_input,
                 alpha,
                 sigma_rec=0,
                 w_rec_init='diag',
                 rng=None,
                 reuse=tf.AUTO_REUSE,
                 name=None,
                 mask=None):
        super(FlexibleLeakyStackedRNNCell, self).__init__(_reuse=reuse, name=name)

        assert len(n_rnn_per_layer) == len(activations_per_layer), \
            "Length of n_rnn_per_layer and activations_per_layer must match"

        self._num_layers = len(n_rnn_per_layer)
        self._cells = []

        input_size = n_input
        for i, (num_units, activation) in enumerate(zip(n_rnn_per_layer, activations_per_layer)):
            layer_name = f"{name}_layer{i}" if name else f"layer{i}"
            with tf.variable_scope(layer_name):  # Outer scope
                cell = LeakyRNNCell(
                    num_units=num_units,
                    n_input=input_size,
                    alpha=alpha,
                    sigma_rec=sigma_rec,
                    activation=activation,
                    w_rec_init=w_rec_init,
                    rng=rng,
                    reuse=None,  # Don't force reuse here
                    name=layer_name,  # So the inner scope has a unique name too
                    mask=mask if i == 0 else None
                )
                self._cells.append(cell)
                input_size = num_units

        self._multi_cell = tf.nn.rnn_cell.MultiRNNCell(self._cells)

    @property
    def state_size(self):
        return self._multi_cell.state_size

    @property
    def output_size(self):
        return self._multi_cell.output_size

    def call(self, inputs, state):
        """Custom call method that captures outputs from all RNN layers."""
        outputs = []
        new_states = []

        current_input = inputs
        for i, cell in enumerate(self._cells):
            output, new_state = cell(current_input, state[i])
            outputs.append(output)  # store output of current layer
            new_states.append(new_state)
            current_input = output  # feed forward to next layer

        self.hidden_states = outputs  # save for later access (one per layer)

        return outputs[-1], tuple(new_states)  # return top layer output & full state

    def zero_state(self, batch_size, dtype):
        return self._multi_cell.zero_state(batch_size, dtype)

class LeakyGRUCell(RNNCell):
    """Leaky Gated Recurrent Unit cell (cf. https://elifesciences.org/articles/21492).

    Args:
      num_units: int, The number of units in the GRU cell.
      alpha: dt/T, simulation time step over time constant
      sigma_rec: recurrent noise
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 mask=None):
        super(LeakyGRUCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self.mask = mask

        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec

        # info(gryang): allow this to use different initialization

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[
            1].value  # fix: The masking has to be applied on all three relevant structural matrices
        self._gate_kernel = self.add_variable(
            "gates/%s" % 'kernel',
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)

        self._gate_bias = self.add_variable(
            "gates/%s" % 'bias',
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

        self._candidate_kernel = self.add_variable(
            "candidate/%s" % 'kernel',
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)

        self._candidate_bias = self.add_variable(
            "candidate/%s" % 'bias',
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        # info: Apply structural mask only to the recurrent part (w_rec)
        if isinstance(self.mask, np.ndarray):
            # Split the candidate kernel into input and recurrent parts
            w_in, w_rec = tf.split(self._candidate_kernel, [input_depth, self._num_units], axis=0)
            # Split the gate kernel into input and gate parts
            w_in2, w_gate = tf.split(self._gate_kernel, [input_depth, self._num_units], axis=0)
            # Split the gate weights into inhibitory and excitatory parts
            w_inhibi, w_exita = tf.split(w_gate, num_or_size_splits=2, axis=1)

            w_rec = w_rec * self.mask
            w_inhibi = w_inhibi * self.mask
            w_exita = w_exita * self.mask

            # Concatenate the input and masked recurrent weights back together
            self._candidate_kernel = tf.concat([w_in, w_rec], axis=0)
            # Concatenate inhibitory and excitatory parts without duplication
            w_gate = tf.concat([w_inhibi, w_exita], axis=1)
            self._gate_kernel = tf.concat([w_in2, w_gate], axis=0)

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        candidate += tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)

        c = self._activation(candidate)
        # new_h = u * state + (1 - u) * c  # original GRU
        new_h = (1 - self._alpha * u) * state + (
                    self._alpha * u) * c  # info: Where the decision is made which units are inhibitory and which are excitatory

        return new_h, new_h

class LeakyRNNCellSeparateInput(RNNCell):
    """The most basic RNN cell with external inputs separated.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None,
                 mask=None):
        super(LeakyRNNCellSeparateInput, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self.mask = mask

        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start * np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start * tools.gen_ortho_matrix(n_hidden,
                                                                rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden))
        else:
            raise ValueError

        self.w_rnn0 = w_rec0
        self._initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        self._kernel = self.add_variable(
            'kernel',
            shape=[self._num_units, self._num_units],
            initializer=self._initializer)

        # info: Apply structural mask
        if isinstance(self.mask, np.ndarray):
            self._recurrent_kernel = self._recurrent_kernel * self.mask

        self._bias = self.add_variable(
            'bias',
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """output = new_state = act(input + U * state + B)."""

        gate_inputs = math_ops.matmul(state, self._kernel)
        gate_inputs = gate_inputs + inputs  # directly add inputs
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1 - self._alpha) * state + self._alpha * output

        return output, output

class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 hp=None,
                 sigma_rec=None,
                 dt=None):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """

        # Reset tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        # Safe retrieval with a fallback to 0
        hp['seed'] = hp['seed'] if 'seed' in hp else 0
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        if hp['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hp['in_type'])

        self._build(hp)

        self.model_dir = model_dir
        self.hp = hp

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # info: for old models before lerning rate schedule was implemented
        if 'learning_rate_mode' not in hp or hp['learning_rate_mode'] is None:
            # ensure base_lr and max_lr exist for consistency
            if 'learning_rate' not in hp:
                hp['learning_rate'] = 0.0015  # default fallback
            self.learning_rate = hp['learning_rate']

        else:
            # Define global step
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # Define cyclic learning rate
            self.learning_rate = tf.squeeze(
                cyclic_learning_rate(self.global_step, hp['learning_rate_mode'], base_lr=hp['base_lr'],
                                     max_lr=hp['max_lr'], step_size=2000, decay_rate=0.999,
                                     decay_steps=50000))

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate)
        # Set cost
        self.set_optimizer()

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver()

    def _build_fused(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        # Overall, c_mask allows for flexible and selective application of the cost function during training, enabling the
        # model to focus on important parts of the output as determined by the mask. This can be particularly useful in
        # tasks where only certain outputs or time steps are of interest.
        if hp['loss_type'] == 'lsq':
            self.c_mask = tf.placeholder("float", [None, n_output])
        else:
            # Mask on time
            self.c_mask = tf.placeholder("float", [None])

        # Activation functions
        if hp['activation'] == 'power':
            f_act = lambda x: tf.square(tf.nn.relu(x))
        elif hp['activation'] == 'retanh':
            f_act = lambda x: tf.tanh(tf.nn.relu(x))
        elif hp['activation'] == 'relu+':
            f_act = lambda x: tf.nn.relu(x + tf.constant(1.))
        elif hp['activation'] == 'linear':
            f_act = tf.identity  # or f_act = lambda x: x
        else:
            f_act = getattr(tf.nn, hp['activation'])

        # Recurrent activity
        if hp['rnn_type'] == 'NonRecurrent':  # No w_rec is created and therefore no memory property within the model
            # Process each timestep with a dense layer
            flat_input = tf.reshape(self.x, [-1, n_input])
            dense_output = tf.layers.dense(flat_input, units=n_rnn, activation=f_act)
            self.h = tf.reshape(dense_output, [-1, tf.shape(self.x)[1], n_rnn])

        else:
            if hp['rnn_type'] == 'LeakyRNN' and (hp.get('multiLayer') == None or hp.get('multiLayer') == False):
                n_in_rnn = self.x.get_shape().as_list()[-1]
                cell = LeakyRNNCell(n_rnn, n_in_rnn,
                                    hp['alpha'],
                                    sigma_rec=hp['sigma_rec'],
                                    activation=hp['activation'],
                                    w_rec_init=hp['w_rec_init'],
                                    rng=self.rng,
                                    mask=hp['s_mask'],
                                    participant=hp['participant'],
                                    machine=hp['machine'])

            elif hp['rnn_type'] == 'LeakyRNN' and hp.get('multiLayer') == True:
                # Prepare multi-layer LeakyRNN
                n_in_rnn = self.x.get_shape().as_list()[-1]
                cell_stack = []
                input_size = n_in_rnn

                for i, (units, activation) in enumerate(zip(hp['n_rnn_per_layer'], hp['activations_per_layer'])):
                    with tf.variable_scope(f"layer{i}"):
                        cell_i = LeakyRNNCell(
                            num_units=units,
                            n_input=input_size,
                            alpha=hp['alpha'],
                            sigma_rec=hp['sigma_rec'],
                            activation=activation,
                            w_rec_init=hp['w_rec_init'],
                            rng=self.rng,
                            mask=hp['s_mask'] if hp.get('s_mask') is not None else None,
                            participant=hp['participant'],
                            machine=hp['machine']
                        )

                        cell_stack.append(cell_i)
                        input_size = units

                # Manually apply dynamic_rnn layer by layer and store all outputs
                inputs = self.x
                outputs_per_layer = []
                state = [cell.zero_state(tf.shape(self.x)[1], tf.float32) for cell in cell_stack]

                for i, cell in enumerate(cell_stack):
                    with tf.variable_scope(f"rnn_layer_{i}"):
                        out_i, state_i = tf.nn.dynamic_rnn(
                            cell, inputs, initial_state=state[i], time_major=True, dtype=tf.float32
                        )
                        outputs_per_layer.append(out_i)
                        inputs = out_i  # feed into next layer
                self.h_all_layers = outputs_per_layer  # list of [time, batch, units] tensors
                self.h = outputs_per_layer[-1]  # last layer only for output decoding

            elif hp['rnn_type'] == 'LeakyGRU':
                cell = LeakyGRUCell(
                    n_rnn, hp['alpha'],
                    sigma_rec=hp['sigma_rec'], activation=f_act, mask=hp['s_mask']) # fix: s_mask - outdated

            elif hp['rnn_type'] == 'LSTM':
                # cell = tf.contrib.rnn.LSTMCell(n_rnn, activation=f_act)
                # info. alternative function for LSTM model after tf 2.x upgrade
                cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_rnn, activation=f_act)

            elif hp['rnn_type'] == 'GRU':
                # cell = tf.contrib.rnn.GRUCell(n_rnn, activation=f_act, mask=hp['s_mask']) # fix: s_mask - outdated
                # info. alternative function for GRU model after tf 2.x upgrade
                cell = tf.compat.v1.nn.rnn_cell.GRUCell(n_rnn, activation=f_act)

            else:
                raise NotImplementedError("""rnn_type must be one of LeakyRNN,
                        LeakyGRU, EILeakyGRU, LSTM, GRU
                        """)

            if hp.get('multiLayer') == True:
                # Already handled manually above — no need to run dynamic_rnn again
                pass
            else:
                # Only single-layer model: run dynamic_rnn normally
                self.h, states = rnn.dynamic_rnn(cell, self.x, dtype=tf.float32, time_major=True)

        if hp.get('multiLayer') == None or hp.get('multiLayer') == False:
            # Output
            with tf.variable_scope("output"):
                # Using default initialization `glorot_uniform_initializer`
                w_out = tf.get_variable(
                    'weights',
                    [n_rnn, n_output],
                    dtype=tf.float32
                )
                b_out = tf.get_variable(
                    'biases',
                    [n_output],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0, dtype=tf.float32)
                )

            h_shaped = tf.reshape(self.h, (-1, n_rnn))
            y_shaped = tf.reshape(self.y, (-1, n_output))
            # y_hat_ shape (n_time*n_batch, n_unit)
            # y_hat_ = tf.matmul(h_shaped, w_out) + b_out
            y_hat_ = tf.compat.v1.matmul(h_shaped, w_out) + b_out

        elif hp.get('multiLayer') == True:
            # Output
            with tf.variable_scope("output"):
                # Using default initialization `glorot_uniform_initializer`
                w_out = tf.get_variable(
                    'weights',
                    [hp['n_rnn_per_layer'][-1], n_output],
                    dtype=tf.float32
                )
                b_out = tf.get_variable(
                    'biases',
                    [n_output],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0, dtype=tf.float32)
                )

            h_shaped = tf.reshape(self.h, (-1, hp['n_rnn_per_layer'][-1]))
            y_shaped = tf.reshape(self.y, (-1, n_output))
            # y_hat_ shape (n_time*n_batch, n_unit)
            # y_hat_ = tf.matmul(h_shaped, w_out) + b_out
            y_hat_ = tf.compat.v1.matmul(h_shaped, w_out) + b_out


        if hp['loss_type'] == 'lsq':
            # attention: ###############################################################################################
            # Least-square loss
            y_hat = tf.sigmoid(y_hat_)
            # The c_mask is applied element-wise to the squared difference between the predicted and actual values.
            # This means the mask controls which parts of the output contribute to the cost calculation.
            if 'highDim' in hp['data']:
                self.cost_lsq = tf.reduce_mean(tf.square((y_shaped - y_hat) * self.c_mask))

            elif 'lowDim' in hp['data']:
                # Split fixation vs angular part
                y_shaped_fix, y_shaped_ring = tf.split(y_shaped, [1, 2], axis=-1)
                y_hat_fix, y_hat_ring = tf.split(y_hat, [1, 2], axis=-1)
                # Normalize the ring outputs (optional if not normalized already)
                y_shaped_ring = tf.nn.l2_normalize(y_shaped_ring, axis=-1)
                y_hat_ring = tf.nn.l2_normalize(y_hat_ring, axis=-1)
                # Angular loss
                pred_angle = tf.atan2(y_hat_ring[:, 1], y_hat_ring[:, 0])
                true_angle = tf.atan2(y_shaped_ring[:, 1], y_shaped_ring[:, 0])
                angular_loss = tf.square(tf.sin((pred_angle - true_angle) / 2))
                # Mask angular and fixation loss separately
                fixation_loss = tf.square((y_shaped_fix - y_hat_fix) * self.c_mask[:, 0])
                angular_loss_masked = angular_loss * self.c_mask[:, 1]
                # Combine
                self.cost_lsq = tf.reduce_mean(fixation_loss + angular_loss_masked)
        else:
            # sum to 1 is an important preprocessing step for probability optimization
            y_hat = tf.nn.softmax(y_hat_)
            # info: short cut for y_shaped one hot encoding - fundamental for cross entropy
            # Find the index of the maximum value for each row
            indices = tf.argmax(y_shaped, axis=1)
            # Create an identity matrix and index into it to create the one-hot array
            y_shaped_one_hot = tf.one_hot(indices, depth=n_output)

            # Cross-entropy loss
            self.cost_lsq = tf.reduce_mean(self.c_mask * tf.nn.softmax_cross_entropy_with_logits(labels=y_shaped_one_hot, logits=y_hat_))

        self.y_hat = tf.reshape(y_hat, (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_fused(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if hp['rnn_type'] == 'NonRecurrent':
                if 'output' not in v.name:
                    if 'kernel' in v.name:
                        # This could be input or output weights depending on layer naming
                        self.w_in = v if 'dense' in v.name else self.w_in
                    elif 'bias' in v.name:
                        self.b_rec = v if 'dense' in v.name else self.b_rec
                elif 'rnn' in v.name and hp['rnn_type'] != 'NonRecurrent':
                    # Recurrent weight handling for RNN types
                    if 'kernel' in v.name or 'weight' in v.name:
                        self.w_rec = v[n_input:, :]
                        self.w_in = v[:n_input, :]
                    else:
                        self.b_rec = v
                if 'output' in v.name:
                    if 'kernel' in v.name or 'weight' in v.name:
                        self.w_out = v
                    else:
                        self.b_out = v

            elif hp['rnn_type'] == 'LeakyRNN' and hp.get('multiLayer') == True:
                self.w_in = []
                self.w_rec = []
                self.b_rec = []

                rnn_kernels = [v for v in self.var_list if
                               'rnn' in v.name and ('kernel' in v.name or 'weight' in v.name)]
                rnn_biases = [v for v in self.var_list if 'rnn' in v.name and 'bias' in v.name]

                n_layers = len(hp['n_rnn_per_layer'])

                if len(rnn_kernels) != n_layers:
                    raise ValueError(f'Expected {n_layers} recurrent kernels, found {len(rnn_kernels)}')
                if len(rnn_biases) != n_layers:
                    raise ValueError(f'Expected {n_layers} recurrent biases, found {len(rnn_biases)}')

                for i in range(n_layers):
                    n_in = hp['n_input'] if i == 0 else hp['n_rnn_per_layer'][i - 1]
                    n_out = hp['n_rnn_per_layer'][i]
                    kernel = rnn_kernels[i]

                    self.w_in.append(kernel[:n_in, :])
                    self.w_rec.append(kernel[n_in:, :])
                    self.b_rec.append(rnn_biases[i])

                # Find output projection weights
                for v in self.var_list:
                    if 'output' in v.name:
                        if 'kernel' in v.name or 'weight' in v.name:
                            self.w_out = v
                        elif 'bias' in v.name:
                            self.b_out = v

            else:  # for all other rnn_types
                if 'rnn' in v.name:
                    if 'kernel' in v.name or 'weight' in v.name:
                        # info(gryang): For GRU, fix
                        self.w_rec = v[n_input:, :]
                        self.w_in = v[:n_input, :]
                    else:
                        self.b_rec = v
                else:
                    assert 'output' in v.name
                    if 'kernel' in v.name or 'weight' in v.name:
                        self.w_out = v
                    else:
                        self.b_out = v

        if hp['rnn_type'] != 'NonRecurrent' and (hp.get('multiLayer') == None or hp.get('multiLayer') == False):
            # check if the recurrent and output connection has the correct shape
            if self.w_out.shape != (n_rnn, n_output):
                raise ValueError('Shape of w_out should be ' +
                                 str((n_rnn, n_output)) + ', but found ' +
                                 str(self.w_out.shape))
            if hp['rnn_type'] == 'LSTM':  # info: The factor of 4 comes from the LSTM cell having four sets of weights for each of its components (input gate, forget gate, cell state, and output gate), hence a single LSTM cell's weight matrix quadruples in size compared to a simple RNN cell.
                # Special handling for LSTM because it uses a different weight structure
                if self.w_rec.shape != (n_rnn, n_rnn * 4):
                    raise ValueError(
                        f'Expected LSTM w_rec shape to be {(n_rnn, n_rnn * 4)}, but got {self.w_rec.shape}')
                if self.w_in.shape != (n_input, n_rnn * 4):
                    raise ValueError(
                        f'Expected LSTM w_in shape to be {(n_input, n_rnn * 4)}, but got {self.w_in.shape}')
                # LSTM specific code here
            else:
                # Handling for other RNN types (GRU, SimpleRNN, etc.)
                if self.w_rec.shape != (n_rnn, n_rnn):
                    raise ValueError(f'Expected w_rec shape to be {(n_rnn, n_rnn)}, but got {self.w_rec.shape}')
                if self.w_in.shape != (n_input, n_rnn):
                    raise ValueError(f'Expected w_in shape to be {(n_input, n_rnn)}, but got {self.w_in.shape}')

        elif hp['rnn_type'] == 'LeakyRNN' and hp.get('multiLayer') == True:  # info: Extra case for multiLayerRNN
            # Multi-layer LeakyRNN: check shapes layer by layer
            n_layers = len(hp['n_rnn_per_layer'])

            # Count how many kernel & bias pairs exist to sanity check
            rnn_kernels = [v for v in self.var_list if 'rnn' in v.name and ('kernel' in v.name or 'weight' in v.name)]
            rnn_biases = [v for v in self.var_list if 'rnn' in v.name and 'bias' in v.name]

            if len(rnn_kernels) != n_layers:
                raise ValueError(f'Expected {n_layers} recurrent kernels, found {len(rnn_kernels)}')

            if len(rnn_biases) != n_layers:
                raise ValueError(f'Expected {n_layers} recurrent biases, found {len(rnn_biases)}')

            for i in range(n_layers):
                n_in = hp['n_input'] if i == 0 else hp['n_rnn_per_layer'][i - 1]
                n_out = hp['n_rnn_per_layer'][i]

                w = rnn_kernels[i]
                b = rnn_biases[i]

                if w.shape != (n_in + n_out, n_out):
                    raise ValueError(f'Layer {i}: expected kernel shape {(n_in + n_out, n_out)}, but got {w.shape}')
                if b.shape != (n_out,):
                    raise ValueError(f'Layer {i}: expected bias shape {(n_out,)}, but got {b.shape}')

            # Also check output projection
            n_out_final = hp['n_rnn_per_layer'][-1]
            if self.w_out.shape != (n_out_final, hp['n_output']):
                raise ValueError(f'Expected w_out shape {(n_out_final, hp["n_output"])}, but got {self.w_out.shape}')

    def _build_seperate(self, hp):
        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        self.c_mask = tf.placeholder("float", [None, n_output])

        sensory_inputs, rule_inputs = tf.split(
            self.x, [hp['rule_start'], hp['n_rule']], axis=-1)

        sensory_rnn_inputs = tf.layers.dense(sensory_inputs, n_rnn, name='sen_input')

        if 'mix_rule' in hp and hp['mix_rule'] is True:
            # rotate rule matrix
            kernel_initializer = tf.orthogonal_initializer()
            rule_inputs = tf.layers.dense(
                rule_inputs, hp['n_rule'], name='mix_rule',
                use_bias=False, trainable=False,
                kernel_initializer=kernel_initializer)

        rule_rnn_inputs = tf.layers.dense(rule_inputs, n_rnn, name='rule_input', use_bias=False)

        rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs

        # Recurrent activity
        cell = LeakyRNNCellSeparateInput(
            n_rnn, hp['alpha'],
            sigma_rec=hp['sigma_rec'],
            activation=hp['activation'],
            w_rec_init=hp['w_rec_init'],
            rng=self.rng)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, rnn_inputs, dtype=tf.float32, time_major=True)

        # Output
        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat shape (n_time*n_batch, n_unit)
        y_hat = tf.layers.dense(h_shaped, n_output, activation=tf.nn.sigmoid, name='output')
        # Least-square loss
        self.cost_lsq = tf.reduce_mean(tf.square((y_shaped - y_hat) * self.c_mask))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_separate(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rec = v
                else:
                    self.b_rec = v
            elif 'sen_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_sen_in = v
                else:
                    self.b_in = v
            elif 'rule_input' in v.name:
                self.w_rule = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['rule_start'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                             str((hp['rule_start'], n_rnn)) + ', but found ' +
                             str(self.w_sen_in.shape))
        if self.w_rule.shape != (hp['n_rule'], n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((hp['n_rule'], n_rnn)) + ', but found ' +
                             str(self.w_rule.shape))

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def set_optimizer(self, extra_cost=None, var_list=None):
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = self.var_list

        print('Variables being optimized:')
        for v in var_list:
            print(v)

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)

        hp = getattr(self, 'hp', {})
        grad_clip = hp.get('grad_clip', None)
        grad_clip_by = hp.get('grad_clip_by', 'value')

        grads = []
        vars_ = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grads.append(g)
                vars_.append(v)

        clipped_grads = []

        if grad_clip is not None and grad_clip > 0:
            if grad_clip_by == 'global_norm':
                clipped_grads, global_norm = tf.clip_by_global_norm(grads, grad_clip)
                clip_factor = grad_clip / tf.maximum(global_norm, grad_clip)

                # Attach a print to the first clipped gradient (guaranteed to run)
                clipped_grads[0] = tf.Print(
                    clipped_grads[0],
                    [global_norm, clip_factor],
                    message="Global norm, Clip factor: "
                )

            else:
                for g, v in zip(grads, vars_):
                    g_clipped = tf.clip_by_value(g, -grad_clip, grad_clip)
                    # Attach tf.Print to the clipped tensor so it runs during training
                    g_clipped = tf.Print(
                        g_clipped,
                        [tf.reduce_max(g_clipped), tf.reduce_min(g_clipped), tf.reduce_mean(g_clipped)],
                        message=f'Grad stats for {v.name}: max, min, mean = '
                    )
                    clipped_grads.append(g_clipped)
        else:
            clipped_grads = grads

        capped_gvs = list(zip(clipped_grads, vars_))
        self._capped_grads_and_vars = capped_gvs

        if not hasattr(self, 'global_step'):
            self.train_step = self.opt.apply_gradients(capped_gvs)
        else:
            self.train_step = self.opt.apply_gradients(capped_gvs, global_step=self.global_step)

    def lesion_units(self, sess, units, verbose=False):
        """Lesion units given by units

        Args:
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for v in self.var_list:
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'output' in v.name:
                    # output weights
                    v_val[units, :] = 0
                elif 'rnn' in v.name:
                    # recurrent weights
                    v_val[n_input + units, :] = 0
                sess.run(v.assign(v_val))

        if verbose:
            print('Lesioned units:')
            print(units)


