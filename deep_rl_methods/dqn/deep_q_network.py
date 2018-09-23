import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import convolution2d as conv
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import xavier_initializer as xavier


def conv_module(input_layer, convs, activation_fn=tf.nn.relu):
    """ convolutional module
    """
    out = input_layer
    for num_outputs, kernel_size, stride in convs:
        out = conv(
            out,
            num_outputs=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding="VALID",
            activation_fn=activation_fn)
    return out


def fc_module(input_layer, hiddens, activation_fn=tf.nn.relu):
    """ fully connected module
    """
    out = input_layer
    for num_outputs in hiddens:
        out = fc(
            out,
            num_outputs=num_outputs,
            activation_fn=activation_fn,
            weights_initializer=xavier())
    return out


class DeepQNetwork:

    def __init__(
            self,
            num_actions,
            state_shape=[8, 8, 1],
            convs=[[32, 4, 2], [64, 2, 1]],
            hiddens=[128],
            activation_fn=tf.nn.relu,
            optimizer=tf.train.AdamOptimizer(2.5e-4),
            scope="dqn"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            ################### Neural network architecture ###################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(
                dtype=tf.float32, shape=input_shape)
            
            out = conv_module(self.input_states, convs, activation_fn)
            out = layers.flatten(out)
            out = fc_module(out, hiddens, activation_fn)
            self.q_values = fc_module(out, [num_actions], None)

            ##################### Optimization procedure ######################

            # convert input actions to indices for q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack(
                [indices_range, self.input_actions], axis=1)

            # select q-values for input actions
            self.q_values_selected = tf.gather_nd(
                self.q_values, action_indices)

            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.loss = tf.reduce_mean(tf.square(self.q_targets - self.q_values_selected))
            self.train_op = optimizer.minimize(self.loss)
            
        self.vars = tf.trainable_variables(scope)

    def get_q_values(self, sess, states, actions):
        """ 
        Parameters
        ----------
        states: np.array of shape [batch_size] + state_shape
            batch of states
        actions: np.array of shape [batch_size, ]
            batch of actions

        Returns
        -------
        q_values: np.array of shape [batch_size, ]
            q-values for corresponding states and actions
        """
        feed_dict = {self.input_states: states, self.input_actions: actions}
        q_values = sess.run(self.q_values_selected, feed_dict)
        return q_values

    def get_greedy_action(self, sess, states):
        """ 
        Parameters
        ----------
        states: np.array of shape [batch_size] + state_shape
            batch of states

        Returns
        -------
        actions: np.array of shape [batch_size, ]
            actions = argmax Q(states, actions)
        """        
        feed_dict = {self.input_states: states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax    

    def train(self, sess, states, actions, q_targets):
        """ 
        Parameters
        ----------
        states: np.array of shape [batch_size] + state_shape
            batch of states
        actions: np.array of shape [batch_size, ]
            batch of actions
        targets: np.array of shape [batch_size, ]
            batch of TD-targets y = r + gamma * max Q(s,a)

        Returns
        -------
        loss: float scalar
            loss function value on a given batch
        """
        feed_dict = {self.input_states: states,
                     self.input_actions: actions,
                     self.q_targets: q_targets}
        loss, _ = sess.run([self.loss, self.train_op], feed_dict)
        return loss
