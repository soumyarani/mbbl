#######################################################################################
# Reward Class
# Author: Utkarsh Aashu Mishra (utkarsh75477@gmail.com)
# 
# Implementation from:
# A. Zhang, R. McAllister, R. Calandra, Y. Gal, and S. Levine, 
# Learning Invariant Representations for Reinforcement Learning without Reconstruction
# http://arxiv.org/abs/2006.10742.
#######################################################################################

import numpy as np
import tensorflow as tf


class Reward:

    def __init__(self,
                sess=None,
                state_dim = None,
                action_dim = None,
                hidden_sizes = None,
                learning_rate = 0.001,
                hidden_activation = tf.nn.tanh,
                output_activation = tf.nn.tanh,
                w_init=tf.contrib.layers.xavier_initializer(),
                b_init=tf.zeros_initializer()
                ):

        self._sess = sess
        self._state = tf.placeholder(dtype=tf.float32, shape=state_dim, name='state')
        self._action = tf.placeholder(dtype=tf.float32, shape=action_dim, name='action')
        self._reward = tf.placeholder(dtype=tf.float32, shape=1, name='reward')

        ############################# Feature Dynamics Layer ###########################

        self._merged_state_action = tf.concat([self._state, self._action], 0)

        with tf.variable_scope('reward', reuse=tf.AUTO_REUSE):

            layer = tf.layers.dense(inputs=tf.expand_dims(self._merged_state_action, 0),
                                    units=hidden_sizes[0], 
                                    activation=hidden_activation, 
                                    kernel_initializer=w_init,
                                    bias_initializer=b_init,
                                    name='layer_in')

            for i in range(len(hidden_sizes)-1):

                layer = tf.layers.dense(inputs=layer, 
                                        units=hidden_sizes[i+1], 
                                        activation=hidden_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='layer_'+str(i+1))

            layer = tf.layers.dense(inputs=layer, 
                                    units=1, 
                                    activation=output_activation, 
                                    kernel_initializer=w_init,
                                    bias_initializer=b_init,
                                    name='layer_out')

            self.layer_out = tf.layers.flatten(layer)

            self.predicted_reward = tf.squeeze(self.layer_out)

            self._loss = tf.reduce_mean(tf.square(self.predicted_reward - self._reward))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, state, action):
        return self._sess.run(self.predicted_reward, {self._state: state, self._action: action})

    def update(self, state, action, reward):

        self._sess.run(self._train_op, 
                        {self._state: state,
                        self._action: action,
                        self._reward: reward})