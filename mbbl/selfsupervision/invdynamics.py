#######################################################################################
# Inverse Dynamics Class
# Author: Utkarsh Aashu Mishra (utkarsh75477@gmail.com)
#######################################################################################

import numpy as np
import tensorflow as tf


class InverseDynamics:

    def __init__(self,
                sess=None,
                state_dim = None,
                action_dim = None,
                hidden_sizes = None,
                learning_rate = 0.001,
                hidden_activation = tf.nn.relu,
                output_activation = tf.nn.relu,
                w_init=tf.contrib.layers.xavier_initializer(),
                b_init=tf.zeros_initializer()
                ):

        self._sess = sess
        self._state = tf.placeholder(dtype=tf.float32, shape=state_dim, name='state')
        self._action = tf.placeholder(dtype=tf.float32, shape=action_dim, name='action')
        self._next_state = tf.placeholder(dtype=tf.float32, shape=state_dim, name='next_state')

        ############################# Inverse Dynamics Layer ###########################

        self._merged_states = tf.concat([self._state, self._next_state], 0)

        with tf.variable_scope('invdynamics', reuse=tf.AUTO_REUSE):

            with tf.GradientTape() as tape:

                tape.watch([self._state, self._next_state, self._merged_states])

                layer = tf.layers.dense(inputs=tf.expand_dims(self._merged_states, 0),
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
                                        units=action_dim, 
                                        activation=output_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='layer_out')

                self.layer_out = tf.layers.flatten(layer)

                self.predicted_action = tf.squeeze(self.layer_out)

                self._state_gradients = tape.gradient(self.predicted_action, self._merged_states)

            self._loss = tf.reduce_mean(tf.square(self.predicted_action - self._action))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, state, next_state):
        return self._sess.run(self.predicted_action, {self._state: state, self._next_state: next_state})

    def calc_gradients(self, state, next_state):
        return self._sess.run(self._state_gradients, {self._state: state, self._next_state: next_state})

    def update(self, state, action, next_state):

        self._sess.run(self._train_op, 
                        {self._state: state,
                        self._action: action,
                        self._next_state: next_state})