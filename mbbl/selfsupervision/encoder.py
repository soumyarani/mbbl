#######################################################################################
# Encoder Class
# Author: Utkarsh Aashu Mishra (utkarsh75477@gmail.com)
#######################################################################################

import numpy as np
import tensorflow as tf


class Encoder:

    def __init__(self,
                sess=None,
                input_dim = None,
                output_dim = None,
                hidden_sizes = None,
                learning_rate = 0.001,
                hidden_activation = tf.nn.tanh,
                output_activation = tf.nn.tanh,
                w_init=tf.contrib.layers.xavier_initializer(),
                b_init=tf.zeros_initializer()
                ):

        self._sess = sess
        self._state_1 = tf.placeholder(dtype=tf.float32, shape=input_dim, name='state_1')
        self._optim_grads = tf.placeholder(dtype=tf.float32, shape=output_dim, name='action_grads')

        ###################### Encoded Feature Extraction Layer ########################

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):

            with tf.GradientTape() as tape:

                tape.watch([self._state_1])

                layer = tf.layers.dense(inputs=tf.expand_dims(self._state_1, 0),
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
                                        units=output_dim, 
                                        activation=output_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='layer_out')

                self.layer_out_1 = tf.layers.flatten(layer)

                self.encoded_state = tf.squeeze(self.layer_out_1)

                self.feature_gradients = tape.gradient(self.encoded_state, self._state_1)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            params_grad = tf.gradients(self.encoded_state, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), -self._optim_grads)
            grads = zip(params_grad, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

            self._train_op = self._optimizer.apply_gradients(grads)

    def predict(self, state):
        return self._sess.run(self.encoded_state, {self._state_1: state})

    def calc_gradients(self, state):
        return self._sess.run(self.feature_gradients, {self._state_1: state})

    def update(self, state, optim_grads):

        self._sess.run(self._train_op, 
                        {self._state_1: state, 
                        self._optim_grads: optim_grads
                        })