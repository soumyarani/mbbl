#######################################################################################
# Custom Env Wrapper for DBC
# Author: Utkarsh Aashu Mishra (utkarsh75477@gmail.com)
#######################################################################################

import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from encoder import Encoder
from dynamics import Dynamics
from invdynamics import InverseDynamics
from reward import Reward
from replay_buffer import ReplayBuffer

class WrappedEnv(gym.Env):

    def __init__(self,
                sess=None,
                env_name=None,
                feature_dim=None,
                encoder_gamma=None,
                encoder_hidden_size=None,
                dynamics_hidden_size=None,
                invdyn_hidden_size=None,
                encoder_lr=None,
                dynamics_lr=None,
                invdyn_lr=None):

        super(WrappedEnv, self).__init__()

        self._sess = sess
        self._env = gym.make(env_name)
        self._state_dim = self._env.observation_space.shape[0]
        self._action_dim = self._env.action_space.shape[0]
        self._feature_dim = feature_dim
        self._encoder_gamma = encoder_gamma
        self._experience_buffer_size = 50000

        self.observation_space = spaces.Box(np.array([-np.inf] * self._feature_dim),
                                  np.array([np.inf] * self._feature_dim))
        
        self.action_space = self._env.action_space
        
        
        self._num_hidden_layers = 2
        self._encoder_hidden_sizes = [encoder_hidden_size]*self._num_hidden_layers
        self._dynamics_hidden_sizes = [dynamics_hidden_size]*self._num_hidden_layers
        self._invdyn_hidden_sizes = [invdyn_hidden_size]*self._num_hidden_layers


        self._encoder = Encoder(sess=self._sess,
                        input_dim=self._state_dim,
                        output_dim=feature_dim,
                        hidden_sizes=self._encoder_hidden_sizes,
                        learning_rate=encoder_lr)

        self._dynamics = Dynamics(sess=self._sess,
                        state_dim=feature_dim,
                        action_dim=self._action_dim,
                        hidden_sizes=self._dynamics_hidden_sizes,
                        learning_rate=dynamics_lr)

        self._inv_dynamics = InverseDynamics(sess=self._sess,
                        state_dim=feature_dim,
                        action_dim=self._action_dim,
                        hidden_sizes=self._invdyn_hidden_sizes,
                        learning_rate=invdyn_lr)

        self._state = self._env.reset()

    def step(self, action):

        next_state, reward, terminal, info = self._env.step(action)

        encoded_next_state = self._encoder.predict(state=next_state)

        batch_state = self._state
        batch_action = action
        batch_reward = reward
        batch_next_state = next_state

        batch_encoded_state = self._encoder.predict(state=batch_state)
        batch_encoded_next_state = self._encoder.predict(state=batch_next_state)

        self._dynamics.update(state=batch_encoded_state, 
                            action=batch_action,
                            next_state=batch_encoded_next_state)

        self._inv_dynamics.update(state=batch_encoded_state, 
                            action=batch_action,
                            next_state=batch_encoded_next_state)

        dyn_gradients = self._dynamics.calc_gradients(state=batch_encoded_state,
                                                                                action=action)
        invdyn_gradients = self._inv_dynamics.calc_gradients(state=batch_encoded_state,
                                                                                next_state=batch_encoded_next_state)

        encoder_gradients = self._encoder.calc_gradients(state=batch_state)

        dyn_state_gradients = dyn_gradients[:self._feature_dim]
        dyn_action_gradients = dyn_gradients[self._feature_dim:]
        invdyn_state_gradients = invdyn_gradients[:self._feature_dim]
        invdyn_nstate_gradients = invdyn_gradients[self._feature_dim:]

        encoder_optim_grads = np.dot(invdyn_state_gradients, invdyn_nstate_gradients)*dyn_state_gradients

        self._encoder.update(state=batch_state,
                            optim_grads=encoder_optim_grads
                            )

        self._state = next_state


        return encoded_next_state, reward, terminal, info

    def reset(self):

        self._state = self._env.reset()

        encoded_state = self._encoder.predict(state=self._state)
        
        return encoded_state 
        

    def render(self):
        
        self._env.render()
    

    