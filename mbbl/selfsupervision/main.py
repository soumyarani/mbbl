#######################################################################################
# Main File to compare DDPG and PPO performance with and without encoder
# Author: Utkarsh Aashu Mishra (utkarsh75477@gmail.com)
#######################################################################################

import os
import random
import argparse
import math
import numpy as np
import gym
import tensorflow as tf
from wrapped_env import WrappedEnv

# %%
# main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',
                        type=str,
                        default='ddpg',
                        help='Algorithm for training DDPG/PPO')
    parser.add_argument('-a',
                        '--agent',
                        type=str,
                        default='trained_agent',
                        help='the name of the agent to load/store')
    parser.add_argument('-r',
                        '--render',
                        action='store_true',
                        help='enable rendering')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='./results/',
                        help='output directory for reading/writing results')
    parser.add_argument('--use_encoder',
                        action='store_true',
                        help='Use Envioronment Encoder')            

    return parser.parse_args()

def random_run_for_encoder_training(env, num_epochs, num_iters):

    for epoch in range(num_epochs):
        print("Training for Epoch: {}".format(epoch))
        for j in range(num_iters):
            _, _, done, _ = env.step(env.action_space.sample())

            if done:
                env.reset()

    return env


def main():
    args = parse_args()
    algorithm = args.algo
    agent = args.agent
    output = args.output
    use_encoder = args.use_encoder
    time_steps = int(2e6)
    env_name = 'HalfCheetah-v2'

    tf.reset_default_graph()

    with tf.Session() as sess:

        def make_env(use_encoder=True, env_name=env_name):

            if use_encoder:
                return WrappedEnv(sess=sess,
                                env_name=env_name,
                                feature_dim=10,
                                encoder_gamma=0.98,
                                encoder_training_batch=500,
                                encoder_hidden_size=64,
                                dynamics_hidden_size=64,
                                invdyn_hidden_size=64,
                                encoder_lr=0.001,
                                dynamics_lr=0.001,
                                invdyn_lr=0.001)
            else:
                return gym.make(env_name)

        if algorithm=='ddpg':

            from stable_baselines.common.cmd_util import SubprocVecEnv
            from stable_baselines.common.callbacks import CheckpointCallback
            from stable_baselines.ddpg.policies import MlpPolicy
            from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
            from stable_baselines import DDPG

            env = make_env(use_encoder=use_encoder)

            if use_encoder:
                env = random_run_for_encoder_training(env, num_epochs = 200, num_iters=500)

            sess.run(tf.global_variables_initializer())

            n_actions = env.action_space.shape[-1]
            param_noise = None
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=float(0.22) * np.ones(n_actions))

            policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[128, 128])

            model = DDPG(MlpPolicy,
                         env,
                         gamma=0.95,
                         batch_size=256,
                         verbose=1,
                         param_noise=param_noise,
                         action_noise=action_noise,
                         policy_kwargs=policy_kwargs,
                         tensorboard_log=output+algorithm+'_'+str(use_encoder)+'/' + 'log/')

            checkpoint_callback = CheckpointCallback(save_freq=4000,
                                                     save_path=output+algorithm+'_'+str(use_encoder)+'/',
                                                     name_prefix='agent')
            model.set_env(env)
            model.learn(total_timesteps=time_steps,
                        callback=checkpoint_callback,
                        reset_num_timesteps=False)
            model.save(output+algorithm+'_'+str(use_encoder)+'/'+ agent)
            env.close()
            del model

        elif algorithm=='ppo':

            from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
            from stable_baselines import PPO2
            from stable_baselines.common.cmd_util import SubprocVecEnv
            from stable_baselines.common.callbacks import CheckpointCallback
    
            env = make_env(use_encoder=use_encoder)

            if use_encoder:
                env = random_run_for_encoder_training(env, num_epochs = 200, num_iters=500)

            sess.run(tf.global_variables_initializer())

            policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[128, 128])

            model = PPO2(MlpPolicy,
                         env,
                         n_steps = 2048,
                         nminibatches = 32,
                         lam = 0.95,
                         gamma = 0.99,
                         noptepochs = 10,
                         verbose=1,
                         policy_kwargs=policy_kwargs,
                         tensorboard_log=output+algorithm+'_'+str(use_encoder)+'/' + 'log/')

            checkpoint_callback = CheckpointCallback(save_freq=4096,
                                                     save_path=output+algorithm+'_'+str(use_encoder)+'/',
                                                     name_prefix='agent')
            model.learn(total_timesteps=time_steps,
                        callback=checkpoint_callback,
                        reset_num_timesteps=False)
            model.save(output+algorithm+'_'+str(use_encoder)+'/' + agent)
            env.close()
            del model

if __name__ == '__main__':
    main()
