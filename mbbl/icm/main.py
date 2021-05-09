#!/usr/bin/env python
# coding: utf-8

# # Curiosity-Driven Exploration

# In[1]:


import gym


# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


# In[3]:


import collections


# In[4]:


env = gym.make("HalfCheetah-v2")


# In[5]:


#utils
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


# In[6]:


#continuous
class Actor(nn.Module):
    def __init__(self, a_dim, s_dim):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        
        set_init([self.a1, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        return mu, sigma

    def choose_action(self, s):
        self.training = False
        self.mu_p, self.sigma_p = self.forward(s)
        self.m = self.distribution(self.mu_p.view(6, ).data, self.sigma_p.view(6, ).data)
        return self.m.sample()
    
    def action_log_prob(self, action):
        return self.m.log_prob(action)
    


class Critic(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(Critic, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True),
        )
        self.critic = nn.Linear(hidden_dims, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        est_reward = self.critic(features)
        return est_reward


# In[16]:


class InverseModel(nn.Module):
    def __init__(self, n_actions, hidden_dims):
        super(InverseModel, self).__init__()
        
        self.fc = nn.Linear(hidden_dims*2, 1024)
        self.mu = nn.Linear(1024, n_actions)
        self.sigma = nn.Linear(1024, n_actions)

        set_init([self.mu, self.sigma])
        self.distribution = torch.distributions.Normal
        
    def forward(self, features):
        features = features.view(1, -1) # (1, hidden_dims*2)
        hidden = self.fc(features) # (1, n_actions)
        mu = 2 * F.tanh(self.mu(hidden))
        sigma = F.softplus(self.sigma(hidden)) + 0.001      # avoid 0   
        
        return mu, sigma

class ForwardModel(nn.Module):
    def __init__(self, n_actions, hidden_dims):
        super(ForwardModel, self).__init__()
        self.fc = nn.Linear(hidden_dims+n_actions, hidden_dims)
        self.eye = torch.eye(n_actions)
        
    def forward(self, action, features):
        x = torch.cat([action, features], dim=-1) # (1, n_actions+hidden_dims)
        features = self.fc(x) # (1, hidden_dims)
        return features

class FeatureExtractor(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)
        
    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y


# In[17]:


class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()
    
    def forward(self, action_log_prob, reward):
        
        loss = -torch.mean(action_log_prob*reward)
        return loss


# In[18]:


def to_tensor(x, dtype=None):
    return torch.tensor(x, dtype=dtype).unsqueeze(0)


# In[19]:


class ConfigArgs:
    beta = 0.2
    lamda = 0.1
    eta = 100.0 # scale factor for intrinsic reward
    discounted_factor = 0.99
    lr_critic = 0.001
    lr_actor = 0.002
    lr_icm = 0.001
    max_eps = 100000
    sparse_mode = True

args = ConfigArgs()


# In[20]:


# Actor Critic
actor = Actor(env.action_space.shape[0], env.observation_space.shape[0])
critic = Critic(space_dims=env.observation_space.shape[0], hidden_dims=512)


# In[21]:


# ICM
feature_extractor = FeatureExtractor(env.observation_space.shape[0], 512)
forward_model = ForwardModel(env.action_space.shape[0], 512)
inverse_model = InverseModel(env.action_space.shape[0], 512)


# In[22]:


# Actor Critic
a_optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
c_optim = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

# ICM
icm_params = list(feature_extractor.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters())
icm_optim = torch.optim.Adam(icm_params, lr=args.lr_icm)


# In[23]:


pg_loss = PGLoss()
mse_loss = nn.MSELoss()
xe_loss = nn.GaussianNLLLoss()


# In[ ]:


global_step = 0
n_eps = -1
reward_lst = []
mva_lst = []
mva = 0.
avg_ireward_lst = []


while n_eps < args.max_eps:
    n_eps += 1
    next_obs = to_tensor(env.reset(), dtype=torch.float)
    done = False
    score = 0
    ireward_lst = []
    reward_m=0
    
    while not done:
        obs = next_obs
        a_optim.zero_grad()
        c_optim.zero_grad()
        icm_optim.zero_grad()
        
        # estimate action with policy network
        #policy = actor(obs)
        
        
        action = actor.choose_action(obs) 
        log_prob = actor.action_log_prob(action)
        
        # interaction with environment
        #clip
        control = np.clip(action.numpy(), env.action_space.low, env.action_space.high)
        if n_eps%10000 == 0: env.render()
        next_obs, reward, done, info = env.step(control)
        next_obs = to_tensor(next_obs, dtype=torch.float)

        extrinsic_reward = to_tensor([0.], dtype=torch.float) if args.sparse_mode else to_tensor([reward], dtype=torch.float)
        t_action = to_tensor(action)
        
        v = critic(obs)[0]
        next_v = critic(next_obs)[0]
        
        # ICM
        obs_cat = torch.cat([obs, next_obs], dim=0)
        features = feature_extractor(obs_cat) # (2, hidden_dims)
        inverse_mu, inverse_sigma = inverse_model(features) # (n_actions)\
        inverse_var = torch.square(inverse_sigma)
        est_next_features = forward_model(t_action, features[0:1])

        # Loss - ICM
        forward_loss = mse_loss(est_next_features, features[1])        
        inverse_loss = xe_loss(inverse_mu.view(6, ).data, action.detach(), inverse_var.view(6, ).data)
        icm_loss = (1-args.beta)*inverse_loss + args.beta*forward_loss
        
        # Reward
        intrinsic_reward = args.eta*forward_loss.detach()
        if done:
            total_reward = intrinsic_reward
            advantage = total_reward - v
            c_target = total_reward
        else:
            total_reward = extrinsic_reward + intrinsic_reward
            advantage = total_reward + args.discounted_factor*next_v - v
            c_target = total_reward + args.discounted_factor*next_v
        
        # Loss - Actor Critic
        actor_loss = pg_loss(log_prob, to_tensor(advantage, dtype = torch.float).detach())
        critic_loss = mse_loss(v, c_target.detach())
        ac_loss = actor_loss + critic_loss
        
        # Update
        loss = args.lamda*ac_loss + icm_loss
        loss.backward()
        icm_optim.step()
        a_optim.step()
        c_optim.step()
        
        if not done:
            score += reward
        
        ireward_lst.append(intrinsic_reward.item())
        
        global_step += 1
    #env.close()
    avg_intrinsic_reward = sum(ireward_lst) / len(ireward_lst)
    mva = 0.95*mva + 0.05*score
    reward_lst.append(score)
    avg_ireward_lst.append(avg_intrinsic_reward)
    mva_lst.append(mva)
    print('Episodes: {}, AVG Score: {:.3f}, Score: {}, AVG reward i: {:.6f}'.format(n_eps, mva, score, avg_intrinsic_reward))


# ## Visualization

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.plot(reward_lst)
plt.ylabel('Score')
plt.show()


# In[ ]:


plt.plot(mva_lst)
plt.ylabel('Moving Average Score')
plt.show()


# In[ ]:


np.save('curiosity-Half-Cheetah-mva.npy', np.array(mva_lst))

