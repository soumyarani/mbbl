#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_util import device, FLOAT

import numpy as np

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class MLPnetwork1(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, activation=nn.LeakyReLU):
        super(MLPnetwork1, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.value = nn.Sequential(
            nn.Linear(dim_input, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_output)
        )

        self.value.apply(init_weight)

    def forward(self, inputs):
        value = self.value(inputs)
        return value

class MLPnetwork2(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, dim_hidden=128):
        super(MLPnetwork2, self).__init__()

        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.value = nn.Sequential(
            nn.Linear(dim_input1+dim_input2, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_output)
        )

        self.value.apply(init_weight)

    def forward(self, input1, input2):
        combined = torch.cat([input1, input2], dim=1)
        value = self.value(combined)
        return value

class Encoder():

    def __init__(self, state_dim, latent_dim, action_dim):

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        self.encoder = MLPnetwork1(self.state_dim, self.latent_dim, dim_hidden=128).to(device)
        self.forward_dynamics = MLPnetwork2(self.latent_dim, self.action_dim, self.latent_dim, dim_hidden=128).to(device)
        self.inverse_dynamics = MLPnetwork2(self.latent_dim, self.latent_dim, self.action_dim, dim_hidden=128).to(device)

        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=0.0003)
        self.optimizer_fdyn = optim.Adam(self.forward_dynamics.parameters(), lr=0.0003)
        self.optimizer_idyn = optim.Adam(self.inverse_dynamics.parameters(), lr=0.0003)

        self.encoder_loss = 0
        self.fdyn_loss = 0
        self.idyn_loss = 0

    def update_encoder(self, state1, action1, reward1, next_state1, state2, action2, reward2, next_state2):

        encoded_state1 = self.encoder(state1)
        encoded_nstate1 = self.encoder(next_state1)
        pred_encoded_nstate1 = self.forward_dynamics(encoded_state1, action1)
        pred_encoded_action1 = self.inverse_dynamics(encoded_state1, pred_encoded_nstate1)

        encoded_state2 = self.encoder(state2)
        encoded_nstate2 = self.encoder(next_state2)
        pred_encoded_nstate2 = self.forward_dynamics(encoded_state2, action2)
        pred_encoded_action2 = self.inverse_dynamics(encoded_state2, pred_encoded_nstate2)

        self.fdyn_loss = nn.MSELoss()(pred_encoded_nstate1, encoded_nstate1) + nn.MSELoss()(pred_encoded_nstate2, encoded_nstate2)
        self.idyn_loss = nn.MSELoss()(pred_encoded_action1, action1) + nn.MSELoss()(pred_encoded_action2, action2)

        latent_diff = nn.MSELoss()(encoded_state1, encoded_nstate2)
        reward_diff = nn.MSELoss()(reward1, reward2)
        transition_diff = self.fdyn_loss + self.idyn_loss

        self.encoder_loss = nn.MSELoss()(latent_diff, reward_diff + transition_diff)

        self.optimizer_fdyn.zero_grad()
        self.optimizer_idyn.zero_grad()
        self.optimizer_encoder.zero_grad()

        self.fdyn_loss.backward(retain_graph=True)
        self.idyn_loss.backward(retain_graph=True)
        self.encoder_loss.backward(retain_graph=True)
        
        self.optimizer_fdyn.step()
        self.optimizer_idyn.step()
        self.optimizer_encoder.step()

        print(self.forward_dynamics.get_mean_var())

    def update_writer(self, writer, i_iter):

        enco_loss = torch.mean(self.encoder_loss)
        fd_loss = torch.mean(self.fdyn_loss)
        id_loss = torch.mean(self.idyn_loss)

        writer.add_scalar("encodings/encoder_loss", enco_loss, i_iter)
        writer.add_scalar("encodings/forward_dynamics_loss", fd_loss, i_iter)
        writer.add_scalar("encodings/inverse_dynamics_loss", id_loss, i_iter)

        return writer
