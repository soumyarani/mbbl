#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_util import device, FLOAT

import numpy as np

class GaussianModel1(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, max_sigma=1e1, min_sigma=1e-4):

        super().__init__()

        self.fc = nn.Linear(dim_input, dim_hidden)
        self.ln = nn.LayerNorm(dim_hidden)
        self.fc_mu = nn.Linear(dim_hidden, dim_output)
        self.fc_sigma = nn.Linear(dim_hidden, dim_output)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.xavier_normal_(self.fc_sigma.weight)
        nn.init.constant_(self.fc_sigma.bias, 0.0)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma

    def forward(self, input1):
        x = input1
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))
        sigma = (
            self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        )
        return mu, sigma

    def sample_prediction(self, input1):
        mu, sigma = self(input1)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

class GaussianModel2(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, dim_hidden=128, max_sigma=1e1, min_sigma=1e-4):

        super().__init__()

        self.fc = nn.Linear(dim_input1 + dim_input2, dim_hidden)
        self.ln = nn.LayerNorm(dim_hidden)
        self.fc_mu = nn.Linear(dim_hidden, dim_output)
        self.fc_sigma = nn.Linear(dim_hidden, dim_output)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.xavier_normal_(self.fc_sigma.weight)
        nn.init.constant_(self.fc_sigma.bias, 0.0)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))
        sigma = (
            self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        )
        return mu, sigma

    def sample_prediction(self, input1, input2):
        mu, sigma = self(input1, input2)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

class Encoder():

    def __init__(self, state_dim, latent_dim, action_dim):

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.encoder = GaussianModel1(self.state_dim, self.latent_dim, dim_hidden=128).to(device)
        self.forward_dynamics = GaussianModel2(self.latent_dim, self.action_dim, self.latent_dim, dim_hidden=128).to(device)
        self.inverse_dynamics = GaussianModel2(self.latent_dim, self.latent_dim, self.action_dim, dim_hidden=128).to(device)

        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=0.0003)
        self.optimizer_fdyn = optim.Adam(self.forward_dynamics.parameters(), lr=0.0003)
        self.optimizer_idyn = optim.Adam(self.inverse_dynamics.parameters(), lr=0.0003)

        self.encoder_loss = 0
        self.fdyn_loss = 0
        self.idyn_loss = 0

    def update_encoder(self, state1, action1, reward1, next_state1, state2, action2, reward2, next_state2):

        encoded_state1 = self.encoder.sample_prediction(state1)
        encoded_nstate1 = self.encoder.sample_prediction(next_state1)
        pred_encoded_nstate1 = self.forward_dynamics.sample_prediction(encoded_state1, action1)
        pred_encoded_action1 = self.inverse_dynamics.sample_prediction(encoded_state1, pred_encoded_nstate1)

        encoded_state2 = self.encoder.sample_prediction(state2)
        encoded_nstate2 = self.encoder.sample_prediction(next_state2)
        pred_encoded_nstate2 = self.forward_dynamics.sample_prediction(encoded_state2, action2)
        pred_encoded_action2 = self.inverse_dynamics.sample_prediction(encoded_state2, pred_encoded_nstate2)

        self.fdyn_loss = nn.MSELoss()(pred_encoded_nstate1, encoded_nstate1) + nn.MSELoss()(pred_encoded_nstate2, encoded_nstate2)
        self.idyn_loss = nn.MSELoss()(pred_encoded_action1, action1) + nn.MSELoss()(pred_encoded_action2, action2)

        tensor_type = type(reward1)
        enocder_values = tensor_type(reward1.size(0), 1).to(device)
        enocder_targets = tensor_type(reward1.size(0), 1).to(device)

        mu_fd1 , sigma_fd1 = self.forward_dynamics(encoded_state1, action1)
        mu_id1 , sigma_id1 = self.inverse_dynamics(encoded_state1, pred_encoded_nstate1)
        mu_fd2 , sigma_fd2 = self.forward_dynamics(encoded_state2, action2)
        mu_id2 , sigma_id2 = self.inverse_dynamics(encoded_state2, pred_encoded_nstate2)

        for i in range(reward1.size(0)):
            enocder_values[i] = self.calc_L1(encoded_state1[i], encoded_state2[i])
            enocder_targets[i] = torch.abs(reward1[i] - reward2[i]) + \
                                0.01*self.calc_wasserstein(mu_fd1, sigma_fd1, mu_fd2, sigma_fd2) + \
                                0.005*self.calc_wasserstein(mu_id1, sigma_id1, mu_id2, sigma_id2)

        self.encoder_loss = nn.MSELoss()(enocder_values, enocder_targets)

        self.optimizer_fdyn.zero_grad()
        self.optimizer_idyn.zero_grad()
        self.optimizer_encoder.zero_grad()

        self.fdyn_loss.backward(retain_graph=True)
        self.idyn_loss.backward(retain_graph=True)
        self.encoder_loss.backward(retain_graph=True)
        
        self.optimizer_fdyn.step()
        self.optimizer_idyn.step()
        self.optimizer_encoder.step()

    def update_writer(self, writer, i_iter):

        enco_loss = torch.mean(self.encoder_loss)
        fd_loss = torch.mean(self.fdyn_loss)
        id_loss = torch.mean(self.idyn_loss)

        writer.add_scalar("encodings/encoder_loss", enco_loss, i_iter)
        writer.add_scalar("encodings/forward_dynamics_loss", fd_loss, i_iter)
        writer.add_scalar("encodings/inverse_dynamics_loss", id_loss, i_iter)

        return writer

    def calc_wasserstein(self, mu1, sigma1, mu2, sigma2):

        delta_m = torch.linalg.norm(mu1 - mu2)
        delta_s = torch.linalg.norm(torch.sqrt(sigma1) - torch.sqrt(sigma2), 'fro')

        return torch.sum(delta_m + delta_s)

    def calc_L1(self, state1, state2):

        return torch.linalg.norm((state1 - state2), ord=1)