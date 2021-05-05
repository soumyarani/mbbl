#!/usr/bin/env python
import torch.nn as nn

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Policy(nn.Module):
    def __init__(
        self,
        dim_state,
        dim_action,
        max_action=None,
        dim_hidden=256,
        activation=nn.LeakyReLU,
    ):
        super(Policy, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action
        self.max_action = max_action
        self.action = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action),
            nn.Tanh(),
        )
        self.apply(init_weight)

    def forward(self, x):
        action = self.action(x)
        return action * self.max_action

    def get_action_log_prob(self, states):
        action = self.forward(states)
        return action, None
