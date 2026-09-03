# -*- coding: utf-8 -*-
"""Baseline algorithms on the shared SAINT encoders.

SACDiscrete : discrete Soft Actor-Critic - factorized per-sector softmax actor,
              twin per-sector Q critics aggregated with the same VDN-style mean
              as the rest of the DQN family, auto-tuned temperature.
              token_variant selects the encoder: "identity" (baseline row) or
              "tok3" (the probe answering whether the method should train with
              SAC instead of double-DQN). forward() returns actor logits, so
              per-sector argmax (greedy execution and the eval path) works
              unchanged; attach_cache forwards to all three networks.
SAINTQMix   : QMIX - per-sector Q agents plus a monotonic mixing network with
              hypernetworks conditioned on [angles, alpha, progress].
              Decentralized greedy execution is unchanged; only the training
              target mixes, so forward() stays the per-sector Q tensor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACDiscrete(nn.Module):
    name = "saint_sac"

    def __init__(self, num_actions: int, token_dim: int = 64, num_layers: int = 2,
                 num_heads: int = 4, ffn_dim: int = 256, token_variant: str = "identity",
                 sector_map_px: int = 64) -> None:
        super().__init__()
        from saint_policy import SAINTQNetwork, SAINTQNetworkTok3
        kw = dict(num_actions=num_actions, token_dim=token_dim, num_layers=num_layers,
                  num_heads=num_heads, ffn_dim=ffn_dim)
        if token_variant == "tok3":
            cls = SAINTQNetworkTok3
            kw["sector_map_px"] = int(sector_map_px)
        else:
            cls = SAINTQNetwork
        self.token_variant = str(token_variant)
        self.actor = cls(**kw)
        self.q1 = cls(**kw)
        self.q2 = cls(**kw)
        self.log_alpha = nn.Parameter(torch.tensor(float(math.log(0.2))))

    def attach_cache(self, base_dbm, num_angles) -> None:
        for m in (self.actor, self.q1, self.q2):
            if hasattr(m, "attach_cache"):
                m.attach_cache(base_dbm, num_angles)

    def forward(self, *state):
        return self.actor(*state)


class SAINTQMix(nn.Module):
    name = "saint_qmix"

    def __init__(self, num_actions: int, token_dim: int = 64, num_layers: int = 2,
                 num_heads: int = 4, ffn_dim: int = 256, mix_hidden: int = 32,
                 n_agents: int = 9) -> None:
        super().__init__()
        from saint_policy import SAINTQNetwork
        self.qnet = SAINTQNetwork(num_actions=num_actions, token_dim=token_dim,
                                  num_layers=num_layers, num_heads=num_heads, ffn_dim=ffn_dim)
        self.n_agents = int(n_agents)
        self.mix_hidden = int(mix_hidden)
        g = self.n_agents + 2  # [angles(9), alpha(1), progress(1)]
        self.hyper_w1 = nn.Linear(g, self.n_agents * self.mix_hidden)
        self.hyper_b1 = nn.Linear(g, self.mix_hidden)
        self.hyper_w2 = nn.Linear(g, self.mix_hidden)
        self.hyper_b2 = nn.Sequential(nn.Linear(g, self.mix_hidden), nn.ReLU(),
                                      nn.Linear(self.mix_hidden, 1))

    def forward(self, *state):
        return self.qnet(*state)

    def mix(self, q_sel: torch.Tensor, angles: torch.Tensor, alpha: torch.Tensor,
            progress: torch.Tensor) -> torch.Tensor:
        g = torch.cat([angles, alpha, progress], dim=1)
        w1 = torch.abs(self.hyper_w1(g)).view(-1, self.n_agents, self.mix_hidden)
        b1 = self.hyper_b1(g)
        h = F.elu(torch.bmm(q_sel.unsqueeze(1), w1).squeeze(1) + b1)
        w2 = torch.abs(self.hyper_w2(g))
        return (h * w2).sum(dim=1) + self.hyper_b2(g).squeeze(-1)
