#!/usr/bin/env python3
"""Track A policy modules: factorized PPO baselines and plain SAINT-PPO.

All actor-critics share ONE global-state encoder that is a verbatim re-creation
of the ``V1BranchingDQN`` feature stack (map CNN + DeepSets feedback encoder +
angle/alpha/progress MLPs + 248->256->256 trunk), so the state contract is
identical to the BDQN gate arms::

    maps [B,3,64,64] + feedback [B,M,10] + angles/89 [B,9] + alpha [B,1] + t/H [B,1]

Only the *action-side* differs:

* ``FactorizedActorCritic``          : Linear(256, 9*5) logits, Linear(256, 1) value
* ``FactorizedCapMatchedActorCritic``: MLP head sized to match SAINT's extra
  parameters (capacity control)
* ``SAINTActorCritic``               : nine sector tokens = FiLM(h_g) * E_ID + beta,
  2-layer pre-LN Transformer encoder over the 9 tokens, per-token 5-way logits,
  critic on [h_g, mean-pool(tokens)]

Reference: Landers et al., "SAINT: Attention-Based Policies for Discrete
Combinatorial Action Spaces", arXiv:2505.12109 (sub-action tokens as an
unordered set conditioned on the global state).  This is the *plain* variant:
identity tokens only, no sector-specific map features / geometry / attention
bias.  Those are deliberate later ablations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical


NUM_SECTORS = 9
NUM_ACTIONS = 5
GLOBAL_DIM = 256


def small_init_(layer: nn.Linear, gain: float = 0.01) -> nn.Linear:
    """Standard PPO trick: near-uniform initial policy / small initial value.
    Applied identically to every algorithm's final actor and value layers so
    that initial policies and early update magnitudes are comparable."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class FeedbackEncoder(nn.Module):
    """Permutation-invariant DeepSets encoder (identical to the BDQN one)."""

    def __init__(self, report_dim: int = 10, output_dim: int = 64):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(report_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.pool_net = nn.Sequential(
            nn.Linear(128, 96), nn.ReLU(), nn.Linear(96, output_dim), nn.ReLU()
        )

    def forward(self, reports: torch.Tensor) -> torch.Tensor:
        z = self.point_net(reports)
        pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)
        return self.pool_net(pooled)


class GlobalStateEncoder(nn.Module):
    """maps + Y_t + theta_t + alpha + t/H  ->  h_g in R^256 (same as V1BranchingDQN)."""

    def __init__(self, num_sectors: int = NUM_SECTORS, map_channels: int = 3):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feedback_encoder = FeedbackEncoder(report_dim=10, output_dim=64)
        self.angle_net = nn.Sequential(
            nn.Linear(num_sectors, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.alpha_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.progress_net = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(248, GLOBAL_DIM), nn.ReLU(), nn.Linear(GLOBAL_DIM, GLOBAL_DIM), nn.ReLU()
        )

    def forward(
        self,
        maps: torch.Tensor,
        feedback: torch.Tensor,
        angles: torch.Tensor,
        alpha: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            self.map_encoder(maps).flatten(1),
            self.feedback_encoder(feedback),
            self.angle_net(angles),
            self.alpha_net(alpha),
            self.progress_net(progress),
        ]
        return self.trunk(torch.cat(features, dim=1))


class FactorizedActorCritic(nn.Module):
    """Independent per-sector categorical heads on the shared global feature."""

    name = "factorized_ppo"

    def __init__(self, num_sectors: int = NUM_SECTORS, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.num_sectors = int(num_sectors)
        self.num_actions = int(num_actions)
        self.encoder = GlobalStateEncoder(num_sectors)
        self.policy_head = small_init_(nn.Linear(GLOBAL_DIM, self.num_sectors * self.num_actions))
        self.value_head = small_init_(nn.Linear(GLOBAL_DIM, 1), gain=1.0)

    def forward(self, maps, feedback, angles, alpha, progress):
        h = self.encoder(maps, feedback, angles, alpha, progress)
        logits = self.policy_head(h).view(-1, self.num_sectors, self.num_actions)
        return logits, self.value_head(h).squeeze(-1)


class FactorizedCapMatchedActorCritic(nn.Module):
    """Factorized heads with an MLP whose size roughly matches SAINT's extras
    (~170k params), isolating 'more parameters' from 'sector interaction'."""

    name = "factorized_ppo_capmatched"

    def __init__(
        self,
        num_sectors: int = NUM_SECTORS,
        num_actions: int = NUM_ACTIONS,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_sectors = int(num_sectors)
        self.num_actions = int(num_actions)
        self.encoder = GlobalStateEncoder(num_sectors)
        self.policy_head = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            small_init_(nn.Linear(hidden, self.num_sectors * self.num_actions)),
        )
        self.value_head = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 128), nn.ReLU(), small_init_(nn.Linear(128, 1), gain=1.0)
        )

    def forward(self, maps, feedback, angles, alpha, progress):
        h = self.encoder(maps, feedback, angles, alpha, progress)
        logits = self.policy_head(h).view(-1, self.num_sectors, self.num_actions)
        return logits, self.value_head(h).squeeze(-1)


class SAINTActorCritic(nn.Module):
    """Plain SAINT: FiLM-conditioned identity tokens + self-attention + per-token heads."""

    name = "saint_ppo"

    def __init__(
        self,
        num_sectors: int = NUM_SECTORS,
        num_actions: int = NUM_ACTIONS,
        token_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        per_sector_heads: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_sectors = int(num_sectors)
        self.num_actions = int(num_actions)
        self.token_dim = int(token_dim)
        self.encoder = GlobalStateEncoder(num_sectors)
        # e_k^ID  (k = 1..9)
        self.sector_embedding = nn.Parameter(torch.randn(self.num_sectors, self.token_dim) * 0.02)
        # FiLM: gamma(h_g), beta(h_g); gamma initialised around 1.
        self.film = nn.Linear(GLOBAL_DIM, 2 * self.token_dim)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[: self.token_dim].fill_(1.0)
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.per_sector_heads = bool(per_sector_heads)
        if self.per_sector_heads:
            self.actor_heads = nn.ModuleList(
                [small_init_(nn.Linear(self.token_dim, self.num_actions)) for _ in range(self.num_sectors)]
            )
        else:
            self.actor_head = small_init_(nn.Linear(self.token_dim, self.num_actions))
        self.value_head = nn.Sequential(
            nn.Linear(GLOBAL_DIM + self.token_dim, 128), nn.ReLU(), small_init_(nn.Linear(128, 1), gain=1.0)
        )

    def tokens(self, h: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(h).chunk(2, dim=-1)  # [B,d] each
        e = self.sector_embedding[None]  # [1,K,d]
        return gamma[:, None, :] * e + beta[:, None, :]  # [B,K,d]

    def forward(self, maps, feedback, angles, alpha, progress):
        h = self.encoder(maps, feedback, angles, alpha, progress)  # [B,256]
        x = self.transformer(self.tokens(h))  # [B,K,d]
        x = self.token_norm(x)
        if self.per_sector_heads:
            logits = torch.stack(
                [head(x[:, k]) for k, head in enumerate(self.actor_heads)], dim=1
            )
        else:
            logits = self.actor_head(x)  # [B,K,A]
        value = self.value_head(torch.cat([h, x.mean(dim=1)], dim=-1)).squeeze(-1)
        return logits, value


class SAINTQNetwork(nn.Module):
    """SAINT-DQN: the SAINT token/attention stack read as a branching Q-network.

    Same GlobalStateEncoder as V1BranchingDQN / the PPO policies; FiLM-conditioned
    sector identity tokens; self-attention; per-token linear head interpreted as
    per-sector Q-values (not logits).  forward() accepts the extra ``mismatch``
    tensor produced by legacy.state_tensors and ignores it, so it drops into the
    existing (double-)BDQN training loop unchanged.
    """

    name = "saint_dqn"

    def __init__(
        self,
        num_sectors: int = NUM_SECTORS,
        num_actions: int = NUM_ACTIONS,
        token_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_sectors = int(num_sectors)
        self.num_actions = int(num_actions)
        self.token_dim = int(token_dim)
        self.encoder = GlobalStateEncoder(num_sectors)
        self.sector_embedding = nn.Parameter(torch.randn(self.num_sectors, self.token_dim) * 0.02)
        self.film = nn.Linear(GLOBAL_DIM, 2 * self.token_dim)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[: self.token_dim].fill_(1.0)
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=int(num_heads), dim_feedforward=int(ffn_dim),
            dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.q_head = nn.Linear(self.token_dim, self.num_actions)  # standard init: Q magnitudes, not probabilities

    def forward(self, maps, feedback, angles, alpha, progress, mismatch=None):
        h = self.encoder(maps, feedback, angles, alpha, progress)
        gamma, beta = self.film(h).chunk(2, dim=-1)
        x = gamma[:, None, :] * self.sector_embedding[None] + beta[:, None, :]
        x = self.token_norm(self.transformer(x))
        return self.q_head(x)  # [B, K, A] Q-values



class SAINTQNetworkTok(SAINTQNetwork):
    """SAINT-DQN with per-sector token features derived from existing inputs.

    Token k additionally receives, before the transformer:
      * its own commanded angle theta_k (the base network only sees the 9-dim
        angle vector through the global encoder);
      * grouped-reporter features: reporters are assigned to the sector with the
        strongest reported qRSRP (computable from the feedback tensor itself);
        per sector: [share of reporters, mean own-sector qRSRP, mean qSINR,
        masked DeepSets pooling of the group's full reports].
    Everything is a function of the unchanged (maps, feedback, angles, ...) state,
    so the environment/state contract is identical to every other algorithm.
    """

    name = "saint_dqn_tok"

    def __init__(self, *args, group_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_dim = int(group_dim)
        self.group_point = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, self.group_dim), nn.ReLU())
        # per-token side features: angle (1) + share (1) + mean own qRSS (1) + mean qSINR (1) + group pool (group_dim)
        self.side_proj = nn.Linear(4 + self.group_dim, self.token_dim)

    def _sector_side_features(self, feedback: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        B, M, _ = feedback.shape
        K = self.num_sectors
        rss = feedback[..., :K]                    # [B,M,K] normalized qRSRP
        sinr = feedback[..., K]                    # [B,M]
        assign = rss.argmax(dim=-1)                # [B,M]
        one_hot = nn.functional.one_hot(assign, K).float()          # [B,M,K]
        count = one_hot.sum(dim=1)                                  # [B,K]
        share = count / float(M)
        own_rss = (rss.gather(-1, assign[..., None]).squeeze(-1))   # [B,M]
        denom = count.clamp_min(1.0)
        mean_own = torch.einsum("bm,bmk->bk", own_rss, one_hot) / denom
        mean_sinr = torch.einsum("bm,bmk->bk", sinr, one_hot) / denom
        z = self.group_point(feedback)                              # [B,M,D]
        pool = torch.einsum("bmd,bmk->bkd", z, one_hot) / denom[..., None]
        side = torch.cat([angles[..., None], share[..., None], mean_own[..., None], mean_sinr[..., None], pool], dim=-1)
        return self.side_proj(side)                                 # [B,K,token_dim]

    def forward(self, maps, feedback, angles, alpha, progress, mismatch=None):
        h = self.encoder(maps, feedback, angles, alpha, progress)
        gamma, beta = self.film(h).chunk(2, dim=-1)
        x = gamma[:, None, :] * self.sector_embedding[None] + beta[:, None, :]
        x = x + self._sector_side_features(feedback, angles)
        x = self.token_norm(self.transformer(x))
        return self.q_head(x)


class SAINTQNetworkTok2(SAINTQNetworkTok):
    """Tok2: per-sector multi-channel mini-maps encoded by one shared CNN.

    For sector k a 5-channel stack at ``sector_map_px``:
      [ own RSRP map (from the cached per-sector sweep at the CURRENT angle),
        best-server RSRP (global), serving SINR (global), demand density (global),
        "k is best-server" mask ].
    The stacks are RECONSTRUCTED on the fly from (angles, global maps) via the
    attached cache tensor, so the replay buffer is unchanged.  The shared CNN
    output is added to the token alongside Tok's feedback-group features.
    """

    name = "saint_dqn_tok2"

    def __init__(self, *args, sector_map_px: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.sector_map_px = int(sector_map_px)
        self.sector_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 5, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.sector_map_proj = nn.Linear(64, self.token_dim)
        self.register_buffer("sector_sweep", torch.zeros(1), persistent=False)  # replaced by attach_cache
        self._rss_norm = (-115.0, 85.0)

    def attach_cache(self, base_dbm: torch.Tensor, num_angles_task: int) -> None:
        """base_dbm: [A, K, H, W] cached per-sector RSRP sweep (dBm)."""
        px = self.sector_map_px
        A, K, H, W = base_dbm.shape
        down = nn.functional.adaptive_avg_pool2d(base_dbm.reshape(A * K, 1, H, W), (px, px)).reshape(A, K, px, px)
        lo, sc = self._rss_norm
        self.sector_sweep = ((down - lo) / sc).clamp(0.0, 1.0)
        self.task_num_angles = int(num_angles_task)

    def _sector_stacks(self, maps: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        B = maps.shape[0]; K = self.num_sectors; px = self.sector_map_px
        gmap = nn.functional.adaptive_avg_pool2d(maps, (px, px))          # [B,3,px,px]
        idx = (angles * 89.0).round().long().clamp(0, self.sector_sweep.shape[0] - 1)  # [B,K]
        own = self.sector_sweep[idx, torch.arange(K, device=maps.device)[None].expand(B, -1)]  # [B,K,px,px]
        mask = nn.functional.one_hot(own.argmax(dim=1), K).permute(0, 3, 1, 2).float()  # [B,K,px,px]
        g = gmap[:, None].expand(-1, K, -1, -1, -1)                        # [B,K,3,px,px]
        stack = torch.cat([own[:, :, None], g, mask[:, :, None]], dim=2)   # [B,K,5,px,px]
        return stack.reshape(B * K, 5, px, px)

    def forward(self, maps, feedback, angles, alpha, progress, mismatch=None):
        h = self.encoder(maps, feedback, angles, alpha, progress)
        gamma, beta = self.film(h).chunk(2, dim=-1)
        x = gamma[:, None, :] * self.sector_embedding[None] + beta[:, None, :]
        x = x + self._sector_side_features(feedback, angles)
        z = self.sector_cnn(self._sector_stacks(maps, angles)).flatten(1)
        x = x + self.sector_map_proj(z).view(maps.shape[0], self.num_sectors, self.token_dim)
        x = self.token_norm(self.transformer(x))
        return self.q_head(x)


class SAINTQNetworkTok3(SAINTQNetworkTok):
    """Tok3: per-sector demand-derived physical scalars (shape-invariant).

    Tok2 fed each token a raw per-sector map crop and let a CNN interpret it;
    the CNN memorised seen demand-family shapes (OOD collapse).  Tok3 instead
    hands each token low-dimensional aggregates of "how MY crowd is served
    under the current angles" -- quantities the objective is actually made of:

      m_k    demand mass inside my best-server region
      r_k    demand-weighted own RSRP there (normalized state units)
      b_k    demand-weighted best-server RSRP there
      s_k    demand-weighted serving SINR there
      RSRP band-edge masses at fair/good/excellent (-100/-90/-80 dBm)
      SINR band-edge masses at fair/good (0/13 dB)
      deficits: demand below the fair RSRP / fair SINR line in my region

    All are recomputed each forward from (state maps, cached per-sector sweep,
    current angles); the state contract and replay format are unchanged.  The
    demand weight is the state's log-compressed density channel (monotone in
    density), renormalized per sample.  No spatial shapes reach the network,
    so there is nothing family-specific to memorise.
    """

    name = "saint_dqn_tok3"

    # thresholds mapped into the state-map normalizations used by the env:
    # RSRP channel: (dBm + 115) / 85     SINR channel: (dB + 30) / 65
    RSRP_EDGES_NORM = ((-100.0 + 115.0) / 85.0, (-90.0 + 115.0) / 85.0, (-80.0 + 115.0) / 85.0)
    SINR_EDGES_NORM = ((0.0 + 30.0) / 65.0, (13.0 + 30.0) / 65.0)
    RSRP_W = 3.0 / 85.0   # ~3 dB windows
    SINR_W = 3.0 / 65.0

    def __init__(self, *args, sector_map_px: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.sector_map_px = int(sector_map_px)
        n_feat = 4 + len(self.RSRP_EDGES_NORM) + len(self.SINR_EDGES_NORM) + 2
        self.demand_proj = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(), small_init_(nn.Linear(32, self.token_dim), gain=0.1)
        )
        self.register_buffer("sector_sweep", torch.zeros(1), persistent=False)
        self._rss_norm = (-115.0, 85.0)

    def attach_cache(self, base_dbm: torch.Tensor, num_angles_task: int) -> None:
        px = self.sector_map_px
        A, K, H, W = base_dbm.shape
        down = nn.functional.adaptive_avg_pool2d(base_dbm.reshape(A * K, 1, H, W), (px, px)).reshape(A, K, px, px)
        lo, sc = self._rss_norm
        self.sector_sweep = ((down - lo) / sc).clamp(0.0, 1.0)
        self.task_num_angles = int(num_angles_task)

    def _demand_side(self, maps: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        B = maps.shape[0]; K = self.num_sectors; px = self.sector_map_px
        gmap = nn.functional.adaptive_avg_pool2d(maps, (px, px))            # [B,3,px,px]
        dens = gmap[:, 2].clamp_min(0.0)
        dens = dens / dens.flatten(1).sum(dim=1).clamp_min(1e-8)[:, None, None]
        sinr = gmap[:, 1]
        idx = (angles * 89.0).round().long().clamp(0, self.sector_sweep.shape[0] - 1)
        own = self.sector_sweep[idx, torch.arange(K, device=maps.device)[None].expand(B, -1)]  # [B,K,px,px]
        best = own.max(dim=1).values                                        # [B,px,px]
        mask = nn.functional.one_hot(own.argmax(dim=1), K).permute(0, 3, 1, 2).float()
        dm = dens[:, None] * mask                                           # [B,K,px,px]
        m = dm.sum((-2, -1))                                                # [B,K]
        mn = m.clamp_min(1e-6)
        feats = [m,
                 (dm * own).sum((-2, -1)) / mn,
                 (dm * best[:, None]).sum((-2, -1)) / mn,
                 (dm * sinr[:, None]).sum((-2, -1)) / mn]
        for t in self.RSRP_EDGES_NORM:
            feats.append((dm * torch.exp(-(((own - t) / self.RSRP_W) ** 2))).sum((-2, -1)))
        for t in self.SINR_EDGES_NORM:
            feats.append((dm * torch.exp(-(((sinr[:, None] - t) / self.SINR_W) ** 2))).sum((-2, -1)))
        feats.append((dm * torch.sigmoid((self.RSRP_EDGES_NORM[0] - own) / self.RSRP_W)).sum((-2, -1)))
        feats.append((dm * torch.sigmoid((self.SINR_EDGES_NORM[0] - sinr[:, None]) / self.SINR_W)).sum((-2, -1)))
        return self.demand_proj(torch.stack(feats, dim=-1))                 # [B,K,token_dim]

    def forward(self, maps, feedback, angles, alpha, progress, mismatch=None):
        h = self.encoder(maps, feedback, angles, alpha, progress)
        gamma, beta = self.film(h).chunk(2, dim=-1)
        x = gamma[:, None, :] * self.sector_embedding[None] + beta[:, None, :]
        x = x + self._sector_side_features(feedback, angles)
        x = x + self._demand_side(maps, angles)
        x = self.token_norm(self.transformer(x))
        return self.q_head(x)


class NaiveBranchingDQN(nn.Module):
    """Literature-faithful branching DQN (Tavakoli et al.) with a GENERIC encoder.

    Deliberately contains none of this project's design insights: an
    Atari-style CNN over the stacked maps, the report set FLATTENED into an
    MLP (no permutation invariance, no per-report processing), scalars
    concatenated, one shared trunk, per-sector linear branches.  Same inputs
    and training recipe as every other method; ~250k parameters.
    """

    name = "bdqn_naive"

    def __init__(self, num_sectors: int = 9, num_actions: int = 3, num_reports: int = 200, report_dim: int = 10):
        super().__init__()
        self.num_sectors = int(num_sectors)
        self.num_actions = int(num_actions)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fb = nn.Sequential(nn.Linear(num_reports * report_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(32 * 16 + 64 + self.num_sectors + 2, 256), nn.ReLU())
        self.branches = nn.ModuleList([nn.Linear(256, self.num_actions) for _ in range(self.num_sectors)])

    def forward(self, maps, feedback, angles, alpha, progress, mismatch=None):
        z = torch.cat([
            self.cnn(maps).flatten(1),
            self.fb(feedback.flatten(1)),
            angles,
            alpha[:, None] if alpha.ndim == 1 else alpha,
            progress[:, None] if progress.ndim == 1 else progress,
        ], dim=-1)
        h = self.trunk(z)
        return torch.stack([b(h) for b in self.branches], dim=1)  # [B, K, A]


ALGORITHMS: dict[str, type[nn.Module]] = {
    FactorizedActorCritic.name: FactorizedActorCritic,
    FactorizedCapMatchedActorCritic.name: FactorizedCapMatchedActorCritic,
    SAINTActorCritic.name: SAINTActorCritic,
    SAINTQNetwork.name: SAINTQNetwork,
    SAINTQNetworkTok.name: SAINTQNetworkTok,
    SAINTQNetworkTok2.name: SAINTQNetworkTok2,
    SAINTQNetworkTok3.name: SAINTQNetworkTok3,
    NaiveBranchingDQN.name: NaiveBranchingDQN,
}


def build_policy(name: str, **kwargs: Any) -> nn.Module:
    if name not in ALGORITHMS:
        raise KeyError(f"unknown Track-A algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    return ALGORITHMS[name](**kwargs)


def joint_log_prob_and_entropy(
    logits: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of per-sector categorical log-probs / entropies (identical for all algos)."""
    dist = Categorical(logits=logits)
    return dist.log_prob(actions).sum(dim=1), dist.entropy().sum(dim=1)


def sample_actions(logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    if deterministic:
        return torch.argmax(logits, dim=-1)
    return Categorical(logits=logits).sample()


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
