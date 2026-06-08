"""
Object-Centric VQ Encoder.

Instead of CLS-pooling the 7×7 grid into one token, this encoder reads off
the Transformer output at each tracked object's grid cell and quantises each
via its own small VQ head:

  agent head (K=OC_AGENT_K): CLS output — global semantic state
  key head   (K=OC_KEY_K):   spatial output at key cell, or a learned 'absent'
                              embedding when the key is not in the 7×7 FOV
  door head  (K=OC_DOOR_K):  spatial output at door cell, or a learned 'absent'
                              embedding when the door is not in the 7×7 FOV

Each head is an independent VQLayer with EMA + dead-code restart (identical to
the semantic encoder). The three quantised vectors are concatenated to form
the decoder input and the PPO policy input.

Two auxiliary per-head reconstruction heads (key_aux_head, door_aux_head)
provide a direct gradient path that forces the key and door VQ streams to
preserve their object's cell channels, preventing the decoder from routing
all information through the global CLS head.

Factorisation benefit: the agent head does not need to know about key or door
state because the policy also receives the key and door tokens. Each codebook
only needs to cover its own object's state space, so each K stays small and
collapse-resistant.

Comparison with existing conditions:
  OC-Agent (64-dim CLS only) ↔ Condition C (same dim, different encoder)
  OC-All   (192-dim all three) tests whether the added object tokens help
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import (
    VQLayer,
    N_TYPES, N_COLORS, N_STATES,
    N_CELLS,
    D_MODEL, N_HEADS, N_LAYERS,
    COMMITMENT_BETA,
    reconstruction_loss,
)

# MiniGrid type-channel IDs present in the 7×7 partial observation.
# The agent (type 10) is NOT visible in its own egocentric view, so there
# is no agent head that reads a spatial cell — the agent head uses CLS instead.
OC_KEY_TYPE  = 5
OC_DOOR_TYPE = 4

OC_AGENT_K    = 64   # matches existing semantic encoder K for direct comparison
OC_KEY_K      = 16   # ~15 floor positions in DoorKey-5x5 + 'absent' state
OC_DOOR_K     = 8    # locked/open × a handful of positions in FOV
OC_AUX_LAMBDA = 0.5  # weight on per-head auxiliary reconstruction losses


class ObjectCentricVQEncoder(nn.Module):
    """Shared Transformer backbone with three independent VQ readout heads."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.d_model = d_model

        dim_t = d_model // 3 + d_model % 3
        dim_c = d_model // 3
        dim_s = d_model // 3

        self.type_embed  = nn.Embedding(N_TYPES,  dim_t)
        self.color_embed = nn.Embedding(N_COLORS, dim_c)
        self.state_embed = nn.Embedding(N_STATES, dim_s)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Embedding(N_CELLS + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.vq_agent = VQLayer(OC_AGENT_K, d_model, beta=COMMITMENT_BETA)
        self.vq_key   = VQLayer(OC_KEY_K,   d_model, beta=COMMITMENT_BETA)
        self.vq_door  = VQLayer(OC_DOOR_K,  d_model, beta=COMMITMENT_BETA)

        # Used when key or door is absent from the field of view.
        # Small init so they start near the origin alongside the encoder outputs.
        self.key_absent_embed  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.door_absent_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Main decoder: reconstructs full 7×7×3 from concat of all three heads.
        out_dim = N_CELLS * (N_TYPES + N_COLORS + N_STATES)
        self.decoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model * 4), nn.ReLU(),
            nn.Linear(d_model * 4, out_dim),
        )

        # Per-head auxiliary heads: each reconstructs the 3 channels of only
        # its own object's cell. Used during training to force specialisation.
        cell_channels = N_TYPES + N_COLORS + N_STATES
        self.key_aux_head  = nn.Linear(d_model, cell_channels)
        self.door_aux_head = nn.Linear(d_model, cell_channels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backbone(self, obs):
        """
        Shared Transformer pass over the 7×7×3 observation.

        obs : (B, 7, 7, 3) — accepts both long (dataset) and float32 (SB3)
        Returns:
            cls_out     : (B, D)      CLS token output
            spatial_out : (B, 49, D)  per-cell spatial token outputs
        """
        B = obs.shape[0]
        obs_long = obs.long()

        t = self.type_embed(obs_long[..., 0])
        c = self.color_embed(obs_long[..., 1])
        s = self.state_embed(obs_long[..., 2])

        x = torch.cat([t, c, s], dim=-1).view(B, N_CELLS, self.d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pos = torch.arange(N_CELLS + 1, device=obs.device)
        x = x + self.pos_embed(pos)

        h = self.transformer(x)
        return h[:, 0, :], h[:, 1:, :]

    def _object_readout(self, spatial_out, obs_long, obj_type, absent_embed):
        """
        Read the Transformer output at the cell occupied by obj_type.

        Falls back to absent_embed for observations where the object is not in
        the 7×7 FOV (either not visible from this angle, or carried/used).
        The absent_embed is learned alongside the encoder so the VQ head can
        assign it a dedicated code.

        Returns:
            token    : (B, D)  pre-VQ embedding
            present  : (B,)    bool — True where the object was found
            cell_idx : (B,)    flat index in [0, 48]; valid only when present
        """
        B = spatial_out.shape[0]
        mask     = (obs_long[..., 0] == obj_type).view(B, 49)
        present  = mask.any(dim=1)
        cell_idx = mask.float().argmax(dim=1)   # 0 for all-False rows (overridden below)

        gathered = spatial_out[torch.arange(B, device=spatial_out.device), cell_idx]
        absent   = absent_embed.unsqueeze(0).expand(B, -1)

        # torch.where: gradient flows through gathered only where present=True,
        # and through absent_embed only where present=False.
        token = torch.where(present.unsqueeze(1), gathered, absent)
        return token, present, cell_idx

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_all(self, obs):
        """
        Full encode: run backbone, extract and quantise all three heads.

        Returns a dict (rather than a tuple) because the number of meaningful
        outputs makes positional unpacking unreadable).
        """
        cls_out, spatial_out = self._backbone(obs)
        obs_long = obs.long()

        z_key,  key_present,  key_cell  = self._object_readout(
            spatial_out, obs_long, OC_KEY_TYPE,  self.key_absent_embed)
        z_door, door_present, door_cell = self._object_readout(
            spatial_out, obs_long, OC_DOOR_TYPE, self.door_absent_embed)

        z_q_agent, idx_agent, c_agent = self.vq_agent(cls_out)
        z_q_key,   idx_key,   c_key   = self.vq_key(z_key)
        z_q_door,  idx_door,  c_door  = self.vq_door(z_door)

        return {
            'z_q_agent': z_q_agent,  'z_q_key': z_q_key,  'z_q_door': z_q_door,
            'idx_agent': idx_agent,  'idx_key': idx_key,   'idx_door': idx_door,
            'commit': c_agent + c_key + c_door,
            'key_present':  key_present,  'key_cell':  key_cell,
            'door_present': door_present, 'door_cell': door_cell,
        }

    def forward(self, obs):
        """
        Full forward pass: encode all three heads and decode back to 7×7×3.
        Returns (logits, enc_dict) so the training loop can compute losses
        and track per-head metrics without a second Transformer pass.
        """
        enc = self.encode_all(obs)
        combined = torch.cat([enc['z_q_agent'], enc['z_q_key'], enc['z_q_door']], dim=-1)
        logits = self.decoder(combined).view(-1, N_CELLS, N_TYPES + N_COLORS + N_STATES)
        return logits, enc

    def get_tokens(self, obs):
        """No-grad inference: return all three discrete token indices."""
        with torch.no_grad():
            enc = self.encode_all(obs)
        return enc['idx_agent'], enc['idx_key'], enc['idx_door']


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def _cell_ce(head_logits, obs_long_flat, cell_idx, present_mask):
    """
    Cross-entropy reconstruction loss for a single object cell's three channels.

    head_logits   : (B, N_TYPES + N_COLORS + N_STATES)  from a per-head aux linear
    obs_long_flat : (B, 49, 3)
    cell_idx      : (B,) flat index in [0, 48]
    present_mask  : (B,) bool

    Returns a differentiable scalar (zero — with grad — when nothing is present
    in the batch, so the calling code never needs to branch on presence).
    """
    if not present_mask.any():
        return head_logits.sum() * 0.0

    lg     = head_logits[present_mask]
    target = obs_long_flat[present_mask, cell_idx[present_mask], :]

    return (
        F.cross_entropy(lg[:, :N_TYPES],                        target[:, 0])
        + F.cross_entropy(lg[:, N_TYPES:N_TYPES + N_COLORS],   target[:, 1])
        + F.cross_entropy(lg[:, N_TYPES + N_COLORS:],           target[:, 2])
    )


def compute_training_loss(model, obs, logits, enc, lambda_aux=OC_AUX_LAMBDA):
    """
    Total encoder training loss. Accepts pre-computed (logits, enc) from a
    single model.forward() call to avoid a second Transformer pass.

    Components:
      main_recon : full 7×7 reconstruction from concat(z_q_agent, z_q_key, z_q_door)
      key_aux    : key-cell reconstruction from z_q_key alone (specialisation pressure)
      door_aux   : door-cell reconstruction from z_q_door alone
      commit     : sum of commitment losses from all three VQ heads

    Returns:
      total       : scalar tensor to call .backward() on
      breakdown   : dict of float values for logging (detached)
    """
    main_recon = reconstruction_loss(logits, obs)

    obs_long_flat = obs.long().view(obs.shape[0], N_CELLS, 3)
    key_aux  = _cell_ce(
        model.key_aux_head(enc['z_q_key']),
        obs_long_flat, enc['key_cell'], enc['key_present'],
    )
    door_aux = _cell_ce(
        model.door_aux_head(enc['z_q_door']),
        obs_long_flat, enc['door_cell'], enc['door_present'],
    )

    total = main_recon + lambda_aux * (key_aux + door_aux) + enc['commit']
    return total, {
        'recon':    main_recon.item(),
        'key_aux':  key_aux.item(),
        'door_aux': door_aux.item(),
        'commit':   enc['commit'].item(),
    }
