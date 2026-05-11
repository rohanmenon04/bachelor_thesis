"""
Model definitions for thesis experiments.

Implements:
  - VQLayer:              Vector quantization with straight-through gradient estimator
  - TransformerVQEncoder: Transformer + VQ bottleneck (the proposed model)
  - ContinuousEncoder:    Same architecture but continuous bottleneck (Condition B baseline)

Both encoders take a 7x7x3 MiniGrid observation and produce a compact
representation. The VQ version maps to a discrete token index; the
continuous version maps to a real-valued latent vector.

Reference architectures:
  - IRIS (Micheli et al., 2023): VQ tokenizer + Transformer world model
  - VQ-VAE (van den Oord et al., 2017): codebook + straight-through estimator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MiniGrid grid-cell channel ranges (per-channel integer values)
N_TYPES  = 11   # object type:  0-10
N_COLORS =  6   # color:        0-5
N_STATES =  3   # state:        0-2

GRID_H, GRID_W = 7, 7
N_CELLS = GRID_H * GRID_W   # 49 spatial tokens per observation

# Architecture hyperparameters
D_MODEL       = 64
N_HEADS       =  4
N_LAYERS      =  2
CODEBOOK_SIZE = 32
LATENT_DIM    = 32   # continuous AE bottleneck dimension
COMMITMENT_BETA = 0.25


class VQLayer(nn.Module):
    """
    Vector quantization module.

    Maps a continuous vector z to the nearest entry in a learned codebook,
    using the straight-through gradient estimator so gradients flow to the encoder.

    Commitment loss has two terms:
      - Codebook update:  pushes codebook entries toward encoder outputs
      - Encoder update:   pushes encoder outputs toward chosen codebook entry
    """

    def __init__(self, codebook_size=CODEBOOK_SIZE, d_model=D_MODEL, beta=COMMITMENT_BETA):
        super().__init__()
        self.K = codebook_size
        self.D = d_model
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        """
        Args:
            z: (B, D) continuous encoder output
        Returns:
            z_q_st: (B, D)  quantized vector with straight-through gradient
            indices: (B,)   discrete token indices
            commit_loss: scalar
        """
        # Squared Euclidean distances to all codebook entries: (B, K)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * (z @ self.codebook.weight.T)
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)          # (B,)
        z_q = self.codebook(indices)          # (B, D)

        # Straight-through: pass gradients to encoder as if z_q == z
        z_q_st = z + (z_q - z).detach()

        commit_loss = (
            F.mse_loss(z.detach(), z_q)               # move codebook → z
            + self.beta * F.mse_loss(z, z_q.detach()) # move z → codebook
        )
        return z_q_st, indices, commit_loss

    def get_perplexity(self, indices):
        """
        Codebook utilization metric.
        Equals K when all codes are used equally; ~1 indicates collapse.
        """
        probs = F.one_hot(indices, self.K).float().mean(0)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return entropy.exp().item()


class TransformerVQEncoder(nn.Module):
    """
    Proposed model: Transformer encoder + VQ bottleneck.

    Architecture:
      1. Embed each of the 3 discrete grid channels separately, concatenate → d_model
      2. Flatten 7x7 grid → 49 spatial tokens
      3. Prepend trainable [CLS] token, add positional embeddings
      4. Transformer encoder (N_LAYERS layers, N_HEADS attention heads)
      5. Take [CLS] output → VQ → discrete token index
      6. Decoder reconstructs the original 7x7x3 grid (reconstruction signal)
    """

    def __init__(
        self,
        codebook_size=CODEBOOK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ):
        super().__init__()
        self.d_model = d_model

        # Distribute d_model across the three channel embeddings
        dim_t = d_model // 3 + d_model % 3   # absorb remainder here
        dim_c = d_model // 3
        dim_s = d_model // 3

        self.type_embed  = nn.Embedding(N_TYPES,  dim_t)
        self.color_embed = nn.Embedding(N_COLORS, dim_c)
        self.state_embed = nn.Embedding(N_STATES, dim_s)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Embedding(N_CELLS + 1, d_model)   # +1 for CLS

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.vq = VQLayer(codebook_size, d_model)

        out_dim = N_CELLS * (N_TYPES + N_COLORS + N_STATES)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Linear(d_model * 4, out_dim),
        )

    def encode(self, obs):
        """obs: (B, 7, 7, 3) long → (B, D_MODEL) continuous pre-VQ embedding"""
        B = obs.shape[0]
        t = self.type_embed(obs[..., 0])    # (B,7,7,dim_t)
        c = self.color_embed(obs[..., 1])   # (B,7,7,dim_c)
        s = self.state_embed(obs[..., 2])   # (B,7,7,dim_s)

        x = torch.cat([t, c, s], dim=-1).view(B, N_CELLS, self.d_model)  # (B,49,D)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)     # (B,50,D)

        pos = torch.arange(N_CELLS + 1, device=obs.device)
        x = x + self.pos_embed(pos)

        return self.transformer(x)[:, 0, :]  # CLS output

    def forward(self, obs):
        z = self.encode(obs)
        z_q, indices, commit_loss = self.vq(z)
        logits = self.decoder(z_q).view(-1, N_CELLS, N_TYPES + N_COLORS + N_STATES)
        return logits, indices, commit_loss

    def get_token(self, obs):
        """No-grad inference: return discrete token index."""
        with torch.no_grad():
            z = self.encode(obs)
            _, indices, _ = self.vq(z)
        return indices


class ContinuousEncoder(nn.Module):
    """
    Baseline: identical Transformer architecture but continuous bottleneck.

    No VQ layer — the CLS output is projected to latent_dim via a linear layer.
    Decoder reconstructs observations from the continuous latent vector.
    Used as Condition B in the planning efficiency comparison.
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

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

        self.to_latent = nn.Linear(d_model, latent_dim)

        out_dim = N_CELLS * (N_TYPES + N_COLORS + N_STATES)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 4), nn.ReLU(),
            nn.Linear(d_model * 4, out_dim),
        )

    def encode(self, obs):
        B = obs.shape[0]
        t = self.type_embed(obs[..., 0])
        c = self.color_embed(obs[..., 1])
        s = self.state_embed(obs[..., 2])

        x = torch.cat([t, c, s], dim=-1).view(B, N_CELLS, self.d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(N_CELLS + 1, device=obs.device)
        x = x + self.pos_embed(pos)

        return self.to_latent(self.transformer(x)[:, 0, :])   # (B, latent_dim)

    def forward(self, obs):
        z = self.encode(obs)
        logits = self.decoder(z).view(-1, N_CELLS, N_TYPES + N_COLORS + N_STATES)
        return logits, z


def reconstruction_loss(logits, obs):
    """
    Cross-entropy reconstruction loss across all three grid channels.

    Args:
        logits: (B, 49, N_TYPES + N_COLORS + N_STATES)
        obs:    (B, 7, 7, 3) long
    """
    B = obs.shape[0]
    target = obs.view(B, N_CELLS, 3)

    loss_type  = F.cross_entropy(
        logits[:, :, :N_TYPES].reshape(-1, N_TYPES),
        target[:, :, 0].reshape(-1),
    )
    loss_color = F.cross_entropy(
        logits[:, :, N_TYPES:N_TYPES + N_COLORS].reshape(-1, N_COLORS),
        target[:, :, 1].reshape(-1),
    )
    loss_state = F.cross_entropy(
        logits[:, :, N_TYPES + N_COLORS:].reshape(-1, N_STATES),
        target[:, :, 2].reshape(-1),
    )
    return loss_type + loss_color + loss_state
