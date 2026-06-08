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
CODEBOOK_SIZE = 64   # increased from 32; more capacity for semantic specialisation
LATENT_DIM    = 32   # continuous AE bottleneck dimension
COMMITMENT_BETA     = 0.10   # reduced from 0.25; less aggressive encoder compression
EMA_DECAY           = 0.99   # decay factor for EMA codebook update
DEAD_CODE_THRESHOLD = 1.0    # EMA cluster-size below this triggers code restart

# Spatial VQ encoder hyperparameters (Condition I: Dual-VQ)
SPATIAL_CODEBOOK_SIZE = 32   # one code per reachable cell in DoorKey-5x5 (25 cells)
SPATIAL_LATENT_DIM    = 32   # spatial token embedding width
SPATIAL_GRID_SIZE     = 5    # DoorKey-5x5 world is 5×5; positions are ints in [0, 5)


class VQLayer(nn.Module):
    """
    Vector quantization module with EMA codebook updates and dead-code restart.

    Codebook entries are maintained via exponential moving averages (EMA) of
    the encoder outputs assigned to each entry, following van den Oord et al.
    (2017) Appendix A and Razavi et al. (2019). This replaces the original
    gradient-based codebook update, which caused collapse.

    Dead-code restart: any entry whose EMA cluster size falls below
    DEAD_CODE_THRESHOLD is reinitialised to a random encoder output from the
    current batch, immediately returning it to the active pool.

    Commitment loss retains only the encoder-side term (beta * ||z - sg[z_q]||^2)
    because the codebook-side term (||sg[z] - z_q||^2) is redundant when the
    codebook is maintained by EMA.

    See observations/edits_codebook_01.md for full diagnostic rationale.
    """

    def __init__(
        self,
        codebook_size=CODEBOOK_SIZE,
        d_model=D_MODEL,
        beta=COMMITMENT_BETA,
        ema_decay=EMA_DECAY,
        dead_threshold=DEAD_CODE_THRESHOLD,
    ):
        super().__init__()
        self.K = codebook_size
        self.D = d_model
        self.beta = beta
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold

        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.codebook.weight.requires_grad_(False)   # EMA owns all codebook updates

        # EMA state: running count and running sum of assigned encoder outputs
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embed_avg',    self.codebook.weight.data.clone())

    def _ema_update(self, z, indices):
        """Update codebook via EMA and reinitialise any dead codes."""
        one_hot = F.one_hot(indices, self.K).float()            # (B, K)

        cluster_size = one_hot.sum(0)                           # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size * (1.0 - self.ema_decay)
        )

        embed_sum = one_hot.T @ z.detach()                      # (K, D)
        self.ema_embed_avg.mul_(self.ema_decay).add_(
            embed_sum * (1.0 - self.ema_decay)
        )

        # Laplace smoothing: prevents division-by-zero for rarely-used codes
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
        self.codebook.weight.data.copy_(
            self.ema_embed_avg / smoothed.unsqueeze(1)
        )

        # Dead-code restart: reinitialise under-used entries to random batch samples
        dead = self.ema_cluster_size < self.dead_threshold
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            rand_idx = torch.randint(0, z.shape[0], (n_dead,), device=z.device)
            self.codebook.weight.data[dead] = z.detach()[rand_idx]
            self.ema_cluster_size[dead]     = self.dead_threshold
            self.ema_embed_avg[dead]        = z.detach()[rand_idx]

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
        z_q     = self.codebook(indices)      # (B, D)

        # Straight-through: pass gradients to encoder as if z_q == z
        z_q_st = z + (z_q - z).detach()

        if self.training:
            self._ema_update(z, indices)

        # Encoder-only commitment loss; codebook direction is handled by EMA
        commit_loss = self.beta * F.mse_loss(z, z_q.detach())

        return z_q_st, indices, commit_loss

    def get_perplexity(self, indices):
        """
        Codebook utilisation metric.
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


class VQSpatialEncoder(nn.Module):
    """
    Spatial VQ encoder (Condition I) — embedding + cross-entropy variant.

    Built by porting the recipe from the working ``TransformerVQEncoder``:
    discrete inputs through learned embedding tables, VQ bottleneck with
    EMA + dead-code restart, multi-class cross-entropy reconstruction.

    Earlier MLP-on-floats version collapsed to ~3/32 active codes: a 2-float
    input plus an MSE position-prediction target is solvable by very few
    codes (the decoder learns the mean and ignores the bottleneck), and the
    encoder MLP squashes the ~9 reachable cells into a tight latent cluster
    so dead-code restart picks restart targets right next to active codes.
    This version mirrors the semantic encoder pattern instead: cell-coord
    integers ↦ separated embeddings, per-axis cross-entropy ↦ codebook must
    preserve cell identity.

    Input:  pos = (x, y) ∈ [0, GRID_SIZE)^2 as long tensors
    Encode: cat(x_embed, y_embed) → (B, latent_dim)
    Bottleneck: VQLayer (same EMA + dead-code restart used by the semantic encoder)
    Decode: 2*GRID_SIZE logits, trained with cross-entropy on (x, y).
    """

    def __init__(
        self,
        grid_size=SPATIAL_GRID_SIZE,
        codebook_size=SPATIAL_CODEBOOK_SIZE,
        latent_dim=SPATIAL_LATENT_DIM,
        beta=COMMITMENT_BETA,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.K = codebook_size
        self.D = latent_dim

        dim_half = latent_dim // 2
        # Two embedding tables for x and y axes — analogous to the per-channel
        # embeddings (type/color/state) in TransformerVQEncoder.
        self.x_embed = nn.Embedding(grid_size, dim_half)
        self.y_embed = nn.Embedding(grid_size, dim_half)
        nn.init.normal_(self.x_embed.weight, std=0.02)
        nn.init.normal_(self.y_embed.weight, std=0.02)

        self.vq = VQLayer(
            codebook_size=codebook_size, d_model=latent_dim, beta=beta,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4), nn.ReLU(),
            nn.Linear(latent_dim * 4, 2 * grid_size),
        )

    def encode(self, pos_int):
        """pos_int: (B, 2) long with values in [0, grid_size)"""
        x = self.x_embed(pos_int[:, 0])
        y = self.y_embed(pos_int[:, 1])
        return torch.cat([x, y], dim=-1)

    def forward(self, pos_int):
        z = self.encode(pos_int)
        z_q, indices, commit_loss = self.vq(z)
        logits = self.decoder(z_q)
        x_logits = logits[:, :self.grid_size]
        y_logits = logits[:, self.grid_size:]
        return x_logits, y_logits, indices, commit_loss

    def get_token(self, pos_int):
        """No-grad inference: return discrete spatial token index."""
        with torch.no_grad():
            z = self.encode(pos_int)
            _, indices, _ = self.vq(z)
        return indices


def spatial_reconstruction_loss(x_logits, y_logits, pos_int):
    """Per-axis cross-entropy reconstruction for VQSpatialEncoder."""
    loss_x = F.cross_entropy(x_logits, pos_int[:, 0])
    loss_y = F.cross_entropy(y_logits, pos_int[:, 1])
    return loss_x + loss_y


class PositionEmbeddingEncoder(nn.Module):
    """
    Direct position embedding for Condition I-b (alternative to VQSpatialEncoder).

    Maps integer (x, y) grid coordinates to a learned latent vector via a
    lookup table. No VQ bottleneck — each of the grid_size^2 cells gets a
    distinct trainable vector by construction, so codebook collapse is
    impossible.

    Use case: Condition I compares VQSpatialEncoder (VQ-quantised spatial
    token) with this encoder (learned position lookup) to isolate the cost
    of discretising something that is already discrete.

    Pre-training: optional. Run 09_train_spatial_encoder.py --model pos_embed
    to pre-train with a position-reconstruction objective before PPO; or let
    the embedding learn jointly during PPO (the grid_size^2 * latent_dim = 800
    parameters converge very quickly from sparse reward signal).

    get_token() returns the flat grid index (x * grid_size + y) — the cell
    index IS the discrete spatial token; no quantisation is needed.
    """

    def __init__(self, grid_size=SPATIAL_GRID_SIZE, latent_dim=SPATIAL_LATENT_DIM):
        super().__init__()
        self.grid_size  = grid_size
        self.latent_dim = latent_dim
        self.pos_embed  = nn.Embedding(grid_size * grid_size, latent_dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def encode(self, pos_int):
        """pos_int: (B, 2) long with values in [0, grid_size)"""
        flat_idx = pos_int[:, 0] * self.grid_size + pos_int[:, 1]
        return self.pos_embed(flat_idx)

    def forward(self, pos_int):
        return self.encode(pos_int)

    def get_token(self, pos_int):
        """Flat grid index — already discrete, no quantisation needed."""
        with torch.no_grad():
            return pos_int[:, 0] * self.grid_size + pos_int[:, 1]


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
