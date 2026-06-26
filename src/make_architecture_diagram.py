"""
Generates architecture diagram for the Object-Centric VQ Encoder.
Output: plots/architecture_oc_encoder.pdf  (also .png at 200 dpi)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'plots'
OUT_DIR.mkdir(exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
C_OBS    = '#dce8f5'   # input
C_EMBED  = '#e8f4e8'   # embedding
C_TRANS  = '#fff3cd'   # transformer
C_AGENT  = '#d4e6f1'   # agent head
C_KEY    = '#fde8d8'   # key head
C_DOOR   = '#e8f5e9'   # door head
C_VQ     = '#f5e6fa'   # VQ box
C_CONCAT = '#f0f0f0'   # concat / output
C_DECODE = '#fafafa'   # decoder / aux (dashed, training only)
C_LOSS   = '#fff0f0'   # loss

EDGE     = '#333333'
GRAY     = '#888888'
BLUE     = '#2c6fad'
ORANGE   = '#c45f1a'
GREEN    = '#2e7d32'
PURPLE   = '#6a1b9a'

def box(ax, x, y, w, h, text, fc, ec=EDGE, lw=1.0, fontsize=8,
        bold=False, alpha=1.0, ls='-', va='center', extra=''):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.02', facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    label = text if not extra else f'{text}\n{extra}'
    ax.text(x, y, label, ha='center', va=va, fontsize=fontsize,
            fontweight=weight, zorder=4, color='#111111')

def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.0, style='->', dashes=None,
          shrink=2):
    props = dict(arrowstyle=style, color=color, lw=lw,
                 shrinkA=shrink, shrinkB=shrink)
    if dashes:
        props['linestyle'] = (0, dashes)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=props, zorder=2)

def label(ax, x, y, text, color=GRAY, fontsize=6.5, ha='center', va='center',
          rotation=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            rotation=rotation, zorder=5)

# ── figure layout ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

fig.suptitle(
    'Object-Centric VQ Encoder Architecture',
    fontsize=11, fontweight='bold', y=0.97,
)

# ── X positions for three heads ──────────────────────────────────────────────
X_AGENT = 3.5
X_KEY   = 6.5
X_DOOR  = 9.5
X_MID   = (X_AGENT + X_DOOR) / 2   # 6.5 — centre of transformer / input

# ── (1) Input observation ────────────────────────────────────────────────────
Y_OBS = 8.3
box(ax, X_MID, Y_OBS, 4.0, 0.55,
    'Observation  (B × 7 × 7 × 3)',
    C_OBS, bold=True, fontsize=8.5)
label(ax, X_MID, Y_OBS - 0.42,
      'channels: type ∈ [0,10],  color ∈ [0,5],  state ∈ [0,2]',
      fontsize=6.2)

# ── (2) Embedding ────────────────────────────────────────────────────────────
Y_EMB = 7.45
box(ax, X_MID, Y_EMB, 5.6, 0.50,
    'Cell Embedding   cat(type_embed, color_embed, state_embed)  →  64-dim',
    C_EMBED, fontsize=7.5)
label(ax, X_MID + 3.1, Y_EMB, 'dim: 22 + 21 + 21', fontsize=6, ha='left')

arrow(ax, X_MID, Y_OBS - 0.275, X_MID, Y_EMB + 0.25)

# ── (3) CLS prepend + positional embed ──────────────────────────────────────
Y_POS = 6.65
box(ax, X_MID, Y_POS, 5.6, 0.50,
    '[CLS] token  +  49 spatial tokens  +  positional embeddings  →  50 × 64',
    C_EMBED, fontsize=7.5)
label(ax, X_MID + 3.1, Y_POS, 'nn.Embedding(50, 64)', fontsize=6, ha='left')

arrow(ax, X_MID, Y_EMB - 0.25, X_MID, Y_POS + 0.25)

# ── (4) Transformer Encoder ─────────────────────────────────────────────────
Y_TRF = 5.75
box(ax, X_MID, Y_TRF, 5.6, 0.60,
    'Transformer Encoder',
    C_TRANS, bold=True, fontsize=8.5, lw=1.5)
label(ax, X_MID, Y_TRF - 0.17,
      'L=2 layers  ·  H=4 heads  ·  d_ff=256  ·  dropout=0.1',
      fontsize=6.5)

arrow(ax, X_MID, Y_POS - 0.25, X_MID, Y_TRF + 0.30)

# output annotation
label(ax, X_MID, Y_TRF - 0.42,
      'output:  h[0] = CLS token  ·  h[1..49] = per-cell spatial tokens',
      fontsize=6.0, color='#666666')

# ── fan-out arrows from transformer to three columns ─────────────────────────
Y_FAN = Y_TRF - 0.30
Y_RO  = 4.85   # readout row

# agent branch
arrow(ax, X_AGENT, Y_FAN, X_AGENT, Y_RO + 0.25,  color=BLUE)
# key branch
arrow(ax, X_KEY,   Y_FAN, X_KEY,   Y_RO + 0.25,  color=ORANGE)
# door branch
arrow(ax, X_DOOR,  Y_FAN, X_DOOR,  Y_RO + 0.25,  color=GREEN)

# ── (5a) Agent readout ───────────────────────────────────────────────────────
box(ax, X_AGENT, Y_RO, 2.2, 0.44,
    'CLS output  h[0]',
    C_AGENT, ec=BLUE, lw=1.2, fontsize=7.5)
label(ax, X_AGENT, Y_RO - 0.30, 'global state token\n64-dim', fontsize=6.2,
      color=BLUE)

# ── (5b) Key readout ─────────────────────────────────────────────────────────
box(ax, X_KEY, Y_RO, 2.3, 0.44,
    'Key cell readout  h[i_key]',
    C_KEY, ec=ORANGE, lw=1.2, fontsize=7.5)
label(ax, X_KEY, Y_RO - 0.28,
      'scan obs for type=5\n→ h[i_key]  or  key_absent_embed',
      fontsize=5.8, color=ORANGE)

# ── (5c) Door readout ────────────────────────────────────────────────────────
box(ax, X_DOOR, Y_RO, 2.3, 0.44,
    'Door cell readout  h[i_door]',
    C_DOOR, ec=GREEN, lw=1.2, fontsize=7.5)
label(ax, X_DOOR, Y_RO - 0.28,
      'scan obs for type=4\n→ h[i_door]  or  door_absent_embed',
      fontsize=5.8, color=GREEN)

# ── (6) VQ heads ─────────────────────────────────────────────────────────────
Y_VQ = 3.60

arrow(ax, X_AGENT, Y_RO - 0.22, X_AGENT, Y_VQ + 0.25, color=BLUE)
arrow(ax, X_KEY,   Y_RO - 0.22, X_KEY,   Y_VQ + 0.25, color=ORANGE)
arrow(ax, X_DOOR,  Y_RO - 0.22, X_DOOR,  Y_VQ + 0.25, color=GREEN)

box(ax, X_AGENT, Y_VQ, 2.2, 0.55,
    'VQ_agent\n(K = 64)',
    C_VQ, ec=BLUE, lw=1.4, bold=True, fontsize=8)

box(ax, X_KEY, Y_VQ, 2.2, 0.55,
    'VQ_key\n(K = 16)',
    C_VQ, ec=ORANGE, lw=1.4, bold=True, fontsize=8)

box(ax, X_DOOR, Y_VQ, 2.2, 0.55,
    'VQ_door\n(K = 8)',
    C_VQ, ec=GREEN, lw=1.4, bold=True, fontsize=8)

# EMA / straight-through note inside VQ boxes
for xc, note in [(X_AGENT, 'EMA + dead-code restart'),
                 (X_KEY,   'EMA + dead-code restart'),
                 (X_DOOR,  'EMA + dead-code restart')]:
    label(ax, xc, Y_VQ - 0.35, note, fontsize=5.5, color=PURPLE)

# ── (7) VQ outputs — quantised vectors + indices ─────────────────────────────
Y_OUT = 2.70

arrow(ax, X_AGENT, Y_VQ - 0.275, X_AGENT, Y_OUT + 0.22, color=BLUE)
arrow(ax, X_KEY,   Y_VQ - 0.275, X_KEY,   Y_OUT + 0.22, color=ORANGE)
arrow(ax, X_DOOR,  Y_VQ - 0.275, X_DOOR,  Y_OUT + 0.22, color=GREEN)

box(ax, X_AGENT, Y_OUT, 2.2, 0.38,
    'z_q_agent  ·  idx_agent',
    C_AGENT, ec=BLUE, fontsize=7)
label(ax, X_AGENT, Y_OUT - 0.27, '64-dim  ·  idx ∈ {0…63}', fontsize=6, color=BLUE)

box(ax, X_KEY, Y_OUT, 2.2, 0.38,
    'z_q_key  ·  idx_key',
    C_KEY, ec=ORANGE, fontsize=7)
label(ax, X_KEY, Y_OUT - 0.27, '64-dim  ·  idx ∈ {0…15}', fontsize=6, color=ORANGE)

box(ax, X_DOOR, Y_OUT, 2.2, 0.38,
    'z_q_door  ·  idx_door',
    C_DOOR, ec=GREEN, fontsize=7)
label(ax, X_DOOR, Y_OUT - 0.27, '64-dim  ·  idx ∈ {0…7}', fontsize=6, color=GREEN)

# ── (8) Concatenate → policy input ───────────────────────────────────────────
Y_CAT = 1.72

for xc in [X_AGENT, X_KEY, X_DOOR]:
    arrow(ax, xc, Y_OUT - 0.19 - 0.27, xc, Y_CAT + 0.22)

# horizontal connector line
ax.plot([X_AGENT, X_DOOR], [Y_CAT + 0.22, Y_CAT + 0.22],
        color=EDGE, lw=1.0, zorder=2)

box(ax, X_MID, Y_CAT, 5.4, 0.40,
    'concat(z_q_agent, z_q_key, z_q_door)   →   192-dim policy input',
    C_CONCAT, bold=True, fontsize=8, ec='#444444', lw=1.3)

# ── (9) Decoder + aux heads (training only, dashed) ──────────────────────────
Y_DEC  = 1.05
Y_AUX  = 1.05
Y_LOSS = 0.38

# main decoder arrow from concat box
arrow(ax, X_MID, Y_CAT - 0.20, X_MID, Y_DEC + 0.22,
      color='#999999', lw=0.9)

box(ax, X_MID, Y_DEC, 4.4, 0.38,
    'Decoder  Linear(192→256)→ReLU→Linear(256→980)  →  7×7×20 logits',
    C_DECODE, ec=GRAY, lw=0.9, ls='--', fontsize=6.8, alpha=0.85)

# key aux head
box(ax, 3.8, Y_AUX, 1.9, 0.38,
    'key_aux_head\n→ CE on key cell',
    C_DECODE, ec=ORANGE, lw=0.8, ls='--', fontsize=6.2, alpha=0.85)

# door aux head
box(ax, 9.2, Y_AUX, 1.9, 0.38,
    'door_aux_head\n→ CE on door cell',
    C_DECODE, ec=GREEN, lw=0.8, ls='--', fontsize=6.2, alpha=0.85)

# dashed arrows from z_q_key / z_q_door boxes down to aux heads
ax.annotate('', xy=(3.8, Y_AUX + 0.19), xytext=(X_KEY, Y_OUT - 0.46),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8,
                            connectionstyle='arc3,rad=-0.25',
                            linestyle='dashed', shrinkA=2, shrinkB=2),
            zorder=2)
ax.annotate('', xy=(9.2, Y_AUX + 0.19), xytext=(X_DOOR, Y_OUT - 0.46),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.8,
                            connectionstyle='arc3,rad=0.25',
                            linestyle='dashed', shrinkA=2, shrinkB=2),
            zorder=2)

# λ_aux labels
label(ax, 5.0, 1.38, 'λ = 0.5', fontsize=6, color=ORANGE)
label(ax, 8.0, 1.38, 'λ = 0.5', fontsize=6, color=GREEN)

# main recon loss
arrow(ax, X_MID, Y_DEC - 0.19, X_MID, Y_LOSS + 0.13,
      color='#aaaaaa', lw=0.8)
box(ax, X_MID, Y_LOSS, 2.1, 0.24,
    'L_recon  (CE, 7×7×3)',
    C_LOSS, ec='#cccccc', lw=0.8, ls='--', fontsize=6.5, alpha=0.85)

# total loss note
label(ax, X_MID, 0.10,
      'L_total = L_recon + λ(L_key_aux + L_door_aux) + L_commit     (training)',
      fontsize=7.0, color='#444444')

# "training only" brace annotation
ax.annotate('', xy=(11.5, 1.28), xytext=(11.5, 0.28),
            arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=0.8))
label(ax, 11.85, 0.78, 'training\nonly', fontsize=6, color='#aaaaaa', ha='left')

# ── section labels on left margin ────────────────────────────────────────────
for y, txt in [
    (Y_OBS,  '① Input'),
    (Y_EMB,  '② Embed'),
    (Y_POS,  '③ Tokenise'),
    (Y_TRF,  '④ Transformer'),
    (Y_RO,   '⑤ Readout'),
    (Y_VQ,   '⑥ VQ heads'),
    (Y_OUT,  '⑦ Discrete\n    codes'),
    (Y_CAT,  '⑧ Concat'),
    (Y_DEC,  '⑨ Decoder'),
]:
    label(ax, 1.05, y, txt, fontsize=6, ha='right', color='#666666')

# ── column header labels ──────────────────────────────────────────────────────
for xc, txt, col in [
    (X_AGENT, 'Agent head', BLUE),
    (X_KEY,   'Key head',   ORANGE),
    (X_DOOR,  'Door head',  GREEN),
]:
    ax.text(xc, 5.18, txt, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color=col, zorder=5)

# ── legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_AGENT, edgecolor=BLUE,   label='Agent head (CLS)'),
    mpatches.Patch(facecolor=C_KEY,   edgecolor=ORANGE, label='Key head (spatial readout)'),
    mpatches.Patch(facecolor=C_DOOR,  edgecolor=GREEN,  label='Door head (spatial readout)'),
    mpatches.Patch(facecolor=C_VQ,    edgecolor=PURPLE, label='VQ layer (EMA + dead-code restart)'),
    mpatches.Patch(facecolor=C_DECODE,edgecolor=GRAY,   label='Decoder / aux heads (training only)',
                   linestyle='--'),
]
ax.legend(handles=legend_items, loc='lower left', bbox_to_anchor=(0.0, 0.0),
          fontsize=6.2, framealpha=0.9, edgecolor='#cccccc',
          handlelength=1.4, handleheight=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.97])

for ext in ['png', 'pdf']:
    p = OUT_DIR / f'architecture_oc_encoder.{ext}'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f'Saved {p}')

plt.close()
