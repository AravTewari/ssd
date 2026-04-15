"""
DFlash draft model for Qwen3-32B speculative decoding.

Architecture (per DFlash paper, https://arxiv.org/abs/2602.06036):
  - A small stack of Qwen3-style decoder layers (N_draft << N_target).
  - Each attention layer attends over TWO key-value sets:
      (a) context KVs computed from target hidden states at selected layers,
      (b) noise KVs computed from the masked/noised block embedding.
  - Target features are extracted at evenly-spaced intermediate layers,
    concatenated, and projected down to the draft hidden size via an `fc` layer.
  - The draft is conditioned on masked (MASK token) block embeddings, and
    predicts all block tokens in parallel (non-causal within the block).
  - At inference the block is verified with the target model using cascaded
    acceptance (accept prefix up to first mismatch).

Key differences from vanilla BlockDiffusion (https://arxiv.org/abs/2503.09573):
  - Cross-attention to intermediate target hidden states (not just input embeds).
  - Shared lm_head / embed_tokens with the target model — no extra vocab projection.
  - Block-level non-causal self-attention (is_causal=False within block).
  - KV-cache crop after each speculative step for O(1) memory per step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Evenly space `num_draft_layers` tap-points across the target stack."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_target_features(
    all_hidden_states: tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    """
    Gather hidden states at `layer_ids` from the target model's output tuple
    (offset by 1 because index-0 is the embedding, index-k is after layer k-1).
    Returns tensor of shape [B, T, len(layer_ids) * H].
    """
    offset = 1  # all_hidden_states[0] = embedding output, [1] = after layer 0, ...
    selected = [all_hidden_states[lid + offset] for lid in layer_ids]
    return torch.cat(selected, dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,       # [B, n_heads, q_len, head_dim]
    k: torch.Tensor,       # [B, kv_heads, ctx_len+q_len, head_dim]
    cos_q: torch.Tensor,   # [1, q_len, head_dim]  — positions for queries (block only)
    sin_q: torch.Tensor,
    cos_k: torch.Tensor,   # [1, ctx_len+q_len, head_dim]  — positions for keys
    sin_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with separate position tensors for q and k."""
    cos_q = cos_q.unsqueeze(1)   # [1, 1, q_len, D]
    sin_q = sin_q.unsqueeze(1)
    cos_k = cos_k.unsqueeze(1)   # [1, 1, ctx+q, D]
    sin_k = sin_k.unsqueeze(1)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 131072, base: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

    def forward(self, position_ids: torch.LongTensor):
        """
        position_ids: [B, S] — returns (cos, sin) each [1, S, D] indexed by position.
        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len:
            self._build_cache(int(seq_len))
        cos = self.cos_cached[:, position_ids[0], :]   # [1, S, D]
        sin = self.sin_cached[:, position_ids[0], :]
        return cos, sin


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class RMSHeadNorm(nn.Module):
    """Per-head RMS norm applied to Q/K before RoPE (as in Qwen3)."""
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., head_dim]
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# DFlash cross-attentive attention block
# ---------------------------------------------------------------------------

class DFlashAttention(nn.Module):
    """
    Single attention layer that attends over:
      - target context KVs  (from projected target hidden states)
      - noise KVs           (from the noised block embedding)
    Queries are computed only from noise (block) hidden states.
    is_causal=False — all block positions attend to each other + full context.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSHeadNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dropout_p = getattr(config, "attention_dropout", 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,     # [B, block_len, H]  — noised block
        target_hidden: torch.Tensor,     # [B, ctx_len, H]    — projected target features
        position_embeddings: tuple,      # (cos_q, sin_q, cos_k, sin_k)
    ) -> torch.Tensor:
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Queries from noised block
        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        # Keys/values from context AND noised block concatenated
        k_ctx   = self.k_proj(target_hidden).view(B, ctx_len, self.num_kv_heads, self.head_dim)
        k_noise = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
        v_ctx   = self.v_proj(target_hidden).view(B, ctx_len, self.num_kv_heads, self.head_dim)
        v_noise = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

        k = torch.cat([k_ctx, k_noise], dim=1).transpose(1, 2)   # [B, kv_heads, ctx+q, hd]
        v = torch.cat([v_ctx, v_noise], dim=1).transpose(1, 2)

        k = self.k_norm(k)

        # Apply RoPE — separate cos/sin for q (block positions) and k (ctx+block positions)
        cos_q, sin_q, cos_k, sin_k = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k)

        # GQA repeat if needed
        if self.num_heads != self.num_kv_heads:
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        # Flash-attention (non-causal, full block attends to everything)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class DFlashMLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple,   # (cos_q, sin_q, cos_k, sin_k)
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


# ---------------------------------------------------------------------------
# DFlash Draft Model
# ---------------------------------------------------------------------------

class DFlashDraftModel(nn.Module):
    """
    DFlash block-diffusion draft model for a Qwen3 target.

    Parameters
    ----------
    target_config : Qwen3Config
        HF config of the **target** (e.g. Qwen3-32B).
    num_draft_layers : int
        Number of draft transformer layers (paper uses 1–3).
    block_size : int
        Number of tokens generated simultaneously (speculative block width).
    mask_token_id : int
        Token id used as the MASK / [NOISE] token (same as target's mask token).
    """

    def __init__(
        self,
        target_config: Qwen3Config,
        num_draft_layers: int = 1,
        block_size: int = 4,
        mask_token_id: int = 151666,   # Qwen3 uses 151666 as mask token
    ):
        super().__init__()
        self.target_config = target_config
        self.num_draft_layers = num_draft_layers
        self.block_size = block_size
        self.mask_token_id = mask_token_id

        # Draft uses the SAME hidden size as target (shares embed + lm_head)
        H = target_config.hidden_size
        num_target_layers = target_config.num_hidden_layers
        self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)

        # Project concatenated target features → H
        self.fc = nn.Linear(len(self.target_layer_ids) * H, H, bias=False)
        self.hidden_norm = RMSNorm(H, eps=target_config.rms_norm_eps)

        # Draft transformer stack
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(target_config) for _ in range(num_draft_layers)
        ])
        self.norm = RMSNorm(H, eps=target_config.rms_norm_eps)

        # RoPE for the draft (same params as target)
        rope_theta = getattr(target_config, "rope_theta", 1_000_000.0)
        max_pos = getattr(target_config, "max_position_embeddings", 131072)
        head_dim = getattr(target_config, "head_dim",
                           target_config.hidden_size // target_config.num_attention_heads)
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_pos, base=rope_theta)

        # embed_tokens and lm_head are shared from the target at inference time;
        # during training they are passed in explicitly.
        self.embed_tokens: Optional[nn.Embedding] = None
        self.lm_head: Optional[nn.Linear] = None

    def set_shared_modules(self, embed_tokens: nn.Module, lm_head: nn.Module):
        """Bind target embed_tokens and lm_head (called after loading target)."""
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def _project_target(self, target_features: torch.Tensor) -> torch.Tensor:
        """[B, S, n_taps*H] → [B, S, H] with norm."""
        return self.hidden_norm(self.fc(target_features))

    def forward(
        self,
        noise_embedding: torch.Tensor,   # [B, block_size, H] embeddings of masked tokens
        target_features: torch.Tensor,   # [B, ctx_len, n_taps*H] concatenated target hiddens
        position_ids: torch.LongTensor,  # [B, ctx_len + block_size]  full position ids
    ) -> torch.Tensor:
        """
        Returns draft hidden states: [B, block_size, H].
        Caller applies lm_head to get logits.
        """
        B, block_len, H = noise_embedding.shape
        ctx_len = target_features.shape[1]

        target_hidden = self._project_target(target_features)  # [B, ctx_len, H]

        # Build separate position embeddings for q (block only) and k (ctx + block).
        # position_ids: [B, ctx_len + block_len] — ctx positions then block positions.
        block_position_ids = position_ids[:, ctx_len:]        # [B, block_len]
        full_position_ids  = position_ids                      # [B, ctx_len + block_len]

        cos_q, sin_q = self.rotary_emb(block_position_ids)    # [1, block_len, D]
        cos_k, sin_k = self.rotary_emb(full_position_ids)     # [1, ctx_len+block_len, D]

        hidden = noise_embedding
        for layer in self.layers:
            hidden = layer(hidden, target_hidden, (cos_q, sin_q, cos_k, sin_k))

        return self.norm(hidden)   # [B, block_len, H]

    def get_draft_logits(
        self,
        noise_embedding: torch.Tensor,
        target_features: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Convenience: returns [B, block_size, vocab] logits."""
        h = self.forward(noise_embedding, target_features, position_ids)
        assert self.lm_head is not None, "Call set_shared_modules() first"
        return self.lm_head(h)
