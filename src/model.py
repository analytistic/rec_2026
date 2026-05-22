"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union

_HASH_PRIMES = [100003, 200003, 300007, 500009]

from .paired_float_stats import PAIRED_FLOAT_MAX_BY_INT


class MixedNorm(nn.Module):
    """Norm wrapper that runs in float32 then restores input dtype.

    Supports LayerNorm (nn.LayerNorm) and RMSNorm (nn.RMSNorm), selected
    via *norm_type*.
    """
    def __init__(self, dim: int, norm_type: str = 'layer') -> None:
        super().__init__()
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'rms':
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type!r}, expected 'layer' or 'rms'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.float()).to(x.dtype)


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_timestamps: dict  # {domain: tensor [B, L]}, per-event raw timestamps for Fourier
    timestamp: torch.Tensor  # [B], row-level timestamp for NS tokens
    hour: torch.Tensor  # [B], row-level hour of day (1..24)
    dow: torch.Tensor  # [B], row-level day of week (1..7)
    weekend: torch.Tensor  # [B], row-level weekend flag (1=workday, 2=weekend)
    paired_int_feats: torch.Tensor = None   # (B, total_paired_int_dim), optional
    paired_float_feats: torch.Tensor = None  # (B, total_paired_float_dim), optional


class ModelOutput(NamedTuple):
    logits: torch.Tensor      # (B, action_num)
    embeddings: torch.Tensor  # (B, D) final embedding before classifier
    ns_tokens: torch.Tensor   # (B, num_ns, D) NS token representations


class MixerFFNInput(NamedTuple):
    """Input to MixerFFN — pre-split token groups.

    user_token:  (B, user_token_num, D)  — all user-side NS tokens
    item_token:  (B, item_token_num, D)  — all item-side NS tokens
    query_token: dict[str, Tensor] — {domain: (B, Nq, D)} per-domain query tokens
    """
    user_token: torch.Tensor
    item_token: torch.Tensor
    query_token: Dict[str, torch.Tensor]


class MixerFFNOutput(NamedTuple):
    """Output from MixerFFN — each group independently processed."""
    user_token: torch.Tensor
    item_token: torch.Tensor
    query_token: Dict[str, torch.Tensor]


class UIQFFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        user_token_num: int = 0,
        item_token_num: int = 0,
        query_domains: Optional[List[str]] = None,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * hidden_mult
        self.query_domains = query_domains or []

        # User & item NS token FFNs
        self.user_ffn = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
        )
        self.item_ffn = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
        )

        # Shared query token FFN (all domains use the same)
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
        )

        self.ffn_norm = MixedNorm(d_model, norm_type)

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        user_out = self.user_ffn(x.user_token)
        item_out = self.item_ffn(x.item_token)
        query_out = {d: self.query_ffn(x.query_token[d]) for d in x.query_token}
        # Concat outputs + residual from inputs → norm → split back
        out_parts = ([query_out[d] for d in self.query_domains]
                     + [user_out, item_out])
        in_parts = ([x.query_token[d] for d in self.query_domains]
                    + [x.user_token, x.item_token])
        flat = self.ffn_norm(torch.cat(out_parts, dim=1) + torch.cat(in_parts, dim=1))
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = flat[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = flat[:, offset:offset + un, :]
        offset += un
        io = flat[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# Fourier Time Encoding
# ═══════════════════════════════════════════════════════════════════════════════


class FourierTimeEncoding(nn.Module):
    """Multi-frequency sinusoidal time encoding.

    Maps absolute timestamps (seconds) to a hidden-dimensional representation
    via log-spaced Fourier features and a learned linear projection.

    Used as a continuous positional encoding for both seq and NS tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_frequencies: int = 12,
        min_period_seconds: float = 3600.0,
        max_period_seconds: float = 40 * 86400.0,
    ) -> None:
        super().__init__()
        periods = torch.logspace(
            math.log10(min_period_seconds),
            math.log10(max_period_seconds),
            steps=num_frequencies,
        )
        self.register_buffer('periods', periods)
        self.proj = nn.Linear(2 * num_frequencies, d_model, bias=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Computes Fourier time encoding.

        Args:
            timestamps: (B, L) integer seconds, 0 = padding.

        Returns:
            (B, L, D) time encoding, zero at padding positions.
        """
        t = timestamps.to(torch.float32).unsqueeze(-1)  # (B, L, 1)
        angle = t / self.periods * (2 * math.pi)        # (B, L, F)
        feats = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (B, L, 2F)
        out = self.proj(feats)                           # (B, L, D)
        # Zero out padding positions (timestamp == 0)
        mask = (timestamps > 0).unsqueeze(-1).to(out.dtype)
        return out * mask


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre',
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = MixedNorm(d_model, norm_type)
            self.norm_kv = MixedNorm(d_model, norm_type)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankTokenMixer(nn.Module):
    """Token mixing from the RankMixer paper with residual + norm.

    Splits d_model into T subspaces, swaps token and subspace dimensions,
    then residual + layer norm.
    """

    def __init__(self, d_model: int, T: int, norm_type: str = 'layer') -> None:
        super().__init__()
        assert d_model % T == 0, f"d_model={d_model} must be divisible by T={T}"
        self.T = T
        self.d_sub = d_model // T
        self.norm = MixedNorm(d_model, norm_type)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        B, T, D = Q.shape
        Q_split = Q.view(B, T, self.T, self.d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous().view(B, T, D)
        return self.norm(Q_rewired + Q)


class GatedRankTokenMixer(nn.Module):
    """Token mixing with sub-space gating + residual + norm.

    Each output token learns T scalar gates (one per sub-space) to control
    how much of each sub-space from the rewired token is mixed in.
    """

    def __init__(self, d_model: int, T: int, norm_type: str = 'layer') -> None:
        super().__init__()
        assert d_model % T == 0, f"d_model={d_model} must be divisible by T={T}"
        self.T = T
        self.d_sub = d_model // T
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.T),
            nn.Sigmoid(),
        )
        self.norm = MixedNorm(d_model, norm_type)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        B, T, D = Q.shape

        Q_split = Q.view(B, T, self.T, self.d_sub)       # (B, T, T, d_sub)
        Q_rewired_split = Q_split.transpose(1, 2)         # (B, T, T, d_sub)

        gate = self.gate_net(
            Q_rewired_split.contiguous().view(B, T, D))   # (B, T, T)
        gate = gate.unsqueeze(-1)                         # (B, T, T, 1)

        Q_mixed    = Q_rewired_split * gate
        Q_original = Q_split * (1 - gate)

        return self.norm(
            Q_mixed.reshape(B, T, D) + Q_original.reshape(B, T, D))


class PerTokenFFN(nn.Module):
    """Per-token FFN — independent parameters per token position, with residual + norm.

    Accepts MixerFFNInput, concats groups, applies per-token FCs,
    adds residual, applies norm, splits back.
    """

    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.0, num_tokens: int = 0,
                 query_domains: Optional[List[str]] = None,
                 norm_type: str = 'layer', **kwargs) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.query_domains = query_domains or []
        hidden_dim = d_model * hidden_mult
        self.fc1 = nn.ModuleList([
            nn.Linear(d_model, hidden_dim) for _ in range(num_tokens)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(hidden_dim, d_model) for _ in range(num_tokens)
        ])
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = MixedNorm(d_model, norm_type)

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        # Concat groups → flat → per-token FCs → residual+norm
        parts = ([x.query_token[d] for d in self.query_domains]
                 + [x.user_token, x.item_token])
        flat = torch.cat(parts, dim=1)
        outputs = []
        for t in range(self.num_tokens):
            xt = flat[:, t:t+1, :]
            xt = self.fc1[t](xt)
            xt = F.gelu(xt)
            xt = self.dropout(xt)
            xt = self.fc2[t](xt)
            outputs.append(xt)
        out = torch.cat(outputs, dim=1)
        out = self.ffn_norm(out + flat)
        # Split back by input shapes
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = out[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = out[:, offset:offset + un, :]
        offset += un
        io = out[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


class SharedFFN(nn.Module):
    """Shared FFN — single fc1/fc2 for all tokens, with residual + norm.

    Accepts MixerFFNInput, concats groups internally, processes, adds residual,
    applies norm, splits back.
    """

    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.0, query_domains: Optional[List[str]] = None,
                 norm_type: str = 'layer', **kwargs) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.query_domains = query_domains or []
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.ffn_norm = MixedNorm(d_model, norm_type)

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        # Concat groups → flat → process → residual+norm
        parts = ([x.query_token[d] for d in self.query_domains]
                 + [x.user_token, x.item_token])
        flat = torch.cat(parts, dim=1)
        out = self.net(flat)
        out = self.ffn_norm(out + flat)
        # Split back by input shapes
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = out[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = out[:, offset:offset + un, :]
        offset += un
        io = out[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


class DenseMoE(nn.Module):
    """Dense Mixture of Experts with input-dependent routing, residual + norm.

    Accepts MixerFFNInput, concats groups internally, processes, adds residual,
    applies norm, splits back.
    """

    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.0, num_experts: int = 16,
                 query_domains: Optional[List[str]] = None,
                 norm_type: str = 'layer', **kwargs) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.num_experts = num_experts
        self.query_domains = query_domains or []

        self.router = nn.Linear(d_model, num_experts)
        self.W1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.b2 = nn.Parameter(torch.empty(num_experts, d_model))
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = MixedNorm(d_model, norm_type)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def _apply_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Flat dense MoE forward: (B, T, D) → (B, T, D)."""
        weights = F.softmax(self.router(x), dim=-1)
        h = torch.einsum('btd,ndk->btnk', x, self.W1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.einsum('btnk,nkd->btnd', h, self.W2) + self.b2
        return torch.einsum('btn,btnd->btd', weights, h)

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        parts = ([x.query_token[d] for d in self.query_domains]
                 + [x.user_token, x.item_token])
        flat = torch.cat(parts, dim=1)
        out = self.ffn_norm(self._apply_moe(flat) + flat)
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = out[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = out[:, offset:offset + un, :]
        offset += un
        io = out[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


class DenseProtoMoE(nn.Module):
    """Dense MoE with static per-token expert routing, residual + norm.

    Each NS token position learns a fixed set of expert scores (not
    input-dependent). The routing is purely position-based: each token
    slot gets a consistent mixture of experts.
    """

    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.0, num_experts: int = 16,
                 query_domains: Optional[List[str]] = None,
                 num_ns_tokens: int = 0, norm_type: str = 'layer',
                 **kwargs) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.num_experts = num_experts
        self.query_domains = query_domains or []

        # Static per-token routing scores (no input dependence)
        self.ns_token_route = nn.Parameter(torch.empty(num_ns_tokens, num_experts))

        self.W1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.b2 = nn.Parameter(torch.empty(num_experts, d_model))
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = MixedNorm(d_model, norm_type)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.xavier_normal_(self.ns_token_route)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def _apply_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Flat dense MoE forward: (B, T, D) → (B, T, D)."""
        B, T, _ = x.shape
        weights = F.softmax(self.ns_token_route[:T].unsqueeze(0).expand(B, -1, -1), dim=-1)

        h = torch.einsum('btd,ndk->btnk', x, self.W1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.einsum('btnk,nkd->btnd', h, self.W2) + self.b2
        return torch.einsum('btn,btnd->btd', weights, h)

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        parts = ([x.query_token[d] for d in self.query_domains]
                 + [x.user_token, x.item_token])
        flat = torch.cat(parts, dim=1)
        out = self.ffn_norm(self._apply_moe(flat) + flat)
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = out[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = out[:, offset:offset + un, :]
        offset += un
        io = out[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


class SparseMoELossFree(nn.Module):
    """Top-K sparse MoE with loss-free load balancing (DeepSeek-style).

    Routed experts use sigmoid gating + bias-on-scores (paper Algorithm 1).
    Optionally includes shared experts (always activated, summed directly).

    All N routed experts are computed densely (einsum) for N ≤ 8 — GPU
    utilization beats sparse execution. The top-K outputs are gathered per
    token and weighted by bias-free sigmoid scores.
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.0,
        bias_lr: float = 1e-3,
        num_shared_experts: int = 0,
        query_domains: Optional[List[str]] = None,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.bias_lr = bias_lr
        self.num_shared = num_shared_experts
        self.query_domains = query_domains or []

        self.router = nn.Linear(d_model, num_experts)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

        # Routed experts
        self.W1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.b2 = nn.Parameter(torch.empty(num_experts, d_model))

        # Shared experts (always activated, summed directly, no gating)
        if num_shared_experts > 0:
            self.shared_W1 = nn.Parameter(
                torch.empty(num_shared_experts, d_model, hidden_dim))
            self.shared_b1 = nn.Parameter(
                torch.empty(num_shared_experts, hidden_dim))
            self.shared_W2 = nn.Parameter(
                torch.empty(num_shared_experts, hidden_dim, d_model))
            self.shared_b2 = nn.Parameter(
                torch.empty(num_shared_experts, d_model))
        else:
            self.shared_W1 = None

        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = MixedNorm(d_model, norm_type)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        if self.shared_W1 is not None:
            nn.init.xavier_normal_(self.shared_W1)
            nn.init.xavier_normal_(self.shared_W2)
            nn.init.zeros_(self.shared_b1)
            nn.init.zeros_(self.shared_b2)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Flat forward: (B, T, D) → (B, T, D)."""
        B, T, D = x.shape
        N = self.num_experts
        K = self.top_k

        # Shared experts
        if self.shared_W1 is not None:
            S = self.num_shared
            h_s = torch.einsum('btd,sdk->btsk', x, self.shared_W1) + self.shared_b1
            h_s = F.gelu(h_s)
            h_s = self.dropout(h_s)
            h_s = torch.einsum('btsk,skd->btsd', h_s, self.shared_W2) + self.shared_b2
            shared_out = h_s.sum(dim=2)
        else:
            shared_out = 0

        # Routed experts
        router_scores = torch.sigmoid(self.router(x))
        biased_scores = router_scores + self.expert_bias
        _, expert_idx = biased_scores.topk(K, dim=-1)
        gating = router_scores.gather(dim=-1, index=expert_idx)

        h = torch.einsum('btd,ndk->btnk', x, self.W1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.einsum('btnk,nkd->btnd', h, self.W2) + self.b2

        idx_expanded = expert_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        expert_outputs = h.gather(dim=2, index=idx_expanded)
        routed_out = (expert_outputs * gating.unsqueeze(-1)).sum(dim=2)

        if self.training:
            with torch.no_grad():
                total_selections = B * T * K
                target_load = total_selections / N
                load = F.one_hot(expert_idx, N).sum(dim=(0, 1, 2)).to(router_scores.dtype)
                error = target_load - load
                self.expert_bias.add_(self.bias_lr * error.sign())

        return shared_out + routed_out

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        parts = ([x.query_token[d] for d in self.query_domains]
                 + [x.user_token, x.item_token])
        flat = torch.cat(parts, dim=1)
        out = self.ffn_norm(self._forward_flat(flat) + flat)
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = out[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = out[:, offset:offset + un, :]
        offset += un
        io = out[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


class MTmixAttMoE(nn.Module):
    """MTmixAtt-style Shared Dense MoE (Meituan 2025).

    Three key differences from standard MoE:
      1. Sigmoid gating (not softmax) — each expert gets an independent 0-1 gate.
      2. Dense activation — ALL experts activated per token (no top-K).
      3. Fine-grained splitting — each base expert split into *m* smaller
         sub-experts, total FLOPs unchanged.

    Formula (Eq.11-12):
        h_t = Σ α_i · FFN_i(u_t) + Σ β_j · FFN_j(u_t)
              ↑ shared (Ks)         ↑ fine-grained (m×N)
              sigmoid gate          sigmoid gate

    Fine-grained split:
        Original:   N experts,  each hidden_dim = d_model × hidden_mult
        Split:      m×N experts, each hidden_dim = d_model × hidden_mult / m
        Total FLOPs: N × (d × h) = m×N × (d × h/m)
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        num_experts: int = 4,           # N: base experts
        num_shared_experts: int = 2,    # Ks: shared experts
        num_fine_grained: int = 2,      # m: split factor
        query_domains: Optional[List[str]] = None,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        fine_hidden = hidden_dim // num_fine_grained
        num_fine = num_experts * num_fine_grained   # M = m × N
        self.query_domains = query_domains or []

        # Fine-grained experts: sigmoid gate + dense einsum
        self.fine_gate = nn.Linear(d_model, num_fine)
        self.fine_W1 = nn.Parameter(torch.empty(num_fine, d_model, fine_hidden))
        self.fine_b1 = nn.Parameter(torch.empty(num_fine, fine_hidden))
        self.fine_W2 = nn.Parameter(torch.empty(num_fine, fine_hidden, d_model))
        self.fine_b2 = nn.Parameter(torch.empty(num_fine, d_model))

        # Shared experts
        self.num_shared = num_shared_experts
        if num_shared_experts > 0:
            self.shared_gate = nn.Linear(d_model, num_shared_experts)
            self.shared_W1 = nn.Parameter(
                torch.empty(num_shared_experts, d_model, hidden_dim))
            self.shared_b1 = nn.Parameter(
                torch.empty(num_shared_experts, hidden_dim))
            self.shared_W2 = nn.Parameter(
                torch.empty(num_shared_experts, hidden_dim, d_model))
            self.shared_b2 = nn.Parameter(
                torch.empty(num_shared_experts, d_model))
        else:
            self.shared_gate = None
            self.shared_W1 = None

        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = MixedNorm(d_model, norm_type)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.fine_W1)
        nn.init.xavier_normal_(self.fine_W2)
        nn.init.zeros_(self.fine_b1)
        nn.init.zeros_(self.fine_b2)
        if self.shared_W1 is not None:
            nn.init.xavier_normal_(self.shared_W1)
            nn.init.xavier_normal_(self.shared_W2)
            nn.init.zeros_(self.shared_b1)
            nn.init.zeros_(self.shared_b2)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Flat forward: (B, T, D) → (B, T, D)."""
        B, T, D = x.shape
        fine_gates = torch.sigmoid(self.fine_gate(x))
        h = torch.einsum('btd,mdk->btmk', x, self.fine_W1) + self.fine_b1
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.einsum('btmk,mkd->btmd', h, self.fine_W2) + self.fine_b2
        out = (h * fine_gates.unsqueeze(-1)).sum(dim=2)

        if self.shared_W1 is not None:
            shared_gates = torch.sigmoid(self.shared_gate(x))
            h_s = torch.einsum('btd,sdk->btsk', x, self.shared_W1) + self.shared_b1
            h_s = F.gelu(h_s)
            h_s = self.dropout(h_s)
            h_s = torch.einsum('btsk,skd->btsd', h_s, self.shared_W2) + self.shared_b2
            out = out + (h_s * shared_gates.unsqueeze(-1)).sum(dim=2)

        return out

    def forward(self, x: MixerFFNInput) -> MixerFFNOutput:
        parts = ([x.query_token[d] for d in self.query_domains]
                 + [x.user_token, x.item_token])
        flat = torch.cat(parts, dim=1)
        out = self.ffn_norm(self._forward_flat(flat) + flat)
        offset = 0
        qo = {}
        for d in self.query_domains:
            nq = x.query_token[d].shape[1]
            qo[d] = out[:, offset:offset + nq, :]
            offset += nq
        un = x.user_token.shape[1]
        uo = out[:, offset:offset + un, :]
        offset += un
        io = out[:, offset:, :]
        return MixerFFNOutput(user_token=uo, item_token=io, query_token=qo)


class RankMixerBlockV2(nn.Module):
    """Unified RankMixer query boosting block.

    Token mixing → split → FFN (residual + norm inside each FFN).
    Returns MixerFFNOutput(query_token, user_token, item_token).
    """

    FFN_REGISTRY = {
        'shared': SharedFFN,
        'per_token': PerTokenFFN,
        'densemoe': DenseMoE,
        'densemoe_proto': DenseProtoMoE,
        'sparsemoe_lf': SparseMoELossFree,
        'mtmixatt_moe': MTmixAttMoE,
        'uiq': UIQFFN,
    }

    MIXER_REGISTRY = {
        'rank': RankTokenMixer,
        'gated': GatedRankTokenMixer,
    }

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full',
        norm_type: str = 'layer',
        ffn_name: str = 'shared',
        ffn_config: dict = {},
        mixer_type: str = 'rank',
        # Group boundaries for MixerFFNInput split/merge
        user_token_num: int = 0,
        item_token_num: int = 0,
        query_domains: Optional[List[str]] = None,
        num_queries: int = 0,
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode
        self.ffn_name = ffn_name
        self.user_token_num = user_token_num
        self.item_token_num = item_token_num
        self.query_domains = query_domains or []
        self.num_queries = num_queries

        if mode == 'none':
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total}"
                )
            if mixer_type not in self.MIXER_REGISTRY:
                raise ValueError(f"Unknown mixer_type: {mixer_type!r}, "
                                 f"available: {list(self.MIXER_REGISTRY.keys())}")
            mixer_cls = self.MIXER_REGISTRY[mixer_type]
            self.token_mixer = mixer_cls(d_model, n_total, norm_type=norm_type)

        # FFN registry lookup — all variants receive query_domains
        if ffn_name not in self.FFN_REGISTRY:
            raise ValueError(f"Unknown ffn_name: {ffn_name!r}, "
                             f"available: {list(self.FFN_REGISTRY.keys())}")
        ffn_cls = self.FFN_REGISTRY[ffn_name]
        total_ns_tokens = num_queries * len(self.query_domains) + user_token_num + item_token_num
        ffn_kwargs = {**ffn_config, 'query_domains': self.query_domains,
                      'norm_type': norm_type}
        if ffn_name == 'per_token':
            ffn_kwargs.setdefault('num_tokens', n_total)
        if ffn_name == 'densemoe_proto':
            ffn_kwargs.setdefault('num_ns_tokens', total_ns_tokens)
        self.ffn = ffn_cls(d_model=d_model, hidden_mult=hidden_mult,
                           dropout=dropout, **ffn_kwargs)

    def _split_into_groups(self, x: torch.Tensor) -> MixerFFNInput:
        """Split flat (B, T, D) into MixerFFNInput by group boundaries.

        Flat order: [queries (per-domain), user_tokens, item_tokens].
        """
        offset = 0
        query_token = {}
        for domain in self.query_domains:
            query_token[domain] = x[:, offset:offset + self.num_queries, :]
            offset += self.num_queries
        user_token = x[:, offset:offset + self.user_token_num, :]
        offset += self.user_token_num
        item_token = x[:, offset:, :]
        return MixerFFNInput(user_token=user_token, item_token=item_token,
                             query_token=query_token)

    def forward(self, Q: torch.Tensor) -> MixerFFNOutput:
        """Token mixing → split → FFN (residual+norm inside) → MixerFFNOutput."""
        if self.mode == 'none':
            return self._split_into_groups(Q)

        S = self.token_mixer(Q) if self.mode == 'full' else Q

        mixer_input = self._split_into_groups(S)
        return self.ffn(mixer_input)


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = MixedNorm(global_info_dim, norm_type)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    MixedNorm(d_model, norm_type),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).to(seq_tokens_list[i].dtype)  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.norm = MixedNorm(d_model, norm_type)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class SeqSharedFFN(nn.Module):
    """Shared FFN for sequence encoder, Pre-LN norm → net → residual inside.

    Input/output: flat (B, L, D).
    """

    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.0, norm_type: str = 'layer',
                 **kwargs) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.ffn_norm = MixedNorm(d_model, norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.ffn_norm(x)) + x


class SeqDenseMoE(nn.Module):
    """Dense MoE for sequence encoder, Pre-LN norm → moe → residual inside.

    Input/output: flat (B, L, D).
    """

    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.0, num_experts: int = 16,
                 norm_type: str = 'layer', **kwargs) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.num_experts = num_experts

        self.router = nn.Linear(d_model, num_experts)
        self.W1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.b2 = nn.Parameter(torch.empty(num_experts, d_model))
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = MixedNorm(d_model, norm_type)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        inp = self.ffn_norm(x)
        weights = F.softmax(self.router(inp), dim=-1)
        h = torch.einsum('btd,ndk->btnk', inp, self.W1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.einsum('btnk,nkd->btnd', h, self.W2) + self.b2
        return torch.einsum('btn,btnd->btd', weights, h) + x


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    FFN can be either a standard shared FFN or a Dense MoE variant.
    """

    FFN_REGISTRY = {
        'shared': SeqSharedFFN,
        'densemoe': SeqDenseMoE,
    }

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        norm_type: str = 'layer',
        ffn_name: str = 'shared',
        ffn_config: dict = {},
    ) -> None:
        super().__init__()
        self.norm1 = MixedNorm(d_model, norm_type)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        if ffn_name not in self.FFN_REGISTRY:
            raise ValueError(f"Unknown ffn_name: {ffn_name!r}, "
                             f"available: {list(self.FFN_REGISTRY.keys())}")

        ffn_cls = self.FFN_REGISTRY[ffn_name]
        ffn_kwargs = {**ffn_config, 'query_domains': [],
                      'norm_type': norm_type}
        self.ffn = ffn_cls(
            d_model=d_model,
            hidden_mult=hidden_mult,
            dropout=dropout,
            **ffn_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN inside SeqSharedFFN / SeqDenseMoE)
        x = self.ffn(x)

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = MixedNorm(d_model, norm_type)
        self.norm_kv = MixedNorm(d_model, norm_type)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = MixedNorm(d_model, norm_type)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).to(top_k_tokens.dtype)

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
    norm_type: str = 'layer',
    ffn_name: str = 'shared',
    ffn_config: dict = {},
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).
        norm_type: Normalization type ('layer' or 'rms').
        ffn_name: FFN variant name (only used by transformer).
        ffn_config: Extra kwargs for the FFN variant.

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout, norm_type=norm_type)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout,
                                  norm_type=norm_type, ffn_name=ffn_name,
                                  ffn_config=ffn_config)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal, norm_type=norm_type)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        norm_type: str = 'layer',
        # FFN variants (separate for seq encoder vs RankMixer)
        ffn_name: str = 'shared',
        ffn_config: dict = {},
        mixer_type: str = 'rank',
        seq_ffn_name: str = 'shared',
        seq_ffn_config: dict = {},
        # Group info for MixerFFNInput split/merge
        user_token_num: int = 0,
        item_token_num: int = 0,
        seq_domains: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns
        self.seq_domains = seq_domains or []

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal,
                norm_type=norm_type,
                ffn_name=seq_ffn_name,
                ffn_config=seq_ffn_config,
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre',
                norm_type=norm_type,
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlockV2(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
            norm_type=norm_type,
            ffn_name=ffn_name,
            ffn_config=ffn_config,
            mixer_type=mixer_type,
            user_token_num=user_token_num,
            item_token_num=item_token_num,
            query_domains=seq_domains,
            num_queries=num_queries,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting (returns MixerFFNOutput with residual+norm applied)
        mixer_out = self.mixer(combined)

        # 5. Extract per-domain Q and NS from MixerFFNOutput
        next_q_list = [mixer_out.query_token[d] for d in self.seq_domains]
        next_ns = torch.cat([mixer_out.user_token, mixer_out.item_token], dim=1)

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0,
                 norm_type: str = 'layer') -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with MixedNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                MixedNorm(d_model, norm_type),
            )
            for group in groups
        ])

        # Global token: mean pool all raw embeddings → MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        all_fid_embs = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).to(emb_all.dtype).unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
                all_fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(F.silu(proj(cat_emb.to(proj[0].weight.dtype))).unsqueeze(1))
        global_emb = torch.stack(all_fid_embs, dim=0).mean(dim=0)  # (B, emb_dim)
        global_token = self.global_mlp(global_emb).unsqueeze(1)  # (B, 1, d_model)
        return torch.cat(tokens, dim=1), global_token


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        norm_type: str = 'layer',
        hash_config: Optional[Dict[int, Any]] = None,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
            norm_type: Normalization type ('layer' or 'rms').
            hash_config: Optional mapping of fid_idx → {H, k} for multi-hash embedding.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.hash_config = hash_config or {}

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info). Hash features are excluded
        # from embs and stored separately in hash_embs.
        self._hash_multi: Dict[int, dict] = {}
        embs_raw = []
        for fid_idx, (vs, offset, length) in enumerate(feature_specs):
            if fid_idx in self.hash_config:
                cfg = self.hash_config[fid_idx]
                H = cfg['H']
                k = cfg['k']
                chunk_dim = emb_dim // k
                self._hash_multi[fid_idx] = {'H': H, 'k': k, 'chunk_dim': chunk_dim}
                embs_raw.append(None)  # not in embs; handled by hash_embs
            else:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                embs_raw.append(None if skip else nn.Embedding(int(vs) + 1, emb_dim))
        self.embs = nn.ModuleList([e for e in embs_raw if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs_raw:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Separate ModuleList for multi-hash sub-embeddings
        self.hash_embs = nn.ModuleList()
        for fid_idx in sorted(self._hash_multi.keys()):
            cfg = self._hash_multi[fid_idx]
            H, k, chunk_dim = cfg['H'], cfg['k'], cfg['chunk_dim']
            start = len(self.hash_embs)
            for _ in range(k):
                self.hash_embs.append(nn.Embedding(H, chunk_dim))
            cfg['start'] = start
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with MixedNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                MixedNorm(d_model, norm_type),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

        # Global token: mean pool all raw embeddings → MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    if fid_idx in self._hash_multi:
                        cfg = self._hash_multi[fid_idx]
                        k, H = cfg['k'], cfg['H']
                        start = cfg['start']
                        vals = int_feats[:, offset].long()
                        parts = []
                        for j in range(k):
                            emb = self.hash_embs[start + j]
                            hash_idx = (_HASH_PRIMES[j] * (fid_idx + 1) + vals) % (H - 1) + 1
                            hash_idx = torch.where(vals == 0, 0, hash_idx)
                            parts.append(emb(hash_idx))
                        fid_emb = torch.cat(parts, dim=-1)
                    else:
                        fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).to(emb_all.dtype).unsqueeze(-1)
                        count = mask.sum(dim=1)
                        sum_emb = (emb_all * mask).sum(dim=1)
                        pad_emb = emb_layer(torch.zeros(1, dtype=torch.long, device=vals.device))
                        fid_emb = torch.where(
                            count.expand_as(sum_emb) > 0,
                            sum_emb / count.clamp(min=1),
                            pad_emb,
                        )
                all_embs.append(fid_emb)

        # Global token: mean pool all raw embeddings → MLP
        global_emb = torch.stack(all_embs, dim=0).mean(dim=0)  # (B, emb_dim)
        global_token = self.global_mlp(global_emb).unsqueeze(1)  # (B, 1, d_model)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk.to(proj[0].weight.dtype))).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1), global_token  # (B, num_ns_tokens, d_model), (B, 1, d_model)


class SENetProjection(nn.Module):
    """SENet-style per-position feature gating for per-step embeddings.

    For each position in the sequence independently, the S features (each
    ``emb_dim``-dimensional) are squeezed over ``emb_dim`` to produce a
    ``(B, L, S)`` descriptor, then gated through a bottleneck MLP + sigmoid
    to get per-position, per-feature scores.  Features are weighted by
    these scores and aggregated via summation over S.

    An optional Fourier encoding ``(B, L, F)`` is concatenated after
    aggregation before the final ``Linear(emb_dim, d_model)`` projection.

    Args:
        num_features: Number of per-step features (S).
        emb_dim: Per-feature embedding dimension.
        d_model: Output dimension.
        fourier_dim: Dimension of Fourier encoding (0 = disabled).
        reduction: Bottleneck reduction ratio for the gating MLP.
        norm_type: Normalization type ('layer' or 'rms').
    """

    def __init__(
        self,
        num_features: int,
        emb_dim: int,
        d_model: int,
        fourier_dim: int = 0,
        reduction: int = 4,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.fourier_dim = fourier_dim

        reduced_dim = max(num_features // reduction, 4)

        # Per-position, per-feature gate: (B, L, S) → (B, L, S)
        self.gate_net = nn.Sequential(
            nn.Linear(num_features, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_features, bias=False),
            nn.Sigmoid(),
        )

        proj_in_dim = emb_dim + fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
            MixedNorm(d_model, norm_type),
        )

    def forward(
        self,
        feat_stack: torch.Tensor,
        fourier: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            feat_stack: ``(B, L, S, emb_dim)`` stacked per-step embeddings.
            fourier: optional ``(B, L, F)`` Fourier encoding.

        Returns:
            ``(B, L, d_model)`` tensor (pre-gelu, caller applies activation).
        """
        # Squeeze over embedding dim → per-position per-feature descriptor
        squeezed = feat_stack.mean(dim=-1)  # (B, L, S)

        # Gate → per-position per-feature importance scores
        gate = self.gate_net(squeezed)  # (B, L, S)

        # Rescale: (B, L, S, emb_dim) * (B, L, S, 1)
        weighted = feat_stack * gate.unsqueeze(-1)

        # Aggregate over S → (B, L, emb_dim)
        out = weighted.sum(dim=2)

        # Concat Fourier if present
        if fourier is not None and self.fourier_dim > 0:
            out = torch.cat([out, fourier], dim=-1)

        return self.proj(out)


class AutoNSTokenizer(nn.Module):
    """Learnable static feature grouping with concat.

    Each group selects k = n_f // n_g features (top-k by learned weight W),
    scales each by its softmax weight, CONCAT's them (preserving per-feature
    dimensions), then projects to d_model.

    k is derived internally from n_features / num_groups — no extra
    hyperparameter needed.
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        emb_dim: int,
        d_model: int,
        num_groups: int,
        emb_skip_threshold: int = 0,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.emb_dim = emb_dim
        self.num_groups = num_groups
        n_features = len(feature_specs)
        self.k = max(1, n_features // num_groups + 1)

        # Embedding tables
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Static grouping matrix W (n_g × n_f)
        self.W = nn.Parameter(torch.empty(num_groups, n_features))
        nn.init.xavier_normal_(self.W)

        # Concat(k × emb_dim) → d_model
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.k * emb_dim, d_model),
                MixedNorm(d_model, norm_type),
            )
            for _ in range(num_groups)
        ])

        logging.info(
            f"AutoNSTokenizer: {n_features} features → "
            f"{num_groups} groups, k={self.k}"
        )

        # Global token: mean pool all raw embeddings → MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embed → top-k group → concat → project."""
        B = int_feats.shape[0]

        # Embed each feature
        all_embs = []
        for fid_idx, (vs, offset, length) in enumerate(self.feature_specs):
            emb_real_idx = self._emb_index[fid_idx]
            if emb_real_idx == -1:
                all_embs.append(int_feats.new_zeros(B, self.emb_dim))
            else:
                emb_layer = self.embs[emb_real_idx]
                if length == 1:
                    f = emb_layer(int_feats[:, offset].long())
                else:
                    vals = int_feats[:, offset:offset + length].long()
                    emb_all = emb_layer(vals)
                    mask = (vals != 0).to(emb_all.dtype).unsqueeze(-1)
                    count = mask.sum(dim=1).clamp(min=1)
                    f = (emb_all * mask).sum(dim=1) / count
                all_embs.append(f)
        X = torch.stack(all_embs, dim=1)  # (B, n_f, emb_dim)

        # Global token: mean pool all feature embeddings → MLP
        global_token = self.global_mlp(X.mean(dim=1)).unsqueeze(1)  # (B, 1, d_model)

        # Top-k grouping: each group selects its k features
   
        topk_weights = F.softmax(self.W, dim=-1)  # (n_g, n_f)
        topk_scores, topk_idx = torch.topk(topk_weights, self.k, dim=-1)  # (n_g, k)

        # Gather k features per group
        idx = topk_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.emb_dim)
        X_exp = X.unsqueeze(1).expand(-1, self.num_groups, -1, -1)
        grouped = torch.gather(X_exp, 2, idx)  # (B, n_g, k, emb_dim)

        # Scale + concat → project → SiLU
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize scores
        w = topk_scores.unsqueeze(0).unsqueeze(-1)  # (1, n_g, k, 1)
        out = (grouped * w).reshape(B, self.num_groups, -1)  # (B, n_g, k*emb_dim)
        tokens = []
        for i in range(self.num_groups):
            tokens.append(F.silu(self.token_projs[i](out[:, i, :].to(self.token_projs[i][0].weight.dtype))).unsqueeze(1))  # (B, 1, d_model)


        return torch.cat(tokens, dim=1), global_token  # (B, n_g, d_model), (B, 1, d_model)


class PairedProcessor(nn.Module):
    """Process paired int+float features into a single NS token.

    For each paired fid:
      1. Embed int values at each slot: (B, L, emb_dim)
      2. Compute weight = log1p(float_val), zero-weight padding (float_val=0)
      3. Normalize weights over slots: w_i = w_i / sum(w)
      4. Weighted sum: sum(w_i * int_emb_i) → (B, emb_dim)
    Then concat all per-fid embeddings and project to one d_model token.

    Padding positions (float_val=0) naturally get zero weight via the
    normalization, so they contribute nothing to the output token.
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],  # [(vocab_size, offset, length), ...]
        fids: List[int],                            # fid for each spec, used to look up PAIRED_FLOAT_MAX_BY_INT
        emb_dim: int,
        d_model: int,
        norm_type: str = 'layer',
    ) -> None:
        super().__init__()
        self.specs = feature_specs
        self.num_paired = len(feature_specs)

        self.embs = nn.ModuleList()
        for vs, offset, length in feature_specs:
            vs = max(vs, 1)
            self.embs.append(nn.Embedding(vs + 1, emb_dim))

        # Per-int float max lookup tables (frozen Embedding) for count-type fids.
        # Score-type fids get a dummy Embedding(1, 1) and bypass normalization.
        self.max_embs = nn.ModuleList()
        self.is_count: List[bool] = []
        for fid in fids:
            if fid in PAIRED_FLOAT_MAX_BY_INT:
                float_by_int = PAIRED_FLOAT_MAX_BY_INT[fid]
                max_val = max(float_by_int.values())
                if max_val > 10.0:  # count type
                    max_int = max(float_by_int.keys())
                    weight = torch.ones(max_int + 1, 1)  # default 1.0 for unseen int values
                    for int_val, float_max in float_by_int.items():
                        weight[int_val] = float_max
                    self.max_embs.append(nn.Embedding.from_pretrained(weight, freeze=True))
                    self.is_count.append(True)
                else:  # score type (float_max all 0)
                    emb = nn.Embedding(1, 1)
                    emb.requires_grad_(False)
                    self.max_embs.append(emb)
                    self.is_count.append(False)
            else:
                emb = nn.Embedding(1, 1)
                emb.requires_grad_(False)
                self.max_embs.append(emb)
                self.is_count.append(False)

        cat_dim = self.num_paired * emb_dim
        self.proj = nn.Sequential(
            nn.Linear(cat_dim, d_model),
            MixedNorm(d_model, norm_type),
        )

    def forward(
        self,
        paired_int_feats: torch.Tensor,
        paired_float_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, 1, d_model)."""
        fid_embs = []
        for i, (vs, offset, length) in enumerate(self.specs):
            int_vals = paired_int_feats[:, offset:offset + length].long()
            float_vals = paired_float_feats[:, offset:offset + length]

            emb = self.embs[i](int_vals)  # (B, L, emb_dim)

            if self.is_count[i]:
                # Count-type: per-int global normalization (no softmax across slots)
                per_slot_max = self.max_embs[i](int_vals).squeeze(-1)  # (B, L)
                w = float_vals.clamp(min=0)
                w = torch.log1p(w)
                w = w / torch.log1p(per_slot_max)
                w = w * (int_vals != 0).float()
            else:
                # Score-type: raw weights, no normalization
                w = float_vals
            total_w = w.sum(dim=1, keepdim=True)  # (B, 1)
            token = (w.unsqueeze(-1) * emb).sum(dim=1)  # (B, emb_dim)
            pad_emb = self.embs[i](torch.zeros(1, dtype=torch.long, device=int_vals.device)).squeeze(0)
            token = torch.where(total_w > 0, token, pad_emb)
            fid_embs.append(token)

        cat = torch.cat(fid_embs, dim=-1)  # (B, num_paired * emb_dim)
        out = self.proj(cat)               # (B, d_model)
        return out.unsqueeze(1)            # (B, 1, d_model)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Hash embedding config for item features (fid_idx → H)
        item_hash_config: Optional[Dict[int, int]] = None,
        # Hash embedding config for user features (fid_idx → H)
        user_hash_config: Optional[Dict[int, int]] = None,
        # Paired int+float feature specs (empty list = no paired features)
        paired_feature_specs: List[Tuple[int, int, int]] = None,
        paired_fids: Optional[List[int]] = None,
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        embed_dropout_rate: float = 0.01,
        seq_id_dropout_rate: float = 0.02,
        hidden_dropout_rate: float = 0.01,
        norm_type: str = 'layer',
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # Dtype control
        dense_dtype: torch.dtype = torch.float32,
        sparse_dtype: torch.dtype = torch.float32,
        # Time feature controls
        fourier_seq: bool = True,
        fourier_ns: bool = True,
        use_row_time_ns: bool = True,
        # FFN variant for RankMixer
        ffn_name: str = 'shared',
        ffn_config: dict = {},
        # Token mixer type ('rank' or 'gated')
        mixer_type: str = 'rank',
        # FFN variant for Seq Encoder
        seq_ffn_name: str = 'shared',
        seq_ffn_config: dict = {},
        # Sequence projection variant
        seq_proj_type: str = 'linear',
        # Domain embedding flag
        use_domain_emb: bool = False,
        # Seq multi-hash embedding config: {domain: {pos: {H, k}}}
        seq_hash_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.dense_dtype = dense_dtype
        self.sparse_dtype = sparse_dtype
        self.norm_type = norm_type
        self.fourier_seq = fourier_seq
        self.fourier_ns = fourier_ns
        self.ffn_name = ffn_name
        self.ffn_config = ffn_config
        self.mixer_type = mixer_type
        self.seq_ffn_name = seq_ffn_name
        self.seq_ffn_config = seq_ffn_config
        self.seq_proj_type = seq_proj_type
        self.use_row_time_ns = use_row_time_ns

        # ================== NS Tokens Construction ==================

        # Row-level time features: embedded as a dedicated NS token after item
        # tokens (not mixed into user int feats), so time gets its own token.
        self.has_time_ns = use_row_time_ns
        if use_row_time_ns:
            # 8 hour segments: 3-6→0, 7-10→1, 11-12→2, 13-14→3, 15-16→4, 17-19→5, 20-22→6, 23,24,1,2→7
            self.register_buffer(
                'hour_to_segment',
                torch.tensor([0, 7,7, 0,0,0,0, 1,1,1,1, 2,2, 3,3, 4,4, 5,5,5, 6,6,6, 7,7],
                             dtype=torch.long))
            self.time_code_emb = nn.Embedding(16, d_model)

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                norm_type=norm_type,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                norm_type=norm_type,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                norm_type=norm_type,
                hash_config=user_hash_config,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                norm_type=norm_type,
                hash_config=item_hash_config,
            )
            num_item_ns = item_ns_tokens
        elif ns_tokenizer_type == 'auto':
            # Auto mode: use the same tokenizer for both user and item
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = AutoNSTokenizer(
                feature_specs=user_int_feature_specs,
                emb_dim=emb_dim,
                d_model=d_model,
                num_groups=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                norm_type=norm_type,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = AutoNSTokenizer(
                feature_specs=item_int_feature_specs,
                emb_dim=emb_dim,
                d_model=d_model,
                num_groups=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                norm_type=norm_type,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                MixedNorm(d_model, norm_type),
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                MixedNorm(d_model, norm_type),
            )

        # Domain embeddings for NS/Q tokens (so token mixing can distinguish user/item/query)
        self.use_domain_emb = use_domain_emb
        if use_domain_emb:
            self.user_domain_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            self.item_domain_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            self.query_domain_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # Paired int+float processor
        paired_feature_specs = paired_feature_specs or []
        self.has_paired = len(paired_feature_specs) > 0
        if self.has_paired:
            self.paired_processor = PairedProcessor(
                feature_specs=paired_feature_specs,
                fids=paired_fids or [],
                emb_dim=emb_dim,
                d_model=d_model,
                norm_type=norm_type,
            )
        num_paired = len(paired_feature_specs)
        num_paired_tokens = 1 if self.has_paired else 0

        # Global token for user and item (one extra token each)
        num_user_ns += 1
        num_item_ns += 1

        # Total NS token count
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0) + num_paired_tokens
                       + num_item_ns + (1 if self.has_item_dense else 0)
                       + (1 if self.has_time_ns else 0))

        # Token group boundaries for MixerFFNInput split/merge in blocks
        _user_token_num = num_user_ns + (1 if self.has_user_dense else 0) + num_paired_tokens
        _item_token_num = (num_item_ns + (1 if self.has_item_dense else 0)
                           + (1 if self.has_time_ns else 0))

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}. "
                f"Hint: paired features add 1 NS token. "
                f"Set rank_mixer_mode=none in config if you cannot adjust d_model."
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(seq_id_dropout_rate)

        def _make_seq_embs(vocab_sizes, domain_hash_config=None):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0).

            domain_hash_config: {position: {H, k}} for multi-hash features.
            """
            domain_hash_config = domain_hash_config or {}
            embs_raw = []
            is_hash = []
            for pos, vs in enumerate(vocab_sizes):
                if pos in domain_hash_config:
                    embs_raw.append(None)  # handled by multi-hash
                    is_hash.append(True)
                else:
                    is_hash.append(False)
                    skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                    if skip:
                        embs_raw.append(None)
                    else:
                        embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id, is_hash

        # ================== Multi-Hash Seq Embedding Config ==================
        self.seq_hash_config = seq_hash_config or {}
        self._seq_hash_embs = nn.ModuleDict()
        self._seq_hash_emb_index = {}  # {domain: {pos: (start_idx, k)}}

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_is_hash = {}      # domain -> is_hash list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            dhc = self.seq_hash_config.get(domain, {})
            embs, idx_map, is_id, is_hash = _make_seq_embs(vs, dhc)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_is_hash[domain] = is_hash
            self._seq_vocab_sizes[domain] = vs

            # Build multi-hash embeddings for this domain
            if dhc:
                hash_embs = []
                index = {}
                for pos, cfg in sorted(dhc.items()):
                    H, k = cfg['H'], cfg['k']
                    chunk_dim = emb_dim // k
                    start = len(hash_embs)
                    for _ in range(k):
                        hash_embs.append(nn.Embedding(H, chunk_dim))
                    index[pos] = (start, k)
                self._seq_hash_embs[domain] = nn.ModuleList(hash_embs)
                self._seq_hash_emb_index[domain] = index
            else:
                self._seq_hash_embs[domain] = nn.ModuleList()
                self._seq_hash_emb_index[domain] = {}
            if seq_proj_type == 'senet':
                self._seq_proj[domain] = SENetProjection(
                    num_features=len(vs),
                    emb_dim=emb_dim,
                    d_model=d_model,
                    fourier_dim=emb_dim if fourier_seq else 0,
                    norm_type=norm_type,
                )
            else:
                fourier_add_dim = emb_dim if fourier_seq else 0
                self._seq_proj[domain] = nn.Sequential(
                    nn.Linear(len(vs) * emb_dim + fourier_add_dim, d_model),
                    MixedNorm(d_model, norm_type),
                )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model)

        # ================== Fourier Time Encoding ==================
        if fourier_ns:
            self.time_fourier = FourierTimeEncoding(d_model=d_model)
        if fourier_seq:
            self.seq_time_fourier = FourierTimeEncoding(d_model=emb_dim)


        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
            norm_type=norm_type,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=hidden_dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                norm_type=norm_type,
                ffn_name=ffn_name,
                ffn_config=ffn_config,
                mixer_type=mixer_type,
                seq_ffn_name=seq_ffn_name,
                seq_ffn_config=seq_ffn_config,
                user_token_num=_user_token_num,
                item_token_num=_item_token_num,
                seq_domains=self.seq_domains,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base,
                                              dtype=dense_dtype)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            MixedNorm(d_model, norm_type),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(embed_dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            MixedNorm(d_model, norm_type),
            nn.SiLU(),
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Apply mixed-precision dtype conversion
        self._apply_mixed_precision()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for d in self.seq_domains:
            emb_list = self._seq_embs[d]
            vocab_sizes = self._seq_vocab_sizes[d]
            emb_index = self._seq_emb_index[d]
            is_hash = self._seq_is_hash.get(d, [])
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    if is_hash and i < len(is_hash) and is_hash[i]:
                        if int(vs) > cardinality_threshold:
                            start, num_k = self._seq_hash_emb_index[d][i]
                            for j in range(num_k):
                                emb = self._seq_hash_embs[d][start + j]
                                nn.init.xavier_normal_(emb.weight.data)
                                emb.weight.data[0, :] = 0
                                reinit_ptrs.add(emb.weight.data_ptr())
                                reinit_count += 1
                        else:
                            skip_count += 1
                    # Either hash (reinit done) or skipped — no regular emb to process
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    if i in tokenizer._hash_multi and int(vs) > cardinality_threshold:
                        cfg = tokenizer._hash_multi[i]
                        start, k = cfg['start'], cfg['k']
                        for j in range(k):
                            emb = tokenizer.hash_embs[start + j]
                            nn.init.xavier_normal_(emb.weight.data)
                            emb.weight.data[0, :] = 0
                            reinit_ptrs.add(emb.weight.data_ptr())
                            reinit_count += 1
                    else:
                        skip_count += 1
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def _apply_mixed_precision(self) -> None:
        """Convert module parameters to target dtypes.

        Dense modules (Linear, etc.) → self.dense_dtype (e.g., bfloat16)
        Embedding modules → self.sparse_dtype (e.g., float32)
        LayerNorm / RMSNorm → float32 (numerical stability with low precision)
        RotaryEmbedding buffers → float32 (RoPE precision)
        """
        if self.dense_dtype == torch.float32 and self.sparse_dtype == torch.float32:
            return

        # First pass: convert everything except Embedding/Norm to dense_dtype.
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)):
                continue
            module.to(dtype=self.dense_dtype)

        # Second pass: override specific module types.
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.to(dtype=self.sparse_dtype)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                module.to(dtype=torch.float32)

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        fourier_ts: Optional[torch.Tensor] = None,
        is_hash: Optional[List[bool]] = None,
        hash_config: Optional[Dict] = None,
        hash_embs: Optional[nn.ModuleList] = None,
        hash_emb_index: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain and projects to d_model.

        Two projection modes (controlled by ``seq_proj_type``):

        * linear (default): each per-step feature is embedded independently,
          then all S features are concatenated along the last dim together with
          an optional Fourier encoding, and projected via ``Linear(S*emb, d_model)``.

        * senet: features are stacked as ``(B, L, S, emb_dim)`` and passed to
          ``SENetProjection`` together with the Fourier encoding for sample-adaptive
          gating and aggregation.
        """
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                if is_hash and i < len(is_hash) and is_hash[i] and hash_config and i in hash_config:
                    # Multi-hash embedding
                    cfg = hash_config[i]
                    H, k = cfg['H'], cfg['k']
                    chunk_dim = self.emb_dim // k
                    start_idx = hash_emb_index[i][0]
                    vals = seq[:, i, :]  # (B, L)
                    parts = []
                    for j in range(k):
                        emb = hash_embs[start_idx + j]
                        hash_idx = (_HASH_PRIMES[j] * (i + 1) + vals) % (H - 1) + 1
                        hash_idx = torch.where(vals == 0, 0, hash_idx)
                        parts.append(emb(hash_idx))
                    fid_emb = torch.cat(parts, dim=-1)  # (B, L, emb_dim)
                else:
                    # Feature skipped by emb_skip_threshold: output zero vector
                    fid_emb = seq.new_zeros(B, L, self.emb_dim, dtype=self.sparse_dtype)
                emb_list.append(fid_emb)
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)

        # Optional Fourier encoding
        fourier_enc = None
        if fourier_ts is not None and hasattr(self, 'seq_time_fourier'):
            fourier_enc = self.seq_time_fourier(fourier_ts)  # (B, L, emb_dim)

        if self.seq_proj_type == 'senet':
            # SENet: (B, L, S, emb_dim) + optional (B, L, emb_dim) Fourier
            feat_stack = torch.stack(emb_list, dim=2)  # (B, L, S, emb_dim)
            token_emb = F.gelu(
                proj(feat_stack.to(self.dense_dtype),
                     fourier_enc.to(self.dense_dtype) if fourier_enc is not None else None)
            )
        else:
            # Original: flatten all features + optional Fourier
            cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
            if fourier_enc is not None:
                cat_emb = torch.cat([cat_emb, fourier_enc.to(cat_emb.dtype)], dim=-1)
            token_emb = F.gelu(proj(cat_emb.to(self.dense_dtype)))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids).to(self.dense_dtype)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_int_feats = inputs.user_int_feats

        user_ns, user_global = self.user_ns_tokenizer(user_int_feats)
        user_ns = user_ns.to(self.dense_dtype)
        user_global = user_global.to(self.dense_dtype)
        item_ns, item_global = self.item_ns_tokenizer(inputs.item_int_feats)
        item_ns = item_ns.to(self.dense_dtype)
        item_global = item_global.to(self.dense_dtype)

        if self.use_domain_emb:
            user_ns = user_ns + self.user_domain_emb
            user_global = user_global + self.user_domain_emb
            item_ns = item_ns + self.item_domain_emb
            item_global = item_global + self.item_domain_emb

        ns_parts = [user_ns, user_global]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats.to(self.dense_dtype))).unsqueeze(1)  # (B, 1, D)
            if self.use_domain_emb:
                user_dense_tok = user_dense_tok + self.user_domain_emb
            ns_parts.append(user_dense_tok)
        if self.has_paired:
            paired_tokens = self.paired_processor(inputs.paired_int_feats, inputs.paired_float_feats).to(self.dense_dtype)  # (B, num_paired, D)
            if self.use_domain_emb:
                paired_tokens = paired_tokens + self.user_domain_emb
            ns_parts.append(paired_tokens)
        ns_parts.append(item_ns)
        ns_parts.append(item_global)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats.to(self.dense_dtype))).unsqueeze(1)  # (B, 1, D)
            if self.use_domain_emb:
                item_dense_tok = item_dense_tok + self.item_domain_emb
            ns_parts.append(item_dense_tok)
        if self.has_time_ns:
            seg = self.hour_to_segment[inputs.hour]                # (B,)
            # 深夜(1-2点)属于前一天的时段，用前一天的 DOW 判断 workday
            effective_dow = torch.where(
                inputs.hour <= 2,
                (inputs.dow - 2) % 7 + 1,                          # 前一天
                inputs.dow                                          # 当天
            )
            is_workday = (effective_dow <= 5).long()                # Mon-Fri=1, Sat-Sun=0
            t = self.time_code_emb(is_workday * 8 + seg)           # (B, D)
            ns_parts.append(t.unsqueeze(1))                        # (B, 1, D)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                fourier_ts=inputs.seq_timestamps[domain] if self.fourier_seq else None,
                is_hash=self._seq_is_hash.get(domain, []),
                hash_config=self.seq_hash_config.get(domain, {}),
                hash_embs=self._seq_hash_embs[domain] if domain in self._seq_hash_embs else None,
                hash_emb_index=self._seq_hash_emb_index.get(domain, {}),
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        if self.use_domain_emb:
            q_tokens_list = [q + self.query_domain_emb for q in q_tokens_list]

        # 4. Fourier time encoding on NS and Q tokens (seq tokens encode it inside _embed_seq_domain)
        if self.fourier_ns:
            ns_tokens = ns_tokens + self.time_fourier(inputs.timestamp.unsqueeze(-1))
            q_tokens_list = [q + self.time_fourier(inputs.timestamp.unsqueeze(-1)) for q in q_tokens_list]

        # 5. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return ModelOutput(logits=logits, embeddings=output, ns_tokens=ns_tokens)

    def predict(self, inputs: ModelInput) -> ModelOutput:
        """Runs inference without dropout, returning logits, embeddings and ns_tokens."""
        user_int_feats = inputs.user_int_feats

        user_ns, user_global = self.user_ns_tokenizer(user_int_feats)
        user_ns = user_ns.to(self.dense_dtype)
        user_global = user_global.to(self.dense_dtype)
        item_ns, item_global = self.item_ns_tokenizer(inputs.item_int_feats)
        item_ns = item_ns.to(self.dense_dtype)
        item_global = item_global.to(self.dense_dtype)

        if self.use_domain_emb:
            user_ns = user_ns + self.user_domain_emb
            user_global = user_global + self.user_domain_emb
            item_ns = item_ns + self.item_domain_emb
            item_global = item_global + self.item_domain_emb

        ns_parts = [user_ns, user_global]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats.to(self.dense_dtype))).unsqueeze(1)
            if self.use_domain_emb:
                user_dense_tok = user_dense_tok + self.user_domain_emb
            ns_parts.append(user_dense_tok)
        if self.has_paired:
            paired_tokens = self.paired_processor(inputs.paired_int_feats, inputs.paired_float_feats).to(self.dense_dtype)
            if self.use_domain_emb:
                paired_tokens = paired_tokens + self.user_domain_emb
            ns_parts.append(paired_tokens)
        ns_parts.append(item_ns)
        ns_parts.append(item_global)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats.to(self.dense_dtype))).unsqueeze(1)
            if self.use_domain_emb:
                item_dense_tok = item_dense_tok + self.item_domain_emb
            ns_parts.append(item_dense_tok)
        if self.has_time_ns:
            seg = self.hour_to_segment[inputs.hour]
            effective_dow = torch.where(
                inputs.hour <= 2,
                (inputs.dow - 2) % 7 + 1,
                inputs.dow
            )
            is_workday = (effective_dow <= 5).long()
            t = self.time_code_emb(is_workday * 8 + seg)
            ns_parts.append(t.unsqueeze(1))

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                fourier_ts=inputs.seq_timestamps[domain] if self.fourier_seq else None,
                is_hash=self._seq_is_hash.get(domain, []),
                hash_config=self.seq_hash_config.get(domain, {}),
                hash_embs=self._seq_hash_embs[domain] if domain in self._seq_hash_embs else None,
                hash_emb_index=self._seq_hash_emb_index.get(domain, {}),
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        if self.use_domain_emb:
            q_tokens_list = [q + self.query_domain_emb for q in q_tokens_list]

        # Fourier time encoding on NS and Q tokens (seq tokens encode it inside _embed_seq_domain)
        if self.fourier_ns:
            ns_tokens = ns_tokens + self.time_fourier(inputs.timestamp.unsqueeze(-1))
            q_tokens_list = [q + self.time_fourier(inputs.timestamp.unsqueeze(-1)) for q in q_tokens_list]

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return ModelOutput(logits=logits, embeddings=output, ns_tokens=ns_tokens)
