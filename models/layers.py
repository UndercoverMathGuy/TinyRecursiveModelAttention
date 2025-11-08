from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F
from entmax import entmax15

#try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
#except ImportError:
#    # Fallback to FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, dropout: float = 0.01):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

class AttentionMap(nn.Module):
    def __init__(self, emb_dim, hidden_dim, block_size=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.query = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(emb_dim, hidden_dim, bias=False)

    def flash_raw(self, x, qk_release=False):
        # Block-tiled matmul
        B, L, _ = x.shape
        q = self.query(x) * (self.hidden_dim ** -0.5)
        k = self.key(x)

        s = x.new_zeros(B, L, L)
        for i in range(0, L, self.block_size):
            i_end = min(i + self.block_size, L)
            q_blk = q[:, i:i_end]                 # [B, Ii, H]
            for j in range(0, L, self.block_size):
                j_end = min(j + self.block_size, L)
                k_blk = k[:, j:j_end]             # [B, Jj, H]
                s[:, i:i_end, j:j_end] = torch.einsum('b i h, b j h -> b i j', q_blk, k_blk)
        if qk_release:
            return s, q, k
        else:
            return s

    def flash(self, x):
        return flash_attn_func(
            query=self.query(x),
            key=self.key(x),
            value=x,
            qk_release = False,
            dropout=0.0)/self.hidden_dim**0.5

    def forward(self, x):
        return self.flash(x)

class AttentionLinks(nn.Module):
    def __init__ (self, emb_dim, hidden_dim, q = 0.9, single_tau = 0.5, abs_floor = 0.0, C_temp=1.0, F_temp=1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.q = q
        self.single_tau = max(single_tau, abs_floor)
        self.query = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.C_weight = nn.Parameter(torch.ones(1))
        self.F_weight = nn.Parameter(torch.ones(1))
        self.C_temp = C_temp
        self.F_temp = F_temp          
        self.eps = 1e-6

    def query_key(self, x):
        q = self.query(x) 
        k = self.key(x)

        q = self.query_norm(q)
        k = self.key_norm(k)

        q = F.normalize(q, p=2, dim=-1, eps=self.eps)
        k = F.normalize(k, p=2, dim=-1, eps=self.eps)
        return q, k

    def forward(self, x):
        _, L, _ = x.shape

        q, k = self.query_key(x)

        wC = 2 * torch.sigmoid(self.C_weight)
        wF = 2 * torch.sigmoid(self.F_weight)

        Gkk = k.transpose(-2, -1) @ k
        Gkq = k.transpose(-2, -1) @ q
        
        C_raw = q @ Gkk @ q.transpose(-2, -1)
        F_raw = q @ Gkq @ k.transpose(-2, -1)

        C = C_raw - C_raw.mean(dim=-1, keepdim=True)
        F = F_raw - F_raw.mean(dim=-1, keepdim=True)

        pC = torch.clamp(entmax15((wC * C)/C_temp, dim=-1), 0.0, 1.0-self.eps)
        pF = torch.clamp(entmax15((wF * F)/F_temp, dim=-1), 0.0, 1.0-self.eps)

        col_sum = pC.sum(dim=-2, keepdim=True)
        dehub = torch.rsqrt(col_sum + self.eps)
        pC = pC * dehub
        
        H = (2.0 * pC * pF) / (pC + pF + self.eps)
        diag_mask = torch.eye(L, dtype=torch.bool, device=x.device).unsqueeze(0)
        H = H.masked_fill(diag_mask, -torch.inf)
        H = entmax15(H, dim=-1)

        return H, pC, pF

class RotateAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, q, single_tau, abs_floor, C_temp, F_temp):
        super().__init__()
        self.links = AttentionLinks(emb_dim, hidden_dim, q, single_tau, abs_floor, C_temp, F_temp)

        
