from typing import Tuple
import einops
import math
import torch
from torch import nn
import torch.nn.functional as F
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
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S (H D)')  # Flatten directly
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

class PositionBias(nn.Module):
    """
    Positional biases for every pair of values in attention matrix
    """
    def __init__(self, grid_height, grid_width):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        max_rel_h = 2 * grid_height - 1
        max_rel_w = 2 * grid_width - 1
        
        self.bias_table = nn.Parameter(torch.zeros(max_rel_h, max_rel_w))
        trunc_normal_init_(self.bias_table, std=0.02)
        
        coords_h = torch.arange(grid_height)
        coords_w = torch.arange(grid_width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.reshape(2, -1)
        
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords[0] += grid_height - 1
        relative_coords[1] += grid_width - 1
        
        self.register_buffer('rel_idx', relative_coords[0] * max_rel_w + relative_coords[1])
        
    def forward(self):
        return self.bias_table.view(-1)[self.rel_idx]

class FlashLinks(nn.Module):
    def __init__(self, hidden_size, head_dim, grid_height, grid_width, num_hops = 2, use_entmax = False, eps = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_hops = num_hops
        self.eps = eps
        self.scale = math.sqrt(head_dim)
        self.seq_len = grid_height * grid_width
        
        self.q_proj = CastedLinear(hidden_size, head_dim, bias=False)
        self.k_proj = CastedLinear(hidden_size, head_dim, bias=False)
        self.v_proj = CastedLinear(hidden_size, head_dim, bias=False)
        
        self.pos_bias = PositionBias(grid_height, grid_width)
        
        self.rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=self.seq_len, base=10000.0)
        
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3))
            for _ in range(num_hops)
        ])
        self.fusion_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(num_hops)
        ])
        
        self.hop_transforms = nn.ModuleList([
            CastedLinear(head_dim, head_dim, bias=False)
            for _ in range(num_hops)
        ])
        
        self.hop_gate = CastedLinear(hidden_size, num_hops, bias=True)
        self.output_proj = CastedLinear(head_dim, hidden_size, bias=False)
        self.temperature = nn.Parameter(torch.ones(1))
        
    
    def prob_log(self, p):
        p = p.clamp(min=self.eps, max=1.0 - self.eps)
        return torch.log(p / (1 - p + self.eps))
    
    def log_prob(self, logit):
        return torch.sigmoid(logit)

    def mat_exp(self, M, h):
        if h <= 1:
            return M
        result = M
        for _ in range(h - 1):
            result = torch.bmm(result, M)
        return result
    
    def base_attention(self, Q, K):
        scores = torch.bmm(Q, K.transpose(-1, -2)) / self.scale
        scores = scores + self.pos_bias().unsqueeze(0)
        scores = scores / self.temperature.clamp(min=0.1)
        A = F.softmax(scores, dim=-1)
        return A.to(Q.dtype)

    def motif_attention(self, A, hop):
        A_T = A.transpose(-1, -2)
        h = hop + 1
        
        A_seq = self.mat_exp(A, h)
        
        AA_T = torch.bmm(A, A_T)
        A_coattn = self.mat_exp(AA_T, h)
        
        A_T_A = torch.bmm(A_T, A)
        A_split = self.mat_exp(A_T_A, h)
        
        A_seq = A_seq / (A_seq.sum(dim=-1, keepdim=True) + self.eps)
        A_coattn = A_coattn / (A_coattn.sum(dim=-1, keepdim=True) + self.eps)
        A_split = A_split / (A_split.sum(dim=-1, keepdim=True) + self.eps)

        return A_seq, A_coattn, A_split

    
    def fuse_motifs(self, A_seq, A_coattn, A_split, hop):
        logit_seq = self.prob_log(A_seq)
        logit_coattn = self.prob_log(A_coattn)
        logit_split = self.prob_log(A_split)
        
        w = self.fusion_weights[hop]
        b = self.fusion_bias[hop]
        
        fused_logit = b + w[0] * logit_seq + w[1] * logit_coattn + w[2] * logit_split
        
        fused_prob = self.log_prob(fused_logit)
        
        fused_attn = fused_prob / (fused_prob.sum(dim=-1, keepdim=True) + self.eps)
        
        return fused_attn
    
    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        cos, sin = self.rope()
        Q, K = apply_rotary_pos_emb(Q.unsqueeze(2), K.unsqueeze(2), cos, sin)
        Q, K = Q.squeeze(2), K.squeeze(2)
        
        A = self.base_attention(Q, K)
        
        hop_outputs = []
        for hop in range(self.num_hops):
            hop_v = self.hop_transforms[hop](V)
            
            A_seq, A_coattn, A_split = self.motif_attention(A, hop=hop)
            
            fused_attn = self.fuse_motifs(A_seq, A_coattn, A_split, hop)
            
            hop_out = torch.bmm(fused_attn, hop_v)
            hop_outputs.append(hop_out)
        
        hop_weights = F.softmax(self.hop_gate(x), dim=-1)
        stacked = torch.stack(hop_outputs, dim=-1)
        combined = (stacked * hop_weights.unsqueeze(2)).sum(dim=-1)
        
        return self.output_proj(combined)