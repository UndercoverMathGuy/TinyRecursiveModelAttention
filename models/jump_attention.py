import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import trunc_normal_init_
from models.layers import CastedLinear, RotaryEmbedding, apply_rotary_pos_emb

class JumpAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        grid_height: int,
        grid_width: int,
        max_seq_len: int = 256  # Safety limit
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.seq_len = grid_height * grid_width
        self.max_seq_len = max_seq_len
        self.scale = math.pow(head_dim, -0.5)
        
        if self.seq_len > max_seq_len:
            print(f"WARNING: JumpAttention seq_len={self.seq_len} > max={max_seq_len}, will be slow!")
        
        # Projections
        self.q_proj = CastedLinear(hidden_size, head_dim, bias=False)
        self.k_proj = CastedLinear(hidden_size, head_dim, bias=False)
        self.v_proj = CastedLinear(hidden_size, head_dim, bias=False)
        
        # Jump projection (for the second key in triadic)
        self.k2_proj = CastedLinear(hidden_size, head_dim, bias=False)
        
        # Triadic combination weights
        self.triadic_weight = nn.Parameter(torch.ones(3) / 3)  # [direct, jump1, jump2]
        
        # Output
        self.output_proj = CastedLinear(head_dim, hidden_size, bias=False)
        
        # Position bias
        self.pos_bias = PositionBias(grid_height, grid_width)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        Q = self.q_proj(hidden_states)  # [B, L, d]
        K = self.k_proj(hidden_states)  # [B, L, d]
        K2 = self.k2_proj(hidden_states)  # [B, L, d]
        V = self.v_proj(hidden_states)  # [B, L, d]
        
        # Standard attention: A[i,j] = softmax(Q[i] · K[j])
        std_scores = torch.bmm(Q, K.transpose(-1, -2)) * self.scale
        std_scores = std_scores + self.pos_bias().unsqueeze(0)
        A_std = F.softmax(std_scores, dim=-1)
        out_std = torch.bmm(A_std, V)
        
        # Jump attention: For each i, aggregate over all (j,k) pairs
        # T[i,j,k] = Q[i] · (K[j] + K2[k]) / 2
        # This is O(L³) in the explicit form
        
        # Efficient approximation: A_jump1[i,j] based on second-order key similarity
        # K[j] · K2[k] averaged over k
        K_K2_sim = torch.bmm(K, K2.transpose(-1, -2))  # [B, L, L]
        K_K2_weights = F.softmax(K_K2_sim * self.scale, dim=-1)
        K_aggregated = torch.bmm(K_K2_weights, K2)  # [B, L, d] - K enhanced by K2
        
        jump_scores = torch.bmm(Q, K_aggregated.transpose(-1, -2)) * self.scale
        jump_scores = jump_scores + self.pos_bias().unsqueeze(0)
        A_jump = F.softmax(jump_scores, dim=-1)
        out_jump1 = torch.bmm(A_jump, V)
        
        # Reverse jump
        K2_K_sim = torch.bmm(K2, K.transpose(-1, -2))
        K2_K_weights = F.softmax(K2_K_sim * self.scale, dim=-1)
        K2_aggregated = torch.bmm(K2_K_weights, K)
        
        jump_scores2 = torch.bmm(Q, K2_aggregated.transpose(-1, -2)) * self.scale
        jump_scores2 = jump_scores2 + self.pos_bias().unsqueeze(0)
        A_jump2 = F.softmax(jump_scores2, dim=-1)
        out_jump2 = torch.bmm(A_jump2, V)
        
        # Combine
        w = F.softmax(self.triadic_weight, dim=0)
        output = w[0] * out_std + w[1] * out_jump1 + w[2] * out_jump2
        
        return self.output_proj(output)