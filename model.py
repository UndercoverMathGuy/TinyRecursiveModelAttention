import torch.nn as nn
import torch.nn.functional as F
from data.data_aug import DataTensor, TensorAugmented

class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0, out_dim=None):
        super().__init__()
        self.out_dim = in_dim if out_dim is None else out_dim
        self.fc_in = nn.Linear(in_dim, 2 * hidden_dim)   # -> [*, 2H]
        self.fc_out = nn.Linear(hidden_dim, self.out_dim) # -> [*, out_dim]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        u, v = self.fc_in(x).chunk(2, dim=-1)
        x = v * F.silu(u)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

class InputQuestionEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 10
        embed_dim = 512
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)
    
class OutputEmbed_Load(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 9
        output_dim = 512
        self.linear = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class OutputEmbed_Y(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 512
        output_dim = 9
        self.linear = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class MLP_Mixer(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.norm_token = nn.LayerNorm(512)
        self.norm_channel = nn.LayerNorm(512)
        self.token_mlp = nn.Sequential(
            SwiGLU(in_dim=81, out_dim=81, dropout=dropout, hidden_dim=324)
        )
        self.channel_mlp = nn.Sequential(
            SwiGLU(in_dim=512, out_dim=512, dropout=dropout, hidden_dim=2048)
        )

    def token_mix(self, x):
        x_orig = x
        x = self.norm_token(x)
        x = x.permute(0,2,1) # [B, L, D] -> [B, D, L]
        x = self.token_mlp(x)
        x = x.permute(0,2,1) # [B, D, L] -> [B, L, D]
        x = x + x_orig
        return x

    def channel_mix(self, x):
        x_orig = x
        x = self.norm_channel(x)
        x = self.channel_mlp(x)
        x = x + x_orig
        return x
    
    def forward(self, x):
        x = self.token_mix(x)
        x = self.channel_mix(x)
        return x

class MLP_Block(nn.Module): # Fusion of x, y_t, h_t happens before - x is projected sum
    def __init__(self, dropout=0.0):
        super().__init__()
        self.output_embed = OutputEmbed_Y()
        self.mlp_mixer = MLP_Mixer(dropout=dropout)
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        h = self.mlp_mixer(x)
        y = self.output_embed(self.norm(h))
        return h, y