import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from data_sudoku.data_aug import DataTensor, TensorAugmented

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
    
class MergeInputs(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 512
        self.weight_x = nn.Linear(dim, dim, bias=False)
        self.weight_y = nn.Linear(dim, dim, bias=False)
        self.weight_h = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, y, h):
        x = self.weight_x(x)
        y = self.weight_y(y)
        h = self.weight_h(h)
        merged = x + y + h
        merged = self.norm(merged)
        return merged
    
class SudokuModel(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.merge_inputs = MergeInputs()
        self.mlp_block = MLP_Block(dropout=dropout)
        self.input_embed = InputQuestionEmbed()
        self.output_embed = OutputEmbed_Load()

    def forward(self, x, n_r):
        x_embed = self.input_embed(x) # [B, 81, 512]
        h = torch.zeros_like(x_embed) # [B, 81, 512]
        y_state = torch.zeros_like(x_embed) # [B, 81, 512]
        y_logits = x_embed.new_zeros((x_embed.shape[0], 81, 9)) # [B, 81, 9]

        for _ in range(n_r):
            mlp_inputs = self.merge_inputs(x_embed, y_state, h)
            h, y_logits = self.mlp_block(mlp_inputs)
            y_state = self.output_embed(y_logits)

        return y_logits
    
def train(epochs, permutations):
    dataset = pd.read_parquet('sudoku/train.parquet')
    Tensor = DataTensor(dataset)
    AugTensor = TensorAugmented(Tensor)
    AugTensor.permute_loop(permutations)
    train_tensor = AugTensor.to_tensors(device='mps', col_question=True)
    test_tensor = AugTensor.to_tensors(device='mps', col_question=False)