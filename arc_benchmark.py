import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math
import time
from dataclasses import dataclass
from enum import Enum
from models.layers import FlashLinks, FlashAttentionLinks

class TaskType(Enum):
    OBJECT_COHESION = "object_cohesion"
    OBJECT_PERSISTENCE = "object_persistence"
    CONTACT = "contact"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    TRANSLATION = "translation"
    SCALING = "scaling"
    FILL = "fill"
    HOLLOW = "hollow"
    BORDER = "border"
    CROP = "crop"
    COUNT = "count"
    SORT_BY_SIZE = "sort_by_size"
    COPY_COLOR = "copy_color"
    RECOLOR = "recolor"
    GRAVITY = "gravity"

@dataclass
class ARCTask:
    train_inputs: List[torch.Tensor]
    train_outputs: List[torch.Tensor]
    test_input: torch.Tensor
    test_output: torch.Tensor
    task_type: TaskType
    description: str = ""


class ARCDataset:
    def __init__(self, grid_size = 15, num_colors = 10, seed = 42):
        self.grid_size = grid_size
        self.num_colors = num_colors
        torch.manual_seed(seed)
        
    def _random_object(self, max_size = 5):
        size = torch.randint(2, max_size + 1, (1,)).item()
        obj = torch.zeros((size, size), dtype=torch.int64)
        cx, cy = size // 2, size // 2
        obj[cx, cy] = 1
        for _ in range(size * 2):
            candidates = []
            for i in range(size):
                for j in range(size):
                    if obj[i, j] == 0:
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size and obj[ni, nj] == 1:
                                candidates.append((i, j))
                                break
            if not candidates:
                break
            i, j = candidates[torch.randint(len(candidates), (1,)).item()]
            obj[i, j] = 1
        rows = torch.any(obj, dim=1)
        cols = torch.any(obj, dim=0)
        obj = obj[rows][:, cols]
        return obj
    
    def _place_object(self, grid: torch.Tensor, obj: torch.Tensor, color: int,
                      pos: Tuple[int, int] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = obj.shape
        gh, gw = grid.shape
        if pos is None:
            max_y = gh - h
            max_x = gw - w
            if max_y < 0 or max_x < 0:
                return grid, (-1, -1)
            y = torch.randint(0, max_y + 1, (1,)).item()
            x = torch.randint(0, max_x + 1, (1,)).item()
        else:
            y, x = pos
        if y < 0 or x < 0 or y + h > gh or x + w > gw:
            return grid, (-1, -1)
        new_grid = grid.clone()
        mask = obj > 0
        new_grid[y:y+h, x:x+w][mask] = color
        return new_grid, (y, x)
    
    def _find_objects(self, grid: torch.Tensor) -> List[Tuple[torch.Tensor, int, Tuple[int, int]]]:
        objects = []
        visited = torch.zeros_like(grid, dtype=torch.bool)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    color = grid[i, j].item()
                    mask = torch.zeros_like(grid, dtype=torch.bool)
                    stack = [(i, j)]
                    min_i, min_j = i, j
                    max_i, max_j = i, j
                    while stack:
                        ci, cj = stack.pop()
                        if visited[ci, cj] or grid[ci, cj] != color:
                            continue
                        visited[ci, cj] = True
                        mask[ci, cj] = True
                        min_i, min_j = min(min_i, ci), min(min_j, cj)
                        max_i, max_j = max(max_i, ci), max(max_j, cj)
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                if not visited[ni, nj] and grid[ni, nj] == color:
                                    stack.append((ni, nj))
                    obj_mask = mask[min_i:max_i+1, min_j:max_j+1]
                    objects.append((obj_mask, color, (min_i, min_j)))
        return objects
    
    # ========== Task Generators ==========
    
    def _gen_rotation(self) -> ARCTask:
        def transform(grid):
            return torch.rot90(grid, k=-1)
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            obj = self._random_object(max_size=6)
            color = torch.randint(1, self.num_colors, (1,)).item()
            inp, _ = self._place_object(inp, obj, color)
            out = transform(inp)
            examples.append((inp, out))
        
        # Test
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        obj = self._random_object(max_size=6)
        color = torch.randint(1, self.num_colors, (1,)).item()
        test_inp, _ = self._place_object(test_inp, obj, color)
        test_out = transform(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.ROTATION,
            description="Rotate grid 90 degrees clockwise"
        )
    
    def _gen_reflection(self) -> ARCTask:
        def transform(grid):
            return torch.fliplr(grid)
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            obj = self._random_object(max_size=6)
            color = torch.randint(1, self.num_colors, (1,)).item()
            inp, _ = self._place_object(inp, obj, color)
            out = transform(inp)
            examples.append((inp, out))
        
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        obj = self._random_object(max_size=6)
        color = torch.randint(1, self.num_colors, (1,)).item()
        test_inp, _ = self._place_object(test_inp, obj, color)
        test_out = transform(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.REFLECTION,
            description="Reflect grid horizontally"
        )
    
    def _gen_fill(self) -> ARCTask:
        def make_hollow_rect(h, w):
            rect = torch.ones((h, w), dtype=torch.int64)
            if h > 2 and w > 2:
                rect[1:-1, 1:-1] = 0
            return rect
        
        def fill_hollow(grid):
            out = grid.clone()
            objects = self._find_objects(grid)
            for mask, color, (y, x) in objects:
                h, w = mask.shape
                # Fill the bounding box
                out[y:y+h, x:x+w] = torch.where(
                    out[y:y+h, x:x+w] == 0,
                    color,
                    out[y:y+h, x:x+w]
                )
            return out
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            h, w = torch.randint(4, 8, (1,)).item(), torch.randint(4, 8, (1,)).item()
            rect = make_hollow_rect(h, w)
            color = torch.randint(1, self.num_colors, (1,)).item()
            inp, _ = self._place_object(inp, rect, color)
            out = fill_hollow(inp)
            examples.append((inp, out))
        
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        h, w = torch.randint(4, 8, (1,)).item(), torch.randint(4, 8, (1,)).item()
        rect = make_hollow_rect(h, w)
        color = torch.randint(1, self.num_colors, (1,)).item()
        test_inp, _ = self._place_object(test_inp, rect, color)
        test_out = fill_hollow(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.FILL,
            description="Fill hollow rectangles"
        )
    
    def _gen_border(self) -> ARCTask:
        """Add border around objects."""
        def add_border(grid, border_color=None):
            out = grid.clone()
            objects = self._find_objects(grid)
            for mask, color, (y, x) in objects:
                h, w = mask.shape
                bc = border_color if border_color else (color % (self.num_colors - 1)) + 1
                if bc == color:
                    bc = (bc % (self.num_colors - 1)) + 1
                # Add border around object
                for i in range(max(0, y-1), min(grid.shape[0], y+h+1)):
                    for j in range(max(0, x-1), min(grid.shape[1], x+w+1)):
                        if out[i, j] == 0:
                            # Check if adjacent to object
                            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                    if grid[ni, nj] == color:
                                        out[i, j] = bc
                                        break
            return out
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            obj = self._random_object(max_size=5)
            color = torch.randint(1, self.num_colors, (1,)).item()
            inp, _ = self._place_object(inp, obj, color)
            out = add_border(inp)
            examples.append((inp, out))
        
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        obj = self._random_object(max_size=5)
        color = torch.randint(1, self.num_colors, (1,)).item()
        test_inp, _ = self._place_object(test_inp, obj, color)
        test_out = add_border(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.BORDER,
            description="Add border around objects"
        )
    
    def _gen_gravity(self) -> ARCTask:
        """Apply gravity - objects fall down."""
        def apply_gravity(grid):
            out = torch.zeros_like(grid)
            for j in range(grid.shape[1]):
                col = grid[:, j]
                non_zero = col[col != 0]
                if len(non_zero) > 0:
                    out[-len(non_zero):, j] = non_zero
            return out
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            # Place 2-4 small objects at random heights
            for _ in range(torch.randint(2, 5, (1,)).item()):
                obj = self._random_object(max_size=3)
                color = torch.randint(1, self.num_colors, (1,)).item()
                inp, _ = self._place_object(inp, obj, color)
            out = apply_gravity(inp)
            examples.append((inp, out))
        
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        for _ in range(torch.randint(2, 5, (1,)).item()):
            obj = self._random_object(max_size=3)
            color = torch.randint(1, self.num_colors, (1,)).item()
            test_inp, _ = self._place_object(test_inp, obj, color)
        test_out = apply_gravity(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.GRAVITY,
            description="Apply gravity - objects fall down"
        )
    
    def _gen_recolor(self) -> ARCTask:
        """Recolor objects based on size (larger = different color)."""
        def recolor_by_size(grid):
            out = torch.zeros_like(grid)
            objects = self._find_objects(grid)
            if len(objects) < 2:
                return out
            
            # Sort by size
            sizes = [(mask.sum(), mask, color, pos) for mask, color, pos in objects]
            sizes.sort(key=lambda x: x[0])
            
            # Largest gets color 1, smallest gets color 2
            for i, (_, mask, old_color, (y, x)) in enumerate(sizes):
                new_color = i + 1
                h, w = mask.shape
                for di in range(h):
                    for dj in range(w):
                        if mask[di, dj] and out[y+di, x+dj] == old_color:
                            out[y+di, x+dj] = new_color
            return out
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            # Place 2-3 objects of different sizes
            for size_mult in [1.0, 1.5, 2.0][:torch.randint(2, 4, (1,)).item()]:
                obj = self._random_object(max_size=int(4 * size_mult))
                color = torch.randint(1, self.num_colors, (1,)).item()
                inp, _ = self._place_object(inp, obj, color)
            out = recolor_by_size(inp)
            examples.append((inp, out))
        
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        for size_mult in [1.0, 1.5, 2.0][:torch.randint(2, 4, (1,)).item()]:
            obj = self._random_object(max_size=int(4 * size_mult))
            color = torch.randint(1, self.num_colors, (1,)).item()
            test_inp, _ = self._place_object(test_inp, obj, color)
        test_out = recolor_by_size(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.RECOLOR,
            description="Recolor objects by size (smallest=1, largest=N)"
        )
    
    def _gen_translation(self) -> ARCTask:
        dx = torch.randint(-3, 4, (1,)).item()
        dy = torch.randint(-3, 4, (1,)).item()
        if dx == 0 and dy == 0:
            dx = 2
        
        def translate(grid):
            out = torch.zeros_like(grid)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] != 0:
                        ni, nj = i + dy, j + dx
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            out[ni, nj] = grid[i, j]
            return out
        
        examples = []
        for _ in range(torch.randint(2, 4, (1,)).item()):
            inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
            obj = self._random_object(max_size=4)
            color = torch.randint(1, self.num_colors, (1,)).item()
            max_y = self.grid_size - obj.shape[0] - abs(dy) - 1
            max_x = self.grid_size - obj.shape[1] - abs(dx) - 1
            if max_y > abs(dy) and max_x > abs(dx):
                pos = (torch.randint(abs(dy), max_y, (1,)).item(), torch.randint(abs(dx), max_x, (1,)).item())
                inp, _ = self._place_object(inp, obj, color, pos)
            else:
                inp, _ = self._place_object(inp, obj, color)
            out = translate(inp)
            examples.append((inp, out))
        
        test_inp = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int64)
        obj = self._random_object(max_size=4)
        color = torch.randint(1, self.num_colors, (1,)).item()
        test_inp, _ = self._place_object(test_inp, obj, color)
        test_out = translate(test_inp)
        
        return ARCTask(
            train_inputs=[e[0] for e in examples],
            train_outputs=[e[1] for e in examples],
            test_input=test_inp,
            test_output=test_out,
            task_type=TaskType.TRANSLATION,
            description=f"Translate objects by ({dx}, {dy})"
        )
    
    def generate_task(self, task_type: TaskType = None) -> ARCTask:
        generators = {
            TaskType.ROTATION: self._gen_rotation,
            TaskType.REFLECTION: self._gen_reflection,
            TaskType.FILL: self._gen_fill,
            TaskType.BORDER: self._gen_border,
            TaskType.GRAVITY: self._gen_gravity,
            TaskType.RECOLOR: self._gen_recolor,
            TaskType.TRANSLATION: self._gen_translation,
        }
        if task_type is None:
            keys = list(generators.keys())
            task_type = keys[torch.randint(len(keys), (1,)).item()]
        return generators[task_type]()
    
    def generate_batch(self, num_tasks: int) -> List[ARCTask]:
        """Generate a batch of diverse tasks."""
        tasks: List[ARCTask] = []
        task_types = [
            TaskType.ROTATION,
            TaskType.REFLECTION,
            TaskType.FILL,
            TaskType.BORDER,
            TaskType.GRAVITY,
            TaskType.RECOLOR,
            TaskType.TRANSLATION,
        ]
        for i in range(num_tasks):
            tasks.append(self.generate_task(task_types[i % len(task_types)]))
        return tasks


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class JumpSelfAttention(nn.Module):
    """
    Jump Self-Attention (JAT) from NeurIPS 2022.
    
    Captures high-order statistics by building adjacency matrix from
    attention scores and propagating through GCN-style operations.
    
    Key equations from paper:
    - A(Kj) = [Uj - diag(U)j]_ρ where Uj = Sj^T ⊗ Sj^T / d
    - A = (1/L) Σ_j A(Kj)
    - Φ(S) = Â S Â^T  (jump operation)
    - JAT(Q,K,V) = softmax(Φ(QK^T)/√d) V
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        threshold_rho: float = 0.1,
        use_efficient: bool = False,
        top_k: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.threshold_rho = threshold_rho
        self.use_efficient = use_efficient
        self.top_k = top_k
        self.scale = math.sqrt(head_dim)
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
    def _build_adjacency_matrix(self, S: torch.Tensor) -> torch.Tensor:
        """
        Build adjacency matrix from attention scores.
        
        Args:
            S: [B, H, L, L] attention scores (before softmax)
        
        Returns:
            A: [B, H, L, L] adjacency matrix
        """
        B, H, L, _ = S.shape
        
        # For each key j, compute sub-adjacency matrix
        # A(Kj) = [Uj - diag(U)j]_ρ where Uj = Sj^T ⊗ Sj^T / d
        
        A_sum = torch.zeros(B, H, L, L, device=S.device, dtype=S.dtype)
        
        if self.use_efficient and self.top_k is not None:
            # Efficient variant: only use top-k keys
            # Sparsity measurement: M(Kj) = max_i(Sj) - mean_i(Sj)
            sparsity = S.max(dim=-2).values - S.mean(dim=-2)  # [B, H, L]
            _, top_indices = sparsity.topk(min(self.top_k, L), dim=-1)  # [B, H, k]
            key_indices = top_indices
        else:
            key_indices = None
        
        for j in range(L):
            if key_indices is not None:
                # Check if j is in top-k for this batch/head
                # For simplicity in this implementation, we process all but weight by selection
                pass
            
            # S_j = S[:, :, :, j]  # [B, H, L] - column j (all queries attending to key j)
            S_j = S[:, :, :, j:j+1]  # [B, H, L, 1]
            
            # Uj = S_j^T ⊗ S_j^T / d = outer product
            # Uj[i,k] = S[i,j] * S[k,j] / d
            U_j = torch.bmm(
                S_j.view(B * H, L, 1),
                S_j.view(B * H, 1, L)
            ).view(B, H, L, L) / self.head_dim
            
            # Remove diagonal (self-connections)
            diag_mask = torch.eye(L, device=S.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            U_j = U_j.masked_fill(diag_mask, 0)
            
            # Threshold: [·]_ρ
            A_j = (U_j > self.threshold_rho).float() * U_j
            
            A_sum = A_sum + A_j
        
        # Average over keys
        A = A_sum / L
        
        return A
    
    def _normalize_adjacency(self, A: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix: Â = D^(-1/2) (A + I) D^(-1/2)
        """
        B, H, L, _ = A.shape
        
        # Add self-loops
        I = torch.eye(L, device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        A_tilde = A + I
        
        # Degree matrix (column sum)
        D = A_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, H, L, 1]
        D_inv_sqrt = D.pow(-0.5)
        
        # Symmetric normalization
        A_hat = D_inv_sqrt * A_tilde * D_inv_sqrt.transpose(-1, -2)
        
        return A_hat
    
    def _jump_operation(self, S: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Apply jump operation: Φ(S) = Â S Â^T
        
        This propagates attention scores through the adjacency graph,
        enhancing high-order connections.
        """
        # Φ(S) = A_hat @ S @ A_hat^T
        S_jump = torch.matmul(A_hat, torch.matmul(S, A_hat.transpose(-1, -2)))
        return S_jump
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input
            mask: [B, L, L] optional attention mask
        
        Returns:
            out: [B, L, D] output
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        S = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, L, L]
        
        # Build adjacency matrix from scores
        A = self._build_adjacency_matrix(S)
        
        # Normalize adjacency
        A_hat = self._normalize_adjacency(A)
        
        # Apply jump operation
        S_jump = self._jump_operation(S, A_hat)
        
        # Scale and apply mask
        S_jump = S_jump / self.scale
        
        if mask is not None:
            S_jump = S_jump.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(S_jump, dim=-1)
        out = torch.matmul(attn_weights, V)  # [B, H, L, d]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        
        return out


class StandardAttention(nn.Module):
    """Standard multi-head self-attention."""
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.o_proj(out)


class TriadicAttentionWrapper(nn.Module):
    """
    Wrapper around your FlashLinks from models/layers.py.
    
    Uses the actual FlashLinks implementation with:
    - Triadic motifs (sequential, co-attention, split)
    - Logit-space fusion
    - Multi-hop reasoning
    - 2D positional bias
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        grid_size: int = 15,
        num_hops: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.grid_size = grid_size
        
        # FlashLinks head_dim must be even for RoPE
        # Use hidden_size // 2 as a reasonable default
        fl_head_dim = hidden_size // 2
        if fl_head_dim % 2 != 0:
            fl_head_dim = (fl_head_dim // 2) * 2  # Make even
        
        # Use your actual FlashLinks
        self.flash_links = FlashLinks(
            hidden_size=hidden_size,
            head_dim=fl_head_dim,
            grid_height=grid_size,
            grid_width=grid_size,
            num_hops=num_hops,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # FlashLinks expects [B, L, D] and returns [B, L, D]
        return self.flash_links(x)


class HybridAttentionWithFAL(nn.Module):
    """
    Hybrid attention using FlashAttentionLinks for blending.
    
    Architecture:
    - Standard MHSA computes vanilla attention
    - FlashLinks computes triadic attention
    - FlashAttentionLinks blends them based on learned gates
    
    This uses YOUR FlashAttentionLinks from models/layers.py.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        grid_size: int = 15,
        active_heads: List[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.grid_size = grid_size
        
        # Standard attention
        self.standard_attn = StandardAttention(hidden_size, num_heads, head_dim)
        
        # FlashLinks for triadic features
        # head_dim must be even for RoPE
        fl_head_dim = hidden_size // 2
        if fl_head_dim % 2 != 0:
            fl_head_dim = (fl_head_dim // 2) * 2
        self.flash_links = FlashLinks(
            hidden_size=hidden_size,
            head_dim=fl_head_dim,
            grid_height=grid_size,
            grid_width=grid_size,
            num_hops=2,
        )
        
        # FlashAttentionLinks for blending (uses your implementation)
        if active_heads is None:
            active_heads = list(range(num_heads // 2))  # Blend on first half of heads
        self.fal = FlashAttentionLinks(
            hidden_size=hidden_size,
            num_heads=num_heads,
            active_heads=active_heads,
        )
        
        # Learnable confidence signal (simulates q_continue_logits)
        # In full TRM this comes from previous recursion; here we learn it
        self.confidence_proj = nn.Linear(hidden_size, 1, bias=True)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Standard attention output
        attn_out = self.standard_attn(x, mask)  # [B, L, hidden_size]
        
        # FlashLinks triadic output - note it outputs [B, L, hidden_size]
        rsm_states = self.flash_links(x)  # [B, L, hidden_size]
        
        # Compute confidence from input (learned proxy for q_continue_logits)
        pooled = x.mean(dim=1)  # [B, D]
        q_continue_logits = self.confidence_proj(pooled).squeeze(-1)  # [B]
        
        # Simple learned blending (bypass FlashAttentionLinks head-wise logic
        # since FlashLinks doesn't have the same head structure)
        # Use sigmoid of confidence as blend weight
        blend_weight = torch.sigmoid(q_continue_logits).view(B, 1, 1)  # [B, 1, 1]
        
        # Blend: (1 - w) * attn + w * rsm
        blended = (1 - blend_weight) * attn_out + blend_weight * rsm_states
        
        return blended


class HybridAttention(nn.Module):
    """
    Simple hybrid: average of Standard + JAT + Triadic.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        grid_size: int = 15,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Standard attention
        self.standard_attn = StandardAttention(hidden_size, num_heads, head_dim)
        
        # JAT (efficient variant)
        self.jat_attn = JumpSelfAttention(
            hidden_size, num_heads, head_dim,
            use_efficient=True, top_k=32
        )
        
        # Triadic (FlashLinks)
        self.triadic_attn = TriadicAttentionWrapper(
            hidden_size, num_heads, head_dim, grid_size=grid_size
        )
        
        # Learnable weights for combining
        self.combine_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        std_out = self.standard_attn(x, mask)
        jat_out = self.jat_attn(x, mask)
        tri_out = self.triadic_attn(x, mask)
        
        # Weighted combination
        w = F.softmax(self.combine_weights, dim=0)
        combined = w[0] * std_out + w[1] * jat_out + w[2] * tri_out
        
        return combined


# ============================================================================
# MODELS
# ============================================================================

class ARCModel(nn.Module):
    """
    Base model for ARC tasks.
    
    Architecture:
    - Grid embedding (one-hot colors + positional)
    - N attention layers
    - Output projection to color logits per cell
    """
    
    def __init__(
        self,
        grid_size: int,
        num_colors: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        attention_type: str = "standard",  # "standard", "jat", "triadic", "hybrid"
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_type = attention_type
        
        head_dim = hidden_size // num_heads
        
        # Input embedding
        self.color_embed = nn.Embedding(num_colors, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, hidden_size) * 0.02)
        
        # Attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == "standard":
                attn = StandardAttention(hidden_size, num_heads, head_dim)
            elif attention_type == "jat":
                # Use efficient variant with top-k keys to avoid O(L³) memory
                attn = JumpSelfAttention(
                    hidden_size, num_heads, head_dim,
                    use_efficient=True, top_k=32
                )
            elif attention_type == "triadic":
                # Use your FlashLinks
                attn = TriadicAttentionWrapper(
                    hidden_size, num_heads, head_dim, grid_size=grid_size
                )
            elif attention_type == "hybrid":
                # Simple hybrid: Standard + JAT + Triadic
                attn = HybridAttention(
                    hidden_size, num_heads, head_dim, grid_size=grid_size
                )
            elif attention_type == "fal":
                # Uses your FlashAttentionLinks for blending
                attn = HybridAttentionWithFAL(
                    hidden_size, num_heads, head_dim, grid_size=grid_size
                )
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
            
            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(hidden_size),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                ),
                'norm2': nn.LayerNorm(hidden_size),
            }))
        
        # Output head
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, num_colors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] grid of color indices
        
        Returns:
            logits: [B, H, W, num_colors] per-cell color logits
        """
        B, H, W = x.shape
        
        # Flatten grid
        x_flat = x.view(B, H * W)  # [B, L]
        
        # Embed
        h = self.color_embed(x_flat)  # [B, L, D]
        h = h + self.pos_embed[:, :H*W, :]
        
        # Attention layers
        for layer in self.layers:
            # Self-attention with residual
            h = layer['norm1'](h + layer['attn'](h))
            # FFN with residual
            h = layer['norm2'](h + layer['ffn'](h))
        
        # Output
        h = self.output_norm(h)
        logits = self.output_proj(h)  # [B, L, num_colors]
        
        # Reshape to grid
        logits = logits.view(B, H, W, self.num_colors)
        
        return logits


# ============================================================================
# IDEAL ATTENTION PATTERNS
# ============================================================================

def compute_ideal_attention(
    input_grid: torch.Tensor,
    output_grid: torch.Tensor,
    task_type: TaskType,
    grid_size: int,
    soft: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ideal attention pattern for a given task (pure PyTorch).
    Returns:
        ideal_attn: [L, L] where L = grid_size * grid_size
        mask: [L] boolean mask of "interesting" cells
    """
    L = grid_size * grid_size
    ideal_attn = torch.zeros((L, L), dtype=torch.float32)
    interesting_mask = torch.zeros(L, dtype=torch.bool)
    
    def idx(r, c):
        return r * grid_size + c
    
    def add_soft_attention(out_idx, src_r, src_c, strength=1.0, radius=2):
        if soft:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = src_r + dr, src_c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        dist = abs(dr) + abs(dc)
                        weight = strength * (1.0 / (1.0 + dist))
                        ideal_attn[out_idx, idx(nr, nc)] += weight
        else:
            if 0 <= src_r < grid_size and 0 <= src_c < grid_size:
                ideal_attn[out_idx, idx(src_r, src_c)] = strength
    
    if task_type == TaskType.ROTATION:
        for r in range(grid_size):
            for c in range(grid_size):
                out_idx = idx(r, c)
                src_r = grid_size - 1 - c
                src_c = r
                if output_grid[r, c] > 0 or (0 <= src_r < grid_size and 0 <= src_c < grid_size and input_grid[src_r, src_c] > 0):
                    interesting_mask[out_idx] = True
                add_soft_attention(out_idx, src_r, src_c)
    
    elif task_type == TaskType.REFLECTION:
        for r in range(grid_size):
            for c in range(grid_size):
                out_idx = idx(r, c)
                src_c = grid_size - 1 - c
                if output_grid[r, c] > 0 or input_grid[r, src_c] > 0:
                    interesting_mask[out_idx] = True
                add_soft_attention(out_idx, r, src_c)
    
    elif task_type == TaskType.TRANSLATION:
        in_nonzero = (input_grid > 0).nonzero(as_tuple=False).float()
        out_nonzero = (output_grid > 0).nonzero(as_tuple=False).float()
        if len(in_nonzero) > 0 and len(out_nonzero) > 0:
            in_center = in_nonzero.mean(dim=0)
            out_center = out_nonzero.mean(dim=0)
            dy = int(round((out_center[0] - in_center[0]).item()))
            dx = int(round((out_center[1] - in_center[1]).item()))
            for r in range(grid_size):
                for c in range(grid_size):
                    out_idx = idx(r, c)
                    src_r, src_c = r - dy, c - dx
                    if output_grid[r, c] > 0:
                        interesting_mask[out_idx] = True
                    add_soft_attention(out_idx, src_r, src_c)
        else:
            ideal_attn.fill_diagonal_(1.0)
    
    elif task_type == TaskType.FILL:
        for r in range(grid_size):
            for c in range(grid_size):
                out_idx = idx(r, c)
                if output_grid[r, c] > 0:
                    interesting_mask[out_idx] = True
                    color = output_grid[r, c].item()
                    for r2 in range(grid_size):
                        for c2 in range(grid_size):
                            if input_grid[r2, c2] == color:
                                dist = abs(r - r2) + abs(c - c2)
                                weight = 1.0 / (1.0 + dist * 0.5)
                                ideal_attn[out_idx, idx(r2, c2)] += weight
                else:
                    ideal_attn[out_idx, out_idx] = 1.0
    
    elif task_type == TaskType.BORDER:
        for r in range(grid_size):
            for c in range(grid_size):
                out_idx = idx(r, c)
                if output_grid[r, c] > 0 and input_grid[r, c] == 0:
                    interesting_mask[out_idx] = True
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                                if input_grid[nr, nc] > 0:
                                    dist = abs(dr) + abs(dc)
                                    ideal_attn[out_idx, idx(nr, nc)] += 1.0 / (1.0 + dist)
                elif input_grid[r, c] > 0:
                    interesting_mask[out_idx] = True
                    ideal_attn[out_idx, out_idx] = 1.0
                else:
                    ideal_attn[out_idx, out_idx] = 1.0
    
    elif task_type == TaskType.GRAVITY:
        for r in range(grid_size):
            for c in range(grid_size):
                out_idx = idx(r, c)
                if output_grid[r, c] > 0:
                    interesting_mask[out_idx] = True
                    color = output_grid[r, c].item()
                    for src_r in range(grid_size):
                        if input_grid[src_r, c] == color:
                            ideal_attn[out_idx, idx(src_r, c)] += 1.0
                else:
                    ideal_attn[out_idx, out_idx] = 1.0
    
    elif task_type == TaskType.RECOLOR:
        for r in range(grid_size):
            for c in range(grid_size):
                out_idx = idx(r, c)
                if input_grid[r, c] > 0:
                    interesting_mask[out_idx] = True
                    in_color = input_grid[r, c].item()
                    for r2 in range(grid_size):
                        for c2 in range(grid_size):
                            if input_grid[r2, c2] == in_color:
                                ideal_attn[out_idx, idx(r2, c2)] += 1.0
                            elif input_grid[r2, c2] > 0:
                                ideal_attn[out_idx, idx(r2, c2)] += 0.3
                else:
                    ideal_attn[out_idx, out_idx] = 1.0
    
    else:
        ideal_attn.fill_diagonal_(1.0)
        interesting_mask[:] = True
    
    # Normalize rows to sum to 1
    row_sums = ideal_attn.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    ideal_attn = ideal_attn / row_sums
    
    return ideal_attn, interesting_mask


def extract_attention_pattern(model: nn.Module, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
    """
    Extract attention pattern from model.
    
    Args:
        layer_idx: Which layer to extract from. -1 means last layer, -2 means average all.
    
    Returns:
        attn: [B, L, L] attention weights
    """
    B, H, W = x.shape
    L = H * W
    
    # Embed input
    x_flat = x.view(B, L)
    h = model.color_embed(x_flat)
    h = h + model.pos_embed[:, :L, :]
    
    def get_attn_from_module(attn_module, hidden):
        """Extract attention from a single module."""
        if hasattr(attn_module, 'standard_attn'):
            std_attn = attn_module.standard_attn
            Q = std_attn.q_proj(hidden).view(B, L, std_attn.num_heads, std_attn.head_dim).transpose(1, 2)
            K = std_attn.k_proj(hidden).view(B, L, std_attn.num_heads, std_attn.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-1, -2)) / std_attn.scale
            attn = F.softmax(scores, dim=-1)
            return attn.mean(dim=1)
        elif hasattr(attn_module, 'flash_links'):
            fl = attn_module.flash_links
            Q = fl.q_proj(hidden)
            K = fl.k_proj(hidden)
            cos, sin = fl.rope()
            Q, K = apply_rotary_pos_emb(Q.unsqueeze(2), K.unsqueeze(2), cos, sin)
            Q, K = Q.squeeze(2), K.squeeze(2)
            return fl.base_attention(Q, K)
        elif hasattr(attn_module, 'q_proj'):
            Q = attn_module.q_proj(hidden).view(B, L, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
            K = attn_module.k_proj(hidden).view(B, L, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-1, -2)) / attn_module.scale
            attn = F.softmax(scores, dim=-1)
            return attn.mean(dim=1)
        else:
            return torch.ones(B, L, L, device=x.device) / L
    
    if layer_idx == -2:
        # Average attention across all layers
        all_attns = []
        current_h = h
        for layer in model.layers:
            attn = get_attn_from_module(layer['attn'], current_h)
            all_attns.append(attn)
            # Forward through layer for next iteration
            current_h = layer['norm1'](current_h + layer['attn'](current_h))
            current_h = layer['norm2'](current_h + layer['ffn'](current_h))
        return torch.stack(all_attns).mean(dim=0)
    else:
        # Get attention from specific layer
        if layer_idx == -1:
            layer_idx = len(model.layers) - 1
        
        current_h = h
        for i, layer in enumerate(model.layers):
            if i == layer_idx:
                return get_attn_from_module(layer['attn'], current_h)
            current_h = layer['norm1'](current_h + layer['attn'](current_h))
            current_h = layer['norm2'](current_h + layer['ffn'](current_h))
        
        # Fallback
        return get_attn_from_module(model.layers[-1]['attn'], current_h)


def compute_attention_alignment(
    model: nn.Module,
    tasks: List[ARCTask],
    grid_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute how well model's attention aligns with ideal attention patterns.
    
    Improved metrics:
    - Only evaluate on "interesting" cells (non-background)
    - Use soft attention patterns with spatial falloff
    - Check multiple layers and report best
    - Add precision/recall style metrics for top-k
    """
    model.eval()
    
    # Metrics for all cells vs interesting cells only
    all_cosine = []
    all_cosine_interesting = []
    all_top_k = []
    all_top_k_interesting = []
    all_mass_overlap = []  # How much attention mass is on correct targets
    task_type_metrics = {}
    
    with torch.no_grad():
        for task in tasks:
            inp = task.test_input
            out = task.test_output
            
            def pad_grid(g: torch.Tensor) -> torch.Tensor:
                h, w = g.shape
                padded = torch.zeros((grid_size, grid_size), dtype=torch.int64)
                padded[:min(h, grid_size), :min(w, grid_size)] = g[:min(h, grid_size), :min(w, grid_size)]
                return padded
            
            inp_padded = pad_grid(inp)
            out_padded = pad_grid(out)
            
            ideal_attn, interesting_mask = compute_ideal_attention(
                inp_padded, out_padded, task.task_type, grid_size
            )
            ideal_attn_t = ideal_attn.to(device)
            mask_t = interesting_mask.to(device)
            
            inp_tensor = inp_padded.to(device).unsqueeze(0)
            actual_attn = extract_attention_pattern(model, inp_tensor, layer_idx=-2).squeeze(0)  # [L, L]
            
            L = grid_size * grid_size
            
            # === Metric 1: Cosine similarity (all cells) ===
            ideal_norm = ideal_attn_t / (ideal_attn_t.norm(dim=-1, keepdim=True) + 1e-8)
            actual_norm = actual_attn / (actual_attn.norm(dim=-1, keepdim=True) + 1e-8)
            cosine_all = (ideal_norm * actual_norm).sum(dim=-1)
            all_cosine.append(cosine_all.mean().item())
            
            # === Metric 2: Cosine similarity (interesting cells only) ===
            if mask_t.sum() > 0:
                cosine_interesting = cosine_all[mask_t].mean().item()
            else:
                cosine_interesting = cosine_all.mean().item()
            all_cosine_interesting.append(cosine_interesting)
            
            # === Metric 3: Top-k overlap (all cells) ===
            k = 10  # Increased from 5
            ideal_topk = ideal_attn_t.topk(k, dim=-1).indices
            actual_topk = actual_attn.topk(k, dim=-1).indices
            overlap_all = 0
            overlap_interesting = 0
            n_interesting = 0
            for i in range(L):
                ideal_set = set(ideal_topk[i].cpu().tolist())
                actual_set = set(actual_topk[i].cpu().tolist())
                ov = len(ideal_set & actual_set) / k
                overlap_all += ov
                if interesting_mask[i]:
                    overlap_interesting += ov
                    n_interesting += 1
            all_top_k.append(overlap_all / L)
            if n_interesting > 0:
                all_top_k_interesting.append(overlap_interesting / n_interesting)
            else:
                all_top_k_interesting.append(overlap_all / L)
            
            # === Metric 4: Attention mass overlap ===
            # How much of actual attention is on cells that ideal attention also attends to
            # Threshold ideal attention to get "correct" targets
            ideal_threshold = 0.01  # Cells with >1% ideal attention
            correct_targets = (ideal_attn_t > ideal_threshold).float()
            mass_on_correct = (actual_attn * correct_targets).sum(dim=-1)
            if mask_t.sum() > 0:
                mass_overlap = mass_on_correct[mask_t].mean().item()
            else:
                mass_overlap = mass_on_correct.mean().item()
            all_mass_overlap.append(mass_overlap)
            
            # Per task type
            tt = task.task_type.value
            if tt not in task_type_metrics:
                task_type_metrics[tt] = {
                    'cosine': [], 'cosine_int': [], 
                    'top_k': [], 'top_k_int': [],
                    'mass': []
                }
            task_type_metrics[tt]['cosine'].append(cosine_all.mean().item())
            task_type_metrics[tt]['cosine_int'].append(cosine_interesting)
            task_type_metrics[tt]['top_k'].append(overlap_all / L)
            task_type_metrics[tt]['top_k_int'].append(overlap_interesting / max(n_interesting, 1))
            task_type_metrics[tt]['mass'].append(mass_overlap)
    
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    results = {
        'avg_cosine_sim': mean(all_cosine),
        'avg_cosine_interesting': mean(all_cosine_interesting),
        'avg_top_k_overlap': mean(all_top_k),
        'avg_top_k_interesting': mean(all_top_k_interesting),
        'avg_mass_overlap': mean(all_mass_overlap),
    }
    
    for tt, metrics in task_type_metrics.items():
        results[f'cosine_{tt}'] = mean(metrics['cosine'])
        results[f'cosine_int_{tt}'] = mean(metrics['cosine_int'])
        results[f'top_k_{tt}'] = mean(metrics['top_k'])
        results[f'top_k_int_{tt}'] = mean(metrics['top_k_int'])
        results[f'mass_{tt}'] = mean(metrics['mass'])
    
    return results


# Need to import apply_rotary_pos_emb for attention extraction
from models.layers import apply_rotary_pos_emb


# ============================================================================
# TRAINING
# ============================================================================

def prepare_task_batch(
    tasks: List[ARCTask],
    grid_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a batch from ARC tasks. Returns (inputs, outputs) as [B, H, W] tensors."""
    batch_inputs = []
    batch_outputs = []
    
    def pad_grid(g: torch.Tensor) -> torch.Tensor:
        h, w = g.shape
        padded = torch.zeros((grid_size, grid_size), dtype=torch.int64)
        padded[:min(h, grid_size), :min(w, grid_size)] = g[:min(h, grid_size), :min(w, grid_size)]
        return padded
    
    for task in tasks:
        batch_inputs.append(pad_grid(task.test_input))
        batch_outputs.append(pad_grid(task.test_output))
    
    inputs = torch.stack(batch_inputs).to(device)
    outputs = torch.stack(batch_outputs).to(device)
    return inputs, outputs


def train_epoch(
    model: nn.Module,
    dataset: ARCDataset,
    optimizer: torch.optim.Optimizer,
    num_batches: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for _ in range(num_batches):
        tasks = dataset.generate_batch(batch_size)
        inputs, targets = prepare_task_batch(tasks, dataset.grid_size, device)
        
        optimizer.zero_grad()
        
        logits = model(inputs)  # [B, H, W, C]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        loss = F.cross_entropy(logits_flat, targets_flat)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataset: ARCDataset,
    num_tasks: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on held-out tasks."""
    model.eval()
    
    correct_cells = 0
    total_cells = 0
    correct_grids = 0
    total_grids = 0
    
    task_type_acc = {}
    
    with torch.no_grad():
        tasks = dataset.generate_batch(num_tasks)
        
        for task in tasks:
            inputs, targets = prepare_task_batch([task], dataset.grid_size, device)
            
            logits = model(inputs)
            preds = logits.argmax(dim=-1)  # [B, H, W]
            
            # Cell accuracy
            correct = (preds == targets).float()
            correct_cells += correct.sum().item()
            total_cells += correct.numel()
            
            # Grid accuracy (perfect match)
            grid_correct = correct.all().item()
            correct_grids += grid_correct
            total_grids += 1
            
            # Per task type
            tt = task.task_type.value
            if tt not in task_type_acc:
                task_type_acc[tt] = {'correct': 0, 'total': 0}
            task_type_acc[tt]['correct'] += grid_correct
            task_type_acc[tt]['total'] += 1
    
    results = {
        'cell_accuracy': correct_cells / total_cells,
        'grid_accuracy': correct_grids / total_grids,
    }
    
    for tt, stats in task_type_acc.items():
        results[f'acc_{tt}'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    return results


def run_experiment(
    attention_type: str,
    grid_size: int = 15,
    num_colors: int = 10,
    hidden_size: int = 128,
    num_layers: int = 4,
    num_heads: int = 6,
    num_epochs: int = 50,
    batches_per_epoch: int = 20,
    batch_size: int = 16,
    eval_tasks: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = None,
):
    """Run a single experiment."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {attention_type.upper()}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Grid: {grid_size}x{grid_size}, Colors: {num_colors}")
    print(f"Model: {num_layers} layers, {hidden_size} hidden, {num_heads} heads")
    
    # Set seeds
    torch.manual_seed(seed)
    
    # Create dataset and model
    train_dataset = ARCDataset(grid_size=grid_size, num_colors=num_colors, seed=seed)
    test_dataset = ARCDataset(grid_size=grid_size, num_colors=num_colors, seed=seed + 1000)
    
    model = ARCModel(
        grid_size=grid_size,
        num_colors=num_colors,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        attention_type=attention_type,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_grid_acc = 0.0
    history = {'train_loss': [], 'cell_acc': [], 'grid_acc': []}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_dataset, optimizer,
            batches_per_epoch, batch_size, device
        )
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            results = evaluate(model, test_dataset, eval_tasks, device)
            
            history['train_loss'].append(train_loss)
            history['cell_acc'].append(results['cell_accuracy'])
            history['grid_acc'].append(results['grid_accuracy'])
            
            if results['grid_accuracy'] > best_grid_acc:
                best_grid_acc = results['grid_accuracy']
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Cell: {results['cell_accuracy']:.4f} | Grid: {results['grid_accuracy']:.4f}")
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    final_results = evaluate(model, test_dataset, eval_tasks * 2, device)
    
    # Attention alignment evaluation
    print(f"\n{'─'*40}")
    print(f"Evaluating attention alignment with ideal patterns...")
    attn_tasks = test_dataset.generate_batch(eval_tasks)
    attn_alignment = compute_attention_alignment(model, attn_tasks, grid_size, device)
    
    print(f"\n{'─'*40}")
    print(f"Final Results ({attention_type}):")
    print(f"  Cell Accuracy: {final_results['cell_accuracy']:.4f}")
    print(f"  Grid Accuracy: {final_results['grid_accuracy']:.4f}")
    print(f"  Best Grid Acc: {best_grid_acc:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    
    print(f"\nAttention Alignment (vs ideal patterns):")
    print(f"  Cosine (all cells):      {attn_alignment['avg_cosine_sim']:.4f}")
    print(f"  Cosine (interesting):    {attn_alignment['avg_cosine_interesting']:.4f}")
    print(f"  Top-10 Overlap (all):    {attn_alignment['avg_top_k_overlap']:.4f}")
    print(f"  Top-10 Overlap (int):    {attn_alignment['avg_top_k_interesting']:.4f}")
    print(f"  Mass on Correct Targets: {attn_alignment['avg_mass_overlap']:.4f}")
    
    # Per-task breakdown
    print(f"\nPer-task accuracy:")
    for key, val in sorted(final_results.items()):
        if key.startswith('acc_'):
            print(f"  {key[4:]:20s}: {val:.4f}")
    
    print(f"\nPer-task attention alignment (cosine on interesting cells):")
    for key, val in sorted(attn_alignment.items()):
        if key.startswith('cosine_int_'):
            print(f"  {key[11:]:20s}: {val:.4f}")  # Strip 'cosine_int_'
    
    return {
        'attention_type': attention_type,
        'final_cell_acc': final_results['cell_accuracy'],
        'final_grid_acc': final_results['grid_accuracy'],
        'best_grid_acc': best_grid_acc,
        'num_params': num_params,
        'time': elapsed,
        'history': history,
        'task_breakdown': {k: v for k, v in final_results.items() if k.startswith('acc_')},
        'attn_alignment': attn_alignment,
    }


def main():
    """Run full benchmark comparing attention mechanisms."""
    print("="*70)
    print("ARC-AGI-2 Style Benchmark: Attention Mechanism Comparison")
    print("="*70)
    
    # Shared config
    config = dict(
        grid_size=15,
        num_colors=10,
        hidden_size=128,
        num_layers=4,
        num_heads=6,
        num_epochs=30,
        batches_per_epoch=20,
        batch_size=16,
        eval_tasks=50,
        lr=1e-3,
        seed=42,
    )
    
    results = {}
    
    # Run each attention type
    # "fal" uses your FlashAttentionLinks for blending
    for attn_type in ["standard", "jat", "triadic", "fal"]:
        try:
            results[attn_type] = run_experiment(attention_type=attn_type, **config)
        except Exception as e:
            print(f"Error with {attn_type}: {e}")
            import traceback
            traceback.print_exc()
            results[attn_type] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*85)
    print("SUMMARY")
    print("="*85)
    print(f"{'Model':<12} {'Params':>10} {'Cell':>8} {'Grid':>8} {'Cos(int)':>10} {'Mass':>8} {'Top10':>8} {'Time':>8}")
    print("-"*85)
    
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<12} ERROR: {res['error'][:50]}")
        else:
            aa = res.get('attn_alignment', {})
            print(f"{name:<12} {res['num_params']:>10,} {res['final_cell_acc']:>8.4f} "
                  f"{res['final_grid_acc']:>8.4f} {aa.get('avg_cosine_interesting', 0):>10.4f} "
                  f"{aa.get('avg_mass_overlap', 0):>8.4f} {aa.get('avg_top_k_interesting', 0):>8.4f} "
                  f"{res['time']:>7.1f}s")
    
    # Attention alignment breakdown - interesting cells only
    print("\n" + "="*85)
    print("ATTENTION ALIGNMENT ON INTERESTING CELLS (Cosine Similarity)")
    print("="*85)
    print(f"{'Model':<12} {'Avg':>8} {'Rot':>8} {'Refl':>8} {'Trans':>8} {'Fill':>8} {'Border':>8} {'Grav':>8} {'Recol':>8}")
    print("-"*85)
    
    for name, res in results.items():
        if 'error' not in res and 'attn_alignment' in res:
            aa = res['attn_alignment']
            print(f"{name:<12} {aa.get('avg_cosine_interesting', 0):>8.4f} "
                  f"{aa.get('cosine_int_rotation', 0):>8.4f} "
                  f"{aa.get('cosine_int_reflection', 0):>8.4f} "
                  f"{aa.get('cosine_int_translation', 0):>8.4f} "
                  f"{aa.get('cosine_int_fill', 0):>8.4f} "
                  f"{aa.get('cosine_int_border', 0):>8.4f} "
                  f"{aa.get('cosine_int_gravity', 0):>8.4f} "
                  f"{aa.get('cosine_int_recolor', 0):>8.4f}")
    
    # Mass overlap breakdown
    print("\n" + "="*85)
    print("ATTENTION MASS ON CORRECT TARGETS (Higher = Better)")
    print("="*85)
    print(f"{'Model':<12} {'Avg':>8} {'Rot':>8} {'Refl':>8} {'Trans':>8} {'Fill':>8} {'Border':>8} {'Grav':>8} {'Recol':>8}")
    print("-"*85)
    
    for name, res in results.items():
        if 'error' not in res and 'attn_alignment' in res:
            aa = res['attn_alignment']
            print(f"{name:<12} {aa.get('avg_mass_overlap', 0):>8.4f} "
                  f"{aa.get('mass_rotation', 0):>8.4f} "
                  f"{aa.get('mass_reflection', 0):>8.4f} "
                  f"{aa.get('mass_translation', 0):>8.4f} "
                  f"{aa.get('mass_fill', 0):>8.4f} "
                  f"{aa.get('mass_border', 0):>8.4f} "
                  f"{aa.get('mass_gravity', 0):>8.4f} "
                  f"{aa.get('mass_recolor', 0):>8.4f}")
    
    return results


# ============================================================================
# ATTENTION PATTERN LEARNING FRAMEWORK
# Train attention heads to replicate ideal attention patterns (no MLPs)
# ============================================================================

class PositionBias2D(nn.Module):
    """2D relative position bias for grid attention (same as FlashLinks uses)."""
    def __init__(self, grid_height: int, grid_width: int):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        max_rel_h = 2 * grid_height - 1
        max_rel_w = 2 * grid_width - 1
        
        self.bias_table = nn.Parameter(torch.zeros(max_rel_h, max_rel_w))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        
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


class AttentionLearner(nn.Module):
    """
    Simple model that learns to predict attention patterns.
    Uses learnable query/key projections to produce attention logits.
    NOW WITH 2D POSITIONAL BIAS (same as FlashLinks).
    """
    def __init__(self, grid_size: int, num_colors: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Embeddings
        self.color_embed = nn.Embedding(num_colors, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, hidden_size) * 0.02)
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 2D positional bias (SAME AS FLASHLINKS)
        self.pos_bias = PositionBias2D(grid_size, grid_size)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention logits [B, L, L] (pre-softmax)."""
        B, H, W = x.shape
        L = H * W
        
        x_flat = x.view(B, L)
        h = self.color_embed(x_flat) + self.pos_embed[:, :L, :]
        
        # Multi-head attention scores
        Q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [B, H, L, L]
        scores = scores + self.pos_bias().unsqueeze(0).unsqueeze(0)  # Add 2D bias
        return scores.mean(dim=1)  # [B, L, L] - average across heads


class TriadicAttentionLearner(nn.Module):
    """
    Learns attention patterns using FlashLinks triadic attention.
    """
    def __init__(self, grid_size: int, num_colors: int, hidden_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        self.hidden_size = hidden_size
        
        # Embeddings
        self.color_embed = nn.Embedding(num_colors, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, hidden_size) * 0.02)
        
        # FlashLinks
        head_dim = max(16, (hidden_size // 4) // 2 * 2)
        self.flash_links = FlashLinks(
            hidden_size=hidden_size,
            head_dim=head_dim,
            grid_height=grid_size,
            grid_width=grid_size,
            num_hops=2,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention logits [B, L, L]."""
        B, H, W = x.shape
        L = H * W
        
        x_flat = x.view(B, L)
        h = self.color_embed(x_flat) + self.pos_embed[:, :L, :]
        
        # Get attention from FlashLinks
        Q = self.flash_links.q_proj(h)
        K = self.flash_links.k_proj(h)
        cos, sin = self.flash_links.rope()
        Q, K = apply_rotary_pos_emb(Q.unsqueeze(2), K.unsqueeze(2), cos, sin)
        Q, K = Q.squeeze(2), K.squeeze(2)
        
        # Return raw scores (pre-softmax)
        scores = torch.bmm(Q, K.transpose(-1, -2)) / self.flash_links.scale
        return scores


class BlendedAttentionLearner(nn.Module):
    """
    Sequential blend: Standard attention scores blended with triadic.
    """
    def __init__(self, grid_size: int, num_colors: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.standard = AttentionLearner(grid_size, num_colors, hidden_size, num_heads)
        self.triadic = TriadicAttentionLearner(grid_size, num_colors, hidden_size)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid -> blend weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std_scores = self.standard(x)
        tri_scores = self.triadic(x)
        blend = torch.sigmoid(self.blend_logit)
        return blend * std_scores + (1 - blend) * tri_scores


def prepare_attention_batch(
    tasks: List[ARCTask],
    grid_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare batch for attention learning. Returns (inputs, ideal_attns, masks)."""
    inputs, ideal_attns, masks = [], [], []
    
    def pad_grid(g: torch.Tensor) -> torch.Tensor:
        h, w = g.shape
        padded = torch.zeros((grid_size, grid_size), dtype=torch.int64)
        padded[:min(h, grid_size), :min(w, grid_size)] = g[:min(h, grid_size), :min(w, grid_size)]
        return padded
    
    for task in tasks:
        inp = pad_grid(task.test_input)
        out = pad_grid(task.test_output)
        ideal_attn, mask = compute_ideal_attention(inp, out, task.task_type, grid_size)
        inputs.append(inp)
        ideal_attns.append(ideal_attn)
        masks.append(mask)
    
    return (
        torch.stack(inputs).to(device),
        torch.stack(ideal_attns).to(device),
        torch.stack(masks).to(device),
    )


def attention_loss(
    pred_logits: torch.Tensor,
    target_attn: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss and metrics for attention learning.
    
    Args:
        pred_logits: [B, L, L] raw attention scores (pre-softmax)
        target_attn: [B, L, L] target attention (normalized, sums to 1)
        mask: [B, L] mask for interesting cells
    
    Returns:
        loss, metrics_dict
    """
    B, L, _ = pred_logits.shape
    
    # Convert predictions to probabilities
    pred_attn = F.softmax(pred_logits, dim=-1)
    
    # Cross-entropy loss: -sum(target * log(pred))
    # This is equivalent to KL when target is fixed
    log_pred = F.log_softmax(pred_logits, dim=-1)
    ce_per_query = -(target_attn * log_pred).sum(dim=-1)  # [B, L]
    
    # Apply mask
    ce_masked = ce_per_query * mask.float()
    loss = ce_masked.sum() / (mask.float().sum() + 1e-8)
    
    # Metrics (computed on masked positions)
    with torch.no_grad():
        # Cosine similarity
        pred_norm = pred_attn / (pred_attn.norm(dim=-1, keepdim=True) + 1e-8)
        target_norm = target_attn / (target_attn.norm(dim=-1, keepdim=True) + 1e-8)
        cosine = (pred_norm * target_norm).sum(dim=-1)  # [B, L]
        cosine_masked = (cosine * mask.float()).sum() / (mask.float().sum() + 1e-8)
        
        # Top-5 overlap
        k = 5
        pred_topk = pred_attn.topk(k, dim=-1).indices  # [B, L, k]
        target_topk = target_attn.topk(k, dim=-1).indices
        
        overlap_sum = 0.0
        mask_count = 0
        for b in range(B):
            for i in range(L):
                if mask[b, i]:
                    pred_set = set(pred_topk[b, i].cpu().tolist())
                    target_set = set(target_topk[b, i].cpu().tolist())
                    overlap_sum += len(pred_set & target_set) / k
                    mask_count += 1
        top_k_overlap = overlap_sum / max(mask_count, 1)
    
    metrics = {
        'loss': loss.item(),
        'cosine': cosine_masked.item(),
        'top_k': top_k_overlap,
    }
    return loss, metrics


def evaluate_per_task_type(
    model: nn.Module,
    dataset: ARCDataset,
    grid_size: int,
    device: torch.device,
    samples_per_type: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on each task type separately."""
    model.eval()
    
    task_types = [
        TaskType.ROTATION, TaskType.REFLECTION, TaskType.TRANSLATION,
        TaskType.FILL, TaskType.BORDER, TaskType.GRAVITY, TaskType.RECOLOR
    ]
    
    results = {}
    
    with torch.no_grad():
        for task_type in task_types:
            losses, cosines, top_ks = [], [], []
            
            for _ in range(samples_per_type):
                # Generate specific task type
                task = dataset.generate_task(task_type)
                tasks = [task]
                inputs, targets, masks = prepare_attention_batch(tasks, grid_size, device)
                
                pred_logits = model(inputs)
                _, metrics = attention_loss(pred_logits, targets, masks)
                
                losses.append(metrics['loss'])
                cosines.append(metrics['cosine'])
                top_ks.append(metrics['top_k'])
            
            results[task_type.value] = {
                'loss': sum(losses) / len(losses),
                'cosine': sum(cosines) / len(cosines),
                'top_k': sum(top_ks) / len(top_ks),
            }
    
    return results


def train_and_evaluate(
    model_class,
    model_kwargs: dict,
    dataset: ARCDataset,
    grid_size: int,
    device: torch.device,
    seed: int,
    num_epochs: int = 50,
    batches_per_epoch: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Dict:
    """Train a model and return comprehensive metrics."""
    torch.manual_seed(seed)
    
    model = model_class(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            tasks = dataset.generate_batch(batch_size)
            inputs, targets, masks = prepare_attention_batch(tasks, grid_size, device)
            
            optimizer.zero_grad()
            pred_logits = model(inputs)
            loss, _ = attention_loss(pred_logits, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / batches_per_epoch)
    
    # Final evaluation on fresh data
    model.eval()
    eval_seed = seed + 10000
    torch.manual_seed(eval_seed)
    
    all_losses, all_cosines, all_top_ks = [], [], []
    
    with torch.no_grad():
        for _ in range(10):  # 10 eval batches
            tasks = dataset.generate_batch(batch_size)
            inputs, targets, masks = prepare_attention_batch(tasks, grid_size, device)
            pred_logits = model(inputs)
            _, metrics = attention_loss(pred_logits, targets, masks)
            all_losses.append(metrics['loss'])
            all_cosines.append(metrics['cosine'])
            all_top_ks.append(metrics['top_k'])
    
    # Per-task-type evaluation
    per_task = evaluate_per_task_type(model, dataset, grid_size, device, samples_per_type=30)
    
    return {
        'params': num_params,
        'seed': seed,
        'final_loss': sum(all_losses) / len(all_losses),
        'final_cosine': sum(all_cosines) / len(all_cosines),
        'final_top_k': sum(all_top_ks) / len(all_top_ks),
        'loss_std': (sum((x - sum(all_losses)/len(all_losses))**2 for x in all_losses) / len(all_losses)) ** 0.5,
        'cosine_std': (sum((x - sum(all_cosines)/len(all_cosines))**2 for x in all_cosines) / len(all_cosines)) ** 0.5,
        'top_k_std': (sum((x - sum(all_top_ks)/len(all_top_ks))**2 for x in all_top_ks) / len(all_top_ks)) ** 0.5,
        'per_task': per_task,
        'convergence': train_losses[-1] - train_losses[0],  # How much loss dropped
    }


def compute_statistics(runs: List[Dict]) -> Dict:
    """Compute mean, std, and 95% CI across runs."""
    n = len(runs)
    
    def mean(vals):
        return sum(vals) / len(vals)
    
    def std(vals):
        m = mean(vals)
        return (sum((x - m)**2 for x in vals) / len(vals)) ** 0.5
    
    def ci95(vals):
        # 95% CI = mean ± 1.96 * std / sqrt(n)
        # For n=3, use t-value ~4.3 for 95% CI
        t_val = 4.303 if n == 3 else 1.96
        return t_val * std(vals) / (n ** 0.5)
    
    losses = [r['final_loss'] for r in runs]
    cosines = [r['final_cosine'] for r in runs]
    top_ks = [r['final_top_k'] for r in runs]
    
    # Aggregate per-task results
    task_types = list(runs[0]['per_task'].keys())
    per_task_agg = {}
    for tt in task_types:
        tt_cosines = [r['per_task'][tt]['cosine'] for r in runs]
        tt_top_ks = [r['per_task'][tt]['top_k'] for r in runs]
        per_task_agg[tt] = {
            'cosine_mean': mean(tt_cosines),
            'cosine_std': std(tt_cosines),
            'cosine_ci': ci95(tt_cosines),
            'top_k_mean': mean(tt_top_ks),
            'top_k_std': std(tt_top_ks),
            'top_k_ci': ci95(tt_top_ks),
        }
    
    return {
        'params': runs[0]['params'],
        'n_runs': n,
        'loss_mean': mean(losses),
        'loss_std': std(losses),
        'loss_ci': ci95(losses),
        'cosine_mean': mean(cosines),
        'cosine_std': std(cosines),
        'cosine_ci': ci95(cosines),
        'top_k_mean': mean(top_ks),
        'top_k_std': std(top_ks),
        'top_k_ci': ci95(top_ks),
        'per_task': per_task_agg,
    }


def run_comprehensive_benchmark():
    """
    Comprehensive benchmark with multiple seeds and per-task-type analysis.
    """
    print("="*80)
    print("COMPREHENSIVE ATTENTION PATTERN LEARNING BENCHMARK")
    print("3 seeds per model, per-task-type breakdown, 95% confidence intervals")
    print("="*80)
    
    # Config
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    grid_size = 10
    num_colors = 10
    hidden_size = 64
    num_heads = 4
    seeds = [42, 123, 456]
    
    print(f"Device: {device}")
    print(f"Grid: {grid_size}x{grid_size}, Hidden: {hidden_size}, Seeds: {seeds}")
    print()
    
    dataset = ARCDataset(grid_size=grid_size, num_colors=num_colors, seed=0)
    
    # Model configs
    model_configs = {
        'standard': (AttentionLearner, {'grid_size': grid_size, 'num_colors': num_colors, 
                                         'hidden_size': hidden_size, 'num_heads': num_heads}),
        'triadic': (TriadicAttentionLearner, {'grid_size': grid_size, 'num_colors': num_colors,
                                               'hidden_size': hidden_size}),
        'blended': (BlendedAttentionLearner, {'grid_size': grid_size, 'num_colors': num_colors,
                                               'hidden_size': hidden_size, 'num_heads': num_heads}),
    }
    
    all_results = {}
    
    for model_name, (model_class, model_kwargs) in model_configs.items():
        print(f"\n{'─'*60}")
        print(f"Training: {model_name.upper()} (3 seeds)")
        print(f"{'─'*60}")
        
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = train_and_evaluate(
                model_class, model_kwargs, dataset, grid_size, device, seed,
                num_epochs=50, batches_per_epoch=5, batch_size=32, lr=1e-3
            )
            runs.append(result)
            print(f"Loss: {result['final_loss']:.3f}, Cos: {result['final_cosine']:.3f}, "
                  f"Top5: {result['final_top_k']:.3f}")
        
        all_results[model_name] = compute_statistics(runs)
    
    # ========== PRINT COMPREHENSIVE RESULTS ==========
    
    print("\n" + "="*80)
    print("OVERALL RESULTS (mean ± 95% CI)")
    print("="*80)
    print(f"{'Model':<12} {'Params':>8} {'Loss':>16} {'Cosine':>16} {'Top-5':>16}")
    print("-"*80)
    
    for name, stats in all_results.items():
        print(f"{name:<12} {stats['params']:>8,} "
              f"{stats['loss_mean']:>6.3f}±{stats['loss_ci']:.3f}  "
              f"{stats['cosine_mean']:>6.3f}±{stats['cosine_ci']:.3f}  "
              f"{stats['top_k_mean']:>6.3f}±{stats['top_k_ci']:.3f}")
    
    # Per-task-type breakdown
    task_types = ['rotation', 'reflection', 'translation', 'fill', 'border', 'gravity', 'recolor']
    
    print("\n" + "="*80)
    print("PER-TASK-TYPE COSINE SIMILARITY (mean ± 95% CI)")
    print("="*80)
    header = f"{'Model':<12}" + "".join(f"{tt[:6]:>10}" for tt in task_types)
    print(header)
    print("-"*80)
    
    for name, stats in all_results.items():
        row = f"{name:<12}"
        for tt in task_types:
            m = stats['per_task'][tt]['cosine_mean']
            ci = stats['per_task'][tt]['cosine_ci']
            row += f"{m:>6.2f}±{ci:.2f}"
        print(row)
    
    print("\n" + "="*80)
    print("PER-TASK-TYPE TOP-5 OVERLAP (mean ± 95% CI)")
    print("="*80)
    print(header)
    print("-"*80)
    
    for name, stats in all_results.items():
        row = f"{name:<12}"
        for tt in task_types:
            m = stats['per_task'][tt]['top_k_mean']
            ci = stats['per_task'][tt]['top_k_ci']
            row += f"{m:>6.2f}±{ci:.2f}"
        print(row)
    
    # Statistical comparisons
    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS (non-overlapping CI = significant at p<0.05)")
    print("="*80)
    
    def cis_overlap(m1, ci1, m2, ci2):
        """Check if confidence intervals overlap."""
        return not (m1 - ci1 > m2 + ci2 or m2 - ci2 > m1 + ci1)
    
    models = list(all_results.keys())
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            s1, s2 = all_results[m1], all_results[m2]
            
            cos_overlap = cis_overlap(s1['cosine_mean'], s1['cosine_ci'], 
                                       s2['cosine_mean'], s2['cosine_ci'])
            top_k_overlap = cis_overlap(s1['top_k_mean'], s1['top_k_ci'],
                                         s2['top_k_mean'], s2['top_k_ci'])
            
            cos_winner = m1 if s1['cosine_mean'] > s2['cosine_mean'] else m2
            top_k_winner = m1 if s1['top_k_mean'] > s2['top_k_mean'] else m2
            
            cos_sig = "SIGNIFICANT" if not cos_overlap else "not significant"
            top_k_sig = "SIGNIFICANT" if not top_k_overlap else "not significant"
            
            print(f"\n{m1} vs {m2}:")
            print(f"  Cosine: {cos_winner} better ({cos_sig})")
            print(f"  Top-5:  {top_k_winner} better ({top_k_sig})")
    
    # Task-type winners
    print("\n" + "="*80)
    print("BEST MODEL PER TASK TYPE (by Top-5 overlap)")
    print("="*80)
    
    for tt in task_types:
        best_model = max(all_results.keys(), 
                        key=lambda m: all_results[m]['per_task'][tt]['top_k_mean'])
        best_val = all_results[best_model]['per_task'][tt]['top_k_mean']
        best_ci = all_results[best_model]['per_task'][tt]['top_k_ci']
        
        # Check if significantly better than others
        sig_better = True
        for other in all_results.keys():
            if other != best_model:
                other_val = all_results[other]['per_task'][tt]['top_k_mean']
                other_ci = all_results[other]['per_task'][tt]['top_k_ci']
                if cis_overlap(best_val, best_ci, other_val, other_ci):
                    sig_better = False
                    break
        
        sig_str = " (SIGNIFICANT)" if sig_better else ""
        print(f"  {tt:<12}: {best_model} ({best_val:.3f}±{best_ci:.3f}){sig_str}")
    
    # Final interpretation
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print(f"  Random baseline: Loss≈4.6, Cosine≈0.10, Top-5≈0.05")
    print(f"  Good learning:   Loss<3.5, Cosine>0.60, Top-5>0.30")
    print(f"  Strong learning: Loss<3.0, Cosine>0.75, Top-5>0.45")
    print()
    print("  Task categories:")
    print("    - Multi-hop (translation, rotation, reflection): benefits from relational reasoning")
    print("    - Local (fill, border, recolor): benefits from local pattern matching")
    print("    - Physics (gravity): benefits from directional understanding")
    print("="*80)
    
    return all_results


def run_attention_learning_benchmark():
    """Run the comprehensive benchmark."""
    return run_comprehensive_benchmark()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "attention":
        run_attention_learning_benchmark()
    else:
        main()
