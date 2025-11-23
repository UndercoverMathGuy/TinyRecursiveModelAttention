import math
import os
import random
import torch
import torch.nn as nn
from layers import FlashLinks, RawAttention
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class JATAdjacency(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        order: int = 1,
        ceof: float = 0.5,
        num_jat_heads: int | None = None,
    ) -> None:
        super().__init__()

        assert d_model > 0
        assert num_heads > 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or (d_model // num_heads)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # JAT order: order=1 corresponds to the basic two-hop triadic closure
        self.order = max(1, int(order))

        self.ceof = float(ceof)

        # How many heads actually use the jump operator; others stay canonical.
        # Default: all heads use JAT.
        if num_jat_heads is None:
            self.num_jat_heads = num_heads
        else:
            self.num_jat_heads = max(0, min(int(num_jat_heads), num_heads))

        # Linear projections for Q and K (values are not needed for adjacency)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

    def _shape_qk(self, x: torch.Tensor) -> torch.Tensor:
        """Project to [B, H, L, D] from [B, L, d_model]."""
        bsz, seq_len, _ = x.shape
        proj = x.view(bsz * seq_len, -1)
        # Apply linear and reshape
        return (
            self.q_proj(x)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

    def _project_qk(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return q, k

    def _build_normalized_adjacency(self, scores: torch.Tensor) -> torch.Tensor:

        bsz, n_heads, seq_len, _ = scores.shape
        device = scores.device
        d_k = float(self.head_dim)

        with torch.no_grad():
            s_detached = scores.detach()
            # U: [B, H, L, L] ~ sum_j S_j S_j^T / (L * d_k)
            U = torch.matmul(s_detached, s_detached.transpose(-1, -2))
            U = U / (seq_len * max(d_k, 1.0))

            # Remove self-connections before graph construction
            eye = torch.eye(seq_len, device=device, dtype=U.dtype).view(1, 1, seq_len, seq_len)
            U = U - U * eye

            # Dense, non-sparse adjacency based on magnitude of high-order
            # similarity. This keeps all edges (no hard thresholding or
            # sampling like JAT†) while ensuring non-negative weights.
            A = torch.abs(U)

            # Add self-loops and normalize: \hat{A} = D^{-1/2} (A + I) D^{-1/2}
            A_tilde = A + eye
            D = A_tilde.sum(dim=-1)  # [B, H, L]
            D = torch.clamp(D, min=1e-6)
            D_inv_sqrt = D.rsqrt().unsqueeze(-1)  # [B, H, L, 1]

            A_hat = A_tilde * D_inv_sqrt * D_inv_sqrt.transpose(-1, -2)

        return A_hat

    def _apply_jump(self, scores: torch.Tensor) -> torch.Tensor:
        A_hat = self._build_normalized_adjacency(scores)

        S_jump = scores
        for _ in range(self.order):
            S_jump = torch.matmul(A_hat, torch.matmul(S_jump, A_hat.transpose(-1, -2)))

        return S_jump

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute JAT-enhanced attention scores.

        Args:
            hidden_states: [B, L, d_model]

        Returns:
            scores: [B, L, L] — aggregated over heads, pre-softmax.
        """

        assert hidden_states.dim() == 3, "hidden_states must be [B, L, d_model]"

        bsz, seq_len, _ = hidden_states.shape

        # Project to multi-head Q/K
        q, k = self._project_qk(hidden_states)
        # q, k: [B, H, L, D]

        # Base dot-product scores S = QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Determine how many heads will use the jump operator
        num_jat = self.num_jat_heads

        if num_jat > 0:
            # Split heads: first num_jat heads use JAT, the rest stay canonical
            scores_jat = scores[:, :num_jat, :, :]
            scores_jump = self._apply_jump(scores_jat)

            # Blend canonical and jump scores on JAT heads
            scores_jat_blend = (1.0 - self.ceof) * scores_jat + self.ceof * scores_jump

            if num_jat < self.num_heads:
                scores_all = torch.cat([scores_jat_blend, scores[:, num_jat:, :, :]], dim=1)
            else:
                scores_all = scores_jat_blend
        else:
            # No JAT heads: everything is canonical
            scores_all = scores

        # Aggregate over heads to get a single structural matrix
        scores_head_avg = scores_all.mean(dim=1)  # [B, L, L]

        return scores_head_avg

class RSM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, seq_len, C_temp, F_temp):
        super().__init__()
        self.links = FlashLinks(emb_dim=emb_dim, hidden_dim=hidden_dim, seq_len=seq_len, C_temp=C_temp, F_temp=F_temp)
        self.attn = RawAttention(hidden_size=hidden_dim, head_dim=emb_dim, seq_len=seq_len, causal=False, softmax=False)
    
    def forward(self, x):
        qclogits = torch.zeros(x.shape(0), device=x.device)
        S = self.attn(x)
        H = self.links(x, S, qclogits)
        return H


def membrane(dim, device):
    N = dim
    if device is None:
        device = torch.device("cpu")

    # 0 = room, 1 = open membrane, 2 = closed membrane
    grid = torch.zeros((N, N), device=device)

    orientation = random.choice(["vertical", "horizontal"])
    stripe_value = torch.tensor(1 if random.random() < 0.5 else 2, device=device)

    def _set_cell(r: int, c: int) -> None:
        grid[r, c] = stripe_value

    if orientation == "vertical":
        col = random.randint(1, N - 2)
        for row in range(N):
            _set_cell(row, col)
            if row == N - 1:
                break
            delta = random.choice([-1, 0, 1])
            col = min(max(1, col + delta), N - 2)
    else:
        row = random.randint(1, N - 2)
        for col in range(N):
            _set_cell(row, col)
            if col == N - 1:
                break
            delta = random.choice([-1, 0, 1])
            row = min(max(1, row + delta), N - 2)

    return grid.long()


def membrane_targets(grid):
    assert grid.dim() == 2 and grid.size(0) == grid.size(1), "grid must be square"

    N = grid.size(0)
    device = grid.device
    grid = grid.to(torch.float32)
    conductive = grid != 2  # closed membrane blocks chains

    L = N * N
    targets = torch.zeros((L, L), device=device, dtype=grid.dtype)
    mask = torch.zeros_like(targets)

    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def flat(r: int, c: int) -> int:
        return r * N + c

    for r in range(N):
        for c in range(N):
            i = flat(r, c)

            # Self-loop
            targets[i, i] = 1.0
            mask[i, i] = 1.0

            # Distance-1 neighbors (always 1.0)
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    j = flat(nr, nc)
                    targets[i, j] = 1.0
                    mask[i, j] = 1.0

            # Distance-2 via a single intermediate node
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    gate_open = conductive[nr, nc]
                    for dr2, dc2 in offsets:
                        jr, jc = nr + dr2, nc + dc2
                        if 0 <= jr < N and 0 <= jc < N:
                            j = flat(jr, jc)
                            if j == i:
                                continue
                            mask[i, j] = 1.0
                            targets[i, j] = 1.0 if gate_open else 0.0

    targets = torch.maximum(targets, targets.transpose(0, 1))
    mask = torch.maximum(mask, mask.transpose(0, 1))

    return targets, mask


def dataset(length: int, size: int, device: torch.device | str) -> TensorDataset:
    device = torch.device(device)
    grids = []
    targets_list = []
    masks = []

    with tqdm(total=length, desc="membrane-dataset", leave=False) as pbar:
        while len(grids) < length:
            grid = membrane(size, device=device)
            targets, mask = membrane_targets(grid)

            grids.append(grid)
            targets_list.append(targets)
            masks.append(mask)
            pbar.update(1)

            if len(grids) >= length:
                break

            grid_invert = torch.where(grid == 2, 1, grid)
            grid_invert = torch.where(grid == 1, 2, grid_invert)
            targets_inv, mask_inv = membrane_targets(grid_invert)

            grids.append(grid_invert)
            targets_list.append(targets_inv)
            masks.append(mask_inv)
            pbar.update(1)

    grids_tensor = torch.stack(grids[:length]) # [L, N, N]
    targets_tensor = torch.stack(targets_list[:length]) # [L, N^2, N^2]
    masks_tensor = torch.stack(masks[:length]) # [L, N^2, N^2]

    return TensorDataset(grids_tensor, targets_tensor, masks_tensor)


def save_membrane_dataset(ds: TensorDataset, path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save({"grids": ds.tensors[0], "targets": ds.tensors[1], "masks": ds.tensors[2]}, path)


def load_membrane_dataset(path: str, device: torch.device | str = "cpu") -> TensorDataset:
    data = torch.load(path, map_location=device)
    return TensorDataset(data["grids"], data["targets"], data["masks"])


def train_JAT(epochs, batch_size):
    torch.manual_seed(1331)
    dataset = load_membrane_dataset("membrane_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # d_model=3: 1 channel for cell type (room/membrane) + 2 for (row, col) coordinates
    model = JATAdjacency(d_model=3, num_heads=1, head_dim=4, order=1, ceof=0.5, num_jat_heads=1)
    model.to("mps")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.01, patience=5
    )
    
    # Precompute positional encodings for a 16x16 grid
    grids_example = dataset.tensors[0]
    N = grids_example.size(-1)
    L = N * N
    rows = torch.linspace(-1.0, 1.0, N)
    cols = torch.linspace(-1.0, 1.0, N)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    pos = torch.stack([rr, cc], dim=-1).view(L, 2).to("mps")  # [L, 2]

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_mask = 0.0

        for batch in dataloader:
            grids, targets, masks = batch
            B = grids.size(0)

            # cell type channel
            cell = grids.view(B, -1, 1).to(dtype=torch.float32, device="mps")  # [B, L, 1]
            # broadcast positional channels
            pos_batch = pos.unsqueeze(0).expand(B, -1, -1)                       # [B, L, 2]
            seq = torch.cat([cell, pos_batch], dim=-1)                           # [B, L, 3]

            targets = targets.to("mps")
            masks = masks.to("mps")

            optimizer.zero_grad()
            outputs = model(seq)
            logits = outputs.view_as(targets)
            loss_full = criterion(logits, targets)

            batch_loss_sum = (loss_full * masks).sum()
            batch_mask_sum = masks.sum()
            loss = batch_loss_sum / batch_mask_sum

            loss.backward()
            optimizer.step()

            epoch_loss += batch_loss_sum.item()
            total_mask += batch_mask_sum.item()

        avg_loss = epoch_loss / max(total_mask, 1.0)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        scheduler.step(avg_loss)
    jat_weights_path = "jat_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        jat_weights_path,
    )

def load_jat_weights(path):
    checkpoint = torch.load(path, map_location="mps")
    model.load_state_dict(checkpoint["model_state_dict"])

if __name__ == "__main__":
    alpha = 2