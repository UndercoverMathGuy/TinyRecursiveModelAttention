FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy lock files first for better caching
COPY pyproject.toml uv.lock .python-version ./

# Install deps
RUN uv sync --frozen --no-dev

# Copy entire repo (preserves folder structure)
COPY . .

# Build dataset, then train
CMD ["sh", "-c", \
     "uv run dataset/build_arc_dataset.py \
        --input_dir kaggle/combined \
        --output_dir data/arc2concept-aug-1000 \
        --augs 1000 && \
     torchrun --nproc-per-node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 \
        pretrain.py \
        arch=trm \
        data_paths=[data/arc2concept-aug-1000] \
        arch.L_layers=2 \
        arch.H_cycles=3 \
        arch.L_cycles=4 \
        run_name=pretrain_att_arc2concept_4 \
        ema=True"]