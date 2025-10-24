import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 1) Read and prep
train = pd.read_csv("datasets/train.csv")
test  = pd.read_csv("datasets/test.csv")

for df in (train, test):
    if 'source' in df.columns:
        df.drop(columns=['source'], inplace=True)

# 2) Parquet write via pyarrow (atomic)
def write_parquet_atomic(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = f"{path}.tmp"
    pq.write_table(table, tmp, compression="snappy")  # or None to be extra safe
    os.replace(tmp, path)  # atomic on most OSes

write_parquet_atomic(train, "datasets/train.parquet")
write_parquet_atomic(test,  "datasets/test.parquet")

# 3) Verify footer
def footer(path):
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return f.read()
print("train footer:", footer("datasets/train.parquet"))  # should be b'PAR1'
print("test footer:", footer("datasets/test.parquet"))

print(pd.read_parquet("datasets/train.parquet").head())