import pandas as pd
import numpy as np
from tqdm.auto import tqdm  # use auto so tqdm.pandas() is available

def clean_sudoku_datasets(train:bool=True):
    path = 'train' if train == True else 'test'
    train = pd.read_csv(f'datasets/{path}.csv')
    train.drop(columns=['source'], inplace=True)
    train['question'] = train['question'].str.replace('.', '0')
    delimiter = ','
    train['question'] = train['question'].apply(lambda x: delimiter.join(x))
    train['answer'] = train['answer'].apply(lambda x: delimiter.join(x))
    print("Cleaning dataset completed.")  # Completion message
    train.to_parquet(f'datasets/{path}.parquet', index=False)

    print(train['question'][0])
    print(train['answer'][0])

def numerise(train:bool=True, save_as_lists: bool=True):
    path = 'train' if train == True else 'test'
    df = pd.read_parquet(f'datasets/{path}.parquet')

    # Convert both columns row-by-row with a single tqdm that updates per row
    question_arrays, answer_arrays = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting rows to numpy arrays", unit="row"):
        q_str = '' if pd.isna(row['question']) else str(row['question'])
        a_str = '' if pd.isna(row['answer']) else str(row['answer'])
        q_arr = pd.to_numeric(pd.Series(q_str.split(',')), errors='coerce').to_numpy(dtype='float32')
        a_arr = pd.to_numeric(pd.Series(a_str.split(',')), errors='coerce').to_numpy(dtype='float32')
        question_arrays.append(q_arr)
        answer_arrays.append(a_arr)

    df['question'] = question_arrays
    df['answer'] = answer_arrays

    if save_as_lists:
        # Progress bar around conversion to Python lists for Parquet compatibility
        tqdm.pandas(desc="Converting numpy arrays to Python lists")
        df['question'] = df['question'].progress_apply(lambda x: x.tolist())
        df['answer'] = df['answer'].progress_apply(lambda x: x.tolist())
        df.to_parquet(f'datasets/{path}.parquet', index=False)
    else:
        # Try saving numpy arrays directly; fall back to lists if the engine refuses
        try:
            df.to_parquet(f'datasets/{path}.parquet', index=False)
        except Exception as e:
            print(f"Parquet write with numpy arrays failed ({e}); falling back to list conversion.")
            tqdm.pandas(desc="Converting numpy arrays to Python lists")
            df['question'] = df['question'].progress_apply(lambda x: x.tolist())
            df['answer'] = df['answer'].progress_apply(lambda x: x.tolist())
            df.to_parquet(f'datasets/{path}.parquet', index=False)

    print(df['question'][0])
    print(df['answer'][0])

clean_sudoku_datasets(train=True)
numerise(train=True, save_as_lists=False)  # set save_as_lists=False to attempt numpy arrays; will auto-fallback if unsupported
clean_sudoku_datasets(train=False)
numerise(train=False, save_as_lists=False)