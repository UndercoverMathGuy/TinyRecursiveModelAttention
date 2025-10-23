import pandas as pd

def clean_sudoku_datasets(train:bool=True):
    path = 'train' if train == True else 'test'
    train = pd.read_csv(f'sudoku_datasets/{path}.csv')
    train.drop(columns=['source'], inplace=True)
    train['question'] = train['question'].str.replace('.', '0')
    delimiter = ','
    train['question'] = train['question'].apply(lambda x: delimiter.join(x))
    train['answer'] = train['answer'].apply(lambda x: delimiter.join(x))
    train.to_parquet(f'sudoku_datasets/{path}.parquet', index=False)

    print(train['question'][0])
    print(train['answer'][0])