import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import pandas
import numpy as np
import random
from typing import Optional

@dataclasses.dataclass
class DataTensor:
    df: pandas.DataFrame

    def __repr__(self):
        return repr(self.df)
    
    def switch_df(self, df):
        self.df = df
        return self
    
    def copy(self):
        return DataTensor(self.df.copy())

    def to_tensors(self, device: torch.device, col_question=True):
        col = 'question' if col_question==True else 'answer'
        tensors = []
        tensor_mask_stack = []
        for item in self.df[col]:
            array = np.array(item, dtype=np.int64)
            ids = torch.tensor(array, dtype=torch.long)
            mask_given = (ids > 0).long()
            if col_question==False:
                ids = ids - 1  # shift to 0-8 for correct one hot on answers (range 1-9)
            one_hot = F.one_hot(ids, num_classes=10 if col_question==True else 9) # [L, C]
            tensors.append(one_hot)
            tensor_mask_stack.append(mask_given)
        one_hots = torch.stack(tensors, dim=0)     # [B, L, C] - one hot
        masks = torch.stack(tensor_mask_stack, dim=0)  # [B, L] - mask per L
        one_hots = one_hots.float().to(device)
        masks = masks.to(device)
        if col_question==True:
            return one_hots, mask_given
        else:
            return one_hots
    
@dataclasses.dataclass
class TensorAugmented:
    tensor: DataTensor
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def permute_numbers(self, i):
        alpha, beta = self._rng.sample(range(1, 10), 2)
        # Remove df.copy() - work on existing df
        answer_arr = np.array(self.tensor.df['answer'][i], dtype=np.int64)
        mask_alpha = answer_arr == alpha
        mask_beta = answer_arr == beta
        answer_arr[mask_alpha] = beta
        answer_arr[mask_beta] = alpha
        self.tensor.df.at[i, 'answer'] = answer_arr

        question_arr = np.array(self.tensor.df['question'][i], dtype=np.int64)
        mask_alpha = question_arr == alpha
        mask_beta = question_arr == beta
        question_arr[mask_alpha] = beta
        question_arr[mask_beta] = alpha
        self.tensor.df.at[i, 'question'] = question_arr

        return self.tensor

    def permute_bands(self, i):
        band_indices = [0, 1, 2]
        self._rng.shuffle(band_indices)
        # Remove df.copy() - work on existing df
        arr = np.array(self.tensor.df['answer'][i], dtype=np.int64).reshape(9, 9)
        new_arr = np.zeros((9, 9), dtype=np.int64)
        for new_band_idx, old_band_idx in enumerate(band_indices):
            new_arr[new_band_idx*3:(new_band_idx+1)*3, :] = arr[old_band_idx*3:(old_band_idx+1)*3, :]
        self.tensor.df.at[i, 'answer'] = new_arr.flatten()

        arr = np.array(self.tensor.df['question'][i], dtype=np.int64).reshape(9, 9)
        new_arr = np.zeros((9, 9), dtype=np.int64)
        for new_band_idx, old_band_idx in enumerate(band_indices):
            new_arr[new_band_idx*3:(new_band_idx+1)*3, :] = arr[old_band_idx*3:(old_band_idx+1)*3, :]
        self.tensor.df.at[i, 'question'] = new_arr.flatten()
        return self.tensor
    
    def permute_rows(self, i):
        band_idx = self._rng.randint(0, 2)
        row_indices = [0, 1, 2]
        self._rng.shuffle(row_indices)
        # Remove df.copy() - work on existing df
        arr = np.array(self.tensor.df['answer'][i], dtype=np.int64).reshape(9, 9)
        new_arr = arr.copy()
        for new_row_idx, old_row_idx in enumerate(row_indices):
            new_arr[band_idx*3 + new_row_idx, :] = arr[band_idx*3 + old_row_idx, :]
        self.tensor.df.at[i, 'answer'] = new_arr.flatten()

        arr = np.array(self.tensor.df['question'][i], dtype=np.int64).reshape(9, 9)
        new_arr = arr.copy()
        for new_row_idx, old_row_idx in enumerate(row_indices):
            new_arr[band_idx*3 + new_row_idx, :] = arr[band_idx*3 + old_row_idx, :]
        self.tensor.df.at[i, 'question'] = new_arr.flatten()
        return self.tensor
    
    def permute_stacks(self, i):
        stack_indices = [0, 1, 2]
        self._rng.shuffle(stack_indices)
        # Remove df.copy() - work on existing df
        arr = np.array(self.tensor.df['answer'][i], dtype=np.int64).reshape(9, 9)
        new_arr = np.zeros((9, 9), dtype=np.int64)
        for new_stack_idx, old_stack_idx in enumerate(stack_indices):
            new_arr[:, new_stack_idx*3:(new_stack_idx+1)*3] = arr[:, old_stack_idx*3:(old_stack_idx+1)*3]
        self.tensor.df.at[i, 'answer'] = new_arr.flatten()

        arr = np.array(self.tensor.df['question'][i], dtype=np.int64).reshape(9, 9)
        new_arr = np.zeros((9, 9), dtype=np.int64)
        for new_stack_idx, old_stack_idx in enumerate(stack_indices):
            new_arr[:, new_stack_idx*3:(new_stack_idx+1)*3] = arr[:, old_stack_idx*3:(old_stack_idx+1)*3]
        self.tensor.df.at[i, 'question'] = new_arr.flatten()
        return self.tensor
    
    def permute_columns(self, i):
        stack_idx = self._rng.randint(0, 2)
        col_indices = [0, 1, 2]
        self._rng.shuffle(col_indices)
        # Remove df.copy() - work on existing df
        arr = np.array(self.tensor.df['answer'][i], dtype=np.int64).reshape(9, 9)
        new_arr = arr.copy()
        for new_col_idx, old_col_idx in enumerate(col_indices):
            new_arr[:, stack_idx*3 + new_col_idx] = arr[:, stack_idx*3 + old_col_idx]
        self.tensor.df.at[i, 'answer'] = new_arr.flatten()

        arr = np.array(self.tensor.df['question'][i], dtype=np.int64).reshape(9, 9)
        new_arr = arr.copy()
        for new_col_idx, old_col_idx in enumerate(col_indices):
            new_arr[:, stack_idx*3 + new_col_idx] = arr[:, stack_idx*3 + old_col_idx]
        self.tensor.df.at[i, 'question'] = new_arr.flatten()
        return self.tensor
    
    def permute(self, i):
        self.permute_numbers(i)
        self.permute_bands(i)
        self.permute_rows(i)
        self.permute_stacks(i)
        self.permute_columns(i)
        return self.tensor
    
    def permute_df(self):
        for i in range(len(self.tensor.df)):
            self.permute(i)
        return self.tensor.df
            
    def permute_loop(self, n_loops):
        original_df = self.tensor.df.copy()
        full_df = original_df.copy()
        frames = [original_df]
        for _ in range(n_loops):
            # Reset to original data before each permutation
            self.tensor.df = original_df.copy()
            self.permute_df()
            frames.append(self.tensor.df)
        full_df = pandas.concat(frames, axis=0, ignore_index=True)
        full_df.drop_duplicates(subset=['question', 'answer'], inplace=True)
        return self.tensor.switch_df(full_df.reset_index(drop=True))