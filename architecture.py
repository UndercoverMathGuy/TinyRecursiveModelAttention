import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import pandas
import numpy as np

@dataclasses.dataclass
class DataTensor:
    df: pandas.DataFrame

    def to_tensors(self, device: torch.device, train=True):
        col = 'question' if train==True else 'answer'
        tensors = []
        tensor_mask_stack = []
        for item in self.df[col]:
            array = np.array(item, dtype=np.int64)
            ids = torch.tensor(array, dtype=torch.long)  # on CPU
            mask = (ids > 0).long()
            one_hot = F.one_hot(ids, num_classes=10)
            tensors.append(one_hot)
            tensor_mask_stack.append(mask)
        one_hots = torch.stack(tensors, dim=0)     # [B, L, C] - one hot
        masks = torch.stack(tensor_mask_stack, dim=0)  # [B, L] - mask per L
        one_hots = one_hots.to(device)
        masks = masks.to(device)
        if train==True:
            return one_hots, masks
        else:
            return one_hots

if __name__ == "__main__":
    df = pandas.read_parquet('datasets/test.parquet')
    df = df[:10]
    test = DataTensor(df)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    one_hots, masks = test.to_tensors(device=device, train=True)
    print(one_hots[0])
    print(masks[0])
    print(one_hots[1])
    print(masks[1])