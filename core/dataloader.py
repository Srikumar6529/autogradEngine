import random
import sys
import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import numpy as np
from core.tensor import Tensor
class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def __getitem__(self, idx):
        pass

class TensorDataset(Dataset):
    def __init__(self,*tensors):
        assert len(tensors) > 0
        self.tensors = tensors
        first_size = len(tensors[0].data)  # Size of first dimension
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"Tensor size mismatch in TensorDataset\n"
                    f"  ❌ Tensor 0 has {first_size} samples, but Tensor {i} has {len(tensor.data)} samples\n"
                    f"  💡 All tensors must have the same size in their first dimension (the sample dimension)\n"
                    f"  🔧 Check your data: features.shape[0] should equal labels.shape[0]\n"
                    f"     Example fix: labels = labels[:{first_size}] or features = features[:{len(tensor.data)}]"
                )
            
    def __len__(self):
        return len(self.tensors[0].data)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)
    
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_batch(batch)

    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        if len(batch) == 0:
            return ()
        num_tensors = len(batch[0])
        batched_tensors = []
        for tensor_idx in range(num_tensors):
            tensor_list = [sample[tensor_idx].data for sample in batch]
            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))
        return tuple(batched_tensors)
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Invalid flip probability: {p}\n"
                f"  ❌ p must be between 0.0 and 1.0\n"
                f"  💡 p is the probability of flipping the image horizontally (p=0.5 means 50% chance)\n"
                f"  🔧 Common values: p=0.0 (never flip), p=0.5 (standard), p=1.0 (always flip)"
            )
        self.p = p
    def __call__(self, x):
        if np.random.random() < self.p:
            is_tensor = isinstance(x, Tensor)
            data = x.data if is_tensor else x
            if data.ndim == 2:
                axis = -1
            elif data.ndim >= 3:
                if data.shape[-1] <= 4:
                    axis = -2
                elif data.shape[-3] <= 4:
                    axis = -1
                else:
                    axis = -1
            else:
                raise ValueError(
                    f"RandomHorizontalFlip requires at least 2D input\n"
                    f"  ❌ Got {data.ndim}D input with shape {data.shape}\n"
                    f"  💡 Images need at least height and width dimensions (H, W) to flip horizontally\n"
                    f"  🔧 Reshape your data: x.reshape(height, width) or x.reshape(1, height, width)"
                )

            flipped = np.flip(data, axis=axis).copy()
            return Tensor(flipped) if is_tensor else flipped
        return x
def _pad_image(data, padding):
    if data.ndim == 2:
        return np.pad(data, padding, mode='constant', constant_values=0)
    elif data.ndim == 3:
        if data.shape[0] <= 4:
            return np.pad(data,
                          ((0, 0), (padding, padding), (padding, padding)),
                          mode='constant', constant_values=0)
        else:
            return np.pad(data,
                          ((padding, padding), (padding, padding), (0, 0)),
                          mode='constant', constant_values=0)
    else:
        raise ValueError(
            f"RandomCrop requires 2D or 3D input\n"
            f"  ❌ Got {data.ndim}D input with shape {data.shape}\n"
            f"  💡 Expected formats: (H, W) for grayscale, (C, H, W) or (H, W, C) for color images\n"
            f"  🔧 Reshape your data:\n"
            f"     - For single grayscale image: x.reshape(height, width)\n"
            f"     - For single color image: x.reshape(channels, height, width)"
        )

def _random_crop_region(padded_h, padded_w, target_h, target_w):
    top = np.random.randint(0, padded_h - target_h + 1)
    left = np.random.randint(0, padded_w - target_w + 1)
    return top, left

class RandomCrop:
    def __init__(self, size, padding=4):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    def __call__(self, x):
        is_tensor = isinstance(x, Tensor)
        data = x.data if is_tensor else x
        target_h, target_w = self.size
        padded = _pad_image(data, self.padding)
        if data.ndim == 2:
            padded_h, padded_w = padded.shape
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[top:top + target_h, left:left + target_w]
        elif data.shape[0] <= 4:
            padded_h, padded_w = padded.shape[1], padded.shape[2]
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[:, top:top + target_h, left:left + target_w]
        else:
            padded_h, padded_w = padded.shape[0], padded.shape[1]
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[top:top + target_h, left:left + target_w, :]

        return Tensor(cropped) if is_tensor else cropped
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x