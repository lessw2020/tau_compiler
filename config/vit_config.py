import time
from dataclasses import dataclass
from typing import Tuple
import os

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch import distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models


@dataclass
class train_config:
    seed: int = 2023
    model_name: str = "vit_large_patch14_224"

    # image size
    image_size: int = 224

    num_classes: int = 1000

    # use synthetic data
    use_synthetic_data: bool = True


def get_dataset():
    """generate both train and val dataset"""
    cfg = train_config()
    if cfg.use_synthetic_data:
        image_size = cfg.image_size

        return GeneratedDataset(image_size=cfg.image_size, num_classes=cfg.num_classes)


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        image_size = kwargs.get("image_size", 224)
        self._input_shape = kwargs.get("input_shape", [3, image_size, image_size])
        self._input_type = kwargs.get("input_type", torch.float32)
        self._len = kwargs.get("len", 1000000)
        self._num_classes = kwargs.get("num_classes", 1000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_image = torch.randn(self._input_shape, dtype=self._input_type)
        label = torch.tensor(data=[index % self._num_classes], dtype=torch.int64)
        return rand_image, label
