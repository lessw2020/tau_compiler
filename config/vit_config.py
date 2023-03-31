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
