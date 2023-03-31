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

    # training
    batch_size_training = 20
    total_steps_to_run: int = 5
    num_epochs: int = 2

    num_workers_dataloader = 2
    print_memory_summary: bool = False

    label_smoothing_value: float = 0.0

    # optimizer
    use_fused_optimizer: bool = True
    learning_rate: float = 8e-4
    weight_decay: float = 0.002

    # monitoring
    log_every: int = 1
    warmup_steps: int = 2


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


def train(
    model,
    data_loader,
    torch_profiler,
    optimizer,
    memmax,
    local_rank,
    tracking_duration,
    total_steps_to_run,
    use_synthetic_data=True,
    use_label_singular=False,
):
    cfg = train_config()
    label_smoothing_amount = cfg.label_smoothing_value
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing_amount)
    t0 = time.perf_counter()
    for batch_index, (batch) in enumerate(data_loader, start=1):
        # print(f"{batch=}")
        if use_synthetic_data:
            inputs, targets = batch
        elif use_label_singular:
            inputs = batch["pixel_values"]
            targets = batch["label"]

        else:
            inputs = batch["pixel_values"]
            targets = batch["labels"]

        inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
            targets.to(torch.cuda.current_device()), -1
        )
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        if optimizer:
            optimizer.step()

        # update durations and memory tracking
        if local_rank == 0:
            mini_batch_time = time.perf_counter() - t0
            tracking_duration.append(mini_batch_time)
            if memmax:
                memmax.update()

        if batch_index % cfg.log_every == 0 and torch.distributed.get_rank() == 0:
            print(
                f"step: {batch_index}: time taken for the last {cfg.log_every} steps is {mini_batch_time}, loss is {loss}"
            )

        # reset timer
        t0 = time.perf_counter()
        if torch_profiler is not None:
            torch_profiler.step()
        if total_steps_to_run is not None and batch_index > total_steps_to_run:
            break
