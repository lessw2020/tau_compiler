# (c) Meta Platforms Inc.

# models we'll test:
# timm_vision_transformer_large
# hf_T5_large
# maybe - DebertaV2ForMaskedLM

from copy import deepcopy
from functools import wraps
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch.distributed._spmd.api import (
    COMPILED_OBJECT_KEY,
    Override,
    Schema,
    SPMD,
    compile,
)
from torch.distributed._spmd.comm_tensor import CommTensor
from torch.distributed._tensor import DeviceMesh, Replicate
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)
import os

import timm
from config.vit_config import train_config


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group()


def cleanup():
    dist.destroy_process_group()


def compiler_main():
    cfg = train_config()
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")
        print(f"--> running with these defaults {cfg}")
        # time_of_run = get_date_of_run()

    # setup_tasks(rank, world_size, cfg)

    setup()
    print(f"hello")
    model = timm.create_model("vit_large_patch14_224", pretrained=False)

    cleanup()


if __name__ == "__main__":
    compiler_main()
