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

# benchmarks/dynamo/huggingface.py
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Config,
)

# from config.vit_config import train_config
from config.t5_config import train_config
from torch.utils.data import DistributedSampler
import config.t5_config as config
import logging
import performance

from typing import Tuple

import colorama
from colorama import Fore

colorama.init(autoreset=True)  # reset after every line


logger: logging.Logger = logging.getLogger("main_training")
from torch.utils._pytree import PyTree, tree_flatten, tree_map, tree_map_only

pytree = torch.utils._pytree

# typ = type(pytree)
# fields = getattr(typ, "_fields", None)
from dataclasses import fields
from transformers.modeling_outputs import Seq2SeqLMOutput


def _flatten_fn(model_input: Seq2SeqLMOutput) -> Tuple[List[Any], pytree.Context]:
    return [getattr(model_input, f.name) for f in fields(model_input)], type(
        model_input
    )


def _unflatten_fn(
    model_input: List[Any],
    context: pytree.Context,
) -> Seq2SeqLMOutput:
    return context(*model_input)


pytree._register_pytree_node(Seq2SeqLMOutput, _flatten_fn, _unflatten_fn)


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")
    torch.cuda.manual_seed(torch.distributed.get_rank())
    torch.manual_seed(torch.distributed.get_rank())


def cleanup(rank):
    dist.barrier()
    logger.info(f"Goodbye from rank {rank}")
    dist.destroy_process_group()


def compiler_main():
    cfg = train_config()

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")
        print(f"--> running with these defaults {cfg}")
        # time_of_run = get_date_of_run()

    setup()
    # setup_tasks(rank, world_size, cfg)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    logger.info(f"hello - starting model building...")

    # setup memory tracking for perf
    if local_rank == 0:
        memmax = performance.Memory_Maximizer()
    else:
        memmax = None

    # ---- Model building ------
    # model = timm.create_model("vit_large_patch14_224", pretrained=False)
    model = config.build_model("t5")

    if local_rank == 0:
        logger.info(f" --> {cfg.model_name} built.")
        num_params = (sum(p.numel() for p in model.parameters())) / 1e6
        logger.info(f" built model with {num_params:0.2f}M params")

    _device = "cuda"
    model.to(_device)
    # model = DDP(model)
    model = SPMD(
        model,
        schema=Schema(
            mesh=DeviceMesh(_device, torch.arange(world_size)),
            placements=[Replicate()],
        ),
        # input_schemas=kwargs["inp_schemas"] if "inp_schemas" in kwargs else None,
    )

    # short cut - run fwd bwd directly

    # if rank == 0:
    #    logger.warning(f"spmd model {model.parameters()=}")

    # ---- optimizer ---------
    use_fused_optimizer = cfg.use_fused_optimizer

    optimizer = None
    """torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=use_fused_optimizer,
    )
    if rank == 0:
        logger.warning(
            f"Running with AdamW optimizer, with fusion set to {use_fused_optimizer}"
        )
    """
    # ----- dataset ---------
    dataset = config.get_dataset()
    train_sampler = DistributedSampler(
        dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True
    )

    # data loader -------------
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size_training,
        num_workers=cfg.num_workers_dataloader,
        pin_memory=False,
        sampler=train_sampler,
    )

    # ---- training loop ------

    # memory and timing tracking
    if local_rank == 0:
        memmax.start()
        # torch.cuda.reset_peak_memory_stats()
        tracking_duration = []
    else:
        tracking_duration = None

    torch_profiler = None

    total_steps = None
    if cfg.total_steps_to_run:
        total_steps = cfg.total_steps_to_run - 1  # fix off by one for step count

    for i in range(cfg.num_epochs):
        if rank == 0:
            print(f"Epoch: {i} starting...")

        config.train(
            model,
            data_loader,
            torch_profiler,
            optimizer,
            memmax,
            local_rank,
            tracking_duration,
            total_steps,
            use_synthetic_data=cfg.use_synthetic_data,
        )
        if cfg.total_steps_to_run is not None:
            break

    # memory summary
    if local_rank == 0:
        # memory monitor
        memmax.stop()  # stop and display info
        # print(f"{tracking_duration=}, {cfg.total_steps_to_run=}")
        """if _stats:
        total_loss_curve = _stats["loss"]
        total_acc_curve = _stats["accuracy"]
        for loss, acc in zip(total_loss_curve, total_acc_curve):
            print(f"{loss=}, {acc=}")

        best_val_acc = 100 * float(max(total_acc_curve))
        print(Fore.GREEN + f"\n--> Highest Val Accuracy =  {best_val_acc}\n")
        """
        if cfg.total_steps_to_run is not None:
            warmup_steps = cfg.warmup_steps
            iters_to_avg = tracking_duration[warmup_steps:]

            stable_sum = sum(iters_to_avg)
            # print(f"len iters_to_avg = {len(iters_to_avg)}")
            total_steps_measured = cfg.total_steps_to_run - warmup_steps
            stable_avg = stable_sum / total_steps_measured
            stable_avg = round(stable_avg, 4)
            print(
                Fore.GREEN
                + f"\n--> Step avg speed based on {total_steps_measured} steps: {stable_avg} seconds"
            )
        # print(f"This was run with TensorParallel? = {cfg.use_tp}")
        print(f"Batch size used = {cfg.batch_size_training}\n")

        print(Fore.LIGHTBLUE_EX + f"\n--> Model Size =  {num_params} M Params\n")
        if cfg.print_memory_summary:
            print(
                f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}"
            )

    cleanup(rank)


if __name__ == "__main__":
    compiler_main()
