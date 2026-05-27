import glob
import os
import random
import sys
import time

import numpy as np
import torch

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from NanoSSM.tools.common.common_utils import get_format_time, common_log, WARNING

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def device_list(s):
    return [int(x.strip()) for x in s.split(",")]

def format_save_dir(save_dir):
    save_dir = os.path.expanduser(save_dir)
    _time = get_format_time(time.time(), "%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, _time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def set_model_checkpoint_callback(save_dir, save_every_epoch, type="site", patience=15):

    if type == "site":
        checkpoint_callback = ModelCheckpoint(
            monitor="val_pearson",
            filename="{step}-{val_pearson:.4f}-{val_mse:.4f}-{val_mae:.4f}",
            save_top_k=3,
            dirpath=save_dir,
            mode="max",
            save_last=True,
            every_n_epochs=save_every_epoch,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_pearson", patience=patience, mode="max"
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_auroc",
            filename="{step}-{val_auroc:.5f}-{val_acc:.5f}",
            save_top_k=3,
            dirpath=save_dir,
            mode="max",
            save_last=True,
            every_n_epochs=save_every_epoch,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_auroc", patience=patience, mode="max"
        )

    return checkpoint_callback, early_stop_callback

def set_trainer_params(args):
    argsdict = dict(training=vars(args))
    common_log(f"[argsdict]: {argsdict}\n")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    common_log(f"[save dir]: {args.save_dir}\n")

    logger = TensorBoardLogger(save_dir=args.save_dir, version=1, name="lightning_logs")
    logger.log_hyperparams(argsdict)

    checkpoint_callback, early_stop_callback = set_model_checkpoint_callback(
        args.save_dir, args.save_every_epoch, args.type, args.patience
    )

    trainer_params = {
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "benchmark": args.benchmark,
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback, early_stop_callback],
        "logger": logger,
        "log_every_n_steps": args.log_every_n_step,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
    }

    if args.accelerator == "gpu":
        trainer_params["devices"] = args.device
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if len(args.device) == 1:
            trainer_params["strategy"] = "auto"
        else:
            if args.type == "site":
                trainer_params["strategy"] = DDPStrategy(find_unused_parameters=False)
            else:
                trainer_params["strategy"] = DDPStrategy(find_unused_parameters=True)

    if not args.no_amp:
        trainer_params["precision"] = "bf16-mixed"

        torch.set_float32_matmul_precision("medium")
    else:
        trainer_params["precision"] = "32-true"

        torch.set_float32_matmul_precision("high")

    return trainer_params

def get_shapes_from_files(path, pattern="train/data_0_*.npz"):

    pattern = os.path.join(path, pattern)

    file = glob.glob(pattern)

    if len(file) > 1:
        raise ValueError("Multiple files found. Please check director correct.")

    try:

        data = np.load(file[0], allow_pickle=True)

        if "signal" in data and "sequence" in data:
            signal = data["signal"]
            sequence = data["sequence"]

            signal_dim = signal.shape[-1]
            sequence_dim = sequence.shape[1]
            kmer = sequence.shape[-1]
        else:
            raise ValueError(
                "'signal' and 'sequence' keys not found in the loaded data."
            )
    except Exception as e:
        sys.stderr.write(f"Error loading file {file}: {e}")
        sys.exit()

    return signal_dim, sequence_dim, kmer


