import inspect
import shutil
import traceback

import lightning as pl
import torch
from lightning.pytorch.profilers import SimpleProfiler

import sys
import os
import time

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from NanoSSM.tools.cli.infer_utils import load_model_train

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from NanoSSM.data.site_level.data_module import MultiDataModule as site_DataModule

from NanoSSM.tools.cli.train_utils import (
    set_trainer_params,
    get_shapes_from_files,
    device_list,
    format_save_dir,
    setup_seed,
)

from NanoSSM.data.site_level.load_data import get_test_loader as site_get_test_loader
from NanoSSM.models.mamba.model import MambaModel
from NanoSSM.tools.common.common_utils import common_log

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, add_help=True
    )
    parser.add_argument("--accelerator", default="gpu", help="cpu|gpu|tpu")
    parser.add_argument(
        "--device", default="0", help="auto | 1 | 1,2,3", type=device_list
    )
    parser.add_argument(
        "--no_amp", action="store_true", default=False, help="disable amp"
    )
    parser.add_argument(
        "--freeze", action="store_true", default=False, help="freeze encoder"
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=False, help="benchmark"
    )
    parser.add_argument(
        "--path",
        default="data/hela",
        help="path to data folders, split multiple paths by ','",
    )
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--epochs", default=10, help="epoch", type=int)
    parser.add_argument("--seed", default=47, type=int)
    parser.add_argument(
        "--save_every_epoch", default=1, type=int, help="save every epoch"
    )
    parser.add_argument("--patience", default=20, type=int, help="early stop patience")
    parser.add_argument(
        "--check_val_every_n_epoch", default=5, type=int, help="check val every n epoch"
    )

    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--log_every_n_step", default=500, type=int)

    parser.add_argument("--type", default="site", help="read | site")

    parser.add_argument("--learning_rate", "--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=0.00001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_mamba_layers", default=2, type=int)

    parser.add_argument("--hidden_dim", default=None, type=int)
    parser.add_argument("--features", default=256, type=int)
    parser.add_argument("--encoder_num_layers", default=1, type=int)
    parser.add_argument("--decoder_num_layers", default=1, type=int)
    parser.add_argument("--kmer", default=31, help="kmer", type=int)
    parser.add_argument(
        "--signal_dim", default=20, help="max signal per base", type=int
    )
    parser.add_argument("--sequence_dim", default=4, help="seq feature num", type=int)
    parser.add_argument("--feature_dim", default=5, help="feature dim", type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--save_dir", default="../../result", help="result save dir")
    parser.add_argument(
        "--model", help="path to pretrain model(xxx.ckpt)", default=None
    )
    parser.add_argument(
        "--norm_path", type=str, default=None, help="overwrite output file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume full training state from checkpoint",
    )

    return parser

def main(args):
    start_time = time.time()
    try:
        setup_seed(args.seed)

        trainer = pl.Trainer(profiler=None, **set_trainer_params(args))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.save_dir, f"{timestamp}_model_logic.py")

        try:

            model_code = inspect.getsource(MambaModel)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"# Auto-backed up at {timestamp}\n")
                f.write(f"# This file contains the architecture of NanoSSM class\n\n")
                f.write(model_code)

            print(f"[Info] Successfully backed up model logic to {save_path}")
        except Exception as e:
            print(f"[Warning] Failed to backup model code via inspect: {e}")

        model = MambaModel(
            signal_dim=args.signal_dim,
            sequence_dim=args.sequence_dim,
            features=args.features,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            wd=args.wd,
            dropout=args.dropout,
            num_mamba_layers=args.num_mamba_layers,
            encoder_num_layers=args.encoder_num_layers,
            decoder_num_layers=args.decoder_num_layers,
            feature_dim=args.feature_dim,
            kmer=args.kmer,
            type=args.type,
            is_finetune=True if args.model else False,
            test_save_path=args.save_dir,
        )

        paths = [p.strip() for p in args.path.split(",")]
        datamodule = site_DataModule(
            paths, args.batch_size, args.num_workers, args.norm_path
        )

        ckpt_path = None
        if args.model:
            common_log(f"> Loading model from {args.model}")
            if args.resume:

                ckpt_path = args.model
            else:

                model = load_model_train(args.model, model)

        if args.freeze:
            common_log("> [freeze] freezing low-level Mamba encoders")
            model.freeze_backbone()

        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

        trainer.test(
            model,
            dataloaders=site_get_test_loader(
                paths, args.batch_size, args.num_workers, norm_path=args.norm_path
            ),
            verbose=True,
            ckpt_path="best",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        common_log(
            f"> An error occurred during processing: {e}, stack trace: {stack_trace}\n"
        )
    finally:
        common_log(f"=======================================================\n")
        common_log(f"> Elapsed time: {time.time() - start_time:.2f} seconds\n")

if __name__ == "__main__":
    main(argparser().parse_args())
