import traceback
import sys
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from NanoSSM.data.site_level.load_data import get_test_loader as site_get_test_loader
from NanoSSM.models.mamba.model import MambaModel
from NanoSSM.tools.common.common_utils import common_log

from NanoSSM.tools.cli.train_utils import device_list
from NanoSSM.tools.cli.infer_utils import load_model, load_infer_model

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, add_help=True
    )
    parser.add_argument("--accelerator", default="gpu", help="cpu|gpu|tpu")
    parser.add_argument(
        "--device", default="0,1", help="auto | 1 | 1,2,3", type=device_list
    )
    parser.add_argument(
        "--no_amp", action="store_true", default=False, help="disable amp"
    )

    parser.add_argument("--test_file", help="h5 data")
    parser.add_argument("--model", help="path to pretrain model(xxx.ckpt)")
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--seed", default=25, type=int)

    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--hparams", default=None, type=str, help="path to hparams.yaml"
    )
    parser.add_argument(
        "--save_dir", default="result/infer/predictions.tsv", help="result .tsv file"
    )

    parser.add_argument("--type", default="site", help="read | site")
    parser.add_argument("--max_read", default=512, type=int, help="max reads per site during inference")


    return parser

def main(args):
    start_time = time.time()
    try:

        model, hparams = load_model(
            args.model, args.hparams, device="cpu", type=args.type
        )
        model.eval()
        logger = TensorBoardLogger(
            save_dir=args.save_dir, version=1, name="lightning_logs_test"
        )
        trainer = pl.Trainer(
            logger=logger, accelerator=args.accelerator, devices=args.device
        )

        trainer.test(
            model,
            dataloaders=site_get_test_loader(
                args.test_file, args.batch_size, args.num_workers, max_read=args.max_read
            ),
            ckpt_path=args.model,
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
