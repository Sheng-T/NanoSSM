import warnings

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import pandas as pd
import sys
import os

import cProfile
import pstats

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from NanoSSM.tools.data.site_process_utils import (
    parallel_index,
    parallel_preprocess_tx as site_parallel_preprocess_tx,
)


NUM_NEIGHBORING_FEATURES = 1

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, add_help=False
    )

    parser.add_argument(
        "--eventalign",
        help="eventalign filepath, the output from nanopolish.", required=True
    )

    parser.add_argument(
        "--out_dir",
        help="output directory.", required=True
    )

    parser.add_argument(
        "--n_processes", help="number of processes to run.", default=1, type=int
    )
    parser.add_argument(
        "--chunk_size",
        help="number of lines from nanopolish eventalign.txt for processing.",
        default=1000000,
        type=int,
    )
    parser.add_argument(
        "--readcount_min", help="minimum read counts per gene", default=20, type=int
    )
    parser.add_argument(
        "--readcount_max", help="maximum read counts per gene", default=1000, type=int
    )
    parser.add_argument(
        "--min_segment_count",
        help="minimum read counts per candidate segment.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--max_signal_len", help="max signal len.", default=15, type=int
    )
    parser.add_argument(
        "--skip_index",
        help="with this argument the program will skip indexing eventalign.txt first.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--n_neighbors",
        help="number of neighboring features to extract.",
        default=NUM_NEIGHBORING_FEATURES,
        type=int,
    )
    parser.add_argument(
        "--compress",
        help="number of neighboring features to extract.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--split", help="split train/val/test", default=False, action="store_true"
    )
    parser.add_argument(
        "--standard_file",
        help="The standard file conforms to the bedmethyl format",
        default=None,
        type=str,
    )
    parser.add_argument("--motif", help="m6a | pseu | a", default="m6a", type=str)
    parser.add_argument("--mode", help="site | read", default="site", type=str)
    parser.add_argument("--train_ratio", default=0.8, help="train ratio", type=float)
    parser.add_argument("--val_ratio", default=0.1, help="val ratio", type=float)
    parser.add_argument("--seed", default=42, help="seed", type=int)
    return parser

def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    if not args.skip_index:
        parallel_index(args.eventalign, args.chunk_size, args.out_dir, args.n_processes)

    motif = args.motif.lower()
    if motif not in ["m6a", "pseu", "a"]:
        raise ValueError("Only support motif m6A, A or pseU")
    if args.mode == "site":
            site_parallel_preprocess_tx(
                args.eventalign,
                args.out_dir,
                args.n_processes,
                args.readcount_min,
                args.readcount_max,
                args.n_neighbors,
                args.min_segment_count,
                args.compress,
                args.max_signal_len,
                args.seed,
                motif,
            )
    else:
        raise ValueError("Only support mode site or read")

if __name__ == "__main__":
    main(argparser().parse_args())
