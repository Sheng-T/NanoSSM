import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from NanoSSM.tools.common.common_utils import common_log
from NanoSSM.tools.data.site_process_utils import (
    index_data_before_process,
    setup_seed,
)

def prepare_data(
    refs_file,
    standard_file,
    signal_file,
    f5c_file,
    out_path,
    motif: str,
    max_signal_len,
    kmer,
    train_ratio=0.8,
    val_ratio=0.1,
    recursive=True,
    n_proc=2,
    index_dir=None,
    no_split=False,
    skip_index=False,
    seed=42,
):
    setup_seed(seed)

    index_dir = index_dir or out_path

    motif = motif.lower()

    train_index_path = (
        os.path.join(index_dir, "train", "index.lmdb")
        if not no_split
        else os.path.join(index_dir, "index.lmdb")
    )
    if not skip_index and not os.path.exists(train_index_path):
        common_log("Start indexing data before processing...")
        index_data_before_process(
            refs_file=refs_file,
            standard_file=standard_file,
            signal_file=signal_file,
            f5c_file=f5c_file,
            out_path=out_path,
            motif=motif,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            no_split=no_split,
            seed=seed,
            kmer=kmer,
            num_workers=n_proc,
            test_transcripts=None,
            max_signal_len=max_signal_len,
        )

    else:
        common_log("Skip indexing data before processing...")

    common_log(f"Start to process data ...")
