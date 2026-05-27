import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd
import torch
import yaml
from torch.nn import DataParallel

from NanoSSM.models.mamba.model import MambaModel
from NanoSSM.tools.common.common_utils import common_log

def load_model_train(path, model):
    model_data = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(model_data["state_dict"])
    return model

def load_model(path: str, hparams_file: str, device: str = "cpu", type: str = "site"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if hparams_file is None:
            hparams_file = os.path.join(
                os.path.dirname(path), "lightning_logs", "version_1", "hparams.yaml"
            )

        with open(hparams_file, "r") as f:
            hparams = yaml.safe_load(f)
        data_params = load_model_params(hparams)

        common_log(f"Loaded model with {data_params}")

        model = MambaModel(**data_params, type=type)

        checkpoint_path = (
            os.path.join(path, "last.ckpt") if os.path.isdir(path) else path
        )
        common_log(f"Loading model from {checkpoint_path}...\n")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model, hparams

def load_model_params(hparams):
    training_config = hparams.get("training", {})

    keys = [
        "signal_dim",
        "sequence_dim",
        "features",
        "hidden_dim",
        "output_dim",
        "encoder_num_layers",
        "decoder_num_layers",
        "num_mamba_layers",
        "dropout",
        "kmer",
        "learning_rate",
        "wd",
    ]

    dic = {key: training_config.get(key) for key in keys}
    if dic["output_dim"] is None:
        dic["output_dim"] = 1

    return dic

def load_infer_model(path: str, hparams_file: str, device: str, gpus: List[int]):
    with warnings.catch_warnings():
        model, hparams = load_model(path, hparams_file, device)

        model = model.to(device)

        if gpus is not None and len(gpus) > 1:
            model = DataParallel(model, device_ids=gpus)

    return model, hparams


def get_mod_pos_ref(strand, ref_start_position, ref_end_position, mod_pos_read):
    if strand == "+":
        return ref_start_position + mod_pos_read
    else:

        return ref_end_position - mod_pos_read - 1

def process_chunk(chunk, chunk_id, temp_dir, motif):
    grouped = chunk.groupby(["chrom", "mod_pos_ref", "strand"], as_index=False).agg(
        score=("read_id", "count"), percent_modified=("prob", "mean")
    )
    grouped["start_position"] = grouped["mod_pos_ref"]
    grouped["end_position"] = grouped["start_position"] + 1
    grouped["motif"] = motif
    grouped["color"] = "0,0,0"
    grouped["N_valid_cov"] = grouped["score"]

    out_cols = [
        "chrom",
        "start_position",
        "end_position",
        "motif",
        "score",
        "strand",
        "start_position",
        "end_position",
        "color",
        "N_valid_cov",
        "percent_modified",
    ]
    grouped = grouped[out_cols]

    temp_file = os.path.join(temp_dir, f"chunk_{chunk_id}.csv")
    grouped.to_csv(temp_file, index=False, header=False)
    return temp_file

def summarize_sites_multiprocess(
    reads_prob_file: str,
    motif: str,
    sites_prob_file: str,
    chunksize: int = 1_000_000,
    num_workers: int = 4,
):
    temp_dir = "temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    dtype = {
        "chrom": "category",
        "mod_pos_ref": "int32",
        "strand": "category",
        "read_id": "category",
        "prob": "float32",
    }

    temp_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        reader = pd.read_csv(
            reads_prob_file,
            usecols=["chrom", "mod_pos_ref", "strand", "read_id", "prob"],
            dtype=dtype,
            chunksize=chunksize,
        )

        for chunk_id, chunk in enumerate(reader):
            future = executor.submit(process_chunk, chunk, chunk_id, temp_dir, motif)
            futures.append(future)

        for future in as_completed(futures):
            temp_file = future.result()
            temp_files.append(temp_file)

    dfs = [
        pd.read_csv(
            f,
            header=None,
            names=[
                "chrom",
                "start_position",
                "end_position",
                "motif",
                "score",
                "strand",
                "start_position_dup",
                "end_position_dup",
                "color",
                "N_valid_cov",
                "percent_modified",
            ],
        )
        for f in temp_files
    ]

    full_df = pd.concat(dfs, ignore_index=True)

    final = full_df.groupby(["chrom", "start_position", "strand"], as_index=False).agg(
        {
            "score": "sum",
            "percent_modified": "mean",
            "motif": "first",
            "end_position": "first",
            "color": "first",
            "N_valid_cov": "sum",
        }
    )

    final = final[
        [
            "chrom",
            "start_position",
            "end_position",
            "motif",
            "score",
            "strand",
            "start_position",
            "end_position",
            "color",
            "N_valid_cov",
            "percent_modified",
        ]
    ]

    final.to_csv(sites_prob_file, index=False, header=False)

    for f in temp_files:
        os.remove(f)
    os.rmdir(temp_dir)

    return final


def write_prob(output_file, infos, prob, type, read_id=0):

    if type == "read":
        for i in range(len(infos["transcript_id"])):
            chrom = infos["transcript_id"][i]
            start_position = infos["position"][i]
            motif = infos["motif"][i]
            read_id = infos["read_id"][i]
            output_file.write(
                f"{chrom}\t{read_id}\t{start_position}\t{motif}\t{prob[i].item():.4f}\n"
            )

    elif type == "site":
        chrom = infos["transcript_id"]
        start_position = infos["position"]
        end_position = int(infos["position"]) + 1
        motif = infos["motif"]
        num = infos["num"]
        output_file.write(
            f"{chrom}\t{start_position}\t{end_position}\t{motif}\t{num}\t+"
            f"\t{start_position}\t{end_position}\t0.0.0\t{num}\t{prob:.4f}\n"
        )

    else:
        chrom = infos["transcript_id"]
        start_position = infos["position"]
        output_file.write(f"{chrom}\t{start_position}\t{str(read_id)}\t{prob:.4f}\n")
