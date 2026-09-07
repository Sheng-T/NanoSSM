#!/usr/bin/env python3
import traceback
import sys
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from contextlib import ExitStack

import torch
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.ckpt")
)
DEFAULT_NORM_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "norm")
)

from NanoSSM.data.site_level.load_data import get_site_dataloader
from NanoSSM.tools.common.common_utils import common_log
from NanoSSM.tools.cli.train_utils import device_list
from NanoSSM.tools.cli.infer_utils import load_infer_model, write_prob


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, add_help=True
    )
    parser.add_argument("--accelerator", default="gpu", help="cpu|gpu|tpu")
    parser.add_argument("--data_path", help="data.json")
    parser.add_argument("--info_path", help="data.info", required=True)
    parser.add_argument(
        "--device", default="0", help="auto | 1 | 1,2,3", type=device_list
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--hparams", default=None, type=str)
    parser.add_argument("--output_dir", default="infer/")
    parser.add_argument("--output_filename", default="infer_site_prob.bed")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--norm_path", type=str, default=DEFAULT_NORM_PATH)
    parser.add_argument("--max_reads", default=512, type=int)
    return parser


def get_output_file(output_dir, filename, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, filename)
    if os.path.exists(out_file):
        if not overwrite:
            raise FileExistsError(out_file)
        os.remove(out_file)
    return out_file


def predict_one_site(net, site_reads_logit, site_features, device):
    n_reads = int(site_reads_logit.numel())
    read_probs = torch.sigmoid(site_reads_logit)

    if n_reads > 512:
        net.site_interaction.cpu()
        interacted = net.site_interaction(
            site_features.unsqueeze(0).cpu().float()
        ).to(device)
        net.site_interaction.to(device)
    else:
        interacted = net.site_interaction(site_features.unsqueeze(0))

    site_features = interacted.squeeze(0)

    p_feat = net.prob_projection(site_reads_logit.unsqueeze(-1))
    f_norm = torch.norm(site_features, dim=-1, keepdim=True)
    n_feat = net.norm_projection(f_norm)
    feat_for_agg = torch.cat([site_features, p_feat, n_feat], dim=-1)

    aggregated_feature, weights = net.aggregator(
        feat_for_agg, return_weights=True
    )

    logit_f = site_reads_logit.float()
    mean_val = logit_f.mean().unsqueeze(0)
    std_val = (
        logit_f.std().unsqueeze(0)
        if n_reads > 1
        else torch.tensor([0.0], device=device)
    )
    q25, q50, q75 = torch.quantile(
        logit_f.cpu(), torch.tensor([0.25, 0.50, 0.75])
    ).to(device)
    log_n_reads = torch.log10(
        torch.tensor([float(n_reads)], device=device)
    )

    stat_features = torch.cat(
        [
            mean_val,
            std_val,
            q25.unsqueeze(0),
            q50.unsqueeze(0),
            q75.unsqueeze(0),
            log_n_reads,
        ]
    ).unsqueeze(0)

    combined_feat = torch.cat([aggregated_feature, stat_features], dim=-1)

    raw_delta = net.final_site_predictor(combined_feat).view(-1)
    delta_scale = float(getattr(net, "delta_scale", 0.5))
    delta_logit = delta_scale * torch.tanh(raw_delta)

    weighted_base_ratio = torch.sum(weights * read_probs)
    base_ratio = weighted_base_ratio.clamp(1e-5, 1.0 - 1e-5)
    final_pred = torch.sigmoid(torch.logit(base_ratio) + delta_logit)

    return final_pred.view(())


def main(args):
    start_time = time.time()

    header = "\t".join(
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
    )

    output_file_name = get_output_file(
        args.output_dir, args.output_filename, args.overwrite
    )

    try:
        gpus = args.device if args.accelerator == "gpu" else None
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            device = "cuda:0"
        else:
            device = "cpu"

        model, hparams = load_infer_model(
            args.model, args.hparams, device, gpus
        )
        model.eval()

        net = (
            model.module
            if isinstance(model, torch.nn.DataParallel)
            else model
        )

        # Fail loudly if an old model implementation is accidentally loaded.
        if not hasattr(net, "delta_scale"):
            raise RuntimeError(
                "Loaded model has no delta_scale attribute. "
                "Make sure NanoSSM.models.mamba.model is the revised "
                "constrained-correction MambaModel."
            )

        data_loader = get_site_dataloader(
            args.data_path,
            args.info_path,
            args.batch_size,
            args.num_workers,
            norm_path=args.norm_path,
            max_reads=args.max_reads,
        )

        with ExitStack() as manager:
            output_file = manager.enter_context(open(output_file_name, "w"))
            output_file.write(header + "\n")
            manager.enter_context(torch.no_grad())
            pbar = manager.enter_context(tqdm())

            for batch in data_loader:
                all_seqs = torch.cat(
                    [b["seq"] for b in batch], dim=0
                ).contiguous().to(device)
                all_signals = (
                    torch.cat([b["signal"] for b in batch], dim=0)
                    .contiguous()
                    .to(device)
                    if batch[0]["signal"] is not None
                    else None
                )
                all_stats = torch.cat(
                    [b["stat"] for b in batch], dim=0
                ).contiguous().to(device)

                infos = [b["info"] for b in batch]

                preds, read_features_all = net.forward_site(
                    all_seqs, all_signals, all_stats
                )

                offset = 0
                out = []
                for site_data in batch:
                    n_reads = int(site_data["seq"].shape[0])
                    site_logits = preds[offset : offset + n_reads]
                    site_features = read_features_all[offset : offset + n_reads]
                    offset += n_reads

                    final_pred = predict_one_site(
                        net, site_logits, site_features, device
                    )
                    out.append(final_pred.cpu().item())

                for i, info in enumerate(infos):
                    write_prob(
                        output_file, info, out[i], type="site", read_id=-1
                    )
                    pbar.update(1)

    except Exception as exc:
        stack_trace = traceback.format_exc()
        common_log(
            f"> An error occurred during processing: {exc}, "
            f"stack trace: {stack_trace}\n"
        )
        raise
    finally:
        common_log("=======================================================\n")
        common_log(f"> Finish processing data, saved to: {output_file_name}\n")
        common_log(f"> Elapsed time: {time.time() - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main(argparser().parse_args())