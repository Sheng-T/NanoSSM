import traceback
import sys
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
from contextlib import ExitStack
from NanoSSM.tools.cli.train_utils import device_list
from NanoSSM.tools.cli.infer_utils import (
    load_infer_model,
    summarize_sites_multiprocess,
    write_prob,
)

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
    parser.add_argument(
        "--no_amp", action="store_true", default=False, help="disable amp"
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--model",
        help="path to pretrain model",
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--seed", default=47, type=int)
    parser.add_argument("--motif", type=str, default="m6A")
    parser.add_argument(
        "--hparams", default=None, type=str, help="path to hparams.yaml"
    )
    parser.add_argument("--type", default="site")
    parser.add_argument("--output_dir", default="infer/", help="result .bed file dir")
    parser.add_argument("--output_filename", default="", help="output file name")
    parser.add_argument(
        "--overwrite", action="store_true", default=False, help="overwrite output file"
    )
    parser.add_argument(
        "--print_read",
        action="store_true",
        default=False,
        help="print read-level to output file",
    )
    parser.add_argument(
        "--norm_path",
        type=str,
        default=DEFAULT_NORM_PATH,
        help="path to normalization statistics directory",
    )
    parser.add_argument("--max_reads", default=1024, type=int, help="max reads per site during inference")

    return parser

def get_output_file(output_dir, output_type, overwrite=False, filename=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if output_type == "site":
        if filename == "":
            filename = "infer_site_prob.bed"
    elif output_type == "print_read":
        if filename == "":
            filename = "infer_site_prob_read.bed"
        else:
            filename = filename.split(".")[0] + "_read.bed"
    out_file = os.path.join(output_dir, filename)
    if os.path.exists(out_file):
        if not overwrite:
            common_log(f"> {out_file} already exists, please delete it first")
            raise FileExistsError
        os.remove(out_file)
    return out_file

def main(args):
    start_time = time.time()

    HEADER_MODKIT_FORMAT = "\t".join(
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
    HEADER_PROBS_READ = "\t".join(["chrom", "start_position", "read_id", "prob"])

    output_file_name = get_output_file(
        args.output_dir, "site", args.overwrite, filename=args.output_filename
    )
    if args.print_read:
        output_file_read_name = get_output_file(
            args.output_dir, "print_read", args.overwrite, filename=args.output_filename
        )
    try:
        gpus = args.device if args.accelerator == "gpu" else None

        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            device = "cuda:0"
        else:
            device = "cpu"

        model, hparams = load_infer_model(args.model, args.hparams, device, gpus)
        model.eval()

        data_loader = get_site_dataloader(
            args.data_path,
            args.info_path,
            args.batch_size,
            args.num_workers,
            norm_path=args.norm_path,
            max_reads=args.max_reads
        )

        with ExitStack() as manager:
            output_file = manager.enter_context(open(output_file_name, "w"))
            output_file.write(f"{HEADER_MODKIT_FORMAT}\n")

            if args.print_read:
                output_file_read = manager.enter_context(
                    open(output_file_read_name, "w")
                )
                output_file_read.write(f"{HEADER_PROBS_READ}\n")

            manager.enter_context(torch.no_grad())
            pbar = manager.enter_context(tqdm())
            for batch in data_loader:
                out = []
                out_pure = []
                all_seqs = (
                    torch.cat([b["seq"] for b in batch], dim=0)
                    .contiguous()
                    .to(device)
                )
                all_signals = (
                    torch.cat([b["signal"] for b in batch], dim=0)
                    .contiguous()
                    .to(device)
                    if batch[0]["signal"] is not None
                    else None
                )
                all_stats = (
                    torch.cat([b["stat"] for b in batch], dim=0)
                    .contiguous()
                    .to(device)
                )

                infos = [b["info"] for b in batch]
                net = (
                    model.module
                    if isinstance(model, torch.nn.DataParallel)
                    else model
                )
                preds, final_alignment_features = net.forward_site(
                    all_seqs, all_signals, all_stats
                )

                offset = 0
                for site_data in batch:
                    n_reads = site_data["seq"].shape[0]
                    site_features = final_alignment_features[offset : offset + n_reads]
                    site_reads_logit = preds[offset : offset + n_reads]
                    offset += n_reads

                    if args.print_read:
                        read_probs = torch.sigmoid(site_reads_logit).view(-1)
                        out_pure.append(read_probs.cpu().tolist())

                    read_probs = torch.sigmoid(site_reads_logit)

                    # site_features_interacted = net.site_interaction(
                    #     site_features.unsqueeze(0)
                    # )
                    if n_reads > 512:
                        net.site_interaction.cpu()
                        site_features_interacted = net.site_interaction(
                            site_features.unsqueeze(0).cpu().float()
                        ).to(device)
                        net.site_interaction.to(device)
                    else:
                        site_features_interacted = net.site_interaction(
                            site_features.unsqueeze(0)
                        )
                    site_features = site_features_interacted.squeeze(0)

                    p_feat = net.prob_projection(site_reads_logit.unsqueeze(-1))
                    f_norm = torch.norm(site_features, dim=-1, keepdim=True)
                    n_feat = net.norm_projection(f_norm)
                    feat_for_agg = torch.cat([site_features, p_feat, n_feat], dim=-1)

                    aggregated_feature, weights = net.aggregator(
                        feat_for_agg, return_weights=True
                    )

                    mean_val = site_reads_logit.mean().unsqueeze(0)
                    std_val = (
                        site_reads_logit.std().unsqueeze(0)
                        if n_reads > 1
                        else torch.tensor([0.0], device=device)
                    )
                    # q25, q50, q75 = torch.quantile(
                    #     site_reads_logit,
                    #     torch.tensor([0.25, 0.5, 0.75], device=device),
                    # )
                    q25, q50, q75 = torch.quantile(site_reads_logit.float().cpu(),torch.tensor([0.25, 0.5, 0.75])).to(device)

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
                    )
                    stat_features = stat_features.detach().unsqueeze(0)
                    combined_feat = torch.cat([aggregated_feature, stat_features], dim=-1)

                    delta_logit = (
                        net.final_site_predictor(combined_feat).view(-1).clamp(-5, 5)
                    )
                    weighted_base_ratio = torch.sum(weights * read_probs)
                    base_ratio = weighted_base_ratio.clamp(1e-5, 1 - 1e-5)
                    base_logit = torch.logit(base_ratio.clamp(1e-5, 1 - 1e-5))
                    final_pred = torch.sigmoid(base_logit + delta_logit)
                    out.append(final_pred.cpu().item())

                for i, info in enumerate(infos):
                    write_prob(output_file, info, out[i], type="site", read_id=-1)
                    if args.print_read:
                        for r_idx, prob in enumerate(out_pure[i]):
                            write_prob(
                                output_file_read,
                                info,
                                prob,
                                type="print_read",
                                read_id=r_idx,
                            )
                    pbar.update(1)

    except Exception as e:
        stack_trace = traceback.format_exc()
        common_log(
            f"> An error occurred during processing: {e}, stack trace: {stack_trace}\n"
        )
    finally:
        common_log(f"=======================================================\n")
        common_log(f"> Finish processing data, saved to: {output_file_name}\n")
        common_log(f"> Elapsed time: {time.time() - start_time:.2f} seconds\n")

if __name__ == "__main__":
    main(argparser().parse_args())
