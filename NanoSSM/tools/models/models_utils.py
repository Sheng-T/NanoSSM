import math

import torch
from functools import partial

from torch import nn
from torch.autograd import Function

import pandas as pd
import numpy as np

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class ACT(Function):
    @staticmethod
    def forward(ctx, inputs):
        output = inputs.new(inputs.size())
        output[inputs > 0.5] = 1
        output[inputs <= 0.5] = 0
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_output[input_ > 1.0] = 0
        grad_output[input_ < 0.0] = 0
        return grad_output

class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

def pearson_loss(pred, target, eps=1e-8):
    pred = pred.view(-1)
    target = target.view(-1)
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_diff = pred - pred.mean()
    target_diff = target - target.mean()

    numerator = torch.sum(pred_diff * target_diff)

    denominator = torch.sqrt(torch.sum(pred_diff**2) + eps) * torch.sqrt(
        torch.sum(target_diff**2) + eps
    )

    pearson = numerator / denominator
    return 1.0 - pearson

def ccc_loss(y_pred, y_true):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    mu_x = torch.mean(y_pred)
    mu_y = torch.mean(y_true)

    var_x = torch.var(y_pred)
    var_y = torch.var(y_true)

    cov = torch.mean((y_pred - mu_x) * (y_true - mu_y))

    ccc = (2 * cov) / (var_x + var_y + (mu_x - mu_y) ** 2 + 1e-8)
    return 1 - ccc

class CombinedLoss(nn.Module):
    def __init__(self, beta=0.1, pearson_weight=0.3):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=beta, reduction="mean")
        self.pearson_weight = pearson_weight

    def forward(self, pred, target):
        huber_loss = self.huber(pred, target)
        p_loss = pearson_loss(pred, target)
        return huber_loss + self.pearson_weight * p_loss

CAL_FEATURE_COLUMNS = [
    "pred_ratio",
    "log_coverage",
    "relative_position",
    "neighbor_score",
    "neighbor_max_ratio",
    "neighbor_std",
    "neighbor_mean_diff",
    "log_neighbor_count",
]

class NanoCalibratorFeatureBuilder:

    def __init__(self, decay_lambda=50, neighbor_window=200):

        self.decay_lambda = decay_lambda
        self.neighbor_window = neighbor_window

    def load_prediction(self, pred_path):

        df = pd.read_csv(pred_path, sep="\t")

        df = df.rename(
            columns={
                "chrom": "transcript_id",
                "start_position": "transcript_position",
                "percent_modified": "pred_ratio",
                "N_valid_cov": "coverage",
            }
        )

        df["log_coverage"] = np.log1p(df["coverage"])

        return df

    def load_truth(self, truth_path):

        truth = pd.read_csv(truth_path)

        truth = truth[["transcript_id", "transcript_position", "ratio"]].rename(
            columns={"ratio": "true_ratio"}
        )

        return truth

    def merge_truth(self, df, truth):

        df = df.merge(truth, on=["transcript_id", "transcript_position"], how="inner")

        return df

    def compute_relative_position(self, df):

        df = df.sort_values(["transcript_id", "transcript_position"])

        transcript_length = (
            df.groupby("transcript_id")["transcript_position"]
            .max()
            .rename("transcript_length")
        )

        df = df.merge(transcript_length, on="transcript_id", how="left")

        df["relative_position"] = df["transcript_position"] / (
            df["transcript_length"] + 1
        )

        df.drop(columns=["transcript_length"], inplace=True)

        return df

    def compute_neighbor_features(self, df):

        df = df.sort_values(["transcript_id", "transcript_position"]).reset_index(
            drop=True
        )

        neighbor_score = np.zeros(len(df))
        neighbor_max_ratio = np.zeros(len(df))
        neighbor_std = np.zeros(len(df))
        neighbor_count = np.zeros(len(df))

        start_idx = 0

        for transcript, group in df.groupby("transcript_id"):

            positions = group["transcript_position"].values
            ratios = group["pred_ratio"].values

            n = len(group)

            left = 0
            right = 0

            for i in range(n):

                center = positions[i]

                while positions[left] < center - self.neighbor_window:
                    left += 1

                while right < n and positions[right] <= center + self.neighbor_window:
                    right += 1

                score = 0
                weight_sum = 0
                max_ratio = 0
                values = []
                count = 0

                for j in range(left, right):

                    if j == i:
                        continue

                    dist = abs(positions[j] - center)

                    weight = np.exp(-dist / self.decay_lambda)

                    score += ratios[j] * weight
                    weight_sum += weight

                    max_ratio = max(max_ratio, ratios[j])
                    values.append(ratios[j])
                    count += 1

                idx = start_idx + i
                neighbor_count[idx] = count

                if weight_sum > 0:
                    neighbor_score[idx] = score / weight_sum
                else:
                    neighbor_score[idx] = 0

                neighbor_max_ratio[idx] = max_ratio

                if len(values) > 1:
                    neighbor_std[idx] = np.std(values)
                else:
                    neighbor_std[idx] = 0

            start_idx += n

        df["neighbor_score"] = neighbor_score
        df["neighbor_max_ratio"] = neighbor_max_ratio
        df["neighbor_std"] = neighbor_std
        df["neighbor_mean_diff"] = df["pred_ratio"] - df["neighbor_score"]
        df["log_neighbor_count"] = np.log1p(neighbor_count)

        return df

    def build_features(self, pred_path, output_path, truth_path=None):

        print("Loading prediction file...")
        df = self.load_prediction(pred_path)

        if truth_path is not None:

            print("Loading truth file...")
            truth = self.load_truth(truth_path)

            print("Merging prediction with truth...")
            df = self.merge_truth(df, truth)

        print("Computing relative position...")
        df = self.compute_relative_position(df)

        print("Computing neighbor features...")
        df = self.compute_neighbor_features(df)

        print("Saving feature table...")
        df.to_csv(output_path, index=False)

        print("Done!")
        print("Total samples:", len(df))
