import os.path

import lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

import math

from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    Specificity,
    AUROC,
    AveragePrecision,
)
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
    SymmetricMeanAbsolutePercentageError,
)

from NanoSSM.models.mamba.layer import (
    SignalEncoder,
    SequenceEncoder,
    MambaEncoder,
    MambaAttentionAggregator,
    GatedResidualNetwork,
    IntraReadAttention,
    StatRefinerFull,
    GatedFusionModule,
    SiteAggregator,
    SiteInteractionModule,
)
from NanoSSM.tools.models.models_utils import (
    get_cosine_schedule_with_warmup,
    pearson_loss,
)
import warnings

warnings.simplefilter("error", RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="Trying to infer the.*batch_size.*ambiguous collection",
)

class MambaModel(pl.LightningModule):
    def __init__(
        self,
        signal_dim: int,
        sequence_dim: int,
        features: int = 256,
        hidden_dim: int = None,
        output_dim: int = 1,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 1,
        num_mamba_layers: int = 2,
        dropout: float = 0.1,
        kmer: int = 5,
        learning_rate: float = 1e-3,
        wd: float = 5e-5,
        type: str = "site",
        is_finetune: bool = False,
        freeze_epochs: int = 3,
        test_save_path: str = None,
        feature_dim: int = 5,
    ):
        super().__init__()

        self.lr = learning_rate
        self.wd = wd
        self.num_mamba_layers = num_mamba_layers
        self.is_finetune = is_finetune
        self.freeze_epochs = freeze_epochs
        self.test_save_path = test_save_path if test_save_path is not None else "./"

        self.loss = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.SmoothL1Loss(beta=0.05, reduction="none")

        self.define_metrics()
        self.type = type
        self.signal_dim = signal_dim

        self._build_model(
            signal_dim,
            sequence_dim,
            features,
            hidden_dim,
            output_dim,
            encoder_num_layers,
            decoder_num_layers,
            num_mamba_layers,
            dropout,
            kmer,
            feature_dim,
        )

        self._init_weights()

    def _build_model(
        self,
        signal_dim,
        seq_dim,
        features,
        hidden_dim,
        output_dim,
        encoder_num_layers,
        decoder_num_layers,
        num_mamba_layers,
        dropout,
        kmer,
        feature_dim=5,
    ):
        if hidden_dim is None:
            hidden_dim = features * 2

        self.seq_embedding = MambaEncoder(in_dim=kmer, out_dim=features)
        self.seq_encoder = SequenceEncoder(
            features,
            hidden_dim,
            features,
            encoder_num_layers,
            dropout,
            num_mamba_layers=num_mamba_layers,
        )

        self.stat_mlp_refiner = StatRefinerFull(
            stat_dim=feature_dim, hidden_dim=features, dropout=dropout
        )

        self.read_norm = nn.LayerNorm(features)

        self.fusion_module = GatedFusionModule(features, dropout=dropout)

        self.fc_mod = nn.Sequential(
            nn.Linear(features, features), nn.ReLU(), nn.Linear(features, output_dim)
        )

        projected_dim = 256

        self.intra_read_pooler = IntraReadAttention(features)

        self.site_interaction = SiteInteractionModule(
            feature_dim=features,
            nhead=4,
            num_layers=1,
            dropout=dropout,
        )

        self.aggregator = SiteAggregator(
            feature_dim=features + 16 + 16,
            projected_dim=projected_dim,
            dropout=dropout,
            temperature=1.5,
        )

        self.prob_projection = nn.Sequential(nn.Linear(1, 16), nn.GELU())

        self.norm_projection = nn.Sequential(nn.Linear(1, 16), nn.GELU())

        self.final_site_predictor = nn.Sequential(
            nn.Linear(projected_dim + 6, projected_dim // 2),
            nn.GELU(),
            nn.LayerNorm(projected_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(projected_dim // 2, 1),
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.wd)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, seqs, signals, stats, type="read"):
        stats = stats.contiguous()
        seqs = seqs.contiguous()
        if signals is not None:
            signals = signals.contiguous()

        seqs_ = self.seq_embedding(seqs)

        stats_feat = self.stat_mlp_refiner(stats)

        fused_feat = self.fusion_module(seqs_, stats_feat)

        refined_seq = self.seq_encoder(fused_feat)

        read_features = self.intra_read_pooler(refined_seq)

        read_features = self.read_norm(read_features)

        out = self.fc_mod(read_features)

        if type == "site":

            return out.squeeze(-1), read_features

        return out.squeeze(-1)

    def forward_site(self, all_seqs, all_signals, all_stats):
        try:

            preds, final_alignment_features = self.forward(
                seqs=all_seqs, signals=all_signals, stats=all_stats, type="site"
            )
        except Exception as e:
            import traceback, sys

            traceback.print_exc()

            def show(t, name):
                if isinstance(t, torch.Tensor):
                    try:
                        print(
                            f"{name}: shape={t.shape}, dtype={t.dtype}, device={t.device}, min={t.min().item()}, max={t.max().item()}"
                        )
                    except:
                        print(
                            f"{name}: shape={t.shape}, dtype={t.dtype}, device={t.device}"
                        )

            show(all_seqs, "all_seqs")
            show(all_signals, "all_signals")
            show(all_stats, "all_stats")

            raise
        return preds, final_alignment_features

    def get_sample_weight(self, ratio):
        weight = torch.ones_like(ratio)

        middle_boost = 1.0 * torch.exp(-((ratio - 0.5) ** 2) / (2 * 0.18**2))

        high_boost = 1.5 * torch.exp(-((ratio - 0.8) ** 2) / (2 * 0.1**2))
        weight += middle_boost + high_boost
        return weight

    def site_shared_step(self, batch, batch_idx, mode="train"):
        if batch is None:
            return torch.tensor(0.0, device=self.device)

        batch_size = len(batch)
        site_preds_reg = []

        ratios = []

        all_seqs = torch.cat([b["seq"] for b in batch], dim=0)
        all_signals = (
            torch.cat([b["signal"] for b in batch], dim=0)
            if batch[0]["signal"] is not None
            else None
        )
        all_stats = torch.cat([b["stat"] for b in batch], dim=0)

        preds, final_alignment_features = self.forward_site(
            all_seqs, all_signals, all_stats
        )

        offset = 0
        read_counts = []
        for site_data in batch:
            n_reads = site_data["seq"].shape[0]
            site_reads_logit = preds[offset : offset + n_reads]
            site_features = final_alignment_features[offset : offset + n_reads]
            offset += n_reads
            read_counts.append(n_reads)
            if mode == "train":

                perm = torch.randperm(n_reads, device=self.device)
                site_features = site_features[perm]
                site_reads_logit = site_reads_logit[perm]

            ratio = site_data["ratio"].to(self.device).view(-1).clone()

            read_probs = torch.sigmoid(site_reads_logit)

            site_features_interacted = self.site_interaction(site_features.unsqueeze(0))

            site_features = site_features_interacted.squeeze(0)

            p_feat = self.prob_projection(site_reads_logit.unsqueeze(-1))

            f_norm = torch.norm(site_features, dim=-1, keepdim=True)
            n_feat = self.norm_projection(f_norm)

            feat_for_agg = torch.cat([site_features, p_feat, n_feat], dim=-1)

            aggregated_feature, weights = self.aggregator(
                feat_for_agg, return_weights=True
            )

            logit_f = site_reads_logit.float()
            mean_val = logit_f.mean().unsqueeze(0)
            std_val = (
                logit_f.std().unsqueeze(0)
                if n_reads > 1
                else torch.tensor([0.0], device=self.device)
            )
            q25, q50, q75 = torch.quantile(
                logit_f, torch.tensor([0.25, 0.5, 0.75], device=self.device)
            )
            log_n_reads = torch.log10(
                torch.tensor([float(n_reads)], device=self.device)
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
            stat_features = stat_features.unsqueeze(0)

            combined_feat = torch.cat([aggregated_feature, stat_features], dim=-1)

            delta_logit = self.final_site_predictor(combined_feat).view(-1).clamp(-5, 5)

            weighted_base_ratio = torch.sum(weights * read_probs)

            base_ratio = weighted_base_ratio.clamp(1e-5, 1 - 1e-5)

            base_logit = torch.logit(base_ratio.clamp(1e-5, 1 - 1e-5))
            final_pred_reg = torch.sigmoid(base_logit + delta_logit)

            site_preds_reg.append(final_pred_reg)
            ratios.append(ratio)

        all_preds = torch.cat(site_preds_reg)
        all_ratios = torch.cat(ratios)

        read_counts_tensor = torch.tensor(
            read_counts, device=self.device, dtype=torch.float
        )
        coverage_weights = torch.log1p(read_counts_tensor)

        coverage_weights = coverage_weights.clamp(
            max=coverage_weights.quantile(0.9).item()
        )

        final_weights = coverage_weights

        loss_dist = self.loss_fn(all_preds, all_ratios)
        weighted_mse_loss = (loss_dist * final_weights).sum() / (
            final_weights.sum() + 1e-8
        )

        p_loss = pearson_loss(all_preds, all_ratios)

        total_loss = weighted_mse_loss + 0.3 * p_loss

        ratios = torch.stack(ratios)
        site_preds_reg = torch.stack(site_preds_reg)

        if mode == "train":
            self.train_mse.update(site_preds_reg, ratios)
            self.train_mae.update(site_preds_reg, ratios)
            self.train_pearson.update(site_preds_reg, ratios)
            self.train_smape.update(site_preds_reg, ratios)
            if self.global_step > 0 and self.global_step % 100 == 0:
                self.log(
                    "train_mse", self.train_mse.compute(), on_step=True, prog_bar=True
                )
                self.log(
                    "train_mae", self.train_mae.compute(), on_step=True, prog_bar=False
                )
            self.log(
                "total_loss",
                total_loss,
                on_step=True,
                prog_bar=True,
                on_epoch=False,
                batch_size=batch_size,
            )

        elif mode == "val":
            self.val_mse.update(site_preds_reg, ratios)
            self.val_mae.update(site_preds_reg, ratios)
            self.val_pearson.update(site_preds_reg, ratios)
            self.val_smape.update(site_preds_reg, ratios)
            self.log(
                "val_loss",
                total_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        elif mode == "test":

            for i, site_data in enumerate(batch):

                info = site_data.get("info", {})

                self.test_step_outputs.append(
                    {
                        "chrom": info.get("transcript_id", "unknown"),
                        "start_position": info.get("position", -1),
                        "true_ratio": ratios[i].item(),
                        "percent_modified": site_preds_reg[i].item(),
                    }
                )

            self.test_mse.update(site_preds_reg, ratios)
            self.test_mae.update(site_preds_reg, ratios)
            self.test_pearson.update(site_preds_reg, ratios)
            self.test_smape.update(site_preds_reg, ratios)

            self.log(
                "test_loss",
                total_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        return total_loss

    def read_shared_step(self, batch, batch_idx, mode="train"):

        seqs = batch["seq"]
        stat = batch["stat"]
        labels = batch["label"]
        signals = batch.get("signals", None)

        logits = self.forward(seqs=seqs, signals=signals, stats=stat)

        loss_total = self.loss(logits, labels)

        probs = torch.sigmoid(logits)

        predicted = (probs >= 0.5).float()
        if mode == "train":
            self.train_acc.update(predicted, labels)
            self.train_auroc.update(probs, labels)

            self.log("train_loss", loss_total, on_step=True, prog_bar=True)

        elif mode == "val":
            self.val_acc.update(predicted, labels)
            self.val_auroc.update(probs, labels)
            self.val_auprc.update(probs, labels.long())
            self.f1.update(predicted, labels)
            self.precision.update(predicted, labels)
            self.recall.update(predicted, labels)
            self.specificity.update(predicted, labels)
            self.log(
                "val_loss", loss_total, on_step=True, sync_dist=True, prog_bar=False
            )

        else:
            self.test_acc.update(predicted, labels)
            self.test_auroc.update(probs, labels)
            self.test_auprc.update(probs, labels.long())
            self.test_f1.update(predicted, labels)
            self.test_precision.update(predicted, labels)
            self.test_recall.update(predicted, labels)
            self.test_specificity.update(predicted, labels)
            self.log("test_loss", loss_total, sync_dist=True)

        return loss_total

    def _init_weights(self, module=None, initialize_mode="xavier"):
        if module is None:
            self.apply(self._init_weights)
            return

        if hasattr(module, "_no_weight_decay"):
            return

        if isinstance(module, nn.Conv1d):
            if initialize_mode == "xavier":
                nn.init.xavier_normal_(module.weight, gain=1.0)
            elif initialize_mode == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, nn.Linear):
            if initialize_mode == "xavier":
                nn.init.xavier_normal_(module.weight, gain=1.0)
            elif initialize_mode == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

        elif isinstance(module, Mamba2):
            for name, p in module.named_parameters():

                if p.dim() < 2:
                    continue

                if "A_log" in name:
                    A_init_range = (1, 16)
                    A = torch.empty_like(p.data).uniform_(*A_init_range)
                    p.data.copy_(torch.log(A))
                elif "dt_bias" in name:
                    dt_min, dt_max = 0.001, 0.1
                    dt = torch.exp(
                        torch.empty_like(p.data).uniform_(
                            math.log(dt_min), math.log(dt_max)
                        )
                    )
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    p.data.copy_(inv_dt)
                elif "D" in name:
                    nn.init.constant_(p, 1.0)
                else:

                    if p.dim() >= 2:
                        if initialize_mode == "xavier":
                            nn.init.xavier_normal_(p, gain=1.0)
                        elif initialize_mode == "kaiming":
                            nn.init.kaiming_normal_(
                                p, mode="fan_out", nonlinearity="relu"
                            )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        if hasattr(self, "final_site_predictor"):

            nn.init.zeros_(self.final_site_predictor[-1].weight)
            nn.init.zeros_(self.final_site_predictor[-1].bias)

    def define_metrics(self):
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")

        self.f1 = F1Score(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.specificity = Specificity(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.val_acc = Accuracy(task="binary")

        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_specificity = Specificity(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.train_pearson = PearsonCorrCoef()
        self.train_smape = SymmetricMeanAbsolutePercentageError()

        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_pearson = PearsonCorrCoef()
        self.val_smape = SymmetricMeanAbsolutePercentageError()

        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.test_pearson = PearsonCorrCoef()
        self.test_smape = SymmetricMeanAbsolutePercentageError()

    def training_step(self, batch, batch_idx):
        if self.type == "site":
            return self.site_shared_step(batch, batch_idx, mode="train")
        return self.read_shared_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        if self.type == "site":
            return self.site_shared_step(batch, batch_idx, mode="val")
        return self.read_shared_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        if self.type == "site":
            return self.site_shared_step(batch, batch_idx, mode="test")
        return self.read_shared_step(batch, batch_idx, mode="test")

    def on_training_epoch_end(self, outputs):

        if self.type == "site":
            self.log(
                "train_mse",
                self.train_mse.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_mae",
                self.train_mae.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_pearson",
                self.train_pearson.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_smape",
                self.train_smape.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.train_mse.reset()
            self.train_mae.reset()
            self.train_pearson.reset()
            self.train_smape.reset()
        else:
            train_acc = self.train_acc.compute()
            train_auroc = self.train_auroc.compute()
            self.log(
                "train_acc",
                train_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "train_auroc",
                train_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            self.train_acc.reset()
            self.train_auroc.reset()

    def on_validation_epoch_end(self):
        if self.type == "site":
            self.log(
                "val_mse",
                self.val_mse.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "val_mae",
                self.val_mae.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "val_pearson",
                self.val_pearson.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "val_smape", self.val_smape.compute(), on_epoch=True, sync_dist=True
            )
            self.val_pearson.reset()
            self.val_smape.reset()
            self.val_mse.reset()
            self.val_mae.reset()
        else:
            self.log(
                "val_acc",
                self.val_acc.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "val_auroc",
                self.val_auroc.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            try:
                value = self.val_auprc.compute()
            except Exception:
                value = torch.tensor(0.0, device=self.device)

            self.log("val_auprc", value, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("val_f1", self.f1.compute(), on_epoch=True, sync_dist=True)
            self.log(
                "val_precision", self.precision.compute(), on_epoch=True, sync_dist=True
            )
            self.log("val_recall", self.recall.compute(), on_epoch=True, sync_dist=True)
            self.log(
                "val_specificity",
                self.specificity.compute(),
                on_epoch=True,
                sync_dist=True,
            )

            self.val_acc.reset()
            self.val_auprc.reset()
            self.f1.reset()
            self.precision.reset()
            self.recall.reset()
            self.specificity.reset()

    def on_test_start(self):

        self.test_step_outputs = []

    def on_test_epoch_end(self):
        if self.type == "site":
            if hasattr(self, "test_step_outputs") and len(self.test_step_outputs) > 0:
                import pandas as pd

                df = pd.DataFrame(self.test_step_outputs)

                save_path = os.path.join(
                    self.test_save_path, f"lightning_test_debug.csv"
                )
                df.to_csv(save_path, index=False, sep="\t")
                print(f">>> Debug results saved to {save_path}")

                self.test_step_outputs.clear()

            self.log("test_mse", self.test_mse.compute(), on_epoch=True, sync_dist=True)
            self.log("test_mae", self.test_mae.compute(), on_epoch=True, sync_dist=True)
            self.log(
                "test_pearson",
                self.test_pearson.compute(),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test_smape", self.test_smape.compute(), on_epoch=True, sync_dist=True
            )
            self.test_mse.reset()
            self.test_mae.reset()
            self.test_pearson.reset()
            self.test_smape.reset()
        else:
            self.log(
                "test_acc",
                self.test_acc.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "test_auroc",
                self.test_auroc.compute(),
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            try:
                value = self.test_auprc.compute()
            except Exception:
                value = torch.tensor(0.0, device=self.device)
            self.log("test_auprc", value, sync_dist=True, on_epoch=True, prog_bar=True)

            self.log("test_f1", self.test_f1.compute(), on_epoch=True, sync_dist=True)
            self.log(
                "test_precision",
                self.test_precision.compute(),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test_recall", self.test_recall.compute(), on_epoch=True, sync_dist=True
            )
            self.log(
                "test_specificity",
                self.test_specificity.compute(),
                on_epoch=True,
                sync_dist=True,
            )

            self.test_acc.reset()
            self.test_auroc.reset()
            self.test_auprc.reset()
            self.test_f1.reset()
            self.test_precision.reset()
            self.test_recall.reset()
            self.test_specificity.reset()

    def backbone_modules(self):
        return [
            self.seq_embedding,
            self.seq_encoder,
            self.stat_mlp_refiner,
            self.read_norm,
            self.fusion_module,
            self.intra_read_pooler,
            self.fc_mod,
        ]

    def head_modules(self):
        return [
            self.aggregator,
            self.final_site_predictor,
        ]

    def freeze_backbone(self):
        for module in self.backbone_modules():
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for module in self.backbone_modules():
            for p in module.parameters():
                p.requires_grad = True
