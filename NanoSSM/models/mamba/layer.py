import random
from typing import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from torch import Tensor

from rotary_embedding_torch import RotaryEmbedding

from mamba_ssm import Mamba2

import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowContextFusion(nn.Module):
    def __init__(self, kmer_dim=5, stat_dim=5, embed_dim=256, dropout=0.1):
        super().__init__()

        self.seq_proj = nn.Linear(kmer_dim, embed_dim)
        self.stat_proj = nn.Linear(stat_dim, embed_dim)

        self.gate_proj = nn.Linear(kmer_dim, embed_dim)

        self.diff_proj = nn.Sequential(
            nn.Linear(stat_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.diff_gate = nn.Parameter(torch.tensor(0.5))

        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.context_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.center_bias = nn.Parameter(torch.ones(1))
        self.context_weight = nn.Parameter(torch.zeros(1))

    def forward(self, seqs, stats):
        batch_size, m, _ = seqs.shape

        x_seq = self.seq_proj(seqs)
        x_stat = self.stat_proj(stats)

        diff = stats[:, 1:, :] - stats[:, :-1, :]
        diff = F.pad(diff, (0, 0, 0, 1))
        x_diff = self.diff_proj(diff)

        gate = torch.sigmoid(self.gate_proj(seqs))

        x = self.fusion_norm(x_seq + (x_stat * gate) + self.diff_gate * x_diff)

        center_idx = m // 2
        center_identity = x[:, center_idx, :].contiguous()
        x_permuted = x.transpose(1, 2)
        conv_out = self.context_conv(x_permuted)
        context_out = self.global_pool(conv_out).squeeze(-1)
        out = (self.center_bias * center_identity) + (self.context_weight * context_out)

        return out

class MambaEncoder(nn.Module):
    def __init__(self, in_dim=15, out_dim=256, expand=4, head_dim=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):

        x = self.embedding(x)
        return x

class SequenceEncoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, dropout, num_mamba_layers
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    input_dim,
                    output_dim,
                    dropout=dropout,
                    num_mamba_layers=num_mamba_layers,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_mamba_layers=2, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList(
            [MambaBlock(input_dim, dropout=dropout) for _ in range(num_mamba_layers)]
        )

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = LinearProjection(input_dim, output_dim, dropout)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)

        return residual + self.proj(self.norm(x))

class MambaBlock(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()

        expand = 2
        head_dim = int(input_dim * expand // 8)

        self.mamba = Mamba2(
            d_model=input_dim,
            d_state=16,
            d_conv=4,
            expand=expand,
            headdim=head_dim,
        )

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))

class StatRefinerFull(nn.Module):
    def __init__(self, stat_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.out_proj = nn.Sequential(
            nn.Linear(stat_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.shortcut = (
            nn.Linear(stat_dim, hidden_dim) if stat_dim != hidden_dim else nn.Identity()
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, stats):

        out = self.out_proj(stats)

        res = self.shortcut(stats)

        return self.final_norm(out + res)

class GatedFusionModule(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()

        self.seq_path = nn.Linear(features, features)

        self.stat_gate = nn.Sequential(nn.Linear(features, features), nn.Sigmoid())

        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_feat, stat_feat):

        gate = self.stat_gate(stat_feat)

        gate = 0.1 + 0.9 * gate
        fused = self.seq_path(seq_feat) + (gate * stat_feat)

        return self.dropout(self.norm(fused))

class IntraReadAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()

        self.norm = nn.LayerNorm(feature_dim)
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x):

        x = self.norm(x)

        scores = self.attn_net(x)

        weights = F.softmax(scores, dim=1)

        context = (x * weights).sum(dim=1)

        return context

class SiteInteractionModule(nn.Module):
    def __init__(self, feature_dim, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):

        out = self.transformer(x)
        return out

class SiteAggregator(nn.Module):

    def __init__(self, feature_dim=257, projected_dim=256, dropout=0.1, temperature=1):
        super().__init__()
        self.tau = temperature

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention_v = nn.Sequential(
            nn.Linear(projected_dim, projected_dim // 2), nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(projected_dim, projected_dim // 2), nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(projected_dim // 2, 1), nn.Dropout(dropout)
        )

        self.post_proj = nn.Sequential(
            nn.Linear(projected_dim, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_weights: bool = False):

        feat = self.input_proj(x)

        a_v = self.attention_v(feat)
        a_u = self.attention_u(feat)

        raw_scores = self.attention_weights(a_v * a_u)

        weights = torch.softmax(raw_scores / self.tau, dim=0)

        site_feature = torch.matmul(weights.transpose(0, 1), feat)

        output = self.post_proj(site_feature)

        if return_weights:
            return output, weights.squeeze(-1)

        return output

class LinearProjection(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class AlignmentDecoder(nn.Module):
    def __init__(
        self, feature_dim, hidden_dim, num_mamba_layers, num_layers=2, dropout=0.1
    ):
        super().__init__()

        self.seq_proj = nn.Linear(feature_dim, hidden_dim)

        self.stat_proj = nn.Linear(feature_dim, hidden_dim)

        self.rotary_emb = RotaryEmbedding(dim=hidden_dim)

        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )

        self.mamba_fusion = nn.ModuleList(
            [
                MambaLayer(
                    hidden_dim,
                    hidden_dim,
                    dropout=dropout,
                    num_mamba_layers=num_mamba_layers,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, stats):

        seqs = self.seq_proj(seqs)

        stats = self.stat_proj(stats)

        seqs = self.rotary_emb.rotate_queries_or_keys(seqs)

        stats = self.rotary_emb.rotate_queries_or_keys(stats)

        x2, _ = self.cross_attn_2(query=seqs, key=stats, value=stats)
        x = self.dropout(x2) + seqs
        x = self.norm(x)

        for mamba_layer in self.mamba_fusion:
            x = mamba_layer(x)

        out = self.output_proj(x)

        return out

class GatedResidualNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.w_main = nn.Linear(input_dim, hidden_dim)
        self.w_gate = nn.Linear(input_dim, hidden_dim)

        self.w_proj = nn.Linear(hidden_dim, input_dim)

        self.dropout_layer = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: Tensor) -> Tensor:

        gate = torch.sigmoid(self.w_gate(x))

        gate = 0.1 + 0.9 * gate

        main_activation = F.gelu(self.w_main(x))

        gated_output = main_activation * gate

        projected_output = self.w_proj(gated_output)

        output = self.norm(x + self.dropout_layer(projected_output))

        return output

class MambaAttentionAggregator(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        projected_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projected_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=projected_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_token = nn.Parameter(torch.randn(1, 1, projected_dim))

        self.attn_norm = nn.LayerNorm(projected_dim)

        self.output_proj = nn.Linear(projected_dim, projected_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, read_features: torch.Tensor, return_weights: bool = False):

        if read_features.dim() != 2:
            raise ValueError(
                f"Input tensor must have 2 dimensions, got {read_features.dim()}"
            )

        projected_read_features = self.projection(read_features)

        query = self.query_token.expand(1, -1, -1)
        key_value = projected_read_features.unsqueeze(0)

        attn_output, attn_weights = self.attention(
            query=query, key=key_value, value=key_value, need_weights=True
        )

        aggregated_feature = self.attn_norm(attn_output).squeeze(0)

        aggregated_feature = self.dropout(self.output_proj(aggregated_feature))

        if return_weights:
            return aggregated_feature, attn_weights.squeeze()

        return aggregated_feature

class SignalEncoder(nn.Module):

    def __init__(self, in_dim, out_dim, num_mamba_layers=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.ms_conv = MultiScaleConv(in_dim, out_dim)

        self.mamba_layers = nn.ModuleList(
            [MambaBlock(out_dim, dropout=0.1) for _ in range(num_mamba_layers)]
        )

    def forward(self, x, mask):
        x = self.norm(x)
        x = self.ms_conv(x)

        x = x * mask.unsqueeze(-1)

        for layer in self.mamba_layers:
            x = layer(x)

        return x * mask.unsqueeze(-1)

class StatRefiner(nn.Module):
    def __init__(self, stat_dim, hidden_dim):
        super().__init__()

        self.seq_encoder = MultiScaleConv(in_dim=stat_dim, out_dim=hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, stats):

        feat_seq = self.seq_encoder(stats)

        center_idx = feat_seq.size(1) // 2
        center_feat = feat_seq[:, center_idx, :]

        global_feat = feat_seq.mean(dim=1)

        feat = center_feat + 0.1 * global_feat

        return self.out_proj(feat)

class PaddingAwareNorm(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mask):

        mask = mask.unsqueeze(-1)

        x_masked = x * mask
        valid = mask.sum(dim=1).clamp(min=1)

        mean = (x_masked.sum(dim=1) / valid).unsqueeze(1)
        var = (((x_masked - mean) ** 2).sum(dim=1) / valid).unsqueeze(1)

        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * mask

class DirectStatFusion(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()

        self.stat_path = nn.Linear(features, features)

        self.seq_path = nn.Linear(features, features)

        self.gate = nn.Sequential(nn.Linear(features, features), nn.Sigmoid())

        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_feat, stat_feat):

        s = self.stat_path(stat_feat)
        c = self.seq_path(seq_feat)

        g = self.gate(stat_feat)

        fused = s + g * c

        return self.dropout(self.norm(fused))

class MultiScaleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv3 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, out_dim, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(in_dim, out_dim, kernel_size=9, padding=4)
        self.proj = nn.Linear(out_dim * 3, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        c3 = F.gelu(self.conv3(x))
        c5 = F.gelu(self.conv5(x))
        c9 = F.gelu(self.conv9(x))
        out = torch.cat([c3, c5, c9], dim=1)
        out = out.transpose(1, 2)
        out = self.proj(out)
        return self.norm(out)

class StatRefinerBalanced(nn.Module):
    def __init__(self, stat_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.proj = nn.Linear(stat_dim, hidden_dim)

        self.act = nn.GELU()

        self.norm = nn.LayerNorm(hidden_dim)

        self.shortcut = (
            nn.Linear(stat_dim, hidden_dim) if stat_dim != hidden_dim else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, stats):

        stats_centered = stats - stats.mean(dim=1, keepdim=True)

        res = self.shortcut(stats_centered)

        out = self.act(self.proj(stats_centered))

        return self.norm(self.dropout(out + res))
