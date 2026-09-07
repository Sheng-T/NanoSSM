import argparse
import os
import sys

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract consensus pseudo-labels from one NanoSSM "
                    "prediction BED and one Dorado bedMethyl file for "
                    "NanoSSM training."
    )

    # ----- inputs -----
    parser.add_argument('--nanossm_bed', type=str, required=True,
                        help='NanoSSM prediction BED file (one per '
                             'pseudo-labeling iteration, as in the paper).')
    parser.add_argument('--dorado_bed', type=str, required=True,
                        help='Dorado/modkit pileup BED (bedMethyl) file. '
                             'Keep this file fixed across iterations.')
    parser.add_argument('--data_info', type=str, required=True,
                        help='Original data.info file providing per-site read '
                             'coverage (n_reads).')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory.')
    parser.add_argument('--output_filename', type=str,
                        default="pseudo_label.info",
                        help='Output file name.')

    # ----- filtering parameters (defaults follow the manuscript) -----
    parser.add_argument('--pos_threshold', type=float, default=0.1,
                        help='Consensus ratio above this defines positive sites.')
    parser.add_argument('--neg_threshold', type=float, default=0.05,
                        help='Consensus ratio below this defines negative sites.')
    parser.add_argument('--consistency', type=float, default=0.1,
                        help='Max allowed sample SD between the two estimators '
                             'for positive sites (0.1 in the manuscript).')
    parser.add_argument('--min_coverage', type=int, default=20,
                        help='Minimum supporting-read coverage (n_reads).')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Negative-to-positive sampling ratio.')
    parser.add_argument('--seed', type=int, default=47,
                        help='Random seed for negative down-sampling.')

    return parser.parse_args()


def load_nanossm_bed(path):
    """Load a NanoSSM prediction BED (11-column format)."""
    try:
        df = pd.read_csv(path, sep="\t", header=None, low_memory=False)
        df.columns = [
            "transcript_id", "start", "end", "motif", "score",
            "strand", "start2", "end2", "color",
            "coverage", "ratio"
        ]
    except Exception as e:
        sys.exit(f"Error loading {path}: {e}\n"
                 "Please check that the BED has 11 columns "
                 "(NanoSSM inference output format).")
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df = df.dropna(subset=["ratio"])
    # key uses 0-based transcript coordinates, consistent with modkit pileup
    df["key"] = df["transcript_id"].astype(str) + "_" + df["start"].astype(str)
    return df[["key", "ratio"]].rename(columns={"ratio": "ratio_nanossm"})


def load_dorado_bed(path):
    """Load a Dorado/modkit pileup bedMethyl file (18 columns)."""
    cols = [
        'chrom', 'start_position', 'end_position', 'pattern', 'score', 'strand',
        'start2', 'end2', 'color', 'Nvalid_cov', 'fraction_modified', 'Npattern',
        'Ncanonical', 'Nother', 'Ndelete', 'Nfail', 'Ndiff', 'Nnocall'
    ]
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=cols, low_memory=False)
    except Exception as e:
        sys.exit(f"Error loading Dorado file {path}: {e}")
    df['fraction_modified'] = pd.to_numeric(df['fraction_modified'], errors='coerce')
    df = df.dropna(subset=['fraction_modified'])
    df['ratio'] = df['fraction_modified'] / 100.0
    df['key'] = df['chrom'].astype(str) + "_" + df['start_position'].astype(str)
    return df[['key', 'ratio']].rename(columns={"ratio": "ratio_dorado"})


def print_pairwise_stats(merged):
    """Print pairwise absolute-difference statistics between estimators."""
    pairs = [("ratio_nanossm", "NanoSSM"), ("ratio_dorado", "Dorado")]
    print(f"\n--- Pairwise differences between estimators ---")
    print(f"{'Model A':<20s} {'Model B':<20s} {'MAD':>8s} {'Median':>8s} "
          f"{'>0.1':>14s} {'>0.2':>14s}")
    print("-" * 84)
    diff = (merged["ratio_nanossm"] - merged["ratio_dorado"]).abs()
    print(f"{'NanoSSM':<20s} {'Dorado':<20s} "
          f"{diff.mean():>8.4f} {diff.median():>8.4f} "
          f"{(diff > 0.1).sum():>7d}({(diff > 0.1).mean()*100:.1f}%) "
          f"{(diff > 0.2).sum():>7d}({(diff > 0.2).mean()*100:.1f}%)")


def print_distribution_stats(merged):
    """Print per-estimator prediction distributions on merged sites."""
    print(f"\n--- Per-estimator prediction distributions ---")
    print(f"{'Model':<20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} "
          f"{'>0.1':>10s} {'>0.5':>10s}")
    print("-" * 66)
    for col, name in [("ratio_nanossm", "NanoSSM"), ("ratio_dorado", "Dorado")]:
        vals = merged[col]
        print(f"{name:<20s} {vals.mean():>8.4f} {vals.median():>8.4f} "
              f"{vals.std():>8.4f} {(vals > 0.1).sum():>9d} {(vals > 0.5).sum():>9d}")


def print_coverage_stats(df, label="Dataset"):
    """Print site counts at several coverage thresholds."""
    thresholds = [20, 50, 100, 200]
    print(f"\n--- Coverage statistics: {label} ---")
    print(f"{'Coverage >=':<15} | {'Site count':<10}")
    print("-" * 30)
    for t in thresholds:
        print(f"{t:<15} | {len(df[df['n_reads'] >= t]):<10}")
    print("-" * 30)


def main():
    args = get_args()

    # =========================================
    # 1. Load the two estimator predictions
    # =========================================
    nanossm_df = load_nanossm_bed(args.nanossm_bed)
    dorado_df = load_dorado_bed(args.dorado_bed)

    # =========================================
    # 2. Inner-join on site key
    # =========================================
    merged = nanossm_df.merge(dorado_df, on="key", how="inner")
    n_dup = merged["key"].duplicated().sum()
    if n_dup > 0:
        print(f"[WARNING] {n_dup} duplicated site keys after merging "
              f"(same transcript/position across strands?). Keeping first occurrence.")
        merged = merged.drop_duplicates(subset="key", keep="first")

    print(f"\nMerged sites (intersection): {len(merged)}")
    if len(merged) == 0:
        sys.exit("[ERROR] No shared sites. Most common cause: coordinate or "
                 "transcript-identifier mismatch between the NanoSSM BED and "
                 "the Dorado bedMethyl (check that both use the same "
                 "transcriptome reference and 0-based starts).")

    # =========================================
    # 3. Consensus ratio and agreement (manuscript formulation)
    #    pandas default ddof=1 matches the sample SD used for agreement
    #    filtering; with two estimators, SD == |r_N - r_D| / sqrt(2)
    # =========================================
    merged["consensus_ratio"] = merged[["ratio_nanossm", "ratio_dorado"]].mean(axis=1)
    merged["agreement_std"] = merged[["ratio_nanossm", "ratio_dorado"]].std(axis=1)

    # =========================================
    # 4. Diagnostics
    # =========================================
    print_pairwise_stats(merged)
    print_distribution_stats(merged)

    # =========================================
    # 5. Align with data.info for read coverage
    # =========================================
    print(f"\nAligning with original info file ...")
    info_df = pd.read_csv(args.data_info)
    info_df["key"] = (info_df["transcript_id"].astype(str) + "_"
                      + info_df["transcript_position"].astype(str))

    combined_df = merged.merge(info_df, on="key", how="inner")
    print(f"  Sites after alignment: {len(combined_df)}")

    print_coverage_stats(combined_df, label="All candidate sites")

    # =========================================
    # 6. Select positive / negative sites
    #    Agreement filtering applies to positives only (as in the paper).
    # =========================================
    agreement_mask = combined_df["agreement_std"] < args.consistency
    coverage_mask = combined_df["n_reads"] >= args.min_coverage

    pos_df = combined_df[
        agreement_mask & coverage_mask &
        (combined_df["consensus_ratio"] > args.pos_threshold)
    ].copy()

    neg_df = combined_df[
        coverage_mask & (combined_df["consensus_ratio"] < args.neg_threshold)
    ].copy()

    print(f"\nSelection done (min_coverage >= {args.min_coverage}, "
          f"agreement SD < {args.consistency})")
    print(f"Positives: {len(pos_df)} | negative candidates: {len(neg_df)}")

    if len(pos_df) > 0:
        print(f"\n--- Positive-site consensus-ratio distribution ---")
        bins = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        labels_bin = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        pos_df["_bin"] = pd.cut(pos_df["consensus_ratio"], bins=bins,
                                labels=labels_bin, include_lowest=True)
        print(pos_df["_bin"].value_counts().sort_index().to_string())
        pos_df.drop(columns=["_bin"], inplace=True)

    # =========================================
    # 7. Down-sample negatives and merge
    # =========================================
    if len(pos_df) == 0 or len(neg_df) == 0:
        sys.exit("[ERROR] Empty positive or negative set; cannot build a "
                 "training subset. Check thresholds / coordinate alignment.")
    n_neg = int(len(pos_df) * args.ratio)
    neg_sampled = neg_df.sample(n=min(n_neg, len(neg_df)), random_state=args.seed)
    final_labeled = pd.concat([pos_df, neg_sampled])

    print_coverage_stats(final_labeled, label="Final training subset")

    # =========================================
    # 8. Save (consensus ratio is the regression target)
    # =========================================
    final_labeled["ratio"] = final_labeled["consensus_ratio"]

    def bin_ratio(x):
        if x < 0.1:
            return "0.0_0.1"
        elif x < 0.5:
            return "0.1_0.5"
        return "0.5_1.0"

    final_labeled["ratio_bin"] = final_labeled["ratio"].apply(bin_ratio)

    output_cols = [
        "transcript_id", "transcript_position", "motif",
        "start", "end", "n_reads", "ratio", "ratio_bin"
    ]

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, args.output_filename)
    final_labeled[output_cols].to_csv(output_path, index=False)

    print(f"\nDone. Training data saved to: {output_path}")
    print(f"  Total sites: {len(final_labeled)} "
          f"(positive: {len(pos_df)}, negative: {len(neg_sampled)})")


if __name__ == "__main__":
    main()
