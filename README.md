# NanoSSM

NanoSSM is a deep learning framework for site-level N6-methyladenosine (m6A) stoichiometry estimation from Oxford Nanopore direct RNA sequencing (DRS) data, with a focus on the RNA004 chemistry. It introduces an interaction-aware aggregation module that explicitly models inter-read interactions at each transcriptomic site, enabling more robust stoichiometry quantification and confidence-based site ranking.

NanoSSM uses a Mamba2-based encoder for efficient read-level feature extraction and employs a cross-method consensus pseudo-labeling strategy to improve cross-cell-line generalization.

## Features

- **Interaction-aware site aggregation**: explicitly models dependencies among reads at the same site, improving stoichiometry estimation, confidence-aware site prioritization, and suppression of false positive predictions
- **Mamba2-based read encoder**: selective state-space modeling for efficient integration of sequence and signal features from nanopore reads
- **RNA004 chemistry**: designed and validated on Oxford Nanopore RNA004 direct RNA sequencing data

---

## Installation

**Requirements**: Python 3.10, CUDA 11.8, [f5c](https://github.com/hasindu2008/f5c)

```bash
git clone https://github.com/Sheng-T/NanoSSM.git
cd NanoSSM
```

```bash
conda create -n nanossm python=3.10
conda activate nanossm
```

PyTorch must be installed before the other dependencies:

```bash
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

>  **Note**: If online installation fails via `pip`, you can download the pre-compiled `.whl` files from the official releases:
> - [**causal-conv1d**](https://github.com/Dao-AILab/causal-conv1d/releases) 
> - [**mamba-ssm**](https://github.com/state-spaces/mamba/releases)
> 
> Make sure to choose the versions that match your Python, PyTorch, and CUDA environment.

Then add the project to your `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/NanoSSM:$PYTHONPATH
```

---

## Usage

```
Raw nanopore reads (RNA004 DRS)
       ↓
  1. f5c eventalign  — signal alignment
       ↓
  2. prepare         — convert eventalign output to NanoSSM format
       ↓
  3. infer           — predict m6A stoichiometry
       ↓
   BED output (bedMethyl format)
```

---

### Step 1 — Signal alignment with f5c

The RNA004 k-mer model (`rna004.nucleotide.5mer.model`) can be downloaded from:
https://raw.githubusercontent.com/hasindu2008/f5c/v1.3/test/rna004-models/rna004.nucleotide.5mer.model

```bash
f5c eventalign --rna --min-mapq 0 \
    -b <sorted.bam> \
    -r <reads.fastq> \
    -g <transcriptome.fa> \
    -o <output.eventalign.tsv> \
    --pore rna004 \
    --slow5 <input.slow5> \
    --signal-index --scale-events \
    --kmer-model rna004.nucleotide.5mer.model \
    -t <threads>
```

---

### Step 2 — Prepare data

```bash
python NanoSSM/cli/prepare.py \
    --eventalign      <output.eventalign.tsv> \
    --out_dir         <output_dir> \
    --n_processes     40 \
    --max_signal_len  64 \
    --min_segment_count 20 \
    --n_neighbors     2
```

`--max_signal_len` controls how many signal points are retained per base position (64 recommended for RNA004). `--min_segment_count` sets the minimum number of signal segments required to keep a read. `--n_neighbors` specifies how many flanking positions are included as context features around each candidate site. `--n_processes` sets the number of parallel worker processes.

---

### Step 3 — Infer

```bash
python NanoSSM/cli/infer.py \
    --model      models/model.ckpt \
    --data_path  <output_dir>/data.json \
    --info_path  <output_dir>/data.info \
    --norm_path  models/norm \
    --output_dir ./result \
    --device     0 \
    --num_workers 8 \
    --overwrite
```

Results are written to `result/infer_site_prob.bed` in bedMethyl-compatible format:

```
chrom  start  end  motif  score  strand  start  end  color  N_valid_cov  percent_modified
```

---

### Train (optional)

```bash
python NanoSSM/cli/train.py \
    --path       <prepared_data_dir> \
    --save_dir   ./result \
    --type       site \
    --device     0 \
    --batch_size 200 \
    --epochs     100
```

`--norm_path` is optional. If not provided, normalization statistics will be computed from the training data and saved to the output directory.

Fine-tuning from a pretrained checkpoint:

```bash
python NanoSSM/cli/train.py \
    --path     <prepared_data_dir> \
    --model    models/model.ckpt \
    --save_dir ./result \
    --type     site \
    --freeze
```
## Citation

If you use NanoSSM in your research, please cite:
> [paper citation placeholder]
