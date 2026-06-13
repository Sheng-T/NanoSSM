import os
import random

import mmap
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import pandas as pd
import orjson, struct

def unpack_tobytes(value: bytes):
    off = 0

    def _read():
        nonlocal off
        ndim, *shape = struct.unpack("4i", value[off : off + 16])
        off += 16
        arr = np.frombuffer(
            value[off : off + np.prod(shape) * 4], dtype=np.float32
        ).reshape(shape)
        off += np.prod(shape) * 4
        return arr

    seq = _read()
    signal = _read()
    stat = _read()
    ratio = struct.unpack("f", value[off : off + 4])[0]
    return seq, signal, stat, ratio

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    return batch if batch else None

def get_test_loader(data_dirs, batch_size=1, num_workers=4, norm_path=None, max_read=512):
    if isinstance(data_dirs, str):
        paths = [p.strip() for p in data_dirs.split(",")]
    elif isinstance(data_dirs, list):
        paths = data_dirs
    else:
        paths = data_dirs

    test_info_paths = []
    json_paths = []

    for p in paths:
        t_info = os.path.join(p, "test_GNS_motif_1.0.info")
        j_file = os.path.join(p, "data.json")

        if os.path.exists(t_info) and os.path.exists(j_file):
            test_info_paths.append(t_info)
            json_paths.append(j_file)

    if not test_info_paths:
        raise ValueError("[Error] No test_info files found!")

    print(f"[INFO] Test datasets: {len(test_info_paths)}")

    test_dataset = JsonIndexedDataset(
        info_paths=test_info_paths,
        json_paths=json_paths,
        split="test",
        norm_path=norm_path,
        infer_max_reads=max_read
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else 2,
        collate_fn=safe_collate,
    )

def get_site_dataloader(
    data_path: str,
    info_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    norm_path: str = None,
    max_reads: int=1024
) -> DataLoader:
    dataset = JsonIndexedDataset(
        info_path, json_paths=data_path, split="infer", norm_path=norm_path,
        infer_max_reads=max_reads
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else 2,
        collate_fn=safe_collate
    )

    return dataloader

class JsonIndexedDataset(Dataset):
    def __init__(
        self,
        info_paths,
        json_paths,
        split="train",
        norm_path=None,
        use_signal=False,
        do_norm=True,
        dorpout=False,
        infer_max_reads=1024,
        train_max_reads=256,
        use_max=True
    ):
        self.info_paths = [info_paths] if isinstance(info_paths, str) else info_paths
        self.json_paths = [json_paths] if isinstance(json_paths, str) else json_paths

        assert len(self.info_paths) == len(self.json_paths), "info != json"

        df_list = []
        for i, p in enumerate(self.info_paths):
            temp_df = pd.read_csv(p)
            temp_df["file_idx"] = i
            df_list.append(temp_df)

        self.use_max = use_max

        self.info_df = pd.concat(df_list, ignore_index=True)
        self.split = split
        self.use_signal = use_signal
        self.do_norm = do_norm
        self.dorpout = dorpout
        self.train_max_reads = train_max_reads
        self.infer_max_reads = infer_max_reads

        self.file_handlers = {}

        if self.do_norm:
            if norm_path:
                norm_dir = norm_path
                prefix = ""
            else:
                norm_dir = os.path.dirname(self.json_paths[0])
                prefix = "merged/" if len(self.json_paths) > 1 else ""

            self.global_norm_path = os.path.join(norm_dir, f"{prefix}global_norm.npy")
            self.kmer_norm_path = os.path.join(norm_dir, f"{prefix}kmer_norm.npy")

            if split == "train":
                if not os.path.exists(self.global_norm_path):
                    self._compute_and_save_global_norm()
                if not os.path.exists(self.kmer_norm_path):
                    self._compute_and_save_kmer_norm()

            gn = np.load(self.global_norm_path, allow_pickle=True).item()
            self.global_mean = torch.tensor(gn["mean"], dtype=torch.float32)
            self.global_std = torch.tensor(gn["std"], dtype=torch.float32)
            self._build_lookup_table()

        if use_signal:
            norm_dir = norm_path if norm_path else os.path.dirname(self.json_paths[0])
            self.global_signal_norm_path = os.path.join(
                norm_dir, "global_signal_norm.npy"
            )
            if split == "train" and not os.path.exists(self.global_signal_norm_path):
                self._compute_and_save_global_signal_norm()
            sn = np.load(self.global_signal_norm_path, allow_pickle=True).item()
            self.g_signal_mean = torch.tensor(sn["mean"], dtype=torch.float32)
            self.g_signal_std = torch.tensor(sn["std"], dtype=torch.float32)

    def _compute_and_save_global_signal_norm(self):
        print("[Norm] Calculating signal norm using streaming...")
        sum_x, sum_x2, n = 0, 0, 0
        with open(self.json_paths[0], "rb") as f:
            for idx in range(len(self.info_df)):
                row = self.info_df.iloc[idx]
                f.seek(int(row["start"]))
                obj = orjson.loads(f.read(int(row["end"]) - int(row["start"])))
                data = obj[row["transcript_id"]][str(row["transcript_position"])][
                    row["motif"]
                ]

                s = np.array(data["signal"], dtype=np.float32).reshape(-1)
                sum_x += np.sum(s)
                sum_x2 += np.sum(s**2)
                n += s.size

        mean = sum_x / n
        std = np.sqrt(max(0, (sum_x2 / n) - (mean**2)))
        np.save(self.global_signal_norm_path, {"mean": mean, "std": std})
        print(f"[Norm] Signal Norm saved. Mean: {mean:.4f}")

    def _compute_and_save_global_norm(self):

        print(f"[Norm] Calculating global norm for {len(self.json_paths)} files...")
        all_stats = []
        for f_idx, group in self.info_df.groupby("file_idx"):
            json_p = self.json_paths[int(f_idx)]
            with open(json_p, "rb") as f:
                for _, row in group.iterrows():
                    f.seek(int(row["start"]))
                    obj = orjson.loads(f.read(int(row["end"]) - int(row["start"])))
                    data = obj[row["transcript_id"]][str(row["transcript_position"])][
                        row["motif"]
                    ]

                    stat = np.array(data["stat"], dtype=np.float32)
                    seq = np.array(data["seq"], dtype=np.int64)
                    n_reads, n_windows, k = seq.shape
                    stat = stat.reshape(-1, stat.shape[-1])
                    seq = seq.reshape(-1, k)

                    mask = np.any(stat != 0, axis=1) & np.any(seq != 0, axis=1)
                    if np.any(mask):
                        all_stats.append(stat[mask])

        all_stats = np.concatenate(all_stats, axis=0)

        all_stats[:, 0] = np.log1p(np.clip(all_stats[:, 0], 0, None))

        gn = {"mean": all_stats.mean(axis=0), "std": all_stats.std(axis=0)}
        os.makedirs(os.path.dirname(self.global_norm_path), exist_ok=True)
        np.save(self.global_norm_path, gn)
        print("[Norm] Global Norm saved.")

    def _compute_and_save_kmer_norm(self):

        print(f"[Norm] Calculating k-mer norm for {len(self.json_paths)} files...")
        kmer_dict = {}
        for f_idx, group in self.info_df.groupby("file_idx"):
            json_p = self.json_paths[int(f_idx)]
            with open(json_p, "rb") as f:
                for _, row in group.iterrows():
                    f.seek(int(row["start"]))
                    obj = orjson.loads(f.read(int(row["end"]) - int(row["start"])))
                    data = obj[row["transcript_id"]][str(row["transcript_position"])][
                        row["motif"]
                    ]

                    stat = np.array(data["stat"], dtype=np.float32)
                    seq = np.array(data["seq"], dtype=np.int64)
                    n_reads, n_windows, k = seq.shape
                    stat = stat.reshape(-1, stat.shape[-1])
                    seq = seq.reshape(-1, k)

                    stat[:, 0] = np.log1p(np.clip(stat[:, 0], 0, None))

                    mask = np.any(stat != 0, axis=1) & np.any(seq != 0, axis=1)

                    v_stat, v_seq = stat[mask], seq[mask]
                    for s, q in zip(v_stat, v_seq):
                        q_tuple = tuple(q)
                        if q_tuple not in kmer_dict:
                            kmer_dict[q_tuple] = []
                        kmer_dict[q_tuple].append(s)

        res = {}
        for k, v in kmer_dict.items():
            if len(v) >= 20:
                arr = np.array(v)
                res[k] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
        os.makedirs(os.path.dirname(self.kmer_norm_path), exist_ok=True)
        np.save(self.kmer_norm_path, res)
        print(f"[Norm] K-mer Norm saved. Count: {len(res)}")

    def _build_lookup_table(self):

        k_norms = np.load(self.kmer_norm_path, allow_pickle=True).item()

        if len(k_norms) > 0:
            first_key = next(iter(k_norms))
            kmer = len(first_key)

        else:
            kmer = 5
            print("[Warning] kmer_norm is empty, defaulting k=5")

        repeat_factors = [6] * kmer + [1]
        view_shape = [1] * kmer + [5]

        g_mean = self.global_mean.detach().cpu().view(-1)
        g_std = self.global_std.detach().cpu().view(-1)

        self.mean_table = g_mean.view(*view_shape).repeat(*repeat_factors)
        self.std_table = g_std.view(*view_shape).repeat(*repeat_factors)

        print(f"[Init] Table initialized with shape: {self.mean_table.shape}")

        count = 0
        for k, v in k_norms.items():
            if len(k) != kmer:
                continue

            if all(0 <= i <= 5 for i in k):

                m = torch.as_tensor(v["mean"], dtype=torch.float32).view(-1)
                s = torch.as_tensor(v["std"], dtype=torch.float32).view(-1)

                try:

                    self.mean_table[k] = m
                    self.std_table[k] = s
                    count += 1
                except Exception as e:
                    print(f"Error indexing with key {k}: {e}")
                    raise e

        print(f"[Init] Successfully filled {count} k-mers into lookup table.")

    def _get_handle(self, file_idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        if worker_id not in self.file_handlers:
            self.file_handlers[worker_id] = {}

        if file_idx not in self.file_handlers[worker_id]:
            self.file_handlers[worker_id][file_idx] = open(
                self.json_paths[file_idx], "rb"
            )

        return self.file_handlers[worker_id][file_idx]

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        row = self.info_df.iloc[idx]
        file_idx = int(row["file_idx"])
        f = self._get_handle(file_idx)

        f.seek(int(row["start"]))
        obj = orjson.loads(f.read(int(row["end"]) - int(row["start"])))

        data = obj[row["transcript_id"]][str(row["transcript_position"])][row["motif"]]

        stat = torch.from_numpy(np.array(data["stat"], dtype=np.float32))
        seq = torch.from_numpy(np.array(data["seq"], dtype=np.int64))
        n, w, _ = stat.shape

        stat[:, :, 0] = torch.log1p(torch.clamp(stat[:, :, 0], min=0))

        if self.do_norm:

            m = self.mean_table[
                seq[..., 0], seq[..., 1], seq[..., 2], seq[..., 3], seq[..., 4]
            ]
            s = self.std_table[
                seq[..., 0], seq[..., 1], seq[..., 2], seq[..., 3], seq[..., 4]
            ]

            stat = (stat - m) / (s + 1e-6)

        valid_mask = (seq != 0).any(dim=-1, keepdim=True)
        stat = stat * valid_mask

        signal = torch.zeros(1)
        if self.use_signal and "signal" in data:
            raw_s = torch.from_numpy(np.array(data["signal"], dtype=np.float32))

            signal = (raw_s - self.g_signal_mean) / (self.g_signal_std + 1e-6)

        if self.dorpout and self.split == "train" and n > 32:
            dropout_rates = {128: 0.2, 64: 0.1, 32: 0.05}
            rate = 0.05
            for threshold, r in dropout_rates.items():
                if n > threshold:
                    rate = r
                    break

            num_keep = max(32, int(n * (1 - rate)))
            indices = torch.randperm(n)[:num_keep]
            stat = stat[indices]
            seq = seq[indices]
            if signal.dim() > 1:
                signal = signal[indices]

        orig_n = n
        if self.use_max:
            MAX_READS = self.train_max_reads if self.split == 'train' else self.infer_max_reads
            if n > MAX_READS:
                indices = torch.randperm(n)[:MAX_READS]
                stat = stat[indices]
                seq = seq[indices]
                if signal.dim() > 1:
                    signal = signal[indices]
                n = MAX_READS

        res = {"seq": seq.float(), "stat": stat, "signal": signal}

        if self.split != "infer":
            res["ratio"] = torch.tensor([float(row["ratio"])], dtype=torch.float32)

        if self.split == "infer" or self.split == "test":
            res["info"] = {
                "transcript_id": row["transcript_id"],
                "position": str(row["transcript_position"]),
                "num": orig_n,
                "motif": row["motif"],
            }

        return res
