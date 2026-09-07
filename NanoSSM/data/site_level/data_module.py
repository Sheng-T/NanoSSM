import os

import lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from NanoSSM.data.site_level.load_data import JsonIndexedDataset
from torch.utils.data.dataloader import default_collate

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    return batch if batch else None

class MultiDataModule(pl.LightningDataModule):
    def __init__(self, data_paths, batch_size=1, num_workers=4, norm_path=None, train_max_reads=256, infer_max_reads=512):
        super().__init__()

        if isinstance(data_paths, str):
            self.data_paths = [path.strip() for path in data_paths.split(",")]
        elif isinstance(data_paths, list):
            self.data_paths = data_paths
        else:
            raise ValueError("data_paths must str or List[str]")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm_path = norm_path
        self.train_max_reads = train_max_reads
        self.infer_max_reads = infer_max_reads

        print(
            f"[INFO] MultiDataModule init，loading {len(self.data_paths)} dataset path"
        )

    def setup(self, stage=None):
        train_info_list = []
        train_json_list = []
        val_info_list = []
        val_json_list = []

        for path in self.data_paths:
            path = path.strip()

            new_train_info_labeled = os.path.join(
                path, "train", "data.labeled.info"
            )
            new_train_info = os.path.join(
                path, "train", "data.info"
            )
            new_train_json = os.path.join(
                path, "train", "data.json"
            )

            new_val_info_labeled = os.path.join(
                path, "val", "data.labeled.info"
            )
            new_val_info = os.path.join(
                path, "val", "data.info"
            )
            new_val_json = os.path.join(
                path, "val", "data.json"
            )

            old_train_info = os.path.join(
                path, "train_GNS_motif_1.0.info"
            )
            old_val_info = os.path.join(
                path, "val_GNS_motif_1.0.info"
            )
            old_json = os.path.join(
                path, "data.json"
            )

            # ---------- train ----------
            if os.path.exists(new_train_json):
                if os.path.exists(new_train_info_labeled):
                    t_info = new_train_info_labeled
                elif os.path.exists(new_train_info):
                    t_info = new_train_info
                else:
                    t_info = None

                if t_info is not None:
                    train_info_list.append(t_info)
                    train_json_list.append(new_train_json)
                    print(f"[INFO] New-layout train: {t_info}")

            elif os.path.exists(old_train_info) and os.path.exists(old_json):
                train_info_list.append(old_train_info)
                train_json_list.append(old_json)
                print(f"[INFO] Legacy train: {old_train_info}")

            # ---------- val ----------
            if os.path.exists(new_val_json):
                if os.path.exists(new_val_info_labeled):
                    v_info = new_val_info_labeled
                elif os.path.exists(new_val_info):
                    v_info = new_val_info
                else:
                    v_info = None

                if v_info is not None:
                    val_info_list.append(v_info)
                    val_json_list.append(new_val_json)
                    print(f"[INFO] New-layout val: {v_info}")

            elif os.path.exists(old_val_info) and os.path.exists(old_json):
                val_info_list.append(old_val_info)
                val_json_list.append(old_json)
                print(f"[INFO] Legacy val: {old_val_info}")

        if train_info_list:
            self.train_ds = JsonIndexedDataset(
                info_paths=train_info_list,
                json_paths=train_json_list,
                split="train",
                norm_path=self.norm_path,
                train_max_reads=self.train_max_reads
            )
        else:
            raise ValueError("[Error] No train_info files found!")

        if val_info_list:
            self.val_ds = JsonIndexedDataset(
                info_paths=val_info_list,
                json_paths=val_json_list,
                split="val",
                norm_path=self.norm_path,
                infer_max_reads=self.infer_max_reads
            )
        else:
            self.val_ds = None
            print("[Warning] No val_info files found!")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):

        if not self.val_ds:
            return None

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=safe_collate,
        )
