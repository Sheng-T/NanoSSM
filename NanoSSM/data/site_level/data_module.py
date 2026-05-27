import os

import lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from NanoSSM.data.site_level.load_data import JsonIndexedDataset
from torch.utils.data.dataloader import default_collate

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    return batch if batch else None

class MultiDataModule(pl.LightningDataModule):
    def __init__(self, data_paths, batch_size=1, num_workers=4, norm_path=None):
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

        print(
            f"[INFO] MultiDataModule init，loading {len(self.data_paths)} dataset path"
        )

    def setup(self, stage=None):
        train_info_list = []
        train_json_list = []
        val_info_list = []
        val_json_list = []

        suffix = "GNS_motif_1.0.info"
        INFO_NAME = "train_" + suffix
        VAL_NAME = "val_" + suffix
        JSON_NAME = "data.json"

        for path in self.data_paths:
            path = path.strip()
            t_info = os.path.join(path, INFO_NAME)
            v_info = os.path.join(path, VAL_NAME)
            j_file = os.path.join(path, JSON_NAME)

            if os.path.exists(t_info) and os.path.exists(j_file):
                train_info_list.append(t_info)
                train_json_list.append(j_file)

            if os.path.exists(v_info) and os.path.exists(j_file):
                val_info_list.append(v_info)
                val_json_list.append(j_file)

        if train_info_list:
            self.train_ds = JsonIndexedDataset(
                info_paths=train_info_list,
                json_paths=train_json_list,
                split="train",
                norm_path=self.norm_path,
            )
        else:
            raise ValueError("[Error] No train_info files found!")

        if val_info_list:
            self.val_ds = JsonIndexedDataset(
                info_paths=val_info_list,
                json_paths=val_json_list,
                split="val",
                norm_path=self.norm_path,
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
            prefetch_factor=4 if self.num_workers > 0 else 2,
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
            prefetch_factor=4 if self.num_workers > 0 else 2,
            collate_fn=safe_collate,
        )
