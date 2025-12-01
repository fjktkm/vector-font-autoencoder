from collections.abc import Sequence
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from torchfont.datasets import FontFolder, GoogleFonts
from torchfont.transforms import (
    Compose,
    LimitSequenceLength,
    Patchify,
)


def collate_fn(
    batch: Sequence[tuple[tuple[Tensor, Tensor], tuple[int, int]]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    types_list = [types for (types, _), _ in batch]
    coords_list = [coords for (_, coords), _ in batch]
    style_label_list = [style for _, (style, _) in batch]
    content_label_list = [content for _, (_, content) in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    style_label_tensor = torch.as_tensor(style_label_list, dtype=torch.long)
    content_label_tensor = torch.as_tensor(content_label_list, dtype=torch.long)

    return types_tensor, coords_tensor, style_label_tensor, content_label_tensor


class LitGoogleFonts(LightningDataModule):
    def __init__(  # noqa: PLR0913
        self,
        root: str | Path = Path("data/google_fonts"),
        ref: str = "main",
        max_len: int = 512,
        patch_size: int = 32,
        batch_size: int = 512,
        num_shards: int = 8,
        num_workers: int = 2,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(root)
        self.ref = ref
        self.max_seq_len = max_len
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_shards = num_shards
        self.prefetch_factor = prefetch_factor
        self.val_ratio = 0.005
        self.test_ratio = 0.005
        self.seed = 37414078

        transform = Compose(
            [
                LimitSequenceLength(max_len=self.max_seq_len),
                Patchify(patch_size=self.patch_size),
            ],
        )
        self.dataset = GoogleFonts(
            root=self.root,
            ref=self.ref,
            transform=transform,
            download=True,
        )
        self.commit_hash = self.dataset.commit_hash
        self.dataset_len = len(self.dataset)
        self.num_style_classes = self.dataset.num_style_classes
        self.num_content_classes = self.dataset.num_content_classes
        self.save_hyperparameters(
            {
                "commit_hash": self.commit_hash,
                "dataset_len": self.dataset_len,
                "num_style_classes": self.num_style_classes,
                "num_content_classes": self.num_content_classes,
            },
        )

    def _get_world_info(self) -> tuple[int, int]:
        if self.trainer is not None:
            return self.trainer.global_rank, self.trainer.world_size
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        length = len(self.dataset)
        rank, world_size = self._get_world_info()
        starts = [(i * length) // self.num_shards for i in range(self.num_shards + 1)]
        g = torch.Generator().manual_seed(self.seed)

        self.train_shards: list[Subset] = []
        self.val_shards: list[Subset] = []
        self.test_shards: list[Subset] = []

        for shard_id in range(self.num_shards):
            if shard_id % world_size != rank:
                continue

            start, end = starts[shard_id], starts[shard_id + 1]
            if end <= start:
                continue

            m = end - start
            n_test = int(m * self.test_ratio)
            n_val = int(m * self.val_ratio)
            n_train = m - n_val - n_test

            perm_local = torch.randperm(m, generator=g).tolist()
            idx_train_local = perm_local[:n_train]
            idx_val_local = perm_local[n_train : n_train + n_val]
            idx_test_local = perm_local[n_train + n_val :]

            idx_train = [start + j for j in idx_train_local]
            idx_val = [start + j for j in idx_val_local]
            idx_test = [start + j for j in idx_test_local]

            self.train_shards.append(Subset(self.dataset, idx_train))
            self.val_shards.append(Subset(self.dataset, idx_val))
            self.test_shards.append(Subset(self.dataset, idx_test))

    def train_dataloader(self) -> CombinedLoader:
        loaders = [
            DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate_fn,
            )
            for subset in self.train_shards
        ]
        return CombinedLoader(loaders)

    def val_dataloader(self) -> CombinedLoader:
        loaders = [
            DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate_fn,
            )
            for subset in self.val_shards
        ]
        return CombinedLoader(loaders)

    def test_dataloader(self) -> CombinedLoader:
        loaders = [
            DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate_fn,
            )
            for subset in self.test_shards
        ]
        return CombinedLoader(loaders)

    def predict_dataloader(self) -> CombinedLoader:
        loaders = [
            DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate_fn,
            )
            for subset in self.test_shards
        ]
        return CombinedLoader(loaders, mode="sequential")

    def interpolate_dataloader(self) -> CombinedLoader:
        transform = Compose(
            [
                LimitSequenceLength(max_len=self.max_seq_len),
                Patchify(patch_size=self.patch_size),
            ],
        )
        paths = [
            "data/google_fonts/ofl/notosansjp",
            "data/google_fonts/ofl/notoserifjp",
        ]
        cp_filter = [
            ord(c)
            for c in "IBIS2025情報論的学習理論ワークショップ那覇文化芸術劇場なはーと"
        ]
        datasets = [
            FontFolder(
                root=path,
                codepoint_filter=cp_filter,
                transform=transform,
            )
            for path in paths
        ]
        loaders = [
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
            )
            for dataset in datasets
        ]

        return CombinedLoader(loaders)
