from multiprocessing import cpu_count

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from dataset import MSMarcoDataset, MSMarcoDatasetTest


class MSMarcoDataModule(L.LightningDataModule):
    def __init__(self, dev: bool = False, seed: int = 42, num_workers: int | None = None):
        super().__init__()

        self.dev = dev
        self.seed = seed
        self.num_workers = cpu_count() if num_workers is None else num_workers

        self.generator = torch.Generator().manual_seed(self.seed)

        self.train = MSMarcoDataset(
            qrels_path="data/qrels.train.tsv" if not dev else "data/qrels.dev.small.tsv",
            queries_path="data/queries.train.tsv" if not dev else "data/queries.dev.small.tsv",
            passages_path="data/collection.tsv" if not dev else "data/collection.small.tsv",
        )
        self.test = MSMarcoDatasetTest(
            dataset_path="data/msmarco-passagetest2019-top1000.tsv" if not dev else "data/msmarco-passagetest2019-top1000.small.tsv",
        )

    def setup(self, stage: str) -> None:
        self.train, self.val = random_split(self.train, [0.9, 0.1], generator=self.generator)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            pin_memory=True,
            batch_size=512,
            shuffle=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            pin_memory=True,
            batch_size=512,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            pin_memory=True,
            batch_size=512,
            num_workers=self.num_workers
        )
