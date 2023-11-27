from multiprocessing import cpu_count

import lightning as L
from torch.utils.data import DataLoader
import webdataset as wds


class MSMarcoDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 256, num_workers: int | None = None, dataset_length: int = 50_000):
        super().__init__()

        self.num_workers = (
            cpu_count() // 2 if num_workers is None else num_workers
        )
        self.batch_size = batch_size

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.dataset_length = dataset_length

    @staticmethod
    def to_string(string: bytes) -> str:
        return string.decode("utf-8")

    @staticmethod
    def to_float(string: bytes) -> float:
        return float(string.decode("utf-8"))

    @staticmethod
    def rating_to_class(rating: bytes) -> float:
        rating = float(rating.decode("utf-8"))

        return 1. if rating >= 1 else 0.

    def setup(self, stage: str) -> None:
        self.train_dataset = (
            wds.WebDataset(
                "data/processed/train-{0..17}.tar", shardshuffle=True
            )
            .with_length(self.dataset_length)
            .shuffle(1000)
            .to_tuple("query.pyd", "passage.pyd", "label.cls")
            .map_tuple(self.to_string, self.to_string, self.to_float)
        )
        self.validation_dataset = (
            wds.WebDataset("data/processed/validation-{0..1}.tar")
            .to_tuple("query.pyd", "passage.pyd", "label.cls")
            .map_tuple(self.to_string, self.to_string, self.to_float)
        )

        self.test_dataset = (
            wds.WebDataset("data/processed/test.tar")
            .to_tuple("query.pyd", "passage.pyd", "rating.cls")
            .map_tuple(self.to_string, self.to_string, self.rating_to_class)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset.batched(self.batch_size),
            pin_memory=True,
            batch_size=None,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset.batched(self.batch_size),
            batch_size=None,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset.batched(self.batch_size),
            batch_size=None,
            pin_memory=True,
            num_workers=self.num_workers,
        )
