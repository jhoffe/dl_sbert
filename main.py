import torch
import lightning as L
from sentence_transformers import SentenceTransformer

from datamodule import MSMarcoDataModule
from sbert_model import SBERT
from lightning.pytorch.loggers import WandbLogger


def train():
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    logger = WandbLogger(
        project="dl_sbert",
        entity="jhoffe",
    )

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        logger=logger
    )

    datamodule = MSMarcoDataModule()

    l_module = SBERT(model, torch.nn.MSELoss())

    trainer.fit(l_module, datamodule)
    trainer.test(l_module, datamodule)


if __name__ == "__main__":
    train()
