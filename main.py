import torch
import lightning as L
from sentence_transformers import SentenceTransformer

from datamodule import MSMarcoDataModule
from sbert_model import SBERT
from lightning.pytorch.loggers import WandbLogger
import click


@click.command()
@click.option("--batch_size", default=256, type=int)
@click.option("--model", default="bert-base-nli-mean-tokens", type=str)
@click.option("--epochs", default=1, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--num_workers", default=None, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--precision", default=None, type=str)
@click.option("--sample_negatives", default=False, type=bool)
def train(batch_size: int, model: str, epochs: int, seed: int, num_workers: int, lr: float,
          precision: str | None = None, sample_negatives: bool = False):
    L.seed_everything(seed)

    model = SentenceTransformer(model)

    logger = WandbLogger(
        project="dl_sbert",
        entity="colodingdongs",
        log_model="all"
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        deterministic=True,
        logger=logger,
        precision=precision
    )

    datamodule = MSMarcoDataModule(batch_size=batch_size, num_workers=num_workers)

    l_module = SBERT(model, torch.nn.MSELoss(), lr=lr)

    trainer.fit(l_module, datamodule)
    trainer.test(l_module, datamodule)


if __name__ == "__main__":
    train()
