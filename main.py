import torch
import lightning as L
from sentence_transformers import SentenceTransformer

from datamodule import MSMarcoDataModule
from sbert_model import SBERT
from lightning.pytorch.loggers import WandbLogger
import click
import os
import wandb


@click.command()
@click.option("--batch_size", default=256, type=int)
@click.option("--model", default="bert-base-uncased", type=str)
@click.option("--epochs", default=1, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--num_workers", default=None, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--precision", default=None, type=str)
@click.option("--dev", default=False, type=bool, is_flag=True)
@click.option("--num_steps", default=5000, type=int)
@click.option("--compile", default=False, type=bool)
def train(
    batch_size: int,
    model: str,
    epochs: int,
    seed: int,
    num_workers: int,
    lr: float,
    precision: str | None = None,
    dev: bool = False,
    num_steps: int = -1,
    compile: bool = True,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    L.seed_everything(seed)

    torch.set_float32_matmul_precision("high")
    model = torch.compile(SentenceTransformer(model), mode="max-autotune", disable=not compile)

    logger = WandbLogger(
        project="dl_sbert", entity="colodingdongs", log_model="all"
    )

    logger.experiment.config.update(
        {
            "batch_size": batch_size,
            "model": model,
            "epochs": epochs,
            "seed": seed,
            "num_workers": num_workers,
            "lr": lr,
            "precision": precision,
            "dev": dev,
            "max_steps": num_steps
        }
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        deterministic=True,
        logger=logger,
        precision=precision,
        fast_dev_run=dev,
        max_steps=num_steps,
    )

    datamodule = MSMarcoDataModule(
        batch_size=batch_size, num_workers=num_workers
    )

    l_module = SBERT(model, torch.nn.MSELoss(), lr=lr)

    trainer.fit(l_module, datamodule)
    trainer.test(l_module, datamodule)


if __name__ == "__main__":
    train()
