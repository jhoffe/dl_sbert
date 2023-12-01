import torch
import lightning as L
from sklearn.metrics import f1_score


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


def find_threshold(trainer: L.Trainer, module: L.LightningModule, data_module: L.LightningDataModule) -> torch.Tensor:
    predictions: list[dict[str, torch.Tensor]] = trainer.predict(module, datamodule=data_module)

    y = torch.cat([batch["y"] for batch in predictions]).cpu().numpy()
    y_hat = torch.cat([batch["y_hat"] for batch in predictions]).cpu().numpy()

    best_threshold = 0
    best_f1 = 0

    for threshold in range(0, 100, 1):
        threshold /= 100

        y_pred = y_hat > threshold

        f1 = f1_score(y, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return torch.Tensor([best_threshold])
