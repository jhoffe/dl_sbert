import torch
import lightning as L
from sklearn.metrics import f1_score


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


@torch.no_grad()
def find_threshold(trainer: L.Trainer, module: L.LightningModule, data_module: L.LightningDataModule, map: bool = False) -> tuple[float, float]:
    module.eval()
    predictions: list[dict[str, torch.Tensor]] = trainer.predict(module, datamodule=data_module)
    module.train()

    y = torch.cat([batch["y"] for batch in predictions]).cpu().numpy()
    y_hat = torch.cat([batch["y_hat"] for batch in predictions]).cpu().numpy()

    if map:
        y_hat = normalize(y_hat)

    best_threshold = 0
    best_f1 = 0

    for threshold in range(0, 100, 1):
        threshold /= 100

        y_pred = y_hat > threshold

        f1 = f1_score(y, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1
