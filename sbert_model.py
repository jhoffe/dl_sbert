import torch
from torch import nn, Tensor
import lightning as L
from sentence_transformers import SentenceTransformer, models
import torchmetrics
from sentence_transformers.util import batch_to_device
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay


class SBERT(L.LightningModule):
    def __init__(
            self,
            model: SentenceTransformer,
            criterion: nn.Module,
            lr: float = 1e-5,
    ) -> None:
        super().__init__()

        self.model = model.to(self.device)
        self.pooling = models.Pooling(model.get_sentence_embedding_dimension())

        self.cosine = nn.CosineSimilarity()
        self.criterion = criterion

        self.is_cosine_embedding_loss = isinstance(self.criterion, nn.CosineEmbeddingLoss)

        self.lr = lr

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "train_accuracy_k1": torchmetrics.Accuracy(task="binary"),
                "train_accuracy_k3": torchmetrics.Accuracy(
                    task="binary", top_k=3
                ),
                "train_accuracy_k5": torchmetrics.Accuracy(
                    task="binary", top_k=5
                ),
            }
        )

        self.validation_metrics = torchmetrics.MetricCollection(
            {
                "val_accuracy_k1": torchmetrics.Accuracy(task="binary"),
                "val_accuracy_k3": torchmetrics.Accuracy(
                    task="binary", top_k=3
                ),
                "val_accuracy_k5": torchmetrics.Accuracy(
                    task="binary", top_k=5
                ),
            }
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "test_accuracy_k1": torchmetrics.Accuracy(task="binary"),
                "test_accuracy_k3": torchmetrics.Accuracy(
                    task="binary", top_k=3
                ),
                "test_accuracy_k5": torchmetrics.Accuracy(
                    task="binary", top_k=5
                ),
            }
        )

        self.y_test = []
        self.y_hat_test = []

        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, x) -> Tensor:
        tokens = self.model.tokenize(x)
        tokens = batch_to_device(tokens, self.device)
        x = self.model(tokens)
        x = self.pooling(x)

        return x

    def training_step(self, batch) -> Tensor:
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        if self.is_cosine_embedding_loss:
            loss = self.criterion(
                embeddings_question, embeddings_answer, y
            )
        else:
            similarity = self.cosine(embeddings_question, embeddings_answer)
            y_hat, y = similarity.to(torch.float32), y.to(torch.float32)
            loss = self.criterion(y_hat, y)
            self.train_metrics(y_hat.detach().cpu(), y.detach().cpu())

            self.log_dict(
                self.train_metrics, on_step=True, on_epoch=True, prog_bar=False
            )

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch):
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        if self.is_cosine_embedding_loss:
            loss = self.criterion(
                embeddings_question, embeddings_answer, y
            )
        else:
            similarity = self.cosine(embeddings_question, embeddings_answer)
            y_hat, y = similarity.to(torch.float32), y.to(torch.float32)

            loss = self.criterion(y_hat, y)
            self.validation_metrics(y_hat.detach().cpu(), y.detach().cpu())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        y_hat, y = similarity.to(torch.float32), y.to(torch.float32)

        if self.is_cosine_embedding_loss:
            loss = self.criterion(
                embeddings_question, embeddings_answer, y
            )
        else:
            loss = self.criterion(y_hat, y)

        self.test_metrics(y_hat.detach().cpu(), y.detach().cpu())
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "y_hat": y_hat.detach().cpu(),
            "y": y.detach().cpu(),
            "loss": loss.detach().cpu()
        }

    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.y_test.append(outputs["y"])
        self.y_hat_test.append(outputs["y_hat"])

    def on_test_end(self) -> None:
        y = torch.cat(self.y_test, dim=0)
        y_hat = torch.cat(self.y_hat_test, dim=0)

        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        y_pred = np.where(y_hat > 0.5, 1, 0)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_hat)
        average_precision = average_precision_score(y, y_hat)

        self.logger.experiment.log(
            {
                "test_confusion_matrix": ConfusionMatrixDisplay.from_predictions(y, y_pred).figure_,
                "test_roc_curve": RocCurveDisplay.from_predictions(y, y_pred).figure_,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_roc_auc": roc_auc,
                "test_average_precision": average_precision
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
