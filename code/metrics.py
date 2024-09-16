from typing import Any, Callable, List

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (f1_score, matthews_corrcoef,
                             precision_recall_curve, accuracy_score,
                             mean_squared_error, mean_absolute_error,
                             roc_auc_score, auc, root_mean_squared_error)
from sklearn.metrics.pairwise import cosine_similarity


class Metrics:
    multilabel = False
    token = False

    def __init__(self):
        self.metrics = {}

    def add_metric(self, name: str, metric: Callable):
        self.metrics[name] = metric

    def get_metrics(self, names: List[str], multilabel: bool = False,
                    token: bool = False):
        metrics = Metrics()
        metrics.multilabel = multilabel
        metrics.token = token
        for name in names:
            if name in self.metrics:
                metrics.add_metric(name, self.metrics[name])
            else:
                raise ValueError(f"Metric: {name} not supported.",
                                 " Please use one of the following: ",
                                 f" {list(self.metrics.keys())}")
        return metrics

    def todict(self) -> dict:
        output = {}
        for name, fun in self.metrics.items():
            output[name] = fun
        return output

    def __call__(self, eval_prediction) -> Any:
        output = {}

        preds = eval_prediction.predictions
        refs = eval_prediction.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        if len(preds.shape) < 2 and not self.token:
            pass
        elif preds.shape[1] > 1 and not self.multilabel and not self.token:
            preds = np.argmax(preds, axis=1)
        elif preds.shape[1] == 1 and not self.multilabel and not self.token:
            preds = preds.squeeze(1)

        if preds.dtype != refs.dtype or self.multilabel and not self.token:
            preds = preds > (1 / preds.shape[1])

        if self.token:
            preds = preds.reshape(preds.shape[0] * preds.shape[1])
            refs = refs.reshape(refs.shape[0] * refs.shape[1])
            refs_mask = refs != -1
            refs = refs[refs_mask]
            preds = preds[refs_mask]

        for key, value in self.metrics.items():
            output[key] = value(preds, refs)[key]
        return output

    def __getitem__(self, idx):
        return list(self.metrics.keys())[idx]


def acc(predictions, references, **kwargs):
    return {"acc": accuracy_score(references, predictions)}


def auroc(predictions, references, **kwargs):
    return {"auroc": roc_auc_score(references, predictions)}


def precision_at_l5(predictions, references, sequence_lengths, **kwargs):
    prev_pos = 0
    correct = 0
    total = 0
    for idx, label in enumerate(references):
        pred = predictions[prev_pos:prev_pos + label.shape[0] ** 2].toarray().reshape(label.shape[0], label.shape[0], 1)
        label = torch.from_numpy(label)
        prev_pos += label.shape[0] ** 2
        length = sequence_lengths[idx]
        pred = torch.from_numpy(pred).float()
        prob = torch.nn.functional.softmax(pred).view(-1)
        most_likely = prob.topk(length // 5, sorted=False)
        selected = label.view(-1).gather(0, most_likely.indices)
        correct += selected.sum().float()
        total += selected.numel()

    return correct / total


def cosine(predictions, references, **kwargs):
    cosine = list(map(lambda x: cosine_similarity(x[0].reshape(1, -1),
                                                  x[1].reshape(1, -1)).item(),
                                                  predictions))
    cosine, references = np.array(cosine), np.array(references)
    score, _ = spearmanr(references, cosine)
    return {'cos': float(score)}


def manhattan(predictions, references, **kwargs):
    dist = list(map(lambda x: cdist(x[0].reshape(1, -1), x[1].reshape(1, -1),
                                    'cityblock'), predictions))
    norm = list(map(lambda x: (np.linalg.norm(x[0].reshape(1, -1), 1) +
                               np.linalg.norm(x[1].reshape(1, -1), 1)),
                               predictions))

    dist = np.array([1 - (d / n).item() for d, n in zip(dist, norm)])
    # dist = np.array(dist) / (np.linalg.norm(predictions) +
    #                          np.linalg.norm(predictions))
    if isinstance(references, list):
        references = np.array(references)
        # references = references.reshape(-1, 1)
    score, _ = spearmanr(dist, references)
    return {'manhattan': float(score)}


def euclidean(predictions, references, **kwargs):
    dist = list(map(lambda x: cdist(x[0].reshape(1, -1), x[1].reshape(1, -1),
                                    'euclidean'), predictions))
    norm = list(map(lambda x: (np.linalg.norm(x[0].reshape(1, -1), 2) +
                               np.linalg.norm(x[1].reshape(1, -1), 2)),
                               predictions))

    dist = np.array([1 - (d / n).item() for d, n in zip(dist, norm)])
    if isinstance(references, list):
        references = np.array(references)

    score, _ = spearmanr(dist, references)
    return {'euclidean': float(score)}


def f1_max(predictions, references, **kwargs):
    beta = 1.0
    fbeta = 0.0
    for i in range(references.shape[-1]):
        if np.sum(references[:, i]) == 0.0:
            continue
        precision, recall, _ = precision_recall_curve(
            y_true=references[:, i], probas_pred=predictions[:, i]
        )
        numerator = (1 + beta**2) * (precision * recall)
        denominator = ((beta**2 * precision) + recall)
        a = np.divide(numerator, denominator,
                      out=np.zeros_like(numerator),
                      where=(denominator != 0))
        fbeta += np.nanmax(a / references.shape[1])
    return {"f1_max": float(fbeta)}


def f1_binary(predictions, references, **kwargs):
    score = f1_score(
        references, predictions, average='binary', zero_division=0
    )
    return {"f1": float(score) if score.size == 1 else score}


def f1_weighted(predictions, references, **kwargs):
    score = f1_score(references, predictions, average='weighted', zero_division=0, **kwargs)
    return {"f1_weighted": float(score) if score.size == 1 else score}


def mcc(predictions, references, **kwargs):
    score = matthews_corrcoef(
        references, predictions
    )
    return {"mcc": float(score)}


def spcc(predictions, references, **kwargs):
    corr, p_value = spearmanr(references, predictions)
    return {"spcc": float(corr)}


def pcc(predictions, references, **kwargs):
    corr, p_value = pearsonr(references, predictions)
    return {"pcc": float(corr)}


def mse(predictions, references, **kwargs):
    return {'mse': mean_squared_error(references, predictions)}


def rmse(predictions, references, **kwargs):
    return {'rmse': root_mean_squared_error(references, predictions)}


def mae(predictions, references, **kwargs):
    return {'mae': mean_absolute_error(references, predictions)}


def aupr(predictions, references, **kwargs):
    precision, recall, _ = precision_recall_curve(
        y_true=references, probas_pred=predictions
    )
    return {'aupr': auc(precision, recall)}


metrics_collection = Metrics()
metrics_collection.add_metric('acc', acc)
metrics_collection.add_metric('auroc', auroc)
metrics_collection.add_metric('f1', f1_binary)
metrics_collection.add_metric('f1_weighted', f1_weighted)
metrics_collection.add_metric('f1_max', f1_max)
metrics_collection.add_metric('mcc', mcc)
metrics_collection.add_metric('spcc', spcc)
metrics_collection.add_metric('pcc', pcc)
metrics_collection.add_metric('euclidean', euclidean)
metrics_collection.add_metric('cosine', cosine)
metrics_collection.add_metric('manhattan', manhattan)
metrics_collection.add_metric('mse', mse)
metrics_collection.add_metric('rmse', rmse)
metrics_collection.add_metric('aupr', aupr)
metrics_collection.add_metric('mae', mae)
metrics_collection.add_metric('precision_at_l5', precision_at_l5)
