"""Métricas implementadas sem funções prontas de machine learning."""

import numpy as np

from libs.kmeans import squared_distances


def classification_metrics(y_true, y_pred):
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    if (
        actual.ndim != 1 or predicted.shape != actual.shape or len(actual) == 0
        or not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all()
    ):
        raise ValueError("Esperava dois vetores binários não vazios e alinhados.")
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())
    # Sem previsões positivas, precisão/F1 são reportadas como zero, explicitamente.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(actual),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def silhouette_score(X, labels):
    """Silhouette euclidiana na amostra explicitamente selecionada pelo notebook."""
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if (
        X.ndim != 2 or X.shape[1] == 0 or labels.ndim != 1
        or len(labels) != len(X) or not np.isfinite(X).all()
    ):
        raise ValueError("X e labels devem ser finitos e alinhados.")
    clusters = np.unique(labels)
    if not 2 <= len(clusters) < len(X):
        raise ValueError("Silhouette requer entre 2 e n-1 grupos.")
    distances = np.sqrt(squared_distances(X, X))
    scores = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() == 1:
            continue
        a = distances[i, own].sum() / (own.sum() - 1)
        b = min(distances[i, labels == cluster].mean() for cluster in clusters if cluster != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) else 0.0
    return float(scores.mean())
