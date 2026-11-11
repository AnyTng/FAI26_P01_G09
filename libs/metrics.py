"""Métricas multiclasse e silhouette implementadas de raiz."""

import numpy as np
from libs.kmeans import squared_distances


def classification_metrics(y_true, y_pred, labels=(0, 1, 2)):
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    labels = np.asarray(labels)
    if (
        actual.ndim != 1 or predicted.shape != actual.shape or len(actual) == 0
        or len(np.unique(labels)) != len(labels)
        or not np.isin(actual, labels).all() or not np.isin(predicted, labels).all()
    ):
        raise ValueError("Vetores não vazios alinhados e classes conhecidas são obrigatórios.")
    matrix = np.array([
        [np.sum((actual == row) & (predicted == col)) for col in labels] for row in labels
    ])
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = matrix.diagonal().astype(float)
    # Uma classe sem previsões tem precisão/F1 zero; mantém-se visível na média macro.
    precision = np.divide(true_positive, predicted_count, out=np.zeros(len(labels)), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(labels)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels)), where=precision + recall > 0)
    return {
        "accuracy": float(true_positive.sum() / len(actual)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix,
        "per_class": [
            {"class": label, "precision": p, "recall": r, "f1": f, "support": int(n)}
            for label, p, r, f, n in zip(labels, precision, recall, f1, support)
        ],
    }


def silhouette_score(X, labels):
    """Silhouette euclidiana na amostra explícita fornecida pelo notebook."""
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or len(labels) != len(X) or not np.isfinite(X).all():
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
        b = min(distances[i, labels == c].mean() for c in clusters if c != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) else 0.0
    return float(scores.mean())
