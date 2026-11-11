"""Árvore binária de classificação: divisões exatas pelo índice de Gini."""

import numpy as np


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10):
        for name, value, minimum in (
            ("max_depth", max_depth, 0),
            ("min_samples_split", min_samples_split, 2),
            ("min_samples_leaf", min_samples_leaf, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} deve ser inteiro >= {minimum}.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X deve ser uma matriz não vazia e finita.")
        if y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1]).all():
            raise ValueError("y deve ser um vetor binário alinhado com X.")
        self.n_features_ = X.shape[1]
        self.root_ = self._grow(X, y.astype(int), depth=0)
        return self

    def _grow(self, X, y, depth):
        n = len(y)
        probability = float(y.mean())
        # Cada folha guarda a proporção positiva para permitir avaliar probabilidades.
        node = {"probability": probability, "samples": n}
        if (
            depth >= self.max_depth or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf or probability in (0.0, 1.0)
        ):
            return node
        best_impurity = 2 * probability * (1 - probability)
        best = None
        for feature in range(self.n_features_):
            order = np.argsort(X[:, feature], kind="stable")
            values, target = X[order, feature], y[order]
            left_n = np.arange(1, n)
            right_n = n - left_n
            left_positive = np.cumsum(target)[:-1]
            right_positive = target.sum() - left_positive
            left_p = left_positive / left_n
            right_p = right_positive / right_n
            impurity = (
                left_n * 2 * left_p * (1 - left_p)
                + right_n * 2 * right_p * (1 - right_p)
            ) / n
            valid = (
                (values[:-1] < values[1:])
                & (left_n >= self.min_samples_leaf)
                & (right_n >= self.min_samples_leaf)
            )
            impurity[~valid] = np.inf
            split = int(np.argmin(impurity))
            # Evita divisões motivadas apenas por erro de arredondamento.
            if impurity[split] < best_impurity - 1e-12:
                best_impurity = float(impurity[split])
                # O maior valor da esquerda preserva a divisão sem calcular um ponto médio.
                best = feature, float(values[split])
        if best is None:
            return node
        feature, threshold = best
        mask = X[:, feature] <= threshold
        node.update(
            feature=feature, threshold=threshold,
            left=self._grow(X[mask], y[mask], depth + 1),
            right=self._grow(X[~mask], y[~mask], depth + 1),
        )
        return node

    def predict_proba(self, X):
        """Devolve um vetor de probabilidades de TARGET=1, uma por pedido."""
        if not hasattr(self, "root_"):
            raise ValueError("Treina a árvore com fit antes de prever.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or not np.isfinite(X).all():
            raise ValueError("X tem esquema incompatível ou valores não finitos.")
        output = np.empty(len(X))
        for i, row in enumerate(X):
            node = self.root_
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            output[i] = node["probability"]
        return output

    def predict(self, X):
        # Empates são resolvidos a favor de 0, a mesma regra da baseline.
        return (self.predict_proba(X) > 0.5).astype(int)
