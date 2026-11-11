"""Melbourne Housing: dados, classes de preço e preparação sem fuga de informação."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans

DATASET = "dansbecker/melbourne-housing-snapshot"
PRICE_LABELS = ("barato", "normal", "caro")
NUMERIC_FEATURES = [
    "Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea",
    "YearBuilt", "Lattitude", "Longtitude",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_housing(data_dir=None, download=True):
    """Lê o CSV original; só descarrega quando ele ainda não existe."""
    folder = Path(data_dir or os.environ.get("MELBOURNE_DATA", "data/melbourne"))
    path = folder / "melb_data.csv"
    if not path.is_file() and download:
        import kagglehub
        kagglehub.dataset_download(DATASET, path="melb_data.csv", output_dir=str(folder))
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path.resolve()}; ativa download ou define MELBOURNE_DATA.")
    frame = pd.read_csv(path)
    missing = {"Price", "Suburb", "Address", *FEATURES} - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias em falta: {sorted(missing)}")
    original_rows = len(frame)
    duplicates = int(frame.duplicated().sum())
    # Remover apenas cópias exatas; vendas distintas do mesmo imóvel mantêm-se juntas na separação.
    frame = frame.drop_duplicates().copy()
    prices = pd.to_numeric(frame["Price"], errors="raise")
    valid = np.isfinite(prices) & (prices > 0)
    invalid_prices = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["Price"] = prices.loc[valid]
    if frame.empty or frame[["Suburb", "Address"]].isna().any().any():
        raise ValueError("São necessários preços válidos e identificadores de imóvel preenchidos.")
    # Preserva o número da linha do CSV, para rastrear a tabela classificada.
    frame["SourceRow"] = frame.index
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    frame.attrs["source"] = {
        "dataset": DATASET, "ficheiro": path.name, "sha256": digest,
        "linhas_originais": original_rows, "duplicados_exatos_removidos": duplicates,
        "preços_inválidos_removidos": invalid_prices, "linhas_utilizadas": len(frame),
    }
    return frame


def split_housing(frame, seed=42):
    """Separa aproximadamente 60/20/20 por imóvel, antes de aprender as classes."""
    # A mesma morada pode ter várias vendas; não pode surgir simultaneamente no treino e teste.
    codes, groups = pd.factorize(pd.MultiIndex.from_frame(frame[["Suburb", "Address"]]), sort=False)
    if len(groups) < 5:
        raise ValueError("São necessários pelo menos cinco imóveis distintos.")
    order = np.random.default_rng(seed).permutation(len(groups))
    a, b = int(0.6 * len(groups)), int(0.8 * len(groups))
    return tuple(frame.loc[np.isin(codes, part)].copy() for part in np.split(order, [a, b]))


class PriceBands:
    """Aprende três grupos no preço de treino e ordena-os economicamente."""

    def __init__(self, seed=42):
        self.seed = seed

    @staticmethod
    def _values(prices):
        values = np.asarray(prices, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Price deve ser um vetor de preços positivos e finitos.")
        return values.reshape(-1, 1)

    def fit(self, prices):
        values = self._values(prices)
        if len(np.unique(values)) < 3:
            raise ValueError("São necessários pelo menos três preços distintos.")
        self.model_ = SklearnKMeans(
            n_clusters=3, n_init=20, random_state=self.seed, algorithm="lloyd"
        ).fit(values)
        # IDs de clusters não têm ordem económica: só os centroides permitem dar nomes.
        order = np.argsort(self.model_.cluster_centers_.ravel())
        self.cluster_to_class_ = np.empty(3, dtype=int)
        self.cluster_to_class_[order] = np.arange(3)
        self.centers_ = self.model_.cluster_centers_.ravel()[order]
        self.boundaries_ = (self.centers_[:-1] + self.centers_[1:]) / 2
        return self

    def predict(self, prices):
        if not hasattr(self, "model_"):
            raise ValueError("Executa fit com preços de treino antes de atribuir classes.")
        return self.cluster_to_class_[self.model_.predict(self._values(prices))]


def prepare_partitions(frame, seed=42):
    """Cria a coluna-alvo com uma única definição, aprendida apenas no treino."""
    parts = split_housing(frame, seed=seed)
    bands = PriceBands(seed=seed).fit(parts[0]["Price"])
    for part in parts:
        part["PriceClass"] = bands.predict(part["Price"])
        part["ClassePreco"] = pd.Categorical.from_codes(
            part["PriceClass"], categories=PRICE_LABELS, ordered=True
        )
    return parts, bands


class HousingPreparation:
    """Aprende medianas e categorias no treino; Price nunca entra nas entradas."""

    def __init__(self, scale=False, log_areas=False):
        self.scale, self.log_areas = scale, log_areas

    def _numeric(self, frame):
        values = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise").copy()
        values = values.replace([np.inf, -np.inf], np.nan)
        if self.log_areas:
            # Áreas extremas podem dominar as distâncias; log1p admite também áreas zero.
            for column in ("Landsize", "BuildingArea"):
                if (values[column] < 0).any():
                    raise ValueError(f"{column} tem áreas negativas; rever os dados.")
                values[column] = np.log1p(values[column])
        return values.to_numpy(dtype=float)

    def fit(self, frame):
        values = self._numeric(frame)
        if len(values) == 0 or np.isnan(values).all(axis=0).any():
            raise ValueError("Não é possível calcular medianas de uma coluna vazia.")
        self.medians_ = np.nanmedian(values, axis=0)
        self.categories_ = {
            column: sorted(frame[column].fillna("desconhecido").astype(str).unique())
            for column in CATEGORICAL_FEATURES
        }
        self.feature_names_ = list(NUMERIC_FEATURES) + [
            f"{column}={value}" for column, categories in self.categories_.items() for value in categories
        ]
        matrix = self._matrix(frame)
        self.mean_, self.std_ = matrix.mean(axis=0), matrix.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def _matrix(self, frame):
        numeric = self._numeric(frame)
        numeric = np.where(np.isnan(numeric), self.medians_, numeric)
        columns = [numeric]
        for column, categories in self.categories_.items():
            values = frame[column].fillna("desconhecido").astype(str).to_numpy()
            # Uma categoria não vista no treino fica sem coluna ativa; não altera o esquema.
            columns.append(np.column_stack([values == category for category in categories]))
        return np.column_stack(columns).astype(float)

    def transform(self, frame):
        if not hasattr(self, "medians_"):
            raise ValueError("Executa fit no treino antes de transform.")
        matrix = self._matrix(frame)
        matrix = (matrix - self.mean_) / self.std_ if self.scale else matrix
        if not np.isfinite(matrix).all():
            raise ValueError("Preparação produziu valores não finitos.")
        return matrix
