"""Leitura e preparação dos dados; os CSV originais permanecem intactos."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION = "home-credit-default-risk"
TRAIN_FILE = "application_train.csv"
DICTIONARY_FILE = "HomeCredit_columns_description.csv"
# Seleção inicial interpretável, a rever após a EDA; não resulta de seleção pelo teste.
FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
]


def load_applications(data_dir=None, download=False):
    """Lê uma origem explícita ou data/home_credit; download é opt-in."""
    folder = Path(data_dir or os.environ.get("HOME_CREDIT_DATA", "data/home_credit"))
    if download and any(not (folder / name).is_file() for name in (TRAIN_FILE, DICTIONARY_FILE)):
        # Só descarrega a tabela principal e o dicionário, não o histórico completo.
        # Credenciais e aceitação das regras pertencem ao utilizador, nunca ao código.
        import kagglehub

        folder.mkdir(parents=True, exist_ok=True)
        for name in (TRAIN_FILE, DICTIONARY_FILE):
            if not (folder / name).is_file():
                kagglehub.competition_download(
                    COMPETITION, path=name, output_dir=str(folder)
                )
    path = folder / TRAIN_FILE
    missing_files = [name for name in (TRAIN_FILE, DICTIONARY_FILE) if not (folder / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Faltam {missing_files} em {folder.resolve()}. Coloca os CSV nessa pasta ou configura "
            "HOME_CREDIT_DATA; consulta as instruções do README."
        )
    frame = pd.read_csv(path)
    required = {"SK_ID_CURR", "TARGET", *FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias em falta: {sorted(missing)}")
    if frame.empty or frame["TARGET"].isna().any() or not frame["TARGET"].isin([0, 1]).all():
        raise ValueError("Esperava pedidos com TARGET preenchido e binário.")
    if frame["SK_ID_CURR"].isna().any() or frame["SK_ID_CURR"].duplicated().any():
        # O mesmo pedido em conjuntos diferentes tornaria a avaliação pouco fiável.
        raise ValueError("SK_ID_CURR deve ser único e estar preenchido.")
    dictionary_path = folder / DICTIONARY_FILE
    try:
        dictionary = pd.read_csv(dictionary_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Ficheiros CSV exportados em Windows podem usar acentos/citações em CP1252.
        # Só muda a codificação após falha explícita; nunca descarta caracteres.
        dictionary = pd.read_csv(dictionary_path, encoding="cp1252")
    return frame, dictionary


def stratified_split(frame, seed=42):
    """Separa 60/20/20 dentro de cada classe, sem reamostrar a classe minoritária."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    if frame["TARGET"].isna().any() or not frame["TARGET"].isin([0, 1]).all():
        raise ValueError("A separação requer TARGET binário preenchido.")
    for value in (0, 1):
        indices = np.flatnonzero(frame["TARGET"].to_numpy() == value)
        if len(indices) < 5:
            raise ValueError("São necessários pelo menos cinco exemplos de cada classe.")
        indices = rng.permutation(indices)
        a, b = int(len(indices) * 0.6), int(len(indices) * 0.8)
        for part, chunk in zip(parts, np.split(indices, [a, b])):
            part.extend(chunk)
    return tuple(frame.iloc[rng.permutation(p)].copy() for p in parts)


class NumericPreparation:
    """Aprende medianas no treino; normalização e log são opcionais para clustering."""

    def __init__(self, scale=False, log_amounts=False):
        self.scale = scale
        self.log_amounts = log_amounts

    def _values(self, frame):
        values = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").copy()
        values = values.replace([np.inf, -np.inf], np.nan)
        if self.log_amounts:
            # Montantes assimétricos podem dominar as distâncias mesmo após escala.
            columns = [c for c in FEATURES if c.startswith("AMT_")]
            if (values[columns] < 0).any().any():
                raise ValueError("Montantes negativos: rever a EDA antes de aplicar log1p.")
            values[columns] = np.log1p(values[columns])
        return values.to_numpy(dtype=float)

    def fit(self, frame):
        values = self._values(frame)
        if len(values) == 0 or np.isnan(values).all(axis=0).any():
            raise ValueError("Não é possível calcular medianas para uma coluna vazia.")
        self.medians_ = np.nanmedian(values, axis=0)
        filled = np.where(np.isnan(values), self.medians_, values)
        self.mean_ = filled.mean(axis=0)
        self.std_ = filled.std(axis=0)
        # Colunas constantes ficam em zero e não provocam divisão por zero.
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, frame):
        if not hasattr(self, "medians_"):
            raise ValueError("Executa fit no treino antes de transform.")
        values = self._values(frame)
        filled = np.where(np.isnan(values), self.medians_, values)
        result = (filled - self.mean_) / self.std_ if self.scale else filled
        if not np.isfinite(result).all():
            raise ValueError("A preparação produziu valores não finitos.")
        return result
