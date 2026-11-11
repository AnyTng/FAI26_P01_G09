"""Testes de cálculos e fronteiras de avaliação; não são resultados do dataset real."""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import unittest

import numpy as np
import pandas as pd

from libs.data import (
    FEATURES, NUMERIC_FEATURES, HousingPreparation, PriceBands,
    load_housing, prepare_partitions, split_housing,
)
from libs.decisiontrees import DecisionTree
from libs.kmeans import KMeans
from libs.metrics import classification_metrics, silhouette_score


def example_frame(size=30):
    frame = pd.DataFrame({name: np.arange(size, dtype=float) + 1 for name in NUMERIC_FEATURES})
    frame["Suburb"] = "Zona"
    frame["Address"] = [f"Rua {i}" for i in range(size)]
    frame["Type"] = ["h", "u", "t"] * (size // 3)
    frame["Price"] = np.repeat([100.0, 500.0, 1000.0], size // 3)
    return frame


class PriceTests(unittest.TestCase):
    def test_economic_order_does_not_depend_on_cluster_ids(self):
        model = PriceBands().fit([100, 100, 500, 500, 1000, 1000])
        np.testing.assert_allclose(model.centers_, [100, 500, 1000])
        np.testing.assert_array_equal(model.predict([90, 510, 990]), [0, 1, 2])
        np.testing.assert_allclose(model.boundaries_, [300, 750])

    def test_test_prices_cannot_change_training_bands(self):
        frame = example_frame()
        parts, original = prepare_partitions(frame)
        altered = frame.copy()
        heldout = parts[1].index.union(parts[2].index)
        altered.loc[heldout, "Price"] = 1e12
        new_parts, after = prepare_partitions(altered)
        np.testing.assert_allclose(original.centers_, after.centers_)
        pd.testing.assert_series_equal(parts[0]["PriceClass"], new_parts[0]["PriceClass"])

    def test_rejects_invalid_or_insufficient_prices(self):
        for prices in [[1, 2], [0, 1, 2], [1, np.nan, 3]]:
            with self.assertRaises(ValueError):
                PriceBands().fit(prices)


class DataTests(unittest.TestCase):
    def test_same_house_stays_in_one_partition(self):
        frame = example_frame()
        frame.loc[1, ["Suburb", "Address"]] = frame.loc[0, ["Suburb", "Address"]]
        parts = split_housing(frame)
        keys = [set(zip(p["Suburb"], p["Address"])) for p in parts]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertTrue(keys[i].isdisjoint(keys[j]))
        self.assertEqual(sum(map(len, parts)), len(frame))
        for a, b in zip(parts, split_housing(frame)):
            pd.testing.assert_frame_equal(a, b)

    def test_medians_and_categories_learn_only_from_training(self):
        frame = example_frame()
        train = frame.iloc[:3].copy()
        train.loc[:, NUMERIC_FEATURES] = np.repeat(
            np.array([1.0, 3.0, np.nan])[:, None], len(NUMERIC_FEATURES), axis=1
        )
        train["Type"] = ["h", "u", "h"]
        prep = HousingPreparation().fit(train)
        heldout = frame.iloc[:1].copy()
        heldout.loc[:, NUMERIC_FEATURES] = np.nan
        heldout["Type"] = "categoria_nova"
        output = prep.transform(heldout)
        np.testing.assert_allclose(output[0, :len(NUMERIC_FEATURES)], 2)
        np.testing.assert_allclose(output[0, len(NUMERIC_FEATURES):], 0)
        self.assertFalse(any("Price" in name for name in prep.feature_names_))

    def test_loader_audits_invalid_prices_duplicates_and_source(self):
        with TemporaryDirectory() as folder:
            frame = example_frame()
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            frame.loc[1, "Price"] = 0
            path = Path(folder) / "melb_data.csv"
            frame.to_csv(path, index=False)
            actual = load_housing(folder, download=False)
            audit = actual.attrs["source"]
            self.assertEqual(audit["duplicados_exatos_removidos"], 1)
            self.assertEqual(audit["preços_inválidos_removidos"], 1)
            self.assertEqual(audit["linhas_utilizadas"], 29)
            self.assertEqual(audit["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_constant_features_scale_without_dividing_by_zero(self):
        frame = example_frame()
        frame[NUMERIC_FEATURES] = 2.0
        frame["Type"] = "h"
        prepared = HousingPreparation(scale=True).fit(frame)
        np.testing.assert_allclose(prepared.transform(frame), 0)

    def test_missing_data_reports_path(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_housing(folder, download=False)


class TreeTests(unittest.TestCase):
    def test_three_classes_and_probability_shape(self):
        X = np.arange(9).reshape(-1, 1)
        y = np.repeat([0, 1, 2], 3)
        model = DecisionTree(3, 2, 1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertEqual(model.predict_proba(X).shape, (9, 3))
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)

    def test_repeated_values_do_not_create_empty_children(self):
        X = [[0], [0], [1], [1], [2], [2]]
        y = [0, 0, 1, 1, 2, 2]
        model = DecisionTree(4, 2, 1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_depth_leaf_and_split_limits(self):
        X, y = [[0], [1], [2], [3]], [0, 0, 1, 2]
        for params in [
            {"max_depth": 0}, {"min_samples_leaf": 3}, {"min_samples_split": 5}
        ]:
            model = DecisionTree(**params).fit(X, y)
            self.assertNotIn("feature", model.root_)

    def test_nonfinite_or_misaligned_inputs_fail(self):
        for X, y in [([[np.nan]], [0]), ([[1]], []), ([], [])]:
            with self.assertRaises(ValueError):
                DecisionTree().fit(X, y)
        with self.assertRaises(ValueError):
            DecisionTree().predict([[1]])


class MetricTests(unittest.TestCase):
    def test_three_class_metrics_known_case(self):
        report = classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["macro_f1"], 0.5)
        self.assertEqual(report["macro_recall"], 0.5)
        np.testing.assert_array_equal(report["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [1, 0, 1]])

    def test_missing_predictions_do_not_hide_a_class(self):
        report = classification_metrics([0, 1, 2], [0, 0, 0])
        self.assertEqual(report["per_class"][2]["recall"], 0)
        self.assertEqual(report["per_class"][2]["precision"], 0)

    def test_silhouette_known_distances(self):
        expected = ((1 - 2 / 11) + (1 - 2 / 9)) / 2
        self.assertAlmostEqual(silhouette_score([[0], [2], [10], [12]], [0, 0, 1, 1]), expected)


class KMeansTests(unittest.TestCase):
    def test_manual_kmeans_known_centers(self):
        model = KMeans(2, tol=1e-10).fit([[0], [2], [10], [12]])
        np.testing.assert_allclose(np.sort(model.centers_.ravel()), [1, 11])
        self.assertAlmostEqual(model.inertia_, 4)
        np.testing.assert_array_equal(model.predict([[0], [2], [10], [12]]), model.labels_)

    def test_degenerate_data_and_reproducibility(self):
        with self.assertRaises(ValueError):
            KMeans(2).fit([[1], [1]])
        a, b = KMeans(2).fit([[0], [1], [10]]), KMeans(2).fit([[0], [1], [10]])
        np.testing.assert_allclose(a.centers_, b.centers_)


if __name__ == "__main__":
    unittest.main()
