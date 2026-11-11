"""Casos artificiais com resultados conhecidos; não representam o dataset Home Credit."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from libs.data import FEATURES, NumericPreparation, load_applications, stratified_split
from libs.decisiontrees import DecisionTree
from libs.kmeans import KMeans
from libs.metrics import classification_metrics, silhouette_score


class DecisionTreeTests(unittest.TestCase):
    def test_separates_known_classes_and_handles_repeated_values(self):
        X = [[0], [0], [1], [1], [2], [2]]
        y = [0, 0, 0, 0, 1, 1]
        model = DecisionTree(max_depth=2, min_samples_split=2, min_samples_leaf=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        np.testing.assert_array_equal(model.predict([[0.5], [2.5]]), [0, 1])

    def test_depth_zero_is_one_leaf(self):
        model = DecisionTree(max_depth=0).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        np.testing.assert_array_equal(model.predict([[0], [3]]), [0, 0])
        np.testing.assert_allclose(model.predict_proba([[0], [3]]), [0.5, 0.5])

    def test_leaf_and_split_constraints(self):
        X, y = [[0], [1], [2], [3]], [0, 0, 0, 1]
        for kwargs in [
            {"min_samples_leaf": 3, "min_samples_split": 2},
            {"min_samples_leaf": 1, "min_samples_split": 5},
        ]:
            model = DecisionTree(**kwargs).fit(X, y)
            self.assertNotIn("feature", model.root_)

    def test_constant_features_and_pure_target_stop(self):
        for X, y in [([[1], [1]], [0, 1]), ([[0], [1]], [1, 1])]:
            model = DecisionTree(min_samples_split=2, min_samples_leaf=1).fit(X, y)
            self.assertNotIn("feature", model.root_)

    def test_rejects_invalid_input(self):
        model = DecisionTree()
        for X, y in [([[np.nan]], [0]), ([[1]], [2]), ([[1]], []), ([], [])]:
            with self.assertRaises(ValueError):
                model.fit(X, y)
        with self.assertRaises(ValueError):
            model.predict([[1]])
        with self.assertRaises(ValueError):
            DecisionTree(min_samples_leaf=0)


class KMeansTests(unittest.TestCase):
    def test_known_centers_and_predictions(self):
        X = np.array([[0.0], [2.0], [10.0], [12.0]])
        model = KMeans(2, n_init=5, tol=1e-10).fit(X)
        np.testing.assert_allclose(np.sort(model.centers_.ravel()), [1, 11])
        self.assertAlmostEqual(model.inertia_, 4)
        self.assertTrue(model.converged_)
        np.testing.assert_array_equal(model.predict(X), model.labels_)
        self.assertNotEqual(model.predict([[0]])[0], model.predict([[12]])[0])

    def test_reproducible_and_identical_points_allowed_with_k_one(self):
        X = np.array([[0, 0], [1, 0], [8, 8], [9, 8]], dtype=float)
        a, b = KMeans(2).fit(X), KMeans(2).fit(X)
        np.testing.assert_allclose(a.centers_, b.centers_)
        model = KMeans(1).fit([[5, 5], [5, 5]])
        self.assertEqual(model.inertia_, 0)

    def test_repeated_points_and_three_clusters(self):
        X = [[0], [0], [2], [2], [10], [10]]
        model = KMeans(3).fit(X)
        self.assertEqual(len(np.unique(model.labels_)), 3)
        self.assertAlmostEqual(model.inertia_, 0)

    def test_rejects_too_many_clusters_and_nonfinite_values(self):
        for X in [[[1], [1]], [[np.nan], [2]]]:
            with self.assertRaises(ValueError):
                KMeans(2).fit(X)
        with self.assertRaises(ValueError):
            KMeans(2).predict([[0]])


class DataTests(unittest.TestCase):
    def test_split_has_no_overlap_is_reproducible_and_preserves_classes(self):
        frame = pd.DataFrame({"SK_ID_CURR": range(100), "TARGET": [0] * 80 + [1] * 20})
        parts = stratified_split(frame)
        self.assertEqual([len(p) for p in parts], [60, 20, 20])
        self.assertEqual([int(p["TARGET"].sum()) for p in parts], [12, 4, 4])
        self.assertEqual(set().union(*(set(p.index) for p in parts)), set(frame.index))
        for i, left in enumerate(parts):
            for right in parts[i + 1:]:
                self.assertTrue(set(left.index).isdisjoint(right.index))
        for actual, expected in zip(parts, stratified_split(frame)):
            pd.testing.assert_frame_equal(actual, expected)

    def test_preparation_learns_only_training_statistics(self):
        train = pd.DataFrame({c: [1.0, 3.0, np.nan] for c in FEATURES})
        test = pd.DataFrame({c: [np.nan, 1000.0] for c in FEATURES})
        prep = NumericPreparation().fit(train)
        output = prep.transform(test)
        np.testing.assert_allclose(output[0], 2)
        np.testing.assert_allclose(output[1], 1000)
        np.testing.assert_allclose(prep.medians_, 2)

    def test_constant_columns_scale_to_zero_and_all_missing_column_fails(self):
        frame = pd.DataFrame({c: [2.0, 2.0] for c in FEATURES})
        prep = NumericPreparation(scale=True).fit(frame)
        np.testing.assert_allclose(prep.transform(frame), 0)
        frame[FEATURES[0]] = np.nan
        with self.assertRaises(ValueError):
            NumericPreparation().fit(frame)

    def test_loader_reports_missing_file_and_duplicate_ids(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_applications(folder)
            frame = pd.DataFrame({c: [1, 2] for c in FEATURES})
            frame["TARGET"], frame["SK_ID_CURR"] = [0, 1], [10, 10]
            path = Path(folder) / "application_train.csv"
            frame.to_csv(path, index=False)
            pd.DataFrame({"Row": ["TARGET"], "Description": ["Fixture artificial"]}).to_csv(
                Path(folder) / "HomeCredit_columns_description.csv", index=False
            )
            with self.assertRaises(ValueError):
                load_applications(folder)
            frame["SK_ID_CURR"] = [10, 11]
            frame.to_csv(path, index=False)
            loaded, dictionary = load_applications(folder)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(dictionary.iloc[0]["Row"], "TARGET")

    def test_partial_download_and_cp1252_dictionary(self):
        with TemporaryDirectory() as folder:
            frame = pd.DataFrame({c: [1, 2] for c in FEATURES})
            frame["TARGET"], frame["SK_ID_CURR"] = [0, 1], [10, 11]
            frame.to_csv(Path(folder) / "application_train.csv", index=False)
            downloaded = []

            def fake_download(handle, path, output_dir):
                downloaded.append(path)
                pd.DataFrame({"Description": ["Descrição de teste"]}).to_csv(
                    Path(output_dir) / path, index=False, encoding="cp1252"
                )

            with patch.dict("sys.modules", {"kagglehub": SimpleNamespace(competition_download=fake_download)}):
                _, dictionary = load_applications(folder, download=True)
            self.assertEqual(downloaded, ["HomeCredit_columns_description.csv"])
            self.assertEqual(dictionary.iloc[0]["Description"], "Descrição de teste")


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual([result[c] for c in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        for metric in ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]:
            self.assertEqual(result[metric], 0.5)

    def test_no_positive_predictions(self):
        result = classification_metrics([0, 1], [0, 0])
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)

    def test_silhouette_known_distances_and_singleton(self):
        # Distâncias intragrupo de 2; médias intergrupo de 9 e 11.
        expected = ((1 - 2 / 11) + (1 - 2 / 9)) / 2
        self.assertAlmostEqual(silhouette_score([[0], [2], [10], [12]], [0, 0, 1, 1]), expected)
        self.assertAlmostEqual(silhouette_score([[0], [2], [10]], [0, 0, 1]), ((1 - 2 / 10) + (1 - 2 / 8)) / 3)

    def test_invalid_metrics_raise(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            silhouette_score([[0], [1]], [0, 0])


if __name__ == "__main__":
    unittest.main()
