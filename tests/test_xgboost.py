"""Contrôles du pipeline XGBoost sur des données synthétiques."""
import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from scripts.train_xgboost import COLUMNS, build_xgboost_pipeline, run


class XGBoostTests(unittest.TestCase):
    def test_pipeline_handles_missing_and_unknown_categories_after_reload(self):
        X = pd.DataFrame({name: np.arange(30, dtype=float) % 3 for name in COLUMNS[:-1]})
        X.loc[0, "age"] = np.nan
        X.loc[1, "thal"] = np.nan
        y = np.arange(30) % 2
        pipeline = clone(build_xgboost_pipeline()).fit(X, y)
        sample = X.iloc[:2].copy()
        sample.loc[0, "thal"] = 99
        probabilities = pipeline.predict_proba(sample)
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.joblib"
            joblib.dump(pipeline, path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(loaded.predict_proba(sample), probabilities)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "metadata.json"
            sentinel.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(directory)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
