import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.export_release import export_release
from scripts.train_xgboost import COLUMNS, run
from scripts.verify_release import validate_release


class XGBoostReleaseTests(unittest.TestCase):
    def test_export_recomputes_scores_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'synthetic.csv'
            rng = np.random.default_rng(42)
            frame = pd.DataFrame(rng.integers(0, 4, size=(303, 14)), columns=COLUMNS)
            frame['num'] = np.arange(303) % 2
            frame.to_csv(data, header=False, index=False)
            with patch('scripts.train_xgboost.DATA_PATH', data), contextlib.redirect_stdout(StringIO()):
                run(root / 'source')
            destination = export_release(root / 'source', data, root / 'releases', 'xgb-test-v1')
            self.assertTrue(validate_release(destination, data)['dataset_checked'])
            with self.assertRaises(FileExistsError):
                export_release(root / 'source', data, root / 'releases', 'xgb-test-v1')
            report_path = root / 'source/test_evaluation.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            report['metrics']['accuracy'] = 1.0
            report_path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                export_release(root / 'source', data, root / 'releases', 'xgb-tampered')
            self.assertFalse((root / 'releases/xgb-tampered').exists())
