from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from cyanoguard_calibration import fit, predict


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class CalibrationTests(unittest.TestCase):
    def test_logistic_inversion_on_synthetic_standards(self):
        concentrations = np.array([0, 0.25, 0.5, 1, 2, 4, 8])
        ratios = 0.3 + 0.9 * concentrations / (2 + concentrations)
        standards = pd.DataFrame(
            {"concentration": concentrations, "green": ratios * 100, "red": 100, "time": 3}
        )
        artifact = fit(standards, "wist_4pl")
        sample = pd.DataFrame({"green": [75], "red": [100], "time": [3]})
        predicted = predict(artifact, sample).iloc[0]
        self.assertEqual(predicted["status"], "ok")
        self.assertAlmostEqual(predicted["predicted_concentration"], 2, places=5)

    def test_joint_model_checks_time_range(self):
        standards = pd.read_csv(EXAMPLES / "synthetic_standards.csv")
        earlier = standards.assign(time=2)
        artifact = fit(pd.concat([earlier, standards]), "joint_time")
        samples = pd.DataFrame({"green": [70, 70], "red": [100, 100], "time": [3, 4]})
        self.assertEqual(predict(artifact, samples)["status"].tolist(), ["ok", "time_out_of_range"])

    def test_installed_cli_from_another_directory(self):
        command = shutil.which("cyanoguard-calibrate")
        self.assertIsNotNone(command, "Install the package before running the tests")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            predictions = root / "predictions.csv"
            subprocess.run(
                [command, "fit", "--model", "freiburg_linear", "--input",
                 str(EXAMPLES / "synthetic_standards.csv"), "--output", str(calibration)],
                cwd=root, check=True,
            )
            infer = [
                command, "predict", "--input", str(EXAMPLES / "synthetic_unknowns.csv"),
                "--artifact", str(calibration), "--output", str(predictions),
            ]
            subprocess.run(infer, cwd=root, check=True)
            output = pd.read_csv(predictions)
            self.assertEqual(output["status"].tolist(),
                             ["ok", "concentration_out_of_range", "invalid_input"])
            self.assertAlmostEqual(output.loc[0, "predicted_concentration"], 1.5)
            self.assertTrue(output.loc[1:, "predicted_concentration"].isna().all())
            previous = predictions.read_bytes()
            duplicate = subprocess.run(infer, cwd=root, capture_output=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual(predictions.read_bytes(), previous)
