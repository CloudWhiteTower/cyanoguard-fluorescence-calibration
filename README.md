# CyanoGuard · Dual-Fluorescence Calibration

Fit concentration calibration models from paired green and red fluorescence
readings. The input uses four columns for reference samples and three for unknown
samples. Fitted parameters are saved to JSON so the same calibration can be
applied to another CSV.

I contributed to the mathematical design and implementation of fluorescence-to-
concentration models in SCAU-China's 2026 iGEM project, CyanoGuard. This
repository packages three exploratory calibration models from that work. The
team Wiki describes the project's separate, time-specific baseline-corrected
standard curves and credits the team members involved.

- [CyanoGuard Model page, section IV](https://2026.igem.wiki/scau-china/dry-lab/model)
- [Team contribution record](https://2026.igem.wiki/scau-china/team/attributions)

## Run the example

Use Python 3.12. From the repository root, install the package in a virtual environment.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in Bash, then run:

```bash
python -m pip install .
cyanoguard-calibrate fit --model freiburg_linear --input examples/synthetic_standards.csv --output calibration.json
cyanoguard-calibrate predict --input examples/synthetic_unknowns.csv --artifact calibration.json --output predictions.csv
python -m unittest discover -s tests -v
```

The synthetic standards are generated from a known linear ratio. The first
sample maps to 1.5, the second exceeds the calibration range, and the third has
an invalid red-channel reading. Only the first sample receives a concentration
estimate. These examples demonstrate the computation with invented signals.

## Models

| Option | Fit and prediction |
| --- | --- |
| `freiburg_linear` | Regresses green/red on concentration at one read time, then inverts the fitted line. |
| `wist_4pl` | Fits and inverts a four-parameter logistic curve at one read time. Prediction reports out-of-range ratios and concentrations. |
| `joint_time` | Fits ridge regression on green, red, time and their interactions to estimate concentration directly. It requires calibration at multiple times. |

The two ratio models use condition means so uneven numbers of technical repeats
do not change each condition's weight. They default to the latest supplied read
time. Pass `--read-time` during fitting to choose another time. The code borrows
method ideas from prior iGEM teams, with independent implementations.

- [Freiburg 2025 Results](https://2025.igem.wiki/freiburg/results/)
- [WIST 2025 Model](https://2025.igem.wiki/wist/model)

## CSV format and interpretation

- Standards require `concentration,green,red,time`. Unknown samples require
  `green,red,time`. Use the same units for standards and unknowns. Estimates
  have the units of the standards' concentration column.
- Prepare comparable channels and handle background correction upstream.
  Positive red intensity is required. Samples outside the accepted range
  receive a status and a blank concentration estimate.
- Inspect the fit artifact's warnings, the valid concentration and time ranges,
  and the prediction status. Evaluation with independent samples is needed to
  establish performance for an assay. Technical repeats of the same standard
  primarily check repeatability.
