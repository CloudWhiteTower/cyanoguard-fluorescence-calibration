"""Three small calibration models; team-inspired adaptations, not team code.

CSV columns: concentration, green, red, time. Units must be consistent.
No background or OD is invented. Fit artifacts explicitly retain provenance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

MODELS = ("freiburg_linear", "wist_4pl", "joint_time")
OUTPUT_COLUMNS = (
    "predicted_concentration",
    "unrestricted_inverse_diagnostic_only",
    "status",
)


def checked(frame, training=True):
    columns = ["green", "red", "time"]
    if training:
        columns += ["concentration"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Required columns: {columns}")
    data = frame.copy()
    data[columns] = data[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError("Inputs must be finite; missing data cannot become zero")
    if (data.time < 0).any() or (training and (data.concentration < 0).any()):
        raise ValueError("Time and concentration must be nonnegative")
    if (data.red <= 0).any():
        raise ValueError("Red must be positive; no pseudocount is added")
    data["ratio"] = data.green / data.red
    if not np.isfinite(data.ratio).all():
        raise ValueError("Nonfinite ratio")
    return data


def hill(c, p):
    q0, qinf, k, n = p
    c = np.asarray(c, dtype=float)
    z = np.zeros_like(c)
    positive = c > 0
    z[positive] = expit(n * (np.log(c[positive]) - np.log(k)))
    return q0 + (qinf - q0) * z


def joint_features(frame, artifact):
    # Dimensionless transforms; scales learned only from the training partition.
    g = np.arcsinh(np.asarray(frame.green, float) / artifact["green_scale"])
    r = np.arcsinh(np.asarray(frame.red, float) / artifact["red_scale"])
    t = np.asarray(frame.time, float) / artifact["time_scale"]
    return np.column_stack([g, r, t, g * t, r * t])


def fit(frame, name, read_time=None):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    data = checked(frame)
    if data.concentration.nunique() < 3:
        raise ValueError("Need at least three distinct concentrations")
    if read_time is None:
        read_time = float(data.time.max())
    used = (
        data
        if name == "joint_time"
        else data[np.isclose(data.time, read_time, rtol=0, atol=1e-9)]
    )
    if used.empty or used.concentration.nunique() < 3:
        raise ValueError("Insufficient concentrations at requested read time")
    # Equal weight per concentration/time condition, independent of repeat counts.
    points = used.groupby(["concentration", "time"], as_index=False).ratio.mean()
    c, t, y = (points[k].to_numpy() for k in ["concentration", "time", "ratio"])
    artifact = {
        "model": name,
        "read_time": float(read_time) if name != "joint_time" else None,
        "concentration_range": [float(c.min()), float(c.max())],
        "time_range": [float(t.min()), float(t.max())],
        "signal": "paired green/red, no implicit blank correction",
        "n_conditions": len(points),
        "n_concentrations": int(points.concentration.nunique()),
        "warnings": [
            "Positive red alone does not establish adequate signal-to-background"
        ],
    }
    if name == "freiburg_linear":
        x = np.column_stack([np.ones(len(c)), c])
        p, _, _rank, _ = np.linalg.lstsq(x, y, rcond=None)
        artifact["parameters"] = p.tolist()
        prediction = x @ p
        artifact["parameter_names"] = ["intercept", "concentration_slope"]
    elif name == "joint_time":
        if points.time.nunique() < 2:
            raise ValueError("Joint model needs multiple times")
        # Direct inverse calibration retains the two channels instead of forcing G/R.
        # One averaged feature vector per condition gives equal condition weights.
        artifact.update(
            time_scale=float(t.max()),
            green_scale=max(float(np.median(np.abs(used.green))), 1e-12),
            red_scale=float(np.median(used.red)),
            ridge_alpha=1.0,
        )
        raw_features = joint_features(used, artifact)
        temp = pd.DataFrame(raw_features, columns=[f"f{i}" for i in range(5)])
        temp["concentration"] = used.concentration.to_numpy()
        temp["time"] = used.time.to_numpy()
        feature_table = temp.groupby(["concentration", "time"], as_index=False).mean()
        features = feature_table[[f"f{i}" for i in range(5)]].to_numpy()
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-12] = 1
        x = (features - mean) / scale
        target_mean = float(c.mean())
        p = np.linalg.solve(
            x.T @ x + artifact["ridge_alpha"] * np.eye(5), x.T @ (c - target_mean)
        )
        artifact.update(
            parameters=p.tolist(),
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            target_mean=target_mean,
            signal="separate green, red and time; no implicit blank correction",
            parameter_names=["green", "red", "time", "green_time", "red_time"],
        )
        artifact["training_concentration_rmse"] = float(
            np.sqrt(np.mean((target_mean + x @ p - c) ** 2))
        )
        artifact["warnings"].append(
            "Direct regression may use concentration-dependent red; not proof of internal-reference correction"
        )
        return artifact
    else:
        if len(np.unique(c)) < 4:
            raise ValueError("4PL needs at least four distinct concentrations")
        span = float(np.ptp(y))
        if span <= 1e-12:
            raise ValueError("Constant response cannot identify 4PL")
        positives = c[c > 0]
        lower = np.array(
            [y.min() - 10 * span, y.min() - 10 * span, positives.min() / 100, 0.2]
        )
        upper = np.array([y.max() + 10 * span, y.max() + 10 * span, c.max() * 100, 8.0])
        order = np.argsort(c)
        fits = []
        for k in [float(np.median(positives)), float(positives.min())]:
            for n in [1.0, 3.0]:
                initial = [y[order[0]], y[order[-1]], k, n]
                result = least_squares(
                    lambda p: (hill(c, p) - y) / span,
                    initial,
                    bounds=(lower, upper),
                    x_scale="jac",
                    max_nfev=2000,
                )
                if result.success and np.isfinite(result.cost):
                    fits.append(result)
        if not fits:
            raise ValueError("All 4PL optimizer starts failed")
        best = min(fits, key=lambda result: result.cost)
        p = best.x
        prediction = hill(c, p)
        condition = float(np.linalg.cond(best.jac))
        artifact.update(
            parameters=p.tolist(),
            parameter_names=["q0", "qinf", "K", "n"],
            bounds=[lower.tolist(), upper.tolist()],
            successful_starts=len(fits),
            jacobian_condition=condition if np.isfinite(condition) else None,
        )
        fraction = (p - lower) / (upper - lower)
        if np.any((fraction < 1e-4) | (fraction > 1 - 1e-4)):
            artifact["warnings"].append("4PL parameter close to imposed bound")
        if len(points) <= 4:
            artifact["warnings"].append(
                "Four or fewer conditions for four parameters: no residual degrees of freedom"
            )
        if not np.isfinite(condition) or condition > 1e6:
            artifact["warnings"].append(
                "Ill-conditioned 4PL Jacobian; parameters weakly identified"
            )
    artifact["training_ratio_rmse"] = float(np.sqrt(np.mean((prediction - y) ** 2)))
    return artifact


def response(artifact, concentration, time):
    c, _t = np.broadcast_arrays(
        np.asarray(concentration, float), np.asarray(time, float)
    )
    p = artifact["parameters"]
    if artifact["model"] == "wist_4pl":
        return hill(c, p)
    if artifact["model"] == "freiburg_linear":
        return p[0] + p[1] * c
    raise ValueError(
        "Joint model directly predicts concentration; it has no forward G/R curve"
    )


def predict(artifact, frame):
    """No concentration label is read when predicting unknown samples."""
    # Row-wise invalid input flags instead of silently removing unknown samples.
    if not {"green", "red", "time"}.issubset(frame.columns):
        raise ValueError("Prediction requires green, red, time")
    outputs = []
    lo, hi = artifact["concentration_range"]
    for _, row in frame.iterrows():
        status, estimate, raw = "ok", np.nan, np.nan
        try:
            g, r, t = (float(row[k]) for k in ["green", "red", "time"])
            if not np.isfinite([g, r, t]).all() or r <= 0 or t < 0:
                status = "invalid_input"
            elif artifact["read_time"] is not None and not np.isclose(
                t, artifact["read_time"], rtol=0, atol=1e-9
            ):
                status = "wrong_read_time"
            elif not artifact["time_range"][0] <= t <= artifact["time_range"][1]:
                status = "time_out_of_range"
            else:
                q = g / r
                p = artifact["parameters"]
                if artifact["model"] == "joint_time":
                    features = joint_features(
                        pd.DataFrame({"green": [g], "red": [r], "time": [t]}), artifact
                    )
                    x = (features - np.array(artifact["feature_mean"])) / np.array(
                        artifact["feature_scale"]
                    )
                    raw = float(artifact["target_mean"] + (x @ np.asarray(p))[0])
                elif artifact["model"] == "wist_4pl":
                    delta = p[1] - p[0]
                    z = (q - p[0]) / delta if abs(delta) > 1e-12 else np.nan
                    if not np.isfinite(z) or z < 0 or z >= 1:
                        status = "outside_asymptotes"
                    else:
                        raw = float(p[2] * (z / (1 - z)) ** (1 / p[3]))
                else:
                    intercept, slope = p
                    if abs(slope) <= 1e-10:
                        status = "flat_response"
                    else:
                        raw = float((q - intercept) / slope)
                if status == "ok":
                    if not np.isfinite(raw) or not lo <= raw <= hi:
                        status = "concentration_out_of_range"
                    else:
                        estimate = raw
        except (ValueError, TypeError, OverflowError):
            status = "invalid_input"
        outputs.append(
            {
                "predicted_concentration": estimate,
                "unrestricted_inverse_diagnostic_only": raw,
                "status": status,
            }
        )
    return pd.DataFrame(outputs, index=frame.index, columns=OUTPUT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--input", required=True)
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--read-time", type=float)
    train.add_argument("--output", required=True)
    infer = sub.add_parser("predict")
    infer.add_argument("--input", required=True)
    infer.add_argument("--artifact", required=True)
    infer.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        data = pd.read_csv(args.input)
        if args.command == "fit":
            artifact = fit(data, args.model, args.read_time)
            text = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
        else:
            if set(OUTPUT_COLUMNS) & set(data.columns):
                raise ValueError("Input columns already contain prediction output fields")
            artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
            text = pd.concat([data, predict(artifact, data)], axis=1).to_csv(index=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
