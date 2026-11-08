
"""
This script fits ARIMA and ARIMAX models, saves the models, and calculates and saves normalized root mean squared error (NRMSE) results.
"""
import pandas as pd
import sys
from pathlib import Path
import joblib

sys.path.append("../")

from model_recurrent import Model


print("RECUERDA ACTIVAR EL VENV")

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    "/home/marianne.pena/mainMarianne/dengue-model-rec/full_data.csv"
)

cantons = [
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    201, -202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 301, 302, 303, 304,
    305, 306, 307, 308, 401, 402, 403, 404, 405, 406,
    407, 408, 409, 410, 501, 502, 503, 504, 505, 506,
    507, 508, 509, 510, 511, 601, 602, 603, 604, 605,
    606, 607, 608, 609, 610, 611, 701, 702, 703, 704,
    705, 706
]

cantons = [str(x) for x in cantons]

# ============================================================
# Configuration
# ============================================================

model_dir = Path("../saved_models/arima")
metrics_dir = Path("../data/metrics")

model_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

output_path = metrics_dir / "nrmse_arima_confint.csv"

n_bootstrap = 500
block_size = 4
epsilon = 0.2
seed = 42
maxiter = 50

results = []

# ============================================================
# Fit models and calculate bootstrap confidence intervals
# ============================================================

for i, canton in enumerate(cantons[0:11]):

    print(f"\n{'=' * 60}")
    print(f"Processing canton {canton} ({i + 1}/{len(cantons)})")
    print(f"{'=' * 60}")

    model_temp = Model(df, canton)

    # --------------------------------------------------------
    # 1. Fit and save ARIMA and best ARIMAX
    # --------------------------------------------------------

    model_temp.fit_and_save_arima_and_best_arimax(
        epsilon=epsilon,
        maxiter=maxiter,
        save_dir=str(model_dir)
    )

    canton_results = {"canton": canton}

    # --------------------------------------------------------
    # 2. Load each saved model and generate predictions
    # --------------------------------------------------------

    for model_name in ["arima", "arimax"]:

        model_path = model_dir / f"{model_name}_{canton}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find saved model: {model_path}"
            )

        artifacts = joblib.load(model_path)

        predictions_df = model_temp.predict_arima_whole_dataset(
            artifacts=artifacts,
            maxiter=maxiter
        )

        # ----------------------------------------------------
        # 3. Calculate 500 bootstrap replicates
        # ----------------------------------------------------

        summary, distributions = model_temp.bootstrap_nrmse(
            predictions_df=predictions_df,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            epsilon=artifacts.get("epsilon", epsilon),
            seed=seed + i,
            output_dir=None
        )

        if summary.empty:
            raise ValueError(
                f"No bootstrap results for canton {canton}, "
                f"model {model_name}."
            )

        # Get the train+validation and test rows from the summary.
        train_summary = summary.loc[
            summary["period"] == "train_validation"
        ]

        test_summary = summary.loc[
            summary["period"] == "test"
        ]

        if train_summary.empty or test_summary.empty:
            raise ValueError(
                f"Missing train_validation or test summary for "
                f"canton {canton}, model {model_name}.\n"
                f"Available periods: {summary['period'].tolist()}"
            )

        train_summary = train_summary.iloc[0]
        test_summary = test_summary.iloc[0]

        # ----------------------------------------------------
        # 4. Store point estimates and confidence intervals
        # ----------------------------------------------------

        if model_name == "arima":
            suffix = "arima"
        else:
            suffix = "arimax"

        canton_results[f"train_point_{suffix}"] = (
            train_summary["point_nrmse"]
        )
        canton_results[f"train_lower_{suffix}"] = (
            train_summary["lower_95"]
        )
        canton_results[f"train_upper_{suffix}"] = (
            train_summary["upper_95"]
        )

        canton_results[f"test_point_{suffix}"] = (
            test_summary["point_nrmse"]
        )
        canton_results[f"test_lower_{suffix}"] = (
            test_summary["lower_95"]
        )
        canton_results[f"test_upper_{suffix}"] = (
            test_summary["upper_95"]
        )

        print(
            f"{model_name}: "
            f"train+validation NRMSE = "
            f"{train_summary['point_nrmse']:.4f} "
            f"[{train_summary['lower_95']:.4f}, "
            f"{train_summary['upper_95']:.4f}]"
        )

        print(
            f"{model_name}: "
            f"test NRMSE = "
            f"{test_summary['point_nrmse']:.4f} "
            f"[{test_summary['lower_95']:.4f}, "
            f"{test_summary['upper_95']:.4f}]"
        )

    results.append(canton_results)

    # Save incrementally so completed cantons are preserved
    # if a later canton fails.
    output_columns = [
        "canton",
        "train_point_arima",
        "train_lower_arima",
        "train_upper_arima",
        "train_point_arimax",
        "train_lower_arimax",
        "train_upper_arimax",
        "test_point_arima",
        "test_lower_arima",
        "test_upper_arima",
        "test_point_arimax",
        "test_lower_arimax",
        "test_upper_arimax"
    ]

    pd.DataFrame(results).reindex(
        columns=output_columns
    ).to_csv(output_path, index=False)

    print(f"Updated results file: {output_path}")

# ============================================================
# Final output
# ============================================================

results_df = pd.DataFrame(results).reindex(
    columns=output_columns
)

results_df.to_csv(output_path, index=False)

print("\nPipeline complete.")
print(f"Results saved to: {output_path}")
print(results_df.head())
