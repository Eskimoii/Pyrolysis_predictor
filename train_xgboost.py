
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SEED = 42

CAT = [
    "Plastic_Type",
    "Catalyst_Type",
    "Reactor_Class"
]

NUM = [
    "Temperature_C",
    "Heating_Rate_C_per_min",
    "Particle_Size_mm",
    "Feed_Rate_kg_per_h",
    "Feed_Mass_kg",
    "Catalyst_Loading_wt_pct"
]

TARGET = "Oil_Yield_wt_pct"

GROUP_COLUMN = "Source_ID"

REQUIRED = CAT + NUM + [TARGET, GROUP_COLUMN]

GROUP = {
    "Plastic_Type": "Plastic type/composition",
    "Temperature_C": "Temperature",
    "Heating_Rate_C_per_min": "Heating rate",
    "Particle_Size_mm": "Particle size",
    "Feed_Rate_kg_per_h": "Feed amount",
    "Feed_Mass_kg": "Feed amount",
    "Catalyst_Type": "Catalyst",
    "Catalyst_Loading_wt_pct": "Catalyst",
    "Reactor_Class": "Reactor type"
}


# --------------------------------------------------
# 2. COMMAND LINE ARGUMENTS
# --------------------------------------------------

def parse_args():

    workspace = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=workspace / "03_FINAL_ML_Dataset (2).csv"
    )

    parser.add_argument(
        "--all-rows",
        action="store_true"
    )

    parser.add_argument(
        "--plot",
        type=Path,
        default=workspace / "xgboost_results.png"
    )

    return parser.parse_args()


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

def load_dataset(path, all_rows=False):

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=["", "NaN", "nan"]
    )

    missing = [col for col in REQUIRED if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use only complete ML rows by default
    if "ML_Core_Complete" in df.columns and not all_rows:
        df = df[df["ML_Core_Complete"].astype(str) == "1"].copy()

    # Convert numerical columns
    for col in NUM + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows without target or source identification
    df = df.dropna(subset=[TARGET, GROUP_COLUMN]).copy()

    if len(df) < 13:
        raise ValueError("Not enough rows for model training and evaluation.")

    if df[GROUP_COLUMN].nunique() < 3:
        raise ValueError(
            "At least 3 unique research sources are needed "
            "for paper-wise splitting and cross-validation."
        )

    print("\nDataset loaded successfully")
    print("Number of rows:", len(df))
    print("Number of research sources:", df[GROUP_COLUMN].nunique())

    return df


# --------------------------------------------------
# 4. CREATE XGBOOST MODEL PIPELINE
# --------------------------------------------------

def make_model():

    # Handle categorical variables
    cat_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CAT)
        ],
        remainder="passthrough"
    )

    # XGBoost regressor
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        random_state=SEED,
        n_jobs=1
    )

    # Complete pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb)
    ])

    return model


# --------------------------------------------------
# 5. PAPER-WISE FEATURE IMPORTANCE
# --------------------------------------------------

def get_group_importance(model):

    preprocessor = model.named_steps["preprocessor"]
    xgb = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = xgb.feature_importances_

    importance_dict = {}

    for name, importance in zip(feature_names, importances):

        # Remove preprocessing prefix
        original_name = name.split("__", 1)[-1]

        # Identify original input column
        matched_column = None

        for col in CAT:
            if original_name == col or original_name.startswith(col + "_"):
                matched_column = col
                break

        if matched_column is None:
            for col in NUM:
                if original_name == col:
                    matched_column = col
                    break

        if matched_column is not None:

            group_name = GROUP[matched_column]

            importance_dict[group_name] = (
                importance_dict.get(group_name, 0) + importance
            )

    return importance_dict


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

def main():

    args = parse_args()

    # Load dataset
    df = load_dataset(args.dataset, args.all_rows)

    # Features and target
    X = df[CAT + NUM]
    y = df[TARGET]

    # Research paper/source groups
    groups = df[GROUP_COLUMN].astype(str)

    print("\nInput columns:", len(CAT + NUM))
    print("Conceptual input variables: 7")

    # --------------------------------------------------
    # PAPER-WISE TRAIN-TEST SPLIT
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Verify no source leakage
    assert set(groups_train).isdisjoint(set(groups_test))

    print("\n--- PAPER-WISE TRAIN-TEST SPLIT ---")

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("Training sources:", groups_train.nunique())
    print("Testing sources:", groups_test.nunique())

    print(
        "Sources shared between train and test:",
        len(set(groups_train) & set(groups_test))
    )

    # --------------------------------------------------
    # PAPER-WISE CROSS VALIDATION
    # --------------------------------------------------

    n_folds = min(5, groups_train.nunique())

    if n_folds < 2:
        raise ValueError(
            "Not enough training sources for cross-validation."
        )

    cv = GroupKFold(n_splits=n_folds)

    model = make_model()

    print(f"\n--- {n_folds}-FOLD PAPER-WISE CROSS VALIDATION ---")

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )

    cv_mae = -cv_scores

    print("Fold MAE scores:", cv_mae)
    print("Mean CV MAE:", cv_mae.mean())
    print("Standard deviation:", cv_mae.std())

    # --------------------------------------------------
    # TRAIN FINAL MODEL
    # --------------------------------------------------

    print("\nTraining final XGBoost model...")

    model.fit(X_train, y_train)

    # Save model
    model_path = Path(__file__).resolve().parent / "xgboost_model.joblib"

    joblib.dump(model, model_path)

    print("Model saved at:", model_path)

    # --------------------------------------------------
    # TEST SET EVALUATION
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- PAPER-WISE TEST RESULTS ---")

    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    importance_dict = get_group_importance(model)

    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=["Feature Group", "Importance"]
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n--- FEATURE IMPORTANCE ---")
    print(importance_df.to_string(index=False))

    # --------------------------------------------------
    # PLOT RESULTS
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs predicted
    axes[0].scatter(y_test, y_pred)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    axes[0].plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    axes[0].set_xlabel("Actual Oil Yield (wt%)")
    axes[0].set_ylabel("Predicted Oil Yield (wt%)")
    axes[0].set_title("Paper-wise Actual vs Predicted")

    # Feature importance
    axes[1].barh(
        importance_df["Feature Group"],
        importance_df["Importance"]
    )

    axes[1].invert_yaxis()

    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title("Feature Importance by Input Group")

    plt.tight_layout()

    args.plot.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(args.plot, dpi=300, bbox_inches="tight")

    print("\nPlot saved at:", args.plot)



# --------------------------------------------------
# 7. RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()
