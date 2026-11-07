
import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Plastic Pyrolysis AI/ML System",
    page_icon="♻️",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "07_ML_Dataset_with_CE_Energy_Scenarios.csv"
)
MODEL_PATH = os.path.join(PROJECT_DIR, "xgboost_model.joblib")
ENERGY_MODEL_PATH = os.path.join(PROJECT_DIR, "energy_surrogate.joblib")

POPULATION_SIZE = 30
N_GENERATIONS = 15
ENERGY_MODEL_VERSION = "2"
BASE_HEAT_LOSS_PERCENT = 20
BASE_HEATER_EFFICIENCY_PERCENT = 70
KNOWN_BAD_HEATING_RATE_SOURCE = "Mastral2006"
KNOWN_BAD_HEATING_RATE = 58620.0

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

FEATURES = CAT + NUM

TARGET = "Oil_Yield_wt_pct"
ENERGY_TARGET = "CE_SEC_MJ_per_kg_feed"


# =====================================================
# 2. LOAD DATA AND EXISTING XGBOOST MODEL
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_oil_model():
    return joblib.load(MODEL_PATH)


df = load_dataset()
oil_model = load_oil_model()


# =====================================================
# 3. PREPARE ENERGY OPTIMIZATION DATA
# =====================================================

def get_eligible_energy_data(data):

    energy_data = data.copy()

    energy_data[ENERGY_TARGET] = pd.to_numeric(
        energy_data[ENERGY_TARGET],
        errors="coerce"
    )
    energy_data["Heating_Rate_C_per_min"] = pd.to_numeric(
        energy_data["Heating_Rate_C_per_min"],
        errors="coerce"
    )

    # Keep only records marked eligible for optimization
    if "CE_Use_In_Optimisation" in energy_data.columns:

        flag = (
            energy_data["CE_Use_In_Optimisation"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        accepted = [
            "TRUE", "1", "YES", "Y",
            "USE", "INCLUDE", "ELIGIBLE"
        ]

        mask = flag.isin(accepted)

        if mask.any():
            energy_data = energy_data[mask]
        else:
            st.warning(
                "No recognized eligible records found in "
                "CE_Use_In_Optimisation. Using records with "
                "available energy calculations instead."
            )

    if "Source_ID" in energy_data.columns:
        known_bad_rate = (
            energy_data["Source_ID"].astype(str).str.strip()
            == KNOWN_BAD_HEATING_RATE_SOURCE
        ) & (
            energy_data["Heating_Rate_C_per_min"]
            == KNOWN_BAD_HEATING_RATE
        )
        energy_data.loc[known_bad_rate, "Heating_Rate_C_per_min"] = np.nan

    energy_data = energy_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    energy_data = energy_data.dropna(
        subset=[ENERGY_TARGET, "Plastic_Type"]
    )

    return energy_data


# =====================================================
# 4. TRAIN ENERGY SURROGATE MODEL
# =====================================================

def build_energy_model(data):

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    numerical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, CAT),
        ("num", numerical_transformer, NUM)
    ])

    energy_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1
            )
        )
    ])

    X = data[FEATURES].copy()
    y = data[ENERGY_TARGET].copy()

    energy_pipeline.fit(X, y)

    return energy_pipeline


energy_data = get_eligible_energy_data(df)


@st.cache_resource
def load_or_build_energy_model(data, model_version):

    if os.path.exists(ENERGY_MODEL_PATH):
        saved_model = joblib.load(ENERGY_MODEL_PATH)
        saved_version = getattr(
            saved_model,
            "_app_preprocessing_version",
            None
        )
        if saved_version == model_version:
            return saved_model

    if len(data) < 10:
        return None

    model = build_energy_model(data)
    model._app_preprocessing_version = model_version
    joblib.dump(model, ENERGY_MODEL_PATH)
    return model


energy_model = load_or_build_energy_model(
    energy_data,
    ENERGY_MODEL_VERSION
)


# =====================================================
# 5. COMMON FUNCTIONS
# =====================================================

def get_categories(column):

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


def get_bounds(column, data=None):

    if data is None:
        data = df

    values = pd.to_numeric(
        data[column],
        errors="coerce"
    ).dropna()

    if (
        column == "Heating_Rate_C_per_min"
        and "Source_ID" in data.columns
    ):
        known_bad_rate = (
            data["Source_ID"].astype(str).str.strip()
            == KNOWN_BAD_HEATING_RATE_SOURCE
        ) & (
            pd.to_numeric(data[column], errors="coerce")
            == KNOWN_BAD_HEATING_RATE
        )
        values = values.loc[~known_bad_rate]

    if values.empty:
        values = pd.to_numeric(
            energy_data[column],
            errors="coerce"
        ).dropna()

    if values.empty:
        return 0.0, 1.0

    low = float(values.min())
    high = float(values.max())

    if low == high:
        high = low + 1.0

    return low, high


def get_optimization_bounds(column, data):

    values = pd.to_numeric(
        data[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        values = pd.to_numeric(
            energy_data[column],
            errors="coerce"
        ).dropna()

    if values.empty:
        return 0.0, 1.0

    low = float(values.min())
    high = float(values.max())

    if low == high:
        high = low + 1.0

    return low, high


def predict_oil(input_df):

    return np.asarray(
        oil_model.predict(input_df[FEATURES])
    ).reshape(-1)


def predict_energy(input_df):

    return np.asarray(
        energy_model.predict(input_df[FEATURES])
    ).reshape(-1)


# =====================================================
# 6. APPLICATION HEADER
# =====================================================

st.title("♻️ Plastic Pyrolysis AI/ML System")

st.markdown(
    """
    Machine Learning Based Oil Yield Prediction
    and Multi-Objective Process Optimization
    """
)

st.divider()


# =====================================================
# 7. TWO MAIN SECTIONS
# =====================================================

tab1, tab2 = st.tabs([
    "🔮 ML Oil Yield Prediction",
    "⚙️ Pyrolysis Optimization using NSGA-II"
])


# =====================================================
# TAB 1: ML OIL YIELD PREDICTION
# =====================================================

with tab1:

    st.header("ML-Based Oil Yield Prediction")

    st.write(
        "Enter the operating conditions to estimate "
        "the oil yield using the trained XGBoost model."
    )

    user_input = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(FEATURES):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            label = feature.replace("_", " ")

            if feature in CAT:

                options = get_categories(feature)

                if not options:
                    st.warning(f"No values available for {label}")
                    continue

                user_input[feature] = st.selectbox(
                    label,
                    options,
                    key=f"predict_{feature}"
                )

            else:

                low, high = get_bounds(feature)

                median = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                ).median()

                if pd.isna(median):
                    median = low

                user_input[feature] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(np.clip(median, low, high)),
                    format="%.4f",
                    key=f"predict_{feature}"
                )

    if st.button(
        "Predict Oil Yield",
        type="primary",
        use_container_width=True
    ):

        try:

            input_df = pd.DataFrame(
                [user_input],
                columns=FEATURES
            )

            prediction = predict_oil(input_df)[0]

            st.success("Oil yield prediction completed!")

            st.metric(
                "Predicted Oil Yield",
                f"{prediction:.2f} wt.%"
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =====================================================
# TAB 2: NSGA-II OPTIMIZATION
# =====================================================

with tab2:

    st.header("Pyrolysis Optimization using NSGA-II")
    st.write(
        "Optimize oil yield and specific energy consumption "
        "simultaneously for the selected plastic type."
    )
    plastic_options = get_categories("Plastic_Type")

    selected_plastic = st.selectbox(
        "Plastic Type",
        plastic_options,
        key="optimization_plastic"
    )

    if energy_model is None:
        st.error(
            "Insufficient eligible energy records to train the energy "
            "surrogate. Check CE_Use_In_Optimisation and "
            "CE_SEC_MJ_per_kg_feed."
        )

    run_optimization = st.button(
        "🚀 Optimize",
        type="primary",
        use_container_width=True,
        disabled=(energy_model is None)
    )

    if run_optimization:

        plastic_data = energy_data[
            energy_data["Plastic_Type"].astype(str) == selected_plastic
        ].copy()

        if len(plastic_data) < 10:
            st.error(
                f"Insufficient eligible records for {selected_plastic}. "
                "Please select another plastic type."
            )
        else:
            categorical_variables = [
                "Catalyst_Type",
                "Reactor_Class"
            ]
            numerical_variables = NUM

            category_maps = {
                column: sorted(
                    plastic_data[column]
                    .fillna("Unknown")
                    .astype(str)
                    .unique()
                    .tolist()
                )
                for column in categorical_variables
            }

            if any(not category_maps[column] for column in categorical_variables):
                st.error(
                    "Missing categorical optimization values in the "
                    "selected plastic dataset."
                )
            else:
                lower_bounds = []
                upper_bounds = []

                for column in categorical_variables:
                    lower_bounds.append(0)
                    upper_bounds.append(len(category_maps[column]) - 1)

                for column in numerical_variables:
                    low, high = get_optimization_bounds(
                        column,
                        plastic_data
                    )
                    lower_bounds.append(low)
                    upper_bounds.append(high)

                lower_bounds = np.asarray(lower_bounds, dtype=float)
                upper_bounds = np.asarray(upper_bounds, dtype=float)

                def decode_candidates(decision_vectors):
                    decision_vectors = np.atleast_2d(decision_vectors)
                    decoded = {
                        "Plastic_Type": np.full(
                            len(decision_vectors),
                            selected_plastic,
                            dtype=object
                        )
                    }

                    for index, column in enumerate(categorical_variables):
                        category_indices = np.rint(
                            decision_vectors[:, index]
                        ).astype(int)
                        category_indices = np.clip(
                            category_indices,
                            0,
                            len(category_maps[column]) - 1
                        )
                        categories = np.asarray(category_maps[column], dtype=object)
                        decoded[column] = categories[category_indices]

                    offset = len(categorical_variables)
                    for index, column in enumerate(numerical_variables):
                        decoded[column] = decision_vectors[:, offset + index]

                    return pd.DataFrame(decoded, columns=FEATURES)

                class PyrolysisProblem(Problem):

                    def __init__(self):
                        super().__init__(
                            n_var=len(categorical_variables) + len(numerical_variables),
                            n_obj=2,
                            n_ieq_constr=2,
                            xl=lower_bounds,
                            xu=upper_bounds
                        )

                    def _evaluate(self, X, out, *args, **kwargs):
                        candidates = decode_candidates(X)
                        oil_predictions = predict_oil(candidates)
                        energy_predictions = predict_energy(candidates)
                        out["F"] = np.column_stack(
                            (-oil_predictions, energy_predictions)
                        )
                        out["G"] = np.column_stack(
                            (-oil_predictions, oil_predictions - 100.0)
                        )

                progress = st.progress(0)
                status = st.empty()

                try:
                    status.info("NSGA-II is searching for optimal conditions...")

                    result = minimize(
                        PyrolysisProblem(),
                        NSGA2(pop_size=POPULATION_SIZE),
                        ("n_gen", N_GENERATIONS),
                        seed=42,
                        verbose=False
                    )

                    progress.progress(100)

                    if result.X is None:
                        st.error(
                            "No solutions satisfy the physical oil-yield "
                            "range of 0% to 100% for this plastic type."
                        )
                    else:
                        pareto_candidates = decode_candidates(result.X)
                        oil_predictions = predict_oil(pareto_candidates)
                        energy_predictions = predict_energy(pareto_candidates)

                        results_df = pareto_candidates.copy()
                        results_df["Predicted Oil Yield (wt%)"] = oil_predictions
                        results_df[
                            "Predicted Energy Consumption (MJ/kg feed)"
                        ] = energy_predictions
                        results_df = results_df.drop_duplicates()
                        results_df = results_df.replace(
                            [np.inf, -np.inf],
                            np.nan
                        ).dropna(
                            subset=[
                                "Predicted Oil Yield (wt%)",
                                "Predicted Energy Consumption (MJ/kg feed)"
                            ]
                        )
                        results_df = results_df[
                            results_df["Predicted Oil Yield (wt%)"].between(
                                0.0,
                                100.0
                            )
                        ]

                        if results_df.empty:
                            st.error(
                                "No physically plausible Pareto solutions "
                                "were found for this plastic type."
                            )
                        else:
                            oil_values = results_df[
                                "Predicted Oil Yield (wt%)"
                            ].to_numpy()
                            energy_values = results_df[
                                "Predicted Energy Consumption (MJ/kg feed)"
                            ].to_numpy()

                            oil_range = max(float(np.ptp(oil_values)), 1e-9)
                            energy_range = max(float(np.ptp(energy_values)), 1e-9)
                            normalized_oil = (
                                oil_values - oil_values.min()
                            ) / oil_range
                            normalized_energy = (
                                energy_values - energy_values.min()
                            ) / energy_range
                            distances = np.sqrt(
                                (1 - normalized_oil) ** 2
                                + normalized_energy ** 2
                            )
                            compromise_index = int(np.argmin(distances))

                            recommendation_indices = []
                            recommendation_labels = []

                            def add_recommendation(index, label):
                                if index not in recommendation_indices:
                                    recommendation_indices.append(index)
                                    recommendation_labels.append(label)

                            add_recommendation(
                                int(np.argmin(energy_values)),
                                "Lowest energy"
                            )
                            add_recommendation(
                                int(np.argmax(oil_values)),
                                "Highest oil yield"
                            )
                            add_recommendation(
                                compromise_index,
                                "Best balanced compromise"
                            )

                            objective_points = np.column_stack(
                                (normalized_energy, normalized_oil)
                            )
                            while (
                                len(recommendation_indices) < min(4, len(results_df))
                            ):
                                remaining_indices = [
                                    index
                                    for index in range(len(results_df))
                                    if index not in recommendation_indices
                                ]
                                selected_points = objective_points[
                                    recommendation_indices
                                ]
                                distances_to_selected = np.linalg.norm(
                                    objective_points[remaining_indices, None, :]
                                    - selected_points[None, :, :],
                                    axis=2
                                )
                                diverse_index = remaining_indices[
                                    int(np.argmax(distances_to_selected.min(axis=1)))
                                ]
                                add_recommendation(
                                    diverse_index,
                                    "Alternative trade-off"
                                )

                            recommended_df = results_df.iloc[
                                recommendation_indices
                            ].copy()
                            recommended_df[
                                "Assumed System Heat Loss (%)"
                            ] = BASE_HEAT_LOSS_PERCENT
                            recommended_df[
                                "Assumed Heater Efficiency (%)"
                            ] = BASE_HEATER_EFFICIENCY_PERCENT
                            recommended_df.insert(
                                0,
                                "Recommendation",
                                recommendation_labels
                            )

                            status.success("Optimization completed.")
                            st.subheader("Recommended Pareto Solutions")
                            st.caption(
                                "Up to four non-dominated options: minimum "
                                "energy, maximum oil yield, the balanced "
                                "compromise, and a distinct alternative."
                            )
                            st.dataframe(
                                recommended_df,
                                use_container_width=True,
                                hide_index=True
                            )

                            csv = recommended_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download Recommended Solutions CSV",
                                data=csv,
                                file_name="NSGA_II_Recommended_Solutions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                except Exception as e:
                    status.error(f"Optimization failed: {e}")


# =====================================================
# END OF APPLICATION
# =====================================================
