
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair


# ============================================================
# 1. FILES AND SETTINGS
# ============================================================

DATASET_FILE = "07_ML_Dataset_with_CE_Energy_Scenarios.csv"
MODEL_FILE = "xgboost_model.joblib"

OUTPUT_FOLDER = Path("09_NSGA2_Results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

RANDOM_SEED = 42
POPULATION_SIZE = 100
GENERATIONS = 200

# Same features used by the existing XGBoost model
CATEGORICAL_FEATURES = [
    "Plastic_Type",
    "Catalyst_Type",
    "Reactor_Class"
]

NUMERICAL_FEATURES = [
    "Temperature_C",
    "Heating_Rate_C_per_min",
    "Particle_Size_mm",
    "Feed_Rate_kg_per_h",
    "Feed_Mass_kg",
    "Catalyst_Loading_wt_pct"
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

SEC_COLUMN = "CE_SEC_MJ_per_kg_feed"


# ============================================================
# 2. LOAD DATA AND EXISTING XGBOOST MODEL
# ============================================================

print("\n========== NSGA-II OPTIMISATION ==========\n")

if not Path(DATASET_FILE).exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

if not Path(MODEL_FILE).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

df = pd.read_csv(DATASET_FILE)
model = joblib.load(MODEL_FILE)

print("Dataset loaded:", df.shape)
print("Existing XGBoost model loaded.")


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURES
    + [
        "ML_Core_Complete",
        "CE_Use_In_Optimisation",
        SEC_COLUMN,
        "ML_Row_ID",
        "Source_ID",
        "Oil_Yield_wt_pct"
    ]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required columns are present.")


# ============================================================
# 4. PREPARE FEASIBLE EXPERIMENTAL CANDIDATES
# ============================================================

candidates = df[
    (df["ML_Core_Complete"] == 1)
    & (df["CE_Use_In_Optimisation"] == 1)
].copy()

print("\nEligible ML + CE experiments:", len(candidates))

# Both proposed operating variables must be available.
# Other missing ML features are retained because the existing
# XGBoost pipeline was trained with its original missing-data setup.

candidates = candidates.dropna(
    subset=[
        "Temperature_C",
        "Particle_Size_mm",
        SEC_COLUMN
    ]
).copy()

# SEC must be a valid positive number.
candidates = candidates[
    np.isfinite(candidates[SEC_COLUMN])
    & (candidates[SEC_COLUMN] > 0)
].copy()

candidates = candidates.reset_index(drop=True)

if len(candidates) < 10:
    raise ValueError(
        "Too few usable candidates for this optimisation."
    )

print("Usable candidate experiments:", len(candidates))

print(
    "Temperature range:",
    candidates["Temperature_C"].min(),
    "to",
    candidates["Temperature_C"].max(),
    "°C"
)

print(
    "Particle-size range:",
    candidates["Particle_Size_mm"].min(),
    "to",
    candidates["Particle_Size_mm"].max(),
    "mm"
)


# ============================================================
# 5. PREDICT OIL YIELD USING EXISTING XGBOOST MODEL
# ============================================================

X = candidates[FEATURES].copy()

print("\nPredicting oil yield...")

predicted_yield = np.asarray(model.predict(X)).reshape(-1)

if len(predicted_yield) != len(candidates):
    raise ValueError("Prediction count does not match candidate count.")

if not np.all(np.isfinite(predicted_yield)):
    raise ValueError("XGBoost produced invalid predictions.")

candidates["Predicted_Oil_Yield_wt_pct"] = predicted_yield

# Existing CE energy values are used for these same experiments.
candidates["Optimisation_SEC_MJ_per_kg_feed"] = (
    candidates[SEC_COLUMN]
)

print("Oil-yield predictions completed.")

print(
    "Predicted oil yield range:",
    round(predicted_yield.min(), 3),
    "to",
    round(predicted_yield.max(), 3)
)

print(
    "SEC range:",
    round(candidates[SEC_COLUMN].min(), 3),
    "to",
    round(candidates[SEC_COLUMN].max(), 3)
)


# Save the complete candidate pool.
candidates.to_csv(
    OUTPUT_FOLDER / "09_All_Optimisation_Candidates.csv",
    index=False
)


# ============================================================
# 6. DEFINE TWO-OBJECTIVE NSGA-II PROBLEM
# ============================================================

# Decision variable:
# Select one experimental row from the candidate pool.
#
# Objective 1: maximise predicted oil yield.
# NSGA-II minimises, so use negative oil yield.
#
# Objective 2: minimise SEC.

oil_values = candidates[
    "Predicted_Oil_Yield_wt_pct"
].to_numpy()

sec_values = candidates[
    "Optimisation_SEC_MJ_per_kg_feed"
].to_numpy()


class PyrolysisOptimisationProblem(Problem):

    def __init__(self):
        super().__init__(
            n_var=1,
            n_obj=2,
            n_ieq_constr=0,
            xl=0,
            xu=len(candidates) - 1,
            vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # Convert selected index into an actual experiment.
        indices = np.clip(
            np.rint(x[:, 0]).astype(int),
            0,
            len(candidates) - 1
        )

        # Objective 1: maximise oil yield.
        objective_1 = -oil_values[indices]

        # Objective 2: minimise SEC.
        objective_2 = sec_values[indices]

        out["F"] = np.column_stack([
            objective_1,
            objective_2
        ])


problem = PyrolysisOptimisationProblem()


# ============================================================
# 7. RUN NSGA-II
# ============================================================

algorithm = NSGA2(
    pop_size=min(POPULATION_SIZE, len(candidates)),
    sampling=IntegerRandomSampling(),

    crossover=SBX(
        prob=0.9,
        eta=15,
        vtype=float,
        repair=RoundingRepair()
    ),

    mutation=PM(
        eta=20,
        vtype=float,
        repair=RoundingRepair()
    ),

    eliminate_duplicates=True
)

print("\nStarting NSGA-II...")
print("Population size:", min(POPULATION_SIZE, len(candidates)))
print("Generations:", GENERATIONS)

result = minimize(
    problem,
    algorithm,
    termination=("n_gen", GENERATIONS),
    seed=RANDOM_SEED,
    verbose=True
)

if result.X is None:
    raise RuntimeError("NSGA-II did not return a solution.")


# ============================================================
# 8. EXTRACT NSGA-II SOLUTIONS
# ============================================================

nsga_indices = np.unique(
    np.clip(
        np.rint(np.asarray(result.X).reshape(-1)).astype(int),
        0,
        len(candidates) - 1
    )
)

nsga_results = candidates.iloc[nsga_indices].copy()

nsga_results = nsga_results.sort_values(
    SEC_COLUMN,
    ascending=True
)

nsga_results.to_csv(
    OUTPUT_FOLDER / "09_NSGA2_Algorithm_Results.csv",
    index=False
)


# ============================================================
# 9. CALCULATE EXACT PARETO FRONT OF CANDIDATE POOL
# ============================================================

# This checks every candidate, ensuring that the final reported
# discrete Pareto front is not limited to candidates visited by
# the stochastic NSGA-II population.

objectives = np.column_stack([
    -oil_values,
    sec_values
])

is_pareto = np.ones(len(objectives), dtype=bool)

for i in range(len(objectives)):

    dominated = np.all(
        objectives <= objectives[i],
        axis=1
    ) & np.any(
        objectives < objectives[i],
        axis=1
    )

    if np.any(dominated):
        is_pareto[i] = False

pareto_results = candidates.loc[is_pareto].copy()

pareto_results = pareto_results.sort_values(
    SEC_COLUMN,
    ascending=True
)

pareto_results.to_csv(
    OUTPUT_FOLDER / "09_NSGA2_Pareto_Front.csv",
    index=False
)


# ============================================================
# 10. SELECT A COMPROMISE SOLUTION
# ============================================================

# Normalise both objectives and find the candidate closest
# to the ideal point: highest yield and lowest SEC.

yield_min = pareto_results[
    "Predicted_Oil_Yield_wt_pct"
].min()

yield_max = pareto_results[
    "Predicted_Oil_Yield_wt_pct"
].max()

sec_min = pareto_results[SEC_COLUMN].min()
sec_max = pareto_results[SEC_COLUMN].max()

yield_range = max(yield_max - yield_min, 1e-12)
sec_range = max(sec_max - sec_min, 1e-12)

normalised_yield = (
    yield_max - pareto_results["Predicted_Oil_Yield_wt_pct"]
) / yield_range

normalised_sec = (
    pareto_results[SEC_COLUMN] - sec_min
) / sec_range

distance = np.sqrt(
    normalised_yield**2 + normalised_sec**2
)

best_index = distance.idxmin()

best_solution = pareto_results.loc[[best_index]].copy()

best_solution.to_csv(
    OUTPUT_FOLDER / "09_NSGA2_Compromise_Solution.csv",
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

display_columns = [
    "ML_Row_ID",
    "Source_ID",
    "Plastic_Type",
    "Reactor_Class",
    "Temperature_C",
    "Particle_Size_mm",
    "Predicted_Oil_Yield_wt_pct",
    SEC_COLUMN
]

print("\n========== NSGA-II RESULTS ==========")

print(
    "\nNumber of solutions returned by NSGA-II:",
    len(nsga_results)
)

print(
    "Number of exact Pareto-optimal candidates:",
    len(pareto_results)
)

print("\n--- PARETO FRONT ---")

print(
    pareto_results[display_columns].to_string(index=False)
)

print("\n--- COMPROMISE SOLUTION ---")

print(
    best_solution[display_columns].to_string(index=False)
)


# ============================================================
# 12. SAVE RUN SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total dataset rows",
        "Eligible ML and CE rows",
        "Usable optimisation candidates",
        "NSGA-II generations",
        "NSGA-II population size",
        "NSGA-II returned solutions",
        "Exact Pareto-optimal candidates",
        "Best predicted oil yield (wt.%)",
        "Lowest SEC (MJ/kg feed)",
        "Compromise predicted oil yield (wt.%)",
        "Compromise SEC (MJ/kg feed)"
    ],

    "Value": [
        len(df),
        len(df[
            (df["ML_Core_Complete"] == 1)
            & (df["CE_Use_In_Optimisation"] == 1)
        ]),
        len(candidates),
        GENERATIONS,
        min(POPULATION_SIZE, len(candidates)),
        len(nsga_results),
        len(pareto_results),
        oil_values.max(),
        sec_values.min(),
        best_solution["Predicted_Oil_Yield_wt_pct"].iloc[0],
        best_solution[SEC_COLUMN].iloc[0]
    ]
})

summary.to_csv(
    OUTPUT_FOLDER / "09_NSGA2_Run_Summary.csv",
    index=False
)

print("\n--- RUN SUMMARY ---")
print(summary.to_string(index=False))

print("\nAll optimisation files saved in:")
print(OUTPUT_FOLDER.resolve())

print("\nNSGA-II optimisation completed.")
