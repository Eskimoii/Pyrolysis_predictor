
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD THE MODIFIED DATASET
# ==========================================

input_file = "07_ML_Dataset_with_CE_Energy_Scenarios.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# ==========================================
# 2. DEFINE ENERGY MODEL PARAMETERS
# ==========================================

# Heat loss fractions
loss_scenarios = [0.10, 0.20, 0.30]

# Heater efficiencies
efficiency_scenarios = [0.85, 0.70, 0.50]

# ==========================================
# 3. CALCULATE USEFUL ENERGY
# ==========================================

# Both values are already expressed per kg of feed
df["CE_Quseful_MJ_per_kg"] = (
    df["CE_Qsens_MJ_per_kg"]
    + df["CE_Qpyr_MJ_per_kg"]
)

# ==========================================
# 4. CALCULATE ALL NINE SCENARIOS
# ==========================================

for loss in loss_scenarios:

    # Heat lost by reactor/system
    q_loss = loss * df["CE_Quseful_MJ_per_kg"]

    # Total useful + lost energy
    q_total = df["CE_Quseful_MJ_per_kg"] + q_loss

    for efficiency in efficiency_scenarios:

        loss_pct = int(loss * 100)
        eta_pct = int(efficiency * 100)

        column_name = (
            f"CE_SEC_loss{loss_pct}_eta{eta_pct}_MJ_per_kg"
        )

        # Energy supplied per kg of feed
        df[column_name] = q_total / efficiency

# ==========================================
# 5. BASE CASE: 20% LOSS, 70% EFFICIENCY
# ==========================================

df["CE_Qloss_base_MJ_per_kg"] = (
    0.20 * df["CE_Quseful_MJ_per_kg"]
)

df["CE_Qtotal_base_MJ_per_kg"] = (
    df["CE_Quseful_MJ_per_kg"]
    + df["CE_Qloss_base_MJ_per_kg"]
)

df["CE_SEC_MJ_per_kg_feed"] = (
    df["CE_Qtotal_base_MJ_per_kg"] / 0.70
)

# ==========================================
# 6. OPTIMISATION ELIGIBILITY FLAG
# ==========================================

valid_energy = (
    df["CE_Qsens_MJ_per_kg"].notna()
    & df["CE_Qpyr_MJ_per_kg"].notna()
    & np.isfinite(df["CE_Quseful_MJ_per_kg"])
    & (df["CE_Quseful_MJ_per_kg"] >= 0)
)

df["CE_Use_In_Optimisation"] = (
    valid_energy.astype(int)
)

# ==========================================
# 7. CHECK RESULTS
# ==========================================

print("\nTotal rows:", len(df))

print(
    "Rows with usable CE energy:",
    df["CE_Use_In_Optimisation"].sum()
)

print(
    "Rows without CE energy:",
    (df["CE_Use_In_Optimisation"] == 0).sum()
)

print(
    "\nMean base-case SEC:",
    df.loc[
        df["CE_Use_In_Optimisation"] == 1,
        "CE_SEC_MJ_per_kg_feed"
    ].mean()
)

print(
    "\nBase-case SEC range:",
    df.loc[
        df["CE_Use_In_Optimisation"] == 1,
        "CE_SEC_MJ_per_kg_feed"
    ].agg(["min", "max"])
)

# ==========================================
# 8. SAVE OUTPUTS
# ==========================================

df.to_csv(
    "08_CE_Model_Results.csv",
    index=False
)

with pd.ExcelWriter(
    "08_CE_Model_Results.xlsx",
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="CE_Dataset",
        index=False
    )

    summary = pd.DataFrame({
        "Scenario": [
            "10% loss, 85% efficiency",
            "10% loss, 70% efficiency",
            "10% loss, 50% efficiency",
            "20% loss, 85% efficiency",
            "20% loss, 70% efficiency",
            "20% loss, 50% efficiency",
            "30% loss, 85% efficiency",
            "30% loss, 70% efficiency",
            "30% loss, 50% efficiency"
        ],
        "Mean_SEC_MJ_per_kg": [
            df.loc[
                df["CE_Use_In_Optimisation"] == 1,
                f"CE_SEC_loss{int(l*100)}_eta{int(e*100)}_MJ_per_kg"
            ].mean()
            for l in loss_scenarios
            for e in efficiency_scenarios
        ]
    })

    summary.to_excel(
        writer,
        sheet_name="Scenario_Summary",
        index=False
    )

print("\nCE model completed.")
print("Files saved successfully.")
