
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("09_NSGA2_Results")

pareto = pd.read_csv(folder / "09_NSGA2_Pareto_Front.csv")
best = pd.read_csv(folder / "09_NSGA2_Compromise_Solution.csv")

yield_col = "Predicted_Oil_Yield_wt_pct"
sec_col = "CE_SEC_MJ_per_kg_feed"

# 1. Pareto front
plt.figure(figsize=(9, 6))

plt.scatter(
    pareto[sec_col],
    pareto[yield_col],
    label="Pareto-optimal solutions"
)

plt.scatter(
    best[sec_col],
    best[yield_col],
    marker="*",
    s=250,
    label="Compromise solution"
)

plt.xlabel("Specific Energy Consumption (MJ/kg feed)")
plt.ylabel("Predicted Oil Yield (wt.%)")
plt.title("NSGA-II Pareto Front - Plastic Pyrolysis")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(folder / "10_Pareto_Front.png", dpi=300)
plt.close()

# 2. Oil yield across Pareto solutions
plt.figure(figsize=(9, 5))
plt.plot(
    range(1, len(pareto) + 1),
    pareto[yield_col],
    marker="o"
)
plt.xlabel("Pareto solution (ordered by SEC)")
plt.ylabel("Predicted Oil Yield (wt.%)")
plt.title("Oil Yield Across Pareto Solutions")
plt.grid(True)
plt.tight_layout()
plt.savefig(folder / "10_Oil_Yield_Comparison.png", dpi=300)
plt.close()

# 3. SEC across Pareto solutions
plt.figure(figsize=(9, 5))
plt.plot(
    range(1, len(pareto) + 1),
    pareto[sec_col],
    marker="o"
)
plt.xlabel("Pareto solution (ordered by SEC)")
plt.ylabel("SEC (MJ/kg feed)")
plt.title("Energy Consumption Across Pareto Solutions")
plt.grid(True)
plt.tight_layout()
plt.savefig(folder / "10_SEC_Comparison.png", dpi=300)
plt.close()

print("All three graphs saved successfully.")
