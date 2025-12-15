import json
import pandas as pd

# Load your JSON (if you have it in a file, replace this with open("file.json"))
path = r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs\all_hyperparameters_SVC.json"

with open(path, "r") as f:
    data = json.load(f)

# Convert JSON to a DataFrame
rows = {}
for test_id, (results, best_model) in data.items():
    # Extract F1 values for each feature space
    row = {feature: values["f1"] for feature, values in results.items()}
    row["best_model"] = best_model  # optional: include the best-performing model name
    rows[test_id] = row

df = pd.DataFrame(rows).T  # transpose so tests are rows

# Sort columns alphabetically or in a custom order (optional)
ordered_cols = [
    "baseline", "expanded+baseline", "expanded_only",
    "time_only", "time_only+exp_FSR",
    "freq_only", "freq_only+exp_FSR",
    "FSR_only", "arm_only", "back_only", "best_model"
]
df = df[ordered_cols]

# Display nicely
print(df.round(4))

df.to_latex(r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs\latex_table.tex")
