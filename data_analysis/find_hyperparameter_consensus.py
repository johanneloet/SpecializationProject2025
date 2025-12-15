import json
from collections import Counter
import pandas as pd

# Path to your JSON file
file_path = "./outputs/all_hyperparameters_NN_resampled2.json"

feature_spaces =  [
                    "baseline",
                    "expanded+baseline",
                    "expanded_only",
                    "time_only",
                    "time_only+exp_FSR",
                    "freq_only",
                    "freq_only+exp_FSR",
                    "FSR_only",
                    "arm_only",
                    "back_only"
                    ]

best_feature_spaces = []
param_rows = []

# Open and load the JSON data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    
for test_id in data.keys():
    print("TEST ID", test_id)
    opt_space = data[test_id][1]
    opt_params = data[test_id][0][opt_space]['parameters']
    
        
    best_feature_spaces.append(opt_space)
    param_rows.append(opt_params)

print("All best spaces")
print(best_feature_spaces)

space_counts = Counter(best_feature_spaces)
overall_best_feature_space, freq = space_counts.most_common(1)[0]
print(f"Most common feature space: {overall_best_feature_space} ({freq} occurrences)")

df_params = pd.DataFrame(param_rows)
consensus_params = df_params.mode().iloc[0].to_dict()
print("Consensus hyperparameters:")
for k, v in consensus_params.items():
    print(f"  {k}: {v}")