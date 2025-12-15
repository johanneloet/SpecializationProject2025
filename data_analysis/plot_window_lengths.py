import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
from get_paths import get_test_file_paths

# --- LOAD DATA ---

test_file_paths =  get_test_file_paths()

all_counts = []  # will store results across all test_ids

for test_id in test_file_paths.keys():
    df = pd.read_csv(test_file_paths[test_id]['arm'])

    # Count samples per (rep_id, label)
    grouped = (
        df.groupby(["rep_id", "label"])
          .size()
          .reset_index(name="num_samples")
    )

    # Add test_id for tracking
    grouped["test_id"] = test_id

    # Append to global list
    all_counts.append(grouped)

# Combine all results
results_df = pd.concat(all_counts, ignore_index=True)

# Save to CSV
output_path = "repetition_sample_counts.csv"
results_df.to_csv(output_path, index=False)

print(f"Saved repetition sample counts to: {output_path}")
df = pd.read_csv(output_path)

# --- CLEANUP ---
df["test_id"] = df["test_id"].astype(str)
df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")

# Drop unwanted labels from the plot
labels_to_exclude = ["standing", "walking", "sitting"]
df = df[~df["label"].isin(labels_to_exclude)]

# Extract the base activity name from rep_id (before the trailing _number)
df["activity"] = df["rep_id"].apply(lambda x: re.sub(r"_\d+$", "", x))

# --- CREATE COLOR MAP BASED ON test_id ---
# Sort test_ids numerically by the number at the end (e.g. "test_1", "test_2", ...)
def extract_num(s):
    m = re.findall(r"\d+", s)
    return int(m[0]) if m else 0

unique_tests = sorted(df["test_id"].unique(), key=extract_num)
# Define your scientific gradient
colors_list = [
    (0.35, 0.55, 0.95),  # Blue
    (0.60, 0.40, 0.95),  # Indigo / Blue-Purple
    (0.85, 0.35, 0.90),  # Purple-Pink
    (0.95, 0.75, 0.30),  # Soft Yellow-Orange (scientific yellow)
    (0.80, 0.80, 0.85),  # Grey
]
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list("blue_purple_pink_grey", colors_list)

# Sample evenly spaced colors for participants
colors = custom_cmap(np.linspace(0, 1, len(unique_tests)))

colors = cm.tab20(np.linspace(0, 1, len(unique_tests)))  # 20 distinct colors
test_id_to_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(unique_tests)}

pid15 = next((tid for tid in unique_tests if extract_num(tid) == 15), None)

# if pid15:
#     # Hot pink (R=1.0, G=0.0, B=0.7)
#     test_id_to_color_map[pid15] = (1.0, 0.0, 0.7, 1.0)

# Map test_id -> "Participant X" for legend
test_id_to_legend = {tid: f"Participant {extract_num(tid)}" for tid in unique_tests}

# --- SCATTER PLOT ---
plt.figure(figsize=(14, 7))

for _, row in df.iterrows():
    activity = row["activity"]
    num_samples = row["num_samples"]
    test_id = row["test_id"]
    plt.plot(activity, num_samples, 'o',
             color=test_id_to_color_map[test_id],
             alpha=0.7)


p15_rows = df[df["test_id"] == pid15]
for _, row in p15_rows.iterrows():
    plt.plot(row["activity"], row["num_samples"], 'o',
             color=test_id_to_color_map[pid15],
             alpha=0.7,
             #markersize=12,   # bigger so it stands out
             #markeredgecolor="black",
             #markeredgewidth=1.0
             )
plt.xlabel("Activity")
plt.ylabel("Number of Samples (repetition length)")
plt.title("Per-Repetition Window Lengths Colored by Participant Insole Data")
plt.xticks(rotation=45, ha="right")

# Legend for participants
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=test_id_to_color_map[tid],
               markersize=8)
    for tid in unique_tests
]
legend_labels = [test_id_to_legend[tid] for tid in unique_tests]

plt.legend(handles, legend_labels, title="Participant",
           bbox_to_anchor=(1.05, 1), loc='upper left')

# --- ADD CUTOFF LINE ---
cutoff = 350
plt.axhline(y=cutoff, color='red', linestyle='--', linewidth=1)

# Add text label slightly above the line
plt.text(
    x=0.02, y=cutoff + 50, s="Cutoff (350 samples)",
    color='red', fontsize=10,
    transform=plt.gca().get_yaxis_transform(),  # keeps text in data-y, axes-x
    ha='left', va='bottom'
)

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("samples_per_participant_insole.pdf", format='pdf', bbox_inches='tight')

plt.show()