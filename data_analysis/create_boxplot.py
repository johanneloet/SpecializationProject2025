import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
import re

CL_path = r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs\all_performance_indexes_testid_and_feat_space_SVC.csv"
#DL_path = r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis_DL\all_metrics_per_test_CNN (1).csv"

DL_path = r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs\all_metrics_per_test_CNN_final.csv"

# read data into dfs
CL_data = pd.read_csv(CL_path,  index_col=0)
DL_data = pd.read_csv(DL_path, index_col=0)

rows = []

CL_space_name_to_plot_mapping = {
    'baseline' : 'CL1',
    'expanded+baseline' : 'CL2',
    'FSR_only' : 'CL4',
    'arm_only+FSR' : 'CL5',
    'back_only+FSR': 'CL6',
    'IMU_only' : 'CL3',
}

DL_space_name_to_plot_mapping = {
    'All' : 'DL1',
    'FSR_only' : 'DL2',
    'IMU_only' : 'DL3',
    'Back_and_FSR' : 'DL4',
    'Arm_and_FSR' : 'DL5'
}

TEST_ID_mapping = {f"test_{i}": f"Test participant {i}" for i in range(1, 21)}


    
# records_CL = []
# for feature_space, row in CL_data.iterrows():
#     for test_id, metrics in row.items():
#         if pd.isna(metrics):
#             continue
#         # metrics is a dict like {'f1': ..., 'accuracy': ...}
#         f1 = metrics.get("f1") or metrics.get("f1_score")
#         if f1 is not None:
#             records_CL.append({
#                 "feature_space": feature_space,
#                 "test_id": test_id,
#                 "f1": f1
#             })

# records_DL = []
# for feature_space, row in DL_data.iterrows():
#     for test_id, metrics in row.items():
#         if pd.isna(metrics):
#             continue
#         # metrics is a dict like {'f1': ..., 'accuracy': ...}
#         f1 = metrics.get("f1") or metrics.get("f1_score")
#         if f1 is not None:
#             records_DL.append({
#                 "feature_space": feature_space,
#                 "test_id": test_id,
#                 "f1": f1
#             })




# # # data structure: { "test_1": [ { ...feature_spaces... }, "best_name" ], ... }
# # for test_name, value in data.items():
# #     result_dict, best_name = value  # first element is dict of feature spaces
    
# #     for feature_space, fs_info in result_dict.items():
# #         f1 = fs_info["f1"]
# #         rows.append({
# #             "test": test_name,
# #             "feature_space": feature_space,
# #             "f1": f1
# #         })

# df = pd.DataFrame(rows)
# print(df.head())

# df['feature_space'] =df['feature_space'].map(space_name_to_plot_mapping)

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10, 5))

# # Boxplot per feature space
# sns.boxplot(x="feature_space", y="f1", data=df, color='lightgray')

# # Overlay individual test F1s as points
# sns.stripplot(
#     x="feature_space", y="f1", data=df,
#     hue="test", alpha=0.6, jitter=True
# )

# plt.xlabel("Feature space") # scenario?
# plt.ylabel("F1 validation score")
# plt.xticks(rotation=30, ha="right")
# plt.legend(
#     title="Left out test ID",
#     bbox_to_anchor=(1.02, 1),
#     loc="upper left",
#     borderaxespad=0.
# )
# plt.tight_layout()
# plt.show()

remove_y_test_fit = re.compile(
    r"'y_test_fit': array\([\s\S]*?\),\s*"   # non-greedy up to the first "),"
)

def parse_metrics(cell):
    """
    cell: either a dict or the string you showed.
    Returns a Python dict with keys: accuracy, f1, precision, recall, hyperparameters, ...
    """
    if isinstance(cell, dict):
        return cell

    if not isinstance(cell, str) or not cell.strip():
        return None

    # Remove the y_test_fit array part so literal_eval is happy
    cleaned = remove_y_test_fit.sub("", cell)

    # Now cleaned should look like:
    # "{'accuracy': 0.6984, 'f1': 0.6038, 'precision': ..., 'recall': ..., 'hyperparameters': {...}}"
    return literal_eval(cleaned)

def extract_records(df, mapping, model_type):
    records = []

    for feature_space, row in df.iterrows():
        print("FEATURE SPACE", feature_space)
        plot_name = mapping.get(feature_space, feature_space)

        for test_id, cell in row.items():
            print(test_id)
            print(cell, " ")
            if pd.isna(cell):
                continue

            metrics = parse_metrics(cell)  # <- use ast.literal_eval *after* cleaning
            if metrics is None:
                continue

            f1 = metrics['f1']  # now it's a real dict
            records.append({
                "model_type": model_type,
                "feature_space": feature_space,
                "feature_space_plot": plot_name,
                "test_id": test_id,
                "f1": f1,
            })

    return records


# -------------------------------------------------------------------
# 5. Build long dataframe for CL and DL
# -------------------------------------------------------------------
records_CL = extract_records(CL_data, CL_space_name_to_plot_mapping, "CL")
records_DL = extract_records(DL_data, DL_space_name_to_plot_mapping, "DL")

perf = pd.DataFrame(records_CL + records_DL)

# -------------------------------------------------------------------
# 6. Order of feature spaces on x-axis (CL1.. + DL1..)
# -------------------------------------------------------------------
cl_order = sorted(CL_space_name_to_plot_mapping.values(), key=lambda s: int(s[2:]))
dl_order = sorted(DL_space_name_to_plot_mapping.values(), key=lambda s: int(s[2:]))

feature_order = cl_order + [""] + dl_order
perf_nonempty = perf[perf["feature_space_plot"] != ""]

perf_nonempty["participant"] = perf_nonempty["test_id"].map(TEST_ID_mapping)

# keep only those actually present
feature_order = [f for f in feature_order if f in perf["feature_space_plot"].unique()]
# keep only those actually present
feature_order = [f for f in feature_order if f in perf["feature_space_plot"].unique()]

# -------------------------------------------------------------------
# 7. Plot: boxplot + colored points by test_id
# -------------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Boxplot: F1 distribution per feature space
sns.boxplot(
    data=perf_nonempty,
    x="feature_space_plot",
    y="f1",
    order=feature_order,
    color="white",
    showfliers=False,
    width=0.3,
    showmeans=True,
    meanprops={"marker":"^",
                       "markerfacecolor":"grey",
                       "markeredgecolor":"grey",
                       "markersize":"8"}
)

# Scatter points: one per test_id, colored by test_id
sns.stripplot(
    data=perf_nonempty,
    x="feature_space_plot",
    y="f1",
    jitter=0.25,      # default ~0.2, reduce to compress horizontally
    size=4, 
    order=feature_order,
    hue="participant",
    dodge=False,
    alpha=0.8,
)

import matplotlib.lines as mlines
mean_legend = mlines.Line2D(
    [], [], 
    color="grey",
    marker="^",        # same marker as meanprops
    markersize=6,
    linestyle="None",
    label="Mean"
)
# add separation between CL and DL
plt.axvline(x=5.5, color='gray', linestyle='--', linewidth=1)

#plt.ylim(0.0, 1.02)


plt.xlabel("Scenario")
plt.ylabel("F1 score")
plt.xticks(rotation=45, ha="right")

# # Move legend outside the plot
# plt.legend(
#     handles=[mean_legend] + plt.gca().get_legend_handles_labels()[0],
#     labels=["Mean F1"] + plt.gca().get_legend_handles_labels()[1],
#     title="Test ID",
#     bbox_to_anchor=(1.02, 1),
#     loc="upper left"
# )

handles, labels = plt.gca().get_legend_handles_labels()

handles = [mean_legend] + handles
labels  = ["Mean F1"] + labels

# plt.legend(
#     handles=handles,
#     labels=labels,
#     title="Test ID",
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.15),  # below the plot
#     ncol=min(len(handles), 6),    # horizontal layout, adjust 6 as you like
#     frameon=True
# )
#plt.legend(title="Test ID", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.legend(
    handles=handles,
    labels=labels,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    fontsize=9,
    markerscale=0.8,
    labelspacing=0.4,
    handletextpad=0.4,
    handlelength=1.2,
    frameon=True
)

plt.tight_layout()
plt.savefig("CL_DL_F1_boxplot.pdf", format="pdf", bbox_inches="tight")
plt.show()