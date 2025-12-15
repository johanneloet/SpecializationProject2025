import pandas as pd 
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap


def save_pca_contributions_headline_style(
    csv_path,
    output_path,
    n_pcs=5,
    top_n=10,
    figsize=(10, 10),
    dpi=300
):
    """
    Creates a visual figure showing PCA feature contributions grouped by PC.
    Each PC appears as a headline with its top N features listed below.
    Automatically splits PCs into two columns if needed.
    """

    # Load and clean data
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df["PC_num"] = df["Principal_Component"].str.extract(r"(\d+)").astype(int)
    df = df[df["PC_num"] <= n_pcs]
    df = df.sort_values(["PC_num", "Contribution_(%)"], ascending=[True, False])

    grouped = df.groupby("PC_num")

    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.text(0.5, 1.03, 
            f"Top {top_n} Feature Contributions per Principal Component",
            ha="center", va="top", fontsize=13, weight="bold")

    # Layout: split into two columns
    num_cols = 2
    pcs_per_col = int(np.ceil(n_pcs / num_cols))
    x_positions = [0.05, 0.55]
    y_start = 0.95
    line_height = 0.03

    pc_counter = 0

    for col in range(num_cols):
        y = y_start
        for i in range(pcs_per_col):
            pc_num = pc_counter + 1
            if pc_num > n_pcs:
                break

            group = grouped.get_group(pc_num)
            ax.text(x_positions[col], y, f"PC{pc_num}", fontsize=12, weight="bold", color="#1f77b4")
            y -= line_height

            for _, row in group.head(top_n).iterrows():
                feature = textwrap.shorten(row["Feature"], width=30, placeholder="...")
                contrib = f"{row['Contribution_(%)']:.1f}%"
                ax.text(x_positions[col] + 0.03, y, feature, fontsize=9, color="black", ha="left")
                ax.text(x_positions[col] + 0.43, y, contrib, fontsize=9, color="black", ha="right")
                y -= line_height

            y -= 0.02  # space between PCs
            pc_counter += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved PCA headline-style figure to: {output_path}")

save_pca_contributions_headline_style(
    csv_path=r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\top_features_feature_importance_test.csv",
    output_path=r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs\PCA_table",
    n_pcs = 6
    
)

