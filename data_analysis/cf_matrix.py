import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def make_confusion_matrix(
        cf,
        categories,
        figsize=(35, 35),
        cmap='Blues',
        title=None,
        fontsize=14,
        savepath="confusion_matrix_colored.pdf"
):

    cf = np.asarray(cf)
    n = cf.shape[0]

    row_sums = cf.sum(axis=1)       # predicted totals (rows)
    col_sums = cf.sum(axis=0)       # true totals (columns)
    total = cf.sum()
    diag = np.diag(cf)

    # ---- METRICS (for your layout: rows=pred, cols=true) ----
    precision = np.divide(diag, row_sums, out=np.zeros_like(row_sums, float), where=row_sums != 0)
    recall    = np.divide(diag, col_sums, out=np.zeros_like(col_sums, float), where=col_sums != 0)
    accuracy  = np.trace(cf) / total if total > 0 else 0

    # ---- EXTENDED GRID ----
    ext = np.zeros((n + 1, n + 1), float)
    ext[:-1, :-1] = cf
    ext[:-1, -1]  = row_sums     # Precision column
    ext[-1, :-1]  = col_sums     # Recall row
    ext[-1, -1]   = total

    # ---- LABELS FOR HEATMAP (only main n×n block) ----
    labels = np.full_like(ext, "", dtype=object)  # start with empty strings

    # main block counts
    for i in range(n):
        for j in range(n):
            labels[i, j] = f"{cf[i, j]:.0f}"

    # we leave last row/column labels as "" so seaborn doesn't draw counts there

    # ---- PLOTTING ----
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        ext,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=False,
        xticklabels=list(categories) + ["Precision"],
        yticklabels=list(categories) + ["Recall"],
        annot_kws={"fontsize": fontsize, 'fontweight':'bold'}
    )


    # ---- ADD ROW-WISE PERCENTAGES (per predicted class / row) ----
    for i in range(n):
        for j in range(n):
            count = cf[i, j]
            row_total = row_sums[i]
            pct = (count / row_total) * 100 if row_total > 0 else 0

            ax.text(j + 0.5, i + 0.80,        # lower in cell so it does not overlap count
                    f"{pct:.2f}%",
                    color="black",
                    ha='center', va='center',
                    fontsize=fontsize-2,
                    #fontweight='bold'
                    )

    # ---- PRECISION COLUMN (row-based) ----
    # for i in range(n):
    #     p  = precision[i] * 100
    #     fp = (1 - precision[i]) * 100

    #     x = n + 0.5
    #     # three lines: total (black), correct (green), error (red)
    #     ax.text(x, i + 0.35,
    #             f"{row_sums[i]:.0f}",
    #             color="black", ha='center', va='center',
    #             fontsize=fontsize-1)
    #     ax.text(x, i + 0.50,
    #             f"{p:.2f}%",
    #             color="green", ha='center', va='center',
    #             fontsize=fontsize-1)
    #     ax.text(x, i + 0.65,
    #             f"{fp:.2f}%",
    #             color="red", ha='center', va='center',
    #             fontsize=fontsize-1)
    for i in range(n):
        p  = precision[i] * 100
        fp = (1 - precision[i]) * 100

        x = n + 0.5
        ax.text(x, i + 0.2, f"{row_sums[i]:.0f}", color="black",
                ha='center', va='center', fontsize=fontsize-4, fontweight='bold')    # total
        ax.text(x, i + 0.55, f"{p:.2f}%",     color="green",
                ha='center', va='center', fontsize=fontsize-4, fontweight='bold')    # correct%
        ax.text(x, i + 0.80, f"{fp:.2f}%",    color="red",
                ha='center', va='center', fontsize=fontsize-4, fontweight='bold')    # wrong%

    # ---- RECALL ROW (column-based) ----
    # for j in range(n):
    #     r  = recall[j] * 100
    #     fn = (1 - recall[j]) * 100

    #     y = n + 0.5
    #     ax.text(j + 0.35, y,
    #             f"{col_sums[j]:.0f}",
    #             color="black", ha='center', va='center',
    #             fontsize=fontsize-1)
    #     ax.text(j + 0.50, y,
    #             f"\n{r:.2f}%",
    #             color="green", ha='center', va='center',
    #             fontsize=fontsize-1)
    #     ax.text(j + 0.65, y,
    #             f"\n\n{fn:.2f}%",
    #             color="red", ha='center', va='center',
    #             fontsize=fontsize-1)
    for j in range(n):
        r  = recall[j] * 100
        fn = (1 - recall[j]) * 100

        x = j + 0.5
        ax.text(x, n +0.2,       f"{col_sums[j]:.0f}", color="black",
                ha='center', va='center', fontsize=fontsize-4, fontweight='bold')
        ax.text(x, n +0.3,      f"\n{r:.2f}%",        color="green",
                ha='center', va='center', fontsize=fontsize-4,fontweight='bold')
        ax.text(x, n +0.45,       f"\n\n{fn:.2f}%",     color="red",
                ha='center', va='center', fontsize=fontsize-4, fontweight='bold')


    # ---- ACCURACY CELL (bottom-right) ----
    # x = n + 0.5
    # y = n + 0.5
    # ax.text(x, y - 0.15,
    #         f"{total:.0f}",
    #         color="black", ha='center', va='center',
    #         fontsize=fontsize-1)
    # ax.text(x, y,
    #         f"{accuracy*100:.2f}%",
    #         color="green", ha='center', va='center',
    #         fontsize=fontsize-1)
    # ax.text(x, y + 0.15,
    #         f"{(1-accuracy)*100:.2f}%",
    #         color="red", ha='center', va='center',
    #         fontsize=fontsize-1)
    
    x = n + 0.5
    y = n + 0.5

    ax.text(x, y - 0.3, f"{total:.0f}",              color="white",
            ha='center', va='center', fontsize=fontsize-4, fontweight='bold')
    ax.text(x, y , f"{accuracy*100:.2f}%",      color="green",
            ha='center', va='center', fontsize=fontsize-4, fontweight='bold')
    ax.text(x, y + 0.3, f"{(1-accuracy)*100:.2f}%",  color="red",
            ha='center', va='center', fontsize=fontsize-4, fontweight='bold')

    # labels
    ax.set_xlabel("True", fontsize=fontsize)
    ax.set_ylabel("Predicted", fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=45, ha='right', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(savepath, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
