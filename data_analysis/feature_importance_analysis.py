"""
Description of analysis 

Based on LOOCV and average tests scores, the best classifier type is chosen. 
From the generated 'all_hyperparameteres_{clf_name}.json' and the 'find_hyperparameter_consensus.py'-file, 
the optimal feature space (as well as the other hyperparameters, but these are not considered here) is found.

In this analysis, we analyze this feature space in the following manner:

1. for each individual test's feature space, find the correct file and concatenate all tests into one large space.
2. fit a scaler on the space and transform
3. perform PCA with the same explained variance threshold as in the LOOCV-analysis
4. Examine and interpret the principal components and generate plots.  


----- 
The optimal classifier was found to be SVC, with the feature space 'expanded_only', meaning all IMU features 
and only the spatial FSR-features. We therefore examine this feature space.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_feature_importance_long import plot_feature_importance_long
from get_paths import get_feature_paths_for_multiple_spaces
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_correlation_circle(pca, feature_names, pcx=0, pcy=1, top_k=30, figsize=(7,7)):
    """
    Correlation loading plot (a.k.a. correlation circle) for PCx vs PCy.

    Parameters
    ----------
    pca : fitted sklearn.decomposition.PCA (fitted on standardized data)
    feature_names : list[str]  # order must match the columns used to fit PCA
    pcx, pcy : int             # component indices (0-based), e.g. 0,1 for PC1 vs PC2
    top_k : int                # plot only the top-k features by radius to reduce clutter
    figsize : tuple            # figure size
    """
    # correlation loadings: V * sqrt(lambda)
    load = pca.components_.T * np.sqrt(pca.explained_variance_)   # shape (n_features, n_components)

    # take PCx vs PCy coordinates
    xy = load[:, [pcx, pcy]]
    radii = np.linalg.norm(xy, axis=1)  # overall contribution in the 2D plane

    # keep top-k by radius
    if top_k is not None and top_k < len(feature_names):
        idx = np.argsort(radii)[-top_k:]
        xy = xy[idx]
        labels = [feature_names[i] for i in idx]
        radii = radii[idx]
    else:
        labels = feature_names

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    # unit circle
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color='lightgray', lw=1.5)
    ax.axhline(0, color='lightgray', lw=1)
    ax.axvline(0, color='lightgray', lw=1)

    # arrows from origin
    for (x, y), lab in zip(xy, labels):
        ax.arrow(0, 0, x, y, head_width=0.02, head_length=0.03, fc='tab:red', ec='tab:red', alpha=0.85, length_includes_head=True)
        ax.text(x*1.07, y*1.07, lab, fontsize=8, ha='center', va='center')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', 'box')

    ax.set_xlabel(f"PC{pcx+1} corr loadings ({pca.explained_variance_ratio_[pcx]*100:.1f}% var)")
    ax.set_ylabel(f"PC{pcy+1} corr loadings ({pca.explained_variance_ratio_[pcy]*100:.1f}% var)")
    ax.set_title("Correlation Loading Plot (PC1 vs PC2)")
    plt.tight_layout()
    plt.show()
    
    
def plot_grouped_pca_contributions(
    pca,
    feature_labels,
    save_path,
    save_suffix="grouped",
    n_pcs=10,
    use_squared=False,
    group_fn=None,
    annotate_threshold=1.0,
    dpi=300,
    show=True,
):
    """
    Aggregate PCA loadings by (modality × location) groups and plot a heatmap of
    group contributions (% of total loading magnitude per PC).

    Parameters
    ----------
    pca : fitted sklearn.decomposition.PCA
        PCA object already fitted on standardized features.
    feature_labels : list[str]
        Feature names in the same order used to fit the PCA.
    save_path : str
        Directory path where the plot image will be saved.
    save_suffix : str, default "grouped"
        Suffix appended to the saved filename.
    n_pcs : int, default 10
        Number of leading PCs to display.
    use_squared : bool, default False
        If True, aggregate squared loadings (common in PCA contribution plots).
        If False, aggregate absolute loadings.
    group_fn : callable or None
        Optional custom grouping function: group_fn(feature_name) -> "Group Label".
        If None, a default parser will infer modality/location from the name.
    annotate_threshold : float, default 1.0
        Only annotate cells whose percentage >= this threshold.
    dpi : int, default 300
        Save figure DPI.
    show : bool, default True
        If True, display the figure with plt.show().

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns: ['group', 'pc', 'pct'] where pct is the
        percentage contribution of the group to that PC (columns sum to ~100).
    """

    def _trunc(a, decs=1):
        return np.round(a.astype(float), decs)

    # Default grouping: infer modality + location from feature name
    def _default_group(name: str):
        n = name.lower()

        # modality
        if 'gyro' in n:
            modality = 'Gyro'
        elif 'accel' in n:
            modality = 'Accel'
        elif 'mag' in n:
            modality = 'Mag'
        elif 'cop' in n:
            modality = 'CoP'
        elif 'foreheel' in n:
            modality = 'Fore-heel ratio'
        elif 'medlat' in n:
            modality = 'Med-lat ratio'
        elif 'aver' in n:
            modality = 'Avg. insole'
        else:
            modality = 'Other spatial'
        # elif 'activation_pct' in n:
        #     modality = 'Activation percentage'
        # elif 'avg' in n:
        #     modality = 'Average statistical features'
        # elif 'spatial' in n:
        #     modality = 'Spatial statistical features'
        # else:
        #     print("OTHER : ", n)
        #     modality = 'Other'

        # location / side
        if '_arm' in n:
            loc = 'Arm'
        elif '_back' in n:
            loc = 'Back'
        elif '_left' in n:
            loc = 'Left'
        elif '_right' in n:
            loc = 'Right'
        else:
            loc = 'Both/NA'


        return f"{modality} ({loc})"

    if group_fn is None:
        group_fn = _default_group

    # ----- Prepare inputs -----
    n_pcs = min(n_pcs, pca.components_.shape[0])
    loadings = pca.components_.T[:, :n_pcs]  # (n_features, n_pcs)
    if use_squared:
        weight = loadings ** 2
    else:
        weight = np.abs(loadings)

    # Build groups
    groups = [group_fn(f) for f in feature_labels]

    # Desired modality order
    modality_order = [
        "Accel",
        "Gyro",
        "Mag",
        "Avg. insole",
        "CoP",
        "Med-lat ratio",
        "Fore-heel ratio",
    ]

    def modality_key(gname):
        """Extract modality and map to priority."""
        modality = gname.split(" (")[0]        # part before ' (Left)'
        try:
            return modality_order.index(modality)
        except ValueError:
            return len(modality_order)  # unknown categories go last
    
   # Sort unique groups by modality priority
    unique_groups = sorted(set(groups), key=modality_key)

    # Mapping to indices
    g2idx = {g: i for i, g in enumerate(unique_groups)}
    
    modality_colors = {
    "Accel": "#003366",          # deep navy
    "Gyro": "#004C99",           # strong dark blue
    "Mag": "#0066CC",            # vivid medium blue
    "Avg. insole": "#2A6FA8",    # darker mid-blue
    "CoP": "#4F80A8",            # darker blue-grey
    "Med-lat ratio": "#6C8494",  # darker grey-blue
    "Fore-heel ratio": "#8A8A8A",# darker grey
    "Other spatial": "#1A1A1A"   # near-black
}
    
    # Aggregate feature weights into groups
    group_load = np.zeros((len(unique_groups), n_pcs), dtype=float)
    for feat_idx, g in enumerate(groups):
        gi = g2idx[g]
        group_load[gi, :] += weight[feat_idx, :]

    # Normalize per PC to percentage
    col_sums = group_load.sum(axis=0, keepdims=True)
    # Avoid division by zero if a PC has zero total weight (unlikely but safe)
    col_sums[col_sums == 0] = 1.0
    group_pct = (group_load / col_sums) * 100.0
    group_pct_disp = _trunc(group_pct, decs=1)

    # PC labels with explained variance
    evr = pca.explained_variance_ratio_[:n_pcs] * 100.0
    pc_labels = [f"PC{i+1} ({int(round(e))}%)" for i, e in enumerate(evr)]

    # ----- Build tidy dataframe to return -----
    df = pd.DataFrame(group_pct, index=unique_groups, columns=[f"PC{i+1}" for i in range(n_pcs)])
    df_tidy = (
        df.reset_index()
          .melt(id_vars='index', var_name='pc', value_name='pct')
          .rename(columns={'index': 'group'})
          .sort_values(['pc', 'pct'], ascending=[True, False])
          .reset_index(drop=True)
    )

    # ----- Plot -----
    cm = 1 / 2.54
    fig_h = max(8, 0.75 * len(unique_groups)) * cm  # dynamic height
    fig = plt.figure(figsize=(20 * cm, fig_h))
    ax = fig.add_subplot()

    vmax = max(10, np.ceil(group_pct_disp.max()))
    im = ax.imshow(group_pct_disp, cmap='Blues', origin='upper', aspect='auto', vmin=0, vmax=vmax)

    # Annotate cells above threshold
    for i in range(group_pct_disp.shape[0]):
        for j in range(group_pct_disp.shape[1]):
            val = group_pct_disp[i, j]
            if val >= annotate_threshold:
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', color='k', fontsize=8)

    #ax.set_yticks(np.arange(len(unique_groups)), labels=unique_groups)
    #ax.set_yticks(np.arange(len(unique_groups)))
    #ax.set_yticklabels(unique_groups, rotation=0, ha='right', va='bottom')
    ax.set_xticks(np.arange(n_pcs), labels=pc_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', labelrotation=-0)
    
    ax.set_yticks(np.arange(len(unique_groups)))
    yticks = ax.set_yticklabels(unique_groups, rotation=0, ha='right', va='bottom')

    # Color-code labels
    for text, gname in zip(yticks, unique_groups):
        modality = gname.split(" (")[0]  # extract part before "(Left)"
        text.set_color(modality_colors.get(modality, "black"))

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Group contribution to PC (%)", rotation=-90, va="bottom")

    plt.tight_layout()

    out_path = f"{save_path}/grouped_pca_contributions_{save_suffix}.pdf"
    plt.savefig(out_path, format='pdf', dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_tidy

# define best space here
BEST_FEATURE_SPACE = 'expanded+baseline'
# this is very shaky, but just to get some results. weassume we know the order of the feature spaces in the lists
# at each test_id. for the best one here it is 2.

BEST_IDX = 2

FEATURE_PATHS = get_feature_paths_for_multiple_spaces(window_length_sec=8,mean_fsr=True, norm_IMU=False)
#print(FEATURE_PATHS)

feature_dfs = []
for test_id in FEATURE_PATHS.keys():
    for element in FEATURE_PATHS[test_id]:
        if BEST_FEATURE_SPACE in element:
            feature_df = pd.read_csv(element)
            feature_df = feature_df.drop(columns=['label'])
            feature_dfs.append(feature_df)

ALL_TESTS_FEATURES = pd.concat(feature_dfs, axis=0, ignore_index=True)
feature_labels = list(ALL_TESTS_FEATURES.columns)
print(feature_labels)
print("number of features", len(feature_labels))

# Scale data (SD of 1 and mean of 0)
scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(ALL_TESTS_FEATURES)

SCALED_FEATURES = scaler.transform(ALL_TESTS_FEATURES)

# apply PCA
pca = PCA(n_components=0.95)
pca_fit = pca.fit(SCALED_FEATURES)

# Individual and cumulative explained variance
var_ratio = pca_fit.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)

plt.figure(figsize=(8, 5))
x = np.arange(1, len(var_ratio) + 1)

# Bar plot for individual explained variance
plt.bar(x, var_ratio * 100, alpha=0.6, align='center', label='Individual variance')

# Line plot for cumulative explained variance
plt.step(x, cum_var * 100, where='mid', color='red', label='Cumulative variance')

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot of PCA Components')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# plot_correlation_circle(pca=pca_fit, feature_names=feature_labels)

# FEATURES_PCA = pca_fit.transform(SCALED_FEATURES)

# plot_feature_importance_long(
#     pca=pca_fit,
#     feature_labels=feature_labels,
#     save_path=r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs",
#     save_suffix='feature_importance_test')

plot_grouped_pca_contributions(
    pca=pca_fit,
    feature_labels=feature_labels,
    save_path=r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\outputs",
    save_suffix='GROUPED_feature_importance_test',
    )