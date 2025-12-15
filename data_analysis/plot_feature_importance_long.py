import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
# import tikzplotlib
from matplotlib.gridspec import GridSpec

# File from Roya, slightly altered for this project!


def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)

def categorize_features(feature_labels):
        gyro_features = []
        accel_features = []
        other_features = []

        for feature in feature_labels:
            if 'gyro' in feature:
                gyro_features.append(feature)
            elif 'accel' in feature:
                accel_features.append(feature)
            elif 'imu' in feature:
                accel_features.append(feature)
            else:
                other_features.append(feature)
        
        return gyro_features, accel_features, other_features

def shorten_feature_names(features):
    # Example: shorten names by removing 'gyro', 'accel', etc.
    return [f.replace('_gyro', '').replace('_norm_accel', '').replace('_imu','').replace('(hz)_','').strip() for f in features]

def save_top_contributing_features_to_csv(percentage, features, category_name, n_pcs=10, top_n=10, filename="top_features.csv"):
    """
    Saves the top `top_n` contributing features for each PC in a given category to a CSV file.
    """
    results = []

    for i in range(n_pcs):
        # Sort features by their percentage contribution in descending order
        sorted_indices = np.argsort(percentage[:, i])[::-1]
        top_indices = sorted_indices[:top_n]
        
        # Collect the top contributing features for this PC
        for idx in top_indices:
            results.append({
                # "Category": category_name,
                "Principal Component": f"PC{i+1}",
                "Feature": features[idx],
                "Contribution (%)": round(percentage[idx, i], 1)
            })
    
    # Convert the results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nTop features saved to {filename}")

def print_top_contributing_features(percentage, features, category_name, n_pcs=10, top_n=10):
    """
    Prints the top `top_n` contributing features in each PC for a given category of features.
    """
    for i in range(n_pcs):
        # Sort features by their percentage contribution in descending order
        sorted_indices = np.argsort(percentage[:, i])[::-1]
        top_indices = sorted_indices[:top_n]
        
        # Print the top contributing features for this PC
        print(f"\nTop {top_n} contributing {category_name} features for PC{i+1}:")
        for idx in top_indices:
            print(f"Feature: {features[idx]}, Contribution: {percentage[idx, i]:.1f}%")


def plot_feature_importance_long(pca, feature_labels, save_path, save_suffix):
    """ https://stackoverflow.com/questions/67199869/measure-of-feature-importance-in-pca?rq=1
    """
    # cm = 1/2.54
    # r = np.abs(pca.components_.T)
    # percentage = r/r.sum(axis=0)
    # percentage = np.array(percentage)
    # percentage = percentage * 100
    # percentage = trunc(percentage, decs = 2)
   
    # # print(percentage[:, 13].sum()) # to check if the features explaining the feature is close to 1

    # fig = plt.figure(figsize=(30*cm, 35*cm)) #TODO feature importance | find the best figsize

    # # feature_labels
    # # idxes = [x_idx, y_idx, z_idx, remainder_idx]
    # # idxes = [i for i in idxes if len(i) != 0]

    # print(f"Number of features in feature vector: {len(feature_labels)}")
    # explained_var = trunc((pca.explained_variance_ratio_ * 100), decs = 0)
    # explained_var = explained_var.astype(int)

    # sub = fig.add_subplot()
    # im = sub.imshow(percentage, cmap='Blues', 
    #                             origin='upper',
    #                             aspect='auto',
    #                             )
    # temp_var = percentage.shape[0]

    # for i in range(pca.n_components_):
    #     for j in range(temp_var):
    #         text = sub.text(i, j, percentage[j, i],
    #                         ha="center", va="center", color="k")

    # sub.set_yticks(np.arange(len(feature_labels)), labels=feature_labels)
    # sub.set_xticks(np.arange(len(range(pca.n_components_))), labels = [f"PC{i+1}({explained_var[i]}%)" for i in range(pca.n_components_)])
    
    # sub.xaxis.tick_top()

    # sub.tick_params(axis='x', labelrotation = 45, )
    # sub.tick_params(axis='y', labelrotation = 30)

    # cbar = fig.figure.colorbar(im)
    # cbar.ax.set_ylabel("Percentage contribution to PCs", rotation=-90, va="bottom")

    
    # # fig.suptitle(f'Features contribution to PCs')
    # # fig.text(x = 0.08, y = 0.08, s = r"Cell numbers are the percentage of contrubition of explained variance relative in the realationship between the individual PC and Feature", color="k")
    
    # # if cp._first_loop == True and cp.save_plots == True:
    # # plt.savefig(save_path + f"\plot_feature_importance_long_{save_suffix}.png")
    # plt.tight_layout()
    # # if cp.show_explained_variance == True:
    # plt.show()
    # plt.close('all')

    ################################################################################################### Extra Plots for the paper - sorting
    print(feature_labels)
    print(f"All Features: {len(feature_labels)}")

    gyro_features, accel_features, other_features = categorize_features(feature_labels)

    print("Gyroscope Features:", gyro_features)
    print("Accelerometer Features:", accel_features)
    print("Other Features:", other_features)
    # Print counts
    print(f"Gyroscope Features: {len(gyro_features)}")
    print(f"Accelerometer Features: {len(accel_features)}")
    print(f"Other Features: {len(other_features)}")


    # Combine features in the desired order: gyro, accel, other
    sorted_features = gyro_features + accel_features + other_features

    # Get indices to reorder components
    sorted_indices = [feature_labels.index(f) for f in sorted_features]
    n_pcs = 10

    # Reorder PCA components and percentage based on the new sorted indices
    r = np.abs(pca.components_.T)
    r = r[sorted_indices, :n_pcs]
    percentage = r / r.sum(axis=0)
    percentage = np.array(percentage) * 100
    percentage = trunc(percentage, decs=1)

    # Get indices to reorder components for each category
    gyro_indices = [feature_labels.index(f) for f in gyro_features]
    accel_indices = [feature_labels.index(f) for f in accel_features]
    other_indices = [feature_labels.index(f) for f in other_features]

    # Shorten the feature names for Gyroscope, Accelerometer, and Other features
    shortened_gyro_features = shorten_feature_names(gyro_features)
    shortened_accel_features = shorten_feature_names(accel_features)
    shortened_other_features = shorten_feature_names(other_features)
   
    # Gyroscope Features
    percentage_gyro = percentage[:len(gyro_features), :n_pcs]
    # Accelerometer Features
    percentage_accel = percentage[len(gyro_features):len(gyro_features)+len(accel_features), :n_pcs]
    # Other Features
    percentage_other = percentage[len(gyro_features)+len(accel_features):, :n_pcs]


    print_top_contributing_features(percentage, sorted_features, "All", n_pcs=10, top_n=10)
    save_top_contributing_features_to_csv(percentage, sorted_features, "All",n_pcs=10, top_n=10, filename=f"top_features_{save_suffix}.csv")

    # # Print the top 5 contributing features for each category and each PC
    # print_top_contributing_features(percentage_gyro, shortened_gyro_features, "gyroscope")
    # print_top_contributing_features(percentage_accel, shortened_accel_features, "accelerometer")
    # print_top_contributing_features(percentage_other, shortened_other_features, "other")

    ################################################################################################## sorted 5 PCs

    # Explained variance
    explained_var = trunc((pca.explained_variance_ratio_ * 100), decs=0).astype(int)

    # Create figure
    cm = 1 / 2.54
    fig = plt.figure(figsize=(40 * cm, 50 * cm))
    sub = fig.add_subplot()    
    # Plot heatmap
    im = sub.imshow(percentage, cmap='Blues', origin='upper', aspect='auto', vmin=0, vmax=10)

    # Add text annotations
    for i in range(percentage.shape[1]):  # Number of PCs
        for j in range(percentage.shape[0]):  # Number of features
            sub.text(i, j, percentage[j, i], ha="center", va="center", color="k")

    # Set tick labels
    sub.set_yticks(np.arange(len(sorted_features)), labels=sorted_features)
    sub.set_xticks(np.arange(n_pcs), labels=[f"PC{i+1} ({explained_var[i]}%)" for i in range(n_pcs)])

    sub.xaxis.tick_top()
    sub.tick_params(axis='x', labelrotation=45)
    sub.tick_params(axis='y', labelrotation=30)

    # Add colorbar
    cbar = fig.figure.colorbar(im)
    cbar.ax.set_ylabel("Percentage contribution to PCs", rotation=-90, va="bottom")

    # Save the plot
    # plt.savefig(save_path + f"/sorted_plot_feature_importance_{save_suffix}.pdf", format='pdf', dpi=1000)
    plt.savefig(save_path + f"/sorted_plot_feature_importance_{save_suffix}.png")
    plt.tight_layout()
    plt.show()
    # plt.close('all')


    #################################################################################################### V1
#     # Number of principal components (PCs) to plot
#     n_pcs = 5
#     # Set the font size for the annotations
#     font_size = 8  # Adjust this value to make the text smaller or larger

#     # Calculate the combined min and max values across all percentage matrices for consistent color coding
#     min_percentage = min(percentage.min() for percentage in [percentage_gyro, percentage_accel, percentage_other])
#     max_percentage = max(percentage.max() for percentage in [percentage_gyro, percentage_accel, percentage_other])

#    # Create a GridSpec with custom width ratios to give more space to the Gyroscope column
#     gs = GridSpec(n_pcs, 3, width_ratios=[2, 1, 1])  # Gyroscope column gets twice as much space as the others

#     # Create the figure
#     fig = plt.figure(figsize=(10, 5))  # Adjust the figure size as necessary# Set shared color scale for all plots using vmin and vmax

#     vmin = min_percentage
#     vmax = max_percentage

#     # Iterate over PCs and feature categories to plot each separately
#     for i in range(n_pcs):
#         # Gyroscope Features (column 0)
#         axs_gyro = fig.add_subplot(gs[i, 0])
#         im_gyro = axs_gyro.imshow(percentage_gyro[:, i].reshape(-1, 1).T, cmap='Blues', origin='upper', aspect='auto', vmin=vmin, vmax=vmax)
        
#         if i == 0:  # Show xticks for PC1 only
#             axs_gyro.set_xticks(np.arange(len(gyro_features)), labels=shortened_gyro_features)
#             axs_gyro.xaxis.tick_top()
#             axs_gyro.tick_params(axis='x', labelrotation=90)
#         else:
#             axs_gyro.set_xticks([])

#         axs_gyro.set_yticks([0], labels=[f"PC{i+1}"])  # y-ticks only on the left for Gyroscope column
#         axs_gyro.tick_params(axis='y', labelrotation=0)

#         # Add text annotations for each PC and gyroscope feature
#         for j in range(len(gyro_features)):
#             axs_gyro.text(j, 0, percentage_gyro[j, i], ha="center", va="center", color="k", fontsize=6)

#         # Accelerometer Features (column 1)
#         axs_accel = fig.add_subplot(gs[i, 1])
#         im_accel = axs_accel.imshow(percentage_accel[:, i].reshape(-1, 1).T, cmap='Blues', origin='upper', aspect='auto', vmin=vmin, vmax=vmax)
        
#         if i == 0:  # Show xticks for PC1 only
#             axs_accel.set_xticks(np.arange(len(accel_features)), labels=shortened_accel_features)
#             axs_accel.xaxis.tick_top()
#             axs_accel.tick_params(axis='x', labelrotation=90)
#         else:
#             axs_accel.set_xticks([])

#         axs_accel.set_yticks([])  # No y-ticks for Accelerometer column

#         # Add text annotations for each PC and accelerometer feature
#         for j in range(len(accel_features)):
#             axs_accel.text(j, 0, percentage_accel[j, i], ha="center", va="center", color="k", fontsize=font_size)

#         # Other Features (column 2)
#         axs_other = fig.add_subplot(gs[i, 2])
#         im_other = axs_other.imshow(percentage_other[:, i].reshape(-1, 1).T, cmap='Blues', origin='upper', aspect='auto', vmin=vmin, vmax=vmax)
        
#         if i == 0:  # Show xticks for PC1 only
#             axs_other.set_xticks(np.arange(len(other_features)), labels=shortened_other_features)
#             axs_other.xaxis.tick_top()
#             axs_other.tick_params(axis='x', labelrotation=90)
#         else:
#             axs_other.set_xticks([])

#         axs_other.set_yticks([])  # No y-ticks for Other column

#         # Add text annotations for each PC and other feature
#         for j in range(len(other_features)):
#             axs_other.text(j, 0, percentage_other[j, i], ha="center", va="center", color="k", fontsize=font_size)

#     # Add a shared colorbar for all subplots
#     # fig.colorbar(im_gyro, ax=fig.get_axes(), orientation='horizontal', fraction=0.02, pad=0.04).set_label("Percentage contribution to PCs", rotation=0, labelpad=20)

#     # Adjust layout and show the plot
#     plt.tight_layout()
#     # plt.show()
#     plt.close('all')

    # #################################################################################################### V2
    # # Number of principal components (PCs) to plot
    # n_pcs = 5

    # # Get the explained variance ratios for each PC and format them as percentages
    # explained_var = trunc((pca.explained_variance_ratio_[:n_pcs] * 100), decs=1)

    # # Set the font size for the annotations
    # font_size = 8  # Adjust this value to make the text smaller or larger

    # # Create a GridSpec with custom width ratios to give more space to the Gyroscope column
    # gs = GridSpec(n_pcs, 3, width_ratios=[2, 1, 1])  # Gyroscope column gets twice as much space as the others

    # # Create the figure
    # fig = plt.figure(figsize=(7, 3))  # Adjust the figure size as necessary



    # # # Get the gyro feature indices and percentage for PC1
    # # percentage_gyro_pc1 = percentage[:len(gyro_features), 0]  # Only for PC1
    # # # Sort the gyroscope features based on their contribution to PC1
    # # top_gyro_indices_pc1 = np.argsort(percentage_gyro_pc1)[-17:][::-1]  # Top 17 features, sorted in descending order
    # # # Filter the percentage and names to include only the top 17 gyroscope features
    # # top_gyro_features = [gyro_features[i] for i in top_gyro_indices_pc1]
    # # shortened_top_gyro_features = shorten_feature_names(top_gyro_features)
    # # # Update the percentage matrix for PC1-PC5 for the top 17 gyroscope features
    # # percentage_gyro = percentage[top_gyro_indices_pc1, :n_pcs]


    # # Iterate over PCs and feature categories to plot each separately with individual color bars
    # for i in range(n_pcs):
    #     # Calculate vmin and vmax separately for each PC
    #     vmin = min(percentage.min() for percentage in [percentage_gyro[:, i], percentage_accel[:, i], percentage_other[:, i]])
    #     vmax = max(percentage.max() for percentage in [percentage_gyro[:, i], percentage_accel[:, i], percentage_other[:, i]])

    #     # Gyroscope Features (column 0)
    #     axs_gyro = fig.add_subplot(gs[i, 0])
    #     im_gyro = axs_gyro.imshow(percentage_gyro[:, i].reshape(-1, 1).T, cmap='Blues', origin='upper', aspect='auto', vmin=vmin, vmax=vmax)

    #     if i == 0:  # Show xticks for PC1 only
    #         axs_gyro.set_xticks(np.arange(len(gyro_features)), labels=shortened_gyro_features)
    #         axs_gyro.xaxis.tick_top()
    #         axs_gyro.tick_params(axis='x', labelrotation=90)
    #     else:
    #         axs_gyro.set_xticks([])

    #     axs_gyro.set_yticks([0], labels=[f"PC{i+1} ({explained_var[i]}%)"])  # y-ticks only on the left for Gyroscope column
    #     axs_gyro.tick_params(axis='y', labelrotation=90)

    #     # # Add text annotations for each PC and gyroscope feature
    #     # for j in range(len(gyro_features)):
    #     #     axs_gyro.text(j, 0, f"{percentage_gyro[j, i]:.1f}", ha="center", va="center", color="k", fontsize=6)

    #     # Accelerometer Features (column 1)
    #     axs_accel = fig.add_subplot(gs[i, 1])
    #     im_accel = axs_accel.imshow(percentage_accel[:, i].reshape(-1, 1).T, cmap='Blues', origin='upper', aspect='auto', vmin=vmin, vmax=vmax)

    #     if i == 0:  # Show xticks for PC1 only
    #         axs_accel.set_xticks(np.arange(len(accel_features)), labels=shortened_accel_features)
    #         axs_accel.xaxis.tick_top()
    #         axs_accel.tick_params(axis='x', labelrotation=90)
    #     else:
    #         axs_accel.set_xticks([])

    #     axs_accel.set_yticks([])  # No y-ticks for Accelerometer column

    #     # # Add text annotations for each PC and accelerometer feature
    #     # for j in range(len(accel_features)):
    #     #     axs_accel.text(j, 0, f"{percentage_accel[j, i]:.1f}", ha="center", va="center", color="k", fontsize=font_size)

    #     # Other Features (column 2)
    #     axs_other = fig.add_subplot(gs[i, 2])
    #     im_other = axs_other.imshow(percentage_other[:, i].reshape(-1, 1).T, cmap='Blues', origin='upper', aspect='auto', vmin=vmin, vmax=vmax)

    #     if i == 0:  # Show xticks for PC1 only
    #         axs_other.set_xticks(np.arange(len(other_features)), labels=shortened_other_features)
    #         axs_other.xaxis.tick_top()
    #         axs_other.tick_params(axis='x', labelrotation=90)
    #     else:
    #         axs_other.set_xticks([])

    #     axs_other.set_yticks([])  # No y-ticks for Other column

    #     # # Add text annotations for each PC and other feature
    #     # for j in range(len(other_features)):
    #     #     axs_other.text(j, 0, f"{percentage_other[j, i]:.1f}", ha="center", va="center", color="k", fontsize=font_size)

    #     # Add a color bar for each PC below the heatmaps
    #     # cbar = fig.colorbar(im_gyro, ax=[axs_gyro, axs_accel, axs_other], orientation='horizontal', fraction=0.02, pad=0.04)
    #     # cbar.set_label(f"PC{i+1} Contribution (%)", rotation=0, labelpad=10)

    #     # Calculate the sum of percentages for this PC across all features
    #     total_percentage = percentage_gyro[:, i].sum() + percentage_accel[:, i].sum() + percentage_other[:, i].sum()
    #     print(f"Total percentage for PC{i+1}: {total_percentage:.1f}%")  # Check if the sum is 100%


    # print("save_suffix", save_suffix)   
    # plt.savefig(save_path + f"/sorted_plot_feature_importance_{save_suffix}.png")
    # filename = rf"\{save_suffix}.tex"

    # # tikzplotlib.save(f"C:\Roya\TNSRE Paper DigiW-June-2024\RevisedCodes_Aug2024\Table_1_2_revised\Regression_Results"+filename)
    # plt.tight_layout()
    # plt.show()
    # # plt.close('all')

    return
