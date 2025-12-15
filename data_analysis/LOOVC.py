import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from matplotlib.patches import Rectangle
from collections import defaultdict

from get_paths import get_feture_paths, get_feature_paths_for_multiple_spaces
from run_SVC import run_SVC, run_SVC_with_feature_tuning
from run_RFC import run_RFC_with_feature_tuning
from run_NN import  run_NN_with_feature_tuning
import json
from cf_matrix import make_confusion_matrix

def run_loocv_with_pca(label_mapping=None, clf_name = "", window_size=8, norm_IMU=True, mean_fsr=False, hdr=False, class_version=1):
    feature_files = get_feture_paths(window_length_sec=window_size, norm_IMU=norm_IMU, mean_fsr=mean_fsr, hdr=hdr)
    test_ids = list(feature_files.keys())


    start = time.time()
    for leave_out in test_ids:
        print(f"\n Testing on {leave_out}...")

        # Split training and test sets
        train_dfs = [pd.read_csv(path) for test_id, path in feature_files.items() if test_id != leave_out]
        test_df = pd.read_csv(feature_files[leave_out])

        # Combine training sets
        train_df = pd.concat(train_dfs, ignore_index=True)

        # Separate features and labels
        X_train = train_df.drop(columns=["label"])
        Y_train = train_df["label"]

        X_test = test_df.drop(columns=["label"])
        Y_test = test_df["label"]

        if label_mapping is not None:
            Y_train = train_df["label"].map(label_mapping)
            Y_test = test_df["label"].map(label_mapping)
            if label_mapping == label_mapping_v2:
                labels = ["hands_up", "push_pull", "squatting", "lifting", "sit_stand", "walking"]
            elif label_mapping == label_mapping_v3:
                labels = ["hands_up", "push_pull_lift", "squat", "sit", "stand_walk"]
        else:    
            labels = ["hand_up_back", "hands_forward", "hands_up", "push", "pull", "squatting", "lifting", "sitting", "standing", "walking"]

        # Scale data (SD of 1 and mean of 0)
        scaler = StandardScaler().set_output(transform="pandas")
        # Fit scaler on training data
        scaler.fit(X_train)
        # Transfor both train and test set with the scaler
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply pca
        pca = PCA(n_components=0.95)
        pca_fit = pca.fit(X_train_scaled)
        pca_components = pca.n_components_
        print (f"pca components: {pca_components}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")

        X_train_pca = pca_fit.transform(X_train_scaled)
        X_test_pca = pca_fit.transform(X_test_scaled)

        if clf_name == "SVC":
            test_results, train_results = run_SVC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        elif clf_name == "RFC":
            test_results, train_results = run_RFC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        elif clf_name == "NN":
            test_results, train_results, best_params, Y_test, Y_test_fit = run_NN(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        else:
            print("Model name not possible, model set automatic to svc")
            test_results, train_results = run_SVC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        Y_test_fit = test_results[0]
        accuracy_test = test_results[1]
        f1_test = test_results[2]
        precision_test  = test_results[3]
        recall_test = test_results[4]

        accuracy_train = train_results[1]
        f1_test = test_results[2]
        precision_test  = test_results[3]
        recall_test = test_results[4]


        # Evaluate
        print(f"Accuracy for {leave_out}: {accuracy_test:.3f}")
        print(f"Precision for {leave_out}: {precision_test:.3f}")
        print(f"Recall for {leave_out}: {recall_test:.3f}")
        print(f"F1 for {leave_out}: {f1_test:.3f}")
        #print(f"Best hyperparameters {leave_out} : {best_params}")
        #print(f"Accuracy for train {leave_out}: {accuracy_train:.3f}")
        # all_accuracies.append(accuracy_test)
        # all_f1.append(f1_test)
        # all_precision.append(precision_test)
        # all_recall.append(recall_test)

        # all_Y_true.extend(Y_test)
        # all_Y_pred.extend(Y_test_fit)
        # all_hyperparameters.extend(best_params)
        
        cm = confusion_matrix(Y_test, Y_test_fit, labels=labels)

        print(f"\nConfusion matrix for {leave_out}:")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        # Optional: save or plot the heatmap
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='BuPu', xticks_rotation='vertical')
        plt.title(f"Confusion Matrix – {leave_out}")
        plt.tight_layout()
        plt.savefig(f"./plots_use/confmat_{leave_out}.png", dpi=300)
        plt.close()
            
    end = time.time()
    elapsed = end - start
    print(f"\n🕒 Done! Total time uesd: {elapsed:.2f} seconds")

    print(f"\n✅ Mean LOOCV accuracy: {np.mean(all_accuracies):.3f}")
    print(f"\n✅ Mean LOOCV f1: {np.mean(all_f1):.3f}")
    print(f"\n✅ Mean LOOCV precision: {np.mean(all_precision):.3f}")
    print(f"\n✅ Mean LOOCV recall: {np.mean(all_recall):.3f}")

    # Confusion matrix
    save_path="./plots_use"
    os.makedirs(save_path, exist_ok=True)
    
    cm = confusion_matrix(all_Y_true, all_Y_pred, labels=labels, normalize='true')
    print("\n🧮 Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical',cmap=plt.cm.BuPu, values_format=".2f")
    for text in disp.ax_.texts:
        text.set_fontsize(8)
    plt.tight_layout()
    
    save_file1 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM1.png")
    plt.savefig(save_file1, dpi=300)
    plt.close()
    
    cm2 = confusion_matrix(all_Y_true, all_Y_pred, labels=labels)
    df_cm2 = pd.DataFrame(cm2, index=labels, columns=labels)

    df_cm2['Total'] = df_cm2.sum(axis=1)
    totals_row = df_cm2.sum(axis=0)
    totals_row.name = 'Total'

    df_cm2 = pd.concat([df_cm2, totals_row.to_frame().T])
    
    n_rows, n_cols = df_cm2.shape

    mask = np.zeros_like(df_cm2, dtype=bool)
    mask[-1, :] = True   # last row (Total Pred)
    mask[:, -1] = True   # last column (Total True)

    if label_mapping is None:
        plt.figure(figsize=(7, 6))
    else:
        plt.figure(figsize=(5.5, 5)) 
    ax = sns.heatmap(df_cm2, annot=True, fmt='.0f', cmap='BuPu', mask=mask, cbar=True)

    total_bg_color = '#e9ecff'  # matching 'BuPu'
    for i in range(n_rows - 1):
        ax.add_patch(Rectangle((n_cols - 1, i), 1, 1, fill=True, color=total_bg_color, lw=0))
    for j in range(n_cols - 1):
        ax.add_patch(Rectangle((j, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))
    ax.add_patch(Rectangle((n_cols - 1, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))

    # Manually annotate the total row and column
    for i in range(n_rows - 1):  # all rows except last
        val = df_cm2.iat[i, -1]
        ax.text(n_cols - 0.5, i + 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

    for j in range(n_cols - 1):  # all columns except last
        val = df_cm2.iat[-1, j]
        ax.text(j + 0.5, n_rows - 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

    corner_val = df_cm2.iat[-1, -1]
    ax.text(n_cols - 0.5, n_rows - 0.5, f'{corner_val:.0f}', ha='center', va='center', color='black', fontsize=9)

    plt.title('Confusion Matrix with True and Predicted Totals')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    save_file2 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM2.png")
    plt.tight_layout()
    plt.savefig(save_file2, dpi=300)
    plt.close()
    
    return all_accuracies


def run_loocv_with_pca_and_feature_sensor_eval(
    label_mapping=None, 
    clf_name = "", 
    window_size=8, 
    norm_IMU=True, 
    mean_fsr=False, 
    hdr=False, 
    class_version=1,
    feature_spaces = [
    "baseline",
    "expanded+baseline",
    # "expanded_only",
    # "time_only",
    # "time_only+exp_FSR",
    # "freq_only",
    # "freq_only+exp_FSR",
    "FSR_only",
    "arm_only+FSR",
    "back_only+FSR",
    "IMU_only"
    ]):
    feature_files = get_feature_paths_for_multiple_spaces(window_length_sec=8, norm_IMU=False, mean_fsr=True, hdr=False, feature_spaces=feature_spaces)
    test_ids = list(feature_files.keys())

    # all_accuracies = []
    # all_f1 = []
    # all_precision = []
    # all_recall = []
    all_Y_true = []
    all_Y_pred = []
    # all_hyperparameters = {}
    # rows = [] # list of per test metric df rows to concatenate and write to a.csv later

    start = time.time()
    all_tests_metrics_per_space = {}
    for leave_out in test_ids:
        # debug, just test using the first two leave out subjects
        # if leave_out not in ['test_1', 'test_2']:
        #     continue
        print(f"\n Testing on {leave_out}...")
        # Split training and test sets
        train_dicts = []
        test_df_per_feat_space = {}
        for test_id, paths in feature_files.items():
            train_df_per_feat_space = {}
            if test_id == leave_out:
                for i in range(len(paths)): # assume same number of paths and same order as in feature_spaces
                    space = feature_spaces[i]
                    path = paths[i]
                    if space in path:
                        test_df = pd.read_csv(path)
                        test_df_per_feat_space[space] = test_df
                    else:
                        print("MISMATCH BETWEEN FEAT SPACE AND PATH TEST DF CREATION")
                        time.sleep(60)
            else:
                for i in range(len(paths)): # assume same number of paths and same order as in feature_spaces
                    space = feature_spaces[i]
                    path = paths[i]
                    if space in path:
                        print("space is", space)
                        train_df = pd.read_csv(path)
                        print("number of columns is", len(train_df.columns))
                        #train_df['test_id'] = test_id
                        train_df_per_feat_space[space] = train_df
                    else:
                        print("MISMATCH BETWEEN FEAT SPACE AND PATH TRAIN DF CREATION")
                        return None
            train_dicts.append(train_df_per_feat_space)
        
        # create lists of all training tests id dataframes for each given feature space.
        # let feature space be key, and list of dfs to late rbe concatenated value.
        df_list_per_feature_space = defaultdict(list)
        for d in train_dicts:
            for feature_space in d.keys():
                df_list_per_feature_space[feature_space].append(d[feature_space])

        # Combine training sets
        concat_train_df_per_feat_space = {}
        for feature_space in df_list_per_feature_space.keys():
            concat_train_df_per_feat_space[feature_space] = pd.concat(df_list_per_feature_space[feature_space], ignore_index=True)
            

        # # Separate features and labels
        # X_train = train_df.drop(columns=["label"])
        # Y_train = train_df["label"]

        # X_test = test_df.drop(columns=["label"])
        # Y_test = test_df["label"]

        if label_mapping is not None:
            Y_train = train_df["label"].map(label_mapping)
            Y_test = test_df["label"].map(label_mapping)
            labels = ["hands_up", "push_pull", "squat_lift", "stand_sit", "walking"]
        else:    
            labels = ["hand_up_back", "hands_forward", "hands_up", "push", "pull", "squatting", "lifting", "sitting", "standing", "walking"]

        # # Scale data (SD of 1 and mean of 0)
        # scaler = StandardScaler().set_output(transform="pandas")
        # # Fit scaler on training data
        # scaler.fit(X_train)
        # # Transfor both train and test set with the scaler
        # X_train_scaled = scaler.transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        
        # # Apply pca
        # pca = PCA(n_components=0.95)
        # pca_fit = pca.fit(X_train_scaled)
        # pca_components = pca.n_components_
        # print (f"pca components: {pca_components}")
        # print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")

        # X_train_pca = pca_fit.transform(X_train_scaled)
        # X_test_pca = pca_fit.transform(X_test_scaled)

        if clf_name == "SVC":
            space_and_test_metrics, Y_test = run_SVC_with_feature_tuning(concat_train_df_per_feat_space, test_df_per_feat_space, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec", label_mapping=label_mapping)
        elif clf_name == "RFC":
            space_and_test_metrics, Y_test= run_RFC_with_feature_tuning(concat_train_df_per_feat_space, test_df_per_feat_space, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec", label_mapping=label_mapping)
        elif clf_name == "NN":
            space_and_test_metrics, Y_test = run_NN_with_feature_tuning(concat_train_df_per_feat_space, test_df_per_feat_space, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec", label_mapping=label_mapping)
        else:
            print("Model name not possible, model set automatic to svc")
            test_results, train_results, summaries, best, Y_test, Y_test_fit = run_SVC_with_feature_tuning(concat_train_df_per_feat_space, test_df_per_feat_space, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        
        
        all_tests_metrics_per_space[leave_out] = space_and_test_metrics
        
        # Y_test_fit = test_results[0]
        # accuracy_test = test_results[1]
        # f1_test = test_results[2]
        # precision_test  = test_results[3]
        # recall_test = test_results[4]

        # accuracy_train = train_results[1]
        # f1_test = test_results[2]
        # precision_test  = test_results[3]
        # recall_test = test_results[4]

        # Evaluate
        # print(f"Accuracy for {leave_out}: {accuracy_test:.3f}")
        # print(f"Precision for {leave_out}: {precision_test:.3f}")
        # print(f"Recall for {leave_out}: {recall_test:.3f}")
        # print(f"F1 for {leave_out}: {f1_test:.3f}")
        #print(f"Best hyperparameters {leave_out} : {best_params}")
        #print(f"Accuracy for train {leave_out}: {accuracy_train:.3f}")
        # all_accuracies.append(accuracy_test)
        # all_f1.append(f1_test)
        # all_precision.append(precision_test)
        # all_recall.append(recall_test)
        
        # row = {
        #     "Leave_out": leave_out,
        #     "Accuracy": round(accuracy_test, 3),
        #     "Precision": round(precision_test, 3),
        #     "Recall": round(recall_test, 3),
        #     "F1": round(f1_test, 3)
        # }
        
        # row = pd.DataFrame([row])
        # rows.append(row)

        all_Y_true.extend(Y_test)
        
        # cm = confusion_matrix(Y_test, Y_test_fit, 
        #                       #labels=labels
        #                       )

        # print(f"\nConfusion matrix for {leave_out}:")
        # print(pd.DataFrame(cm, index=labels, columns=labels))

        # Optional: save or plot the heatmap
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        # disp.plot(cmap='BuPu', xticks_rotation='vertical')
        # plt.title(f"Confusion Matrix – {leave_out}")
        # plt.tight_layout()
        # plt.savefig(f"./plots_use/confmat_{leave_out}.png", dpi=300)
        # plt.close()
            
    end = time.time()
    elapsed = end - start
    print(f"\n🕒 Done! Total time uesd: {elapsed:.2f} seconds")
    
    label_mapping_suffix = ""
    if label_mapping  is not None:
        label_mapping_suffix = "_relabeled"
        
    
    space_and_test_metrics_df = pd.DataFrame.from_dict(all_tests_metrics_per_space)
    space_and_test_metrics_df.to_csv(f"./outputs/all_performance_indexes_testid_and_feat_space_{clf_name}{label_mapping_suffix}.csv")
    
    all_accuracies = {fs: [] for fs in feature_spaces}
    all_f1 = {fs: [] for fs in feature_spaces}
    all_precision = {fs: [] for fs in feature_spaces}
    all_recall = {fs: [] for fs in feature_spaces}
    
    print("Generating performance summary...")
    for test_id in all_tests_metrics_per_space.keys():
        for feature_space in feature_spaces:
            acc = all_tests_metrics_per_space[test_id][feature_space]['accuracy']
            f1 = all_tests_metrics_per_space[test_id][feature_space]['f1']
            prec = all_tests_metrics_per_space[test_id][feature_space]['precision']
            rec = all_tests_metrics_per_space[test_id][feature_space]['recall']
            
            all_accuracies[feature_space].append(acc)
            all_f1[feature_space].append(f1)
            all_precision[feature_space].append(prec)
            all_recall[feature_space].append(rec)
    
    mean_and_std_metrics_per_space = {}
    for feature_space in all_accuracies.keys():
        mean_acc = np.mean(all_accuracies[feature_space])
        std_acc = np.std(all_accuracies[feature_space], ddof=1)

        mean_f1 = np.mean(all_f1[feature_space])
        std_f1 = np.std(all_f1[feature_space], ddof=1)

        mean_prec = np.mean(all_precision[feature_space])
        std_prec = np.std(all_precision[feature_space], ddof=1)

        mean_rec = np.mean(all_recall[feature_space])
        std_rec = np.std(all_recall[feature_space], ddof=1)

        mean_and_std_metrics_per_space[feature_space] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_precision": mean_prec,
            "std_precision": std_prec,
            "mean_recall": mean_rec,
            "std_recall": std_rec,
        }
    
    mean_std_df = pd.DataFrame.from_dict(mean_and_std_metrics_per_space)
    mean_std_df.to_csv(f"mean_performance_per_space_{clf_name}{label_mapping_suffix}.csv")
    
    best_space = max(
    mean_and_std_metrics_per_space,
    key=lambda fs: mean_and_std_metrics_per_space[fs]["mean_f1"]
    )

    best_f1 = mean_and_std_metrics_per_space[best_space]["mean_f1"]
    
    if label_mapping is not None:
        best_space = 'expanded+baseline' # quickfix to get the correct cm

    print("Best feature space:", best_space)
    print("Mean F1 score:", best_f1)
    
    # create confusion matrix for the best performing space
    for test_id in all_tests_metrics_per_space.keys():
        print(test_id)
        y_pred = all_tests_metrics_per_space[test_id][best_space]['y_test_fit']
        all_Y_pred.extend(y_pred)

    # print("\n✅ LOOCV performance summary:")
    # print(f"Accuracy : {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies, ddof=1):.3f}")
    # print(f"F1-score : {np.mean(all_f1):.3f} ± {np.std(all_f1, ddof=1):.3f}")
    # print(f"Precision: {np.mean(all_precision):.3f} ± {np.std(all_precision, ddof=1):.3f}")
    # print(f"Recall   : {np.mean(all_recall):.3f} ± {np.std(all_recall, ddof=1):.3f}")
    
    
    # summary_rows = {
    #     "Accuracy":  f"{np.mean(all_accuracies):.3f} ± {np.std(all_accuracies, ddof=1):.3f}",
    #     "F1-score":  f"{np.mean(all_f1):.3f}       ± {np.std(all_f1, ddof=1):.3f}",
    #     "Precision": f"{np.mean(all_precision):.3f} ± {np.std(all_precision, ddof=1):.3f}",
    #     "Recall":    f"{np.mean(all_recall):.3f}    ± {np.std(all_recall, ddof=1):.3f}",
    # }

    # df_results = pd.DataFrame.from_dict(summary_rows, orient="index", columns=["value"])
    # df_results.to_csv(f"./outputs/results_summary_{clf_name}.csv")


    # Confusion matrix
    save_path="./plots_use"
    os.makedirs(save_path, exist_ok=True)
    
    # metric_summaries_df = pd.concat(rows)
    
    # metric_summaries_df.to_csv(f"./outputs/all_test_results_{clf_name}.csv")
    
    # summary_save_path = f"./outputs/all_hyperparameters_{clf_name}.json"
    # with open(summary_save_path, "w", encoding="utf-8") as f:
    #     json.dump(all_hyperparameters, f, indent=2, ensure_ascii=False)
    
    cm = confusion_matrix(all_Y_true, all_Y_pred, labels=labels, normalize='true')
    print("\n🧮 Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical',cmap=plt.cm.Blues, values_format=".2f")
    for text in disp.ax_.texts:
        text.set_fontsize(8)
    plt.tight_layout()
    
    save_file1 = os.path.join(save_path, f"{clf_name}_CM1.png")
    plt.savefig(save_file1, dpi=300)
    plt.close()
    
    cm2 = confusion_matrix(all_Y_true, all_Y_pred, labels=labels)
    df_cm2 = pd.DataFrame(cm2, index=labels, columns=labels)

    df_cm2['Total'] = df_cm2.sum(axis=1)
    totals_row = df_cm2.sum(axis=0)
    totals_row.name = 'Total'

    df_cm2 = pd.concat([df_cm2, totals_row.to_frame().T])
    
    n_rows, n_cols = df_cm2.shape

    mask = np.zeros_like(df_cm2, dtype=bool)
    mask[-1, :] = True   # last row (Total Pred)
    mask[:, -1] = True   # last column (Total True)

    if label_mapping is None:
        plt.figure(figsize=(7, 6))
    else:
        plt.figure(figsize=(5.5, 5)) 
    ax = sns.heatmap(df_cm2, annot=True, fmt='.0f', cmap='Blues', mask=mask, cbar=True)

    total_bg_color = "#e8f1ff" # matching 'blues'
    for i in range(n_rows - 1):
        ax.add_patch(Rectangle((n_cols - 1, i), 1, 1, fill=True, color=total_bg_color, lw=0))
    for j in range(n_cols - 1):
        ax.add_patch(Rectangle((j, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))
    ax.add_patch(Rectangle((n_cols - 1, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))

    # Manually annotate the total row and column
    for i in range(n_rows - 1):  # all rows except last
        val = df_cm2.iat[i, -1]
        ax.text(n_cols - 0.5, i + 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

    for j in range(n_cols - 1):  # all columns except last
        val = df_cm2.iat[-1, j]
        ax.text(j + 0.5, n_rows - 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

    corner_val = df_cm2.iat[-1, -1]
    ax.text(n_cols - 0.5, n_rows - 0.5, f'{corner_val:.0f}', ha='center', va='center', color='black', fontsize=9)

    plt.title('Confusion Matrix with True and Predicted Totals')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    save_file2 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM2.png")
    plt.tight_layout()
    plt.savefig(save_file2, dpi=300)
    plt.close()
    
    make_confusion_matrix(cm2, 
                          categories=labels, 
                          figsize=(10,8), 
                          #title=f'Confusion matrix LOOCV {clf_name}-model'
                          )
    
    return all_accuracies



label_mapping = {
    'hand_up_back' : 'hands_up',
    'hands_forward' : 'hands_up',
    'hands_up' : 'hands_up',
    'push' : 'push_pull',
    'pull' : 'push_pull',
    'squatting' : 'squat_lift',
    'lifting' : 'squat_lift',
    'sitting' : 'stand_sit',
    'standing' : 'stand_sit',
    'walking' : 'walking'
}


run_loocv_with_pca_and_feature_sensor_eval(clf_name="SVC", norm_IMU=False,mean_fsr=True, hdr=False, label_mapping=None)