from  Maria_code.data_analysis_DL.create_images_from_timeseries_segments import create_image_dfs_from_timeseries_segments
from  Maria_code.data_analysis_DL.build_CNN_model import create_CNN_model, set_global_seed
from Maria_code.data_analysis.cf_matrix  import make_confusion_matrix

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
import random
import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD

from scikeras.wrappers import KerasClassifier


"""
Pipeline:
    1. Call create_image_dfs_from_timeseries_segments to get labeled segments for each window, sorted by test id
    2. Iterate over test_id's, i.e. keys in the returned dict.
    3. Leave 'current' test_id out, and let the rest be training data.
    4. Fit scaler on training data. And apply scaler to all data.
    5. Convert to grayscale
    6. Use 3-fold CV for hyperparameter, scenarios/sednsor types optimization. Scenarios: (all sensors, IMU only, FSR only, arm+FSR, back+FSR)
    7. Train with best hyoerparameters.
    8. Test -> repeat for all leave-out subjects
    
Note: use random state same as other models and then maybe try a couple other seeds.

Store:
    All validation F1-scores for all feature spaces, average over folds with std. + hyperparameters
    All test accuracies, F1-score, precision, recall
    Average accuracy, F1-score, precision, recall + std
    Confusion matrix over all tests.
    Plots of training accuracy/F1 over epochs
"""
# --- helper ---

def fit_scaler_for_scenario(train_windows, scenario: str):
    """
    train_windows: list of *scaled* windows (or raw windows) with 'label'
    scenario: one of SCENARIOS

    Returns:
        scaler: StandardScaler fitted on scenario-specific columns
        feature_cols: the scenario-specific feature columns (e.g. 50)
    """
    # Concatenate
    train_concat = pd.concat(train_windows, axis=0, ignore_index=True)

    # Select only the columns for this scenario
    feature_cols = select_columns_for_scenario(train_concat, scenario)

    X_train = train_concat[feature_cols].to_numpy()
    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler, feature_cols


def select_columns_for_scenario(df, scenario):
    all_cols = [c for c in df.columns if c != "label"]
    if scenario == "All":
        return all_cols
    elif scenario == "IMU_only":
        return [c for c in all_cols if any(k in c.lower() for k in ("axl", "gyr", "mag"))]
    elif scenario == "FSR_only":
        return [c for c in all_cols if "fsr" in c.lower()]
    elif scenario == "Arm_and_FSR":
        return [c for c in all_cols if "arm" in c.lower() or "fsr" in c.lower()]
    elif scenario == "Back_and_FSR":
        return [c for c in all_cols if "back" in c.lower() or "fsr" in c.lower()]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
def fit_scaler_on_train_windows(train_windows):
    """
    train_windows: list of DataFrames (each window, all sensors, with 'label' col)
    Returns:
        scaler: StandardScaler fitted on ALL feature columns across ALL train windows
        feature_cols: list of feature column names (no 'label')
    """
    # concat all train windows vertically
    train_concat = pd.concat(train_windows, axis=0, ignore_index=True)
    feature_cols = [c for c in train_concat.columns if c != "label"]

    scaler = StandardScaler()
    scaler.fit(train_concat[feature_cols].values)

    return scaler, feature_cols

def scale_window_df(df, scaler: StandardScaler, feature_cols):
    """
    Scale a single window for a given scenario.
    Only uses scenario-specific feature columns.
    """
    X = df[feature_cols].to_numpy()          # (T, n_features)
    X_scaled = scaler.transform(X)

    df_scaled = pd.DataFrame(
        X_scaled,
        columns=feature_cols,
        index=df.index
    )

    if "label" in df.columns:
        df_scaled["label"] = df["label"].values

    return df_scaled


def windows_to_tensor_and_labels(
        windows_scaled,
        scenario: str,
        grayscale_for_training=True,
        debug_plot=True
    ):
    """
    Converts scaled DataFrames into CNN-ready tensors.
    
    Parameters:
    - grayscale_for_training: 
        True  => return float32 in [0,1]
        False => return uint8 0–255 (NOT USED, AND NOT RECOMMENDED)
    - debug_plot:
        If True, plot each window as an image with its label in the title.
    """

    X_list = []
    y_list = []

    for idx, df in enumerate(windows_scaled):
        if "label" not in df.columns:
            raise ValueError("Expected 'label' column in window DataFrame.")
        label = df["label"].iloc[0]
        y_list.append(label)

        feat_cols = select_columns_for_scenario(df, scenario)

        arr = df[feat_cols].values.astype(float)

        #  transpose to channels x time
        arr = arr.T

        # per window grayscale normalization
        arr_min = arr.min(axis=1, keepdims=True)
        arr_max = arr.max(axis=1, keepdims=True)
        denom = arr_max - arr_min
        denom[denom == 0] = 1.0

        arr_norm = (arr - arr_min) / denom     # float32 in [0,1]
        # CNN input = float32
        if grayscale_for_training:
            arr_cnn = arr_norm[..., np.newaxis].astype("float32")

        # actual grayscale 0–255 images  (dont do this)
        else:
            arr_uint8 = (arr_norm * 255).astype("uint8")
            arr_cnn = arr_uint8[..., np.newaxis]

        # Create debug (or report) plots of activity window images 
        if debug_plot:
            plt.figure(figsize=(8, 5))
            plt.imshow(arr_norm, cmap="gray", aspect="auto")
            plt.colorbar()
            #plt.title(f"Window number {idx}, Label: {label}, Scenario: {scenario}")
            plt.title(f"Movement label: {label}")
            plt.xlabel("Time")
            plt.ylabel("Sensor channel")

            # Set y-ticks to channel names
            yticks = [
                0,3,6, # arm accel, gyro, mag 
                9, 12, 15, #back accel, gyro, mag
                18, # left insole
                34, # right insole
            ]
            

            ylabs = ["Accel Arm", 
                     "Gyro Arm",
                     "Magno Arm",
                     "Accel Back", 
                     "Gyro Back",
                     "Magno Back",
                     "Left insole \n(FSR 1–16)", 
                     "Right insole \n(FSR 1–16)"
                     ]

            plt.yticks(yticks, ylabs, fontsize=10)
            plt.ylabel("")  # optional, since labels already explain
            plt.show()

        X_list.append(arr_cnn)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    return X, y

set_global_seed(seed=343)

SCENARIOS = ['All', 'IMU_only', 'FSR_only', 'Arm_and_FSR', 'Back_and_FSR']
test_dict = create_image_dfs_from_timeseries_segments()

outer_results = []
all_y_true = []
all_y_pred = []

scenario_results = []
all_tests_metrics_per_space = {}
for leave_out in test_dict.keys():
    # collect raw train/test windows (with label)
    train_windows_raw = []
    test_windows_raw = test_dict[leave_out]

    for test_id, dfs in test_dict.items():
        if test_id != leave_out:
            train_windows_raw.extend(dfs)
            
    
    # best_scenario = None
    # best_scenario_f1 = -np.inf
    # best_scenario_params = None
    # best_model_final = None
    # X_test_final = None
    # y_test_final = None
    metrics_for_scenario =  {}
    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario}")
        # if scenario != 'All': 
        #     continue

        #Fit scaler for THIS scenario only
        scaler, feature_cols_scen = fit_scaler_for_scenario(train_windows_raw, scenario)
        
        print(feature_cols_scen)

        # 2) Scale train windows
        train_windows_scaled = [
            scale_window_df(df, scaler, feature_cols_scen)
            for df in train_windows_raw
        ]
        
        print("Scaled train wins")
        test_windows_scaled = [
            scale_window_df(df, scaler, feature_cols_scen)
            for df in test_windows_raw
        ]
        print("Scaled test wins")

        X_train, y_train = windows_to_tensor_and_labels(train_windows_scaled, scenario)
        X_test,  y_test  = windows_to_tensor_and_labels(test_windows_scaled, scenario)
        print("Tensors created")

        channels = X_train.shape[1]      
        time = X_train.shape[2]          
        NumOutput = len(np.unique(y_train))

        # Wrap Keras model for GridSearchCV
        clf = KerasClassifier(
            model=create_CNN_model,
            NumOutput=NumOutput,
            original_height=channels,
            original_width=time,

            # MUST include defaults for ALL params you want to tune:
            dropout_rate=0.3,
            filters=32,
            kernel_size=(2,2),
            pool_1=(2,1),
            pool_2=(2,2),

            optimizer="adam",
            lr=1e-3,
            epochs=10,

            verbose=0,
        )

        param_grid = {
            "filters": [16, 32],
            "kernel_size": [(3,3),(5,5)],
            "pool_1": [(2,1), (2,2), (1,2)],
            "dropout_rate": [0.3], #0.5],
            "optimizer": ["adam"], 
            "lr": [3e-4], 
        }

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=343)
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
        
        print("Beginning gridsearch")
        grid = GridSearchCV(
            clf,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            verbose=0,
            n_jobs=4
        )

        grid.fit(
            X_train, 
            y_train,
            callbacks=[early_stop],
            validation_split=0.2)

        best_f1 = grid.best_score_
        best_params = grid.best_params_
        best_model = grid.best_estimator_

        print(f"Best inner-CV F1_macro = {best_f1:.3f}")
        print(f"Best params: {best_params}")
        
        best_params = grid.best_params_
        
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        model = create_CNN_model(
            **best_params,
            NumOutput=NumOutput,
            original_height=channels,
            original_width=time
            )

        history = model.fit(
            X_train,
            y_train_enc,
            validation_split=0.2,
            epochs=60,
            callbacks=[early_stop],
            verbose=0
        )
        
        history_dict = history.history

        plt.figure(figsize=(10,5))
        plt.plot(history_dict['accuracy'], label='Train Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.show()

        
        print(f"Predicting on unseen data for {scenario}")
        y_test_fit      = best_model.predict(X_test)
        accuracy_test   = best_model.score(X_test, y_test)
        f1_test         = f1_score(y_test, y_test_fit, average='macro')
        precision_test  = precision_score(y_test, y_test_fit, average='macro')
        recall_test     = recall_score(y_test, y_test_fit, average='macro')
        
        print(y_test_fit[:5])
        print(y_test_fit.shape)
        
        test_results    = {
                'y_test_fit' : y_test_fit, 
                'accuracy' : accuracy_test, 
                'f1' :  f1_test, 
                'precision' : precision_test,
                'recall' : recall_test,
                'hyperparameters' : best_params}
        
        print(f"Accuracy for {leave_out},{scenario} is {accuracy_test}")
        
        metrics_for_scenario[scenario] = test_results
        all_y_true.extend(y_test)
    
    all_tests_metrics_per_space[leave_out] = metrics_for_scenario

all_tests_metrics_per_space_df = pd.DataFrame.from_dict(all_tests_metrics_per_space)
all_tests_metrics_per_space_df.to_csv("all_metrics_per_test_CNN.csv")
        
all_accuracies = {sc: [] for sc in SCENARIOS}
all_f1 = {sc: [] for sc in SCENARIOS}
all_precision = {sc: [] for sc in SCENARIOS}
all_recall = {sc: [] for sc in SCENARIOS}

print("Generating performance summary...")
for test_id in all_tests_metrics_per_space.keys():
    for scenario in SCENARIOS:
        acc = all_tests_metrics_per_space[test_id][scenario]['accuracy']
        f1 = all_tests_metrics_per_space[test_id][scenario]['f1']
        prec = all_tests_metrics_per_space[test_id][scenario]['precision']
        rec = all_tests_metrics_per_space[test_id][scenario]['recall']
        
        all_accuracies[scenario].append(acc)
        all_f1[scenario].append(f1)
        all_precision[scenario].append(prec)
        all_recall[scenario].append(rec)

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
mean_std_df.to_csv(f"mean_performance_per_space_CNN.csv")

mean_std_df = pd.DataFrame.from_dict(mean_and_std_metrics_per_space)
mean_std_df.to_csv(f"mean_performance_per_space_CNN.csv")

best_space = max(
mean_and_std_metrics_per_space,
key=lambda fs: mean_and_std_metrics_per_space[fs]["mean_f1"]
)

best_f1 = mean_and_std_metrics_per_space[best_space]["mean_f1"]

print("Best feature space:", best_space)
print("Mean F1 score:", best_f1)

# create confusion matrix for the best performing space
for test_id in all_tests_metrics_per_space.keys():
    print(test_id)
    y_pred = all_tests_metrics_per_space[test_id][best_space]['y_test_fit']
    all_y_pred.extend(y_pred)


labels = np.unique(all_y_true)
cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)

make_confusion_matrix(cm, categories=labels, figsize=(10,8), title=f'Confusion matrix LOOCV CNN-model')

