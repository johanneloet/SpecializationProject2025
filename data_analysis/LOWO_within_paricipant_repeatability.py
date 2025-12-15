import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from pathlib import Path
import json


from get_paths import get_feture_paths
from run_NN import run_NN
from run_RFC import run_RFC
from run_SVC import run_SVC

def run_LOWO_for_one_participant(feature_path : str, model_type="NN"):
    """
    Args:
        feature_path (str): path to extracted features for a given participant
        model_type (str, optional): Type of model, possible: 'NN', 'SVC', 'RFC'. Defaults to "NN".
    """
    feature_df = pd.read_csv(feature_path)
    label_col = feature_df['label'].to_list()
    
    labels = list(set(label_col))
    
    all_test_results = []
    all_train_results = []
        
    # lowo loop:
    for i, feature_window in feature_df.iterrows():
        test_df = feature_df.loc[[i]]
        train_df = feature_df.drop(index=i)
        
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        
        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']
        
        # SCALING CODE FROM MARIA
        # Scale data (SD of 1 and mean of 0)
        scaler = StandardScaler().set_output(transform="pandas")
        # Fit scaler on training data
        scaler.fit(X_train)
        # Transfor both train and test set with the scaler
        X_train_scaled = scaler.transform(X_train).to_numpy()
        X_test_scaled = scaler.transform(X_test).to_numpy()
        
        if model_type == "NN":
            test_results, train_results = run_NN(
                X_train=X_train_scaled,
                y_train=y_train,
                X_test=X_test_scaled,
                y_test=y_test,
                class_names=labels,
                opt=True
            )
            
            y_pred, acc, f1, prec, rec = test_results
            y_true = feature_window['label']
            
            all_test_results.append((y_pred, y_true, acc, f1, prec, rec))
            all_train_results.append(train_results)
            
        elif model_type == "SVC":
            return None, None
        elif model_type == "RFC":
            return None, None
        else:
            print(f"STOPPING RUN: Model type {model_type} not recognized. Enter a valid model type: NN, SVC, or RFC!")
            return None, None
        # Clean the label arrays first
    cleaned_results = []
    for item in all_test_results:
        label_pred, label_true, acc, f1, prec, rec = item
        # flatten label if it's an array
        if isinstance(label_pred, np.ndarray):
            label = label_pred.item() if label_pred.size == 1 else tuple(label_pred.tolist())
        elif isinstance(label_pred, (list, tuple)) and len(label_pred) == 1:
            label = label_pred[0]
        cleaned_results.append((label, label_true, acc, f1, prec, rec))

    # Now make the DataFrame safely
    test_results_df = pd.DataFrame(
        cleaned_results,
        columns=['predicted label', 'true label', 'accuracy', 'f1', 'precision', 'recall']
    )
    
    return test_results_df, all_train_results

if __name__ == '__main__':
    FEAT_PATHS = get_feture_paths(
        window_length_sec=8,
        norm_IMU = False,
        mean_fsr = True,
        hdr = False
        )
    
    for test_id in FEAT_PATHS.keys():
        test_feat_path = FEAT_PATHS[test_id]
        
        test_results_df, all_train_results = run_LOWO_for_one_participant(test_feat_path)
        
        # summary values
        summary = {
            "mean_accuracy_per_label": test_results_df.groupby("true label")["accuracy"].mean().to_dict(),
            "total_mean_accuracy": float(test_results_df["accuracy"].mean())
        }

        # choose filename dynamically, e.g., using test_id
        da_path = Path(r'C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\LOWO_analysis_summaries')
        out_path = da_path / f"LOWO_summary_{test_id}.json"

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"✅ Summary written to {out_path.resolve()}")
    
    