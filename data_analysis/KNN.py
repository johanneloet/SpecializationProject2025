import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from get_paths import get_feture_paths  # your existing helper

def run_LOWO_for_one_participant_knn(
    feature_path: str,
    n_neighbors: int = 5,
    metric: str = "cosine",      # try "euclidean" too
    weights: str = "uniform"     # or "distance"
):
    """
    Leave-One-Window-Out using KNN (classifier-free training; just stores train set).
    Returns a per-fold DataFrame with predicted/true labels and per-fold accuracy.
    """
    # Load & normalize labels
    feature_df = pd.read_csv(feature_path)
    feature_df["label"] = feature_df["label"].astype(str).str.strip()
    feature_df = feature_df.reset_index(drop=True)

    # Keep only numeric features
    X_all = feature_df.drop(columns=["label"]).select_dtypes(include=["number"])
    y_all = feature_df["label"]

    # Stable class list (useful later for reports)
    classes = sorted(y_all.unique().tolist())

    all_test_results = []  # (pred_label, true_label, accuracy, f1, precision, recall)

    # LOWO loop
    for pos, _row in feature_df.iterrows():
        test_df  = feature_df.iloc[[pos]]
        train_df = feature_df.drop(index=pos)

        X_train = train_df.drop(columns=["label"]).select_dtypes(include=["number"])
        y_train = train_df["label"]
        X_test  = test_df.drop(columns=["label"]).select_dtypes(include=["number"])
        y_true  = test_df["label"].iloc[0]

        # Scale (fit on train only)
        scaler = StandardScaler().set_output(transform="pandas").fit(X_train)
        X_train_scaled = scaler.transform(X_train).to_numpy()
        X_test_scaled  = scaler.transform(X_test).to_numpy()

        # Ensure k <= number of training samples
        k_eff = min(n_neighbors, len(y_train))
        if k_eff < 1:
            raise ValueError("Not enough training samples for KNN.")

        # Fit KNN and predict
        knn = KNeighborsClassifier(
            n_neighbors=k_eff,
            metric=metric,
            weights=weights
        )
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)[0]

        # Per-fold accuracy for single test sample: 1 if correct else 0
        acc = float(y_pred == y_true)

        # For a single-sample fold, micro-averaged precision/recall/F1 == accuracy
        prec = rec = f1 = acc

        all_test_results.append((y_pred, y_true, acc, f1, prec, rec))

    # Build results DataFrame
    test_results_df = pd.DataFrame(
        all_test_results,
        columns=["predicted label", "true label", "accuracy", "f1", "precision", "recall"]
    )

    return test_results_df, {"classes": classes, "metric": metric, "n_neighbors": n_neighbors, "weights": weights}


if __name__ == "__main__":
    FEAT_PATHS = get_feture_paths(
        window_length_sec=8,
        norm_IMU=False,
        mean_fsr=True,
        hdr=False
    )

    # Where to save summaries
    da_path = Path(r"C:\Users\Bruker\Master25_code\Master25\Maria_code\data_analysis\LOWO_analysis_summaries")
    da_path.mkdir(parents=True, exist_ok=True)

    for test_id, test_feat_path in FEAT_PATHS.items():
        test_results_df, meta = run_LOWO_for_one_participant_knn(
            test_feat_path,
            n_neighbors=5,
            metric="cosine",
            weights="uniform"
        )

        # Summary values
        mean_acc_per_label = test_results_df.groupby("true label")["accuracy"].mean().to_dict()
        total_mean_accuracy = float(test_results_df["accuracy"].mean())
        summary = {
            "test_id": test_id,
            "meta": meta,
            "mean_accuracy_per_label": mean_acc_per_label,
            "total_mean_accuracy": total_mean_accuracy
        }

        # Save JSON
        out_path = da_path / f"KNN_LOWO_summary_{test_id}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"✅ Summary written to {out_path.resolve()}")

        # (Optional) also save the per-fold table
        per_fold_path = da_path / f"KNN_LOWO_folds_{test_id}.csv"
        test_results_df.to_csv(per_fold_path, index=False)
        print(f"✅ Per-fold results written to {per_fold_path.resolve()}")
