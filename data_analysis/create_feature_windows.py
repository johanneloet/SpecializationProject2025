import numpy as np
import pandas as pd

""" Helpers for creating consistent fixed-size / resampled windows. """
def build_boundaries(
    df: pd.DataFrame,
    fixed_labels=("standing", "sitting", "walking"),
    fixed_len=2800,
    drop_incomplete=True,
):
    labels = df["label"].fillna("").astype(str).str.lower().to_numpy()
    rep_ids = df["rep_id"].fillna("none").astype(str).to_numpy()

    label_chg = np.where(labels[:-1] != labels[1:])[0] + 1
    label_bounds = np.concatenate(([0], label_chg, [len(labels)]))

    rows = []

    for i in range(len(label_bounds) - 1):
        Ls, Le = label_bounds[i], label_bounds[i + 1]
        lab = labels[Ls]  

        if any(lab.startswith(fl) or fl in lab for fl in fixed_labels):
            span_len = Le - Ls
            if span_len < fixed_len and drop_incomplete:
                continue
            n_full = span_len // fixed_len
            for k in range(n_full):
                s = Ls + k * fixed_len
                e = s + fixed_len
                rows.append((s, e, lab, rep_ids[s]))  # rep_id here is just the one at start; label drives this
            if not drop_incomplete and span_len % fixed_len:
                s = Ls + n_full * fixed_len
                e = Le
                rows.append((s, e, lab, rep_ids[s]))
        else:
            # Behavior for non-fixed labels: split by rep_id within this label span
            sub_rep = rep_ids[Ls:Le]
            sub_chg = np.where(sub_rep[:-1] != sub_rep[1:])[0] + 1
            sub_bounds = np.concatenate(([Ls], Ls + sub_chg, [Le]))
            for j in range(len(sub_bounds) - 1):
                s, e = sub_bounds[j], sub_bounds[j + 1]
                rows.append((s, e, lab, rep_ids[s]))

    return pd.DataFrame(rows, columns=["start_idx", "end_idx", "label", "rep_id"])

def drop_last_for_label(df, labels, label_value):
    """
    Drops exactly one row in df (the last occurrence of label_value)
    and the corresponding label from labels.
    Returns updated (df, labels) with correct alignment.
    """
    if not isinstance(labels, list):
        labels = list(labels)

    idxs = [i for i, l in enumerate(labels) if l == label_value]
    if not idxs:
        return df, labels  # nothing to drop

    drop_idx = idxs[-1]

    # Drop without resetting index yet, so we can safely pop label first
    df = df.drop(df.index[drop_idx])

    # Remove the matching label
    labels.pop(drop_idx)

    # Now reindex both so they stay 1:1 aligned
    df = df.reset_index(drop=True)
    return df, labels
