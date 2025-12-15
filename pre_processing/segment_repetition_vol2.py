"""
File purpose: segmentation of repetitions based on manual observations in acceleration plots.

This file is written by Johanne, though the approach is inspired by Maria's repetition segmentation approach.
"""
# Imports
#from get_paths import get_test_file_paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def plot_activity_accelerations_peaks_and_magnitude(
        df,
        activity_label,
        height=1100,
        distance=1100,
        peak_indices=None,
        colors=None,
        figsize=(12, 6)):
    """
    Plot acceleration components (X, Y, Z), magnitude, and detected peaks
    for a given activity label from an IMU DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with accelerometer data and a 'label' column and a 'ReconstructedTime' column. 
    activity_label : str
        The movement label to filter (e.g., 'hand_up_back').
    height : float, optional
        Minimum height (in mg) for peak detection.
    distance : int, optional
        Minimum distance (in samples) between peaks.
    peak_indices : array-like, optional
        Custom peak indices to plot. If None, peaks will be auto-detected.
    colors : dict, optional
        Color dictionary for the curves.
    figsize : tuple, optional
        Figure size for the plot.

    Returns
    -------
    subset : pandas.DataFrame
        Subset of the data corresponding to the selected label.
    peak_indices : np.ndarray
        Indices of the plotted peaks.
    properties : dict or None
        Properties from scipy.signal.find_peaks if peaks were auto-detected.
    """

    if colors is None:
        colors = {
            "mag": "#1F4E99",   # strong, clear blue (focus signal)
            "x":   "#B0C4DE",   # light steel blue (faded)
            "y":   "#C5D1E0",   # very light grey-blue
            "z":   "#D6DFEB"    # near-background blue-grey
        }
        
        
    # Suppose your DataFrame has a non-continuous or meaningful index
    subset = df[df["label"] == activity_label].copy()

    # Compute magnitude
    subset["Accel mag"] = np.sqrt(subset["Axl.X"]**2 + subset["Axl.Y"]**2 + subset["Axl.Z"]**2)

    # Call find_peaks on the NumPy array
    if peak_indices == None:
        local_peak_indices, properties = find_peaks(subset["Accel mag"].to_numpy(), height=1100, distance=1100)

        # Map back to original indices
        original_peak_indices = subset.index[local_peak_indices]

        print("Local:", local_peak_indices[:5])
        print("Original:", original_peak_indices[:5])
    else: 
        local_peak_indices = peak_indices
        original_peak_indices = None
        properties = None
    mid_indices = []
    for i in range(0, len(local_peak_indices) - 1, 2):
        mid_idx = (local_peak_indices[i] + local_peak_indices[i + 1]) // 2
        mid_indices.append(mid_idx)


    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.plot(subset["ReconstructedTime"], subset["Axl.X"],
             label="Axl.X", color=colors["x"], linewidth=1.5)
    plt.plot(subset["ReconstructedTime"], subset["Axl.Y"],
             label="Axl.Y", color=colors["y"], linewidth=1.5)
    plt.plot(subset["ReconstructedTime"], subset["Axl.Z"],
             label="Axl.Z", color=colors["z"], linewidth=1.5)
    plt.plot(subset["ReconstructedTime"], subset["Accel mag"],
            label="Axl. magnitude", color=colors["mag"], linewidth=1.5)

    # Plot peaks
    plt.plot(subset["ReconstructedTime"].iloc[local_peak_indices],
             subset["Accel mag"].iloc[local_peak_indices],
             "rx", label="Peaks")

    # --- Style ---
    plt.title(f"Acceleration Components over Time ({activity_label})",
              fontsize=16, weight='bold')
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Acceleration (mg)", fontsize=13)
    plt.legend(title="Axes", fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    for mid_idx in mid_indices:
        plt.axvline(
            x=subset["ReconstructedTime"].iloc[mid_idx],
            color="red",
            linestyle=":",
            linewidth=1,
            alpha=0.8
        )
    plt.tight_layout()
    plt.savefig("acceleration_peaks.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return subset, original_peak_indices, local_peak_indices



def get_start_stop_times_from_peaks(df, peaks, activity_name, num_reps = 6,time_col="ReconstructedTime"):
    """
    Extract start and stop times for each repetition based on peak indices.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the time column.
    peaks : array-like
        List or array of peak indices (length >= 2).
    activity_name : str
        Activity label prefix (e.g., 'hand_up_back').
    time_col : str, optional
        Name of the time column, by default 'ReconstructedTime'.

    Returns
    -------
    start_stop_times_reps : dict
        Dictionary mapping rep_id -> (start_time, stop_time)
    rep_intervals_df : pandas.DataFrame
        Same info as a DataFrame (optional use)
    """
    peaks = np.sort(np.asarray(peaks, dtype=int))
    if len(peaks) < 2:
        raise ValueError("Need at least two peaks to form intervals.")

    times = df[time_col].to_numpy()

    start_stop_times_reps = {}
    intervals = []
    assert len(peaks) >= num_reps
    for i in range(num_reps):
        rep_start_idx = peaks[i]
        rep_stop_idx = peaks[i + 1]
        rep_start_time = times[rep_start_idx]
        rep_stop_time = times[rep_stop_idx]
        rep_name = f"{activity_name}_{i+1}"

        start_stop_times_reps[rep_name] = (rep_start_time, rep_stop_time)
        intervals.append((rep_name, rep_start_time, rep_stop_time))

    rep_intervals_df = pd.DataFrame(intervals, columns=["rep_id", "start_time", "stop_time"])
    return start_stop_times_reps, rep_intervals_df


if __name__ == '__main__':
    PATH_DICT = get_test_file_paths()
    
    # select test_id ('test_1' - 'test_20')
    test_id = 'test_14'
    
    # Paths to already labelled segments (with rep ids on Push/Pull, so ignore in this file)
    arm_path = PATH_DICT[test_id]['arm']
    back_path = PATH_DICT[test_id]['back']
    left_path = PATH_DICT[test_id]['left']
    right_path = PATH_DICT[test_id]['right']
    
    # Begin with arm
    arm_df = pd.read_csv(arm_path)
    
    plot_activity_accelerations_peaks_and_magnitude(arm_df, 'hand_up_back')
    
    hand_back_df = arm_df[arm_df["label"] == "hand_up_back"].copy()
    hand_back_df['Accel mag'] = np.sqrt(hand_back_df['Axl.X']**2 + hand_back_df['Axl.Y']**2 + hand_back_df['Axl.Z']**2)
    gradient_x = np.gradient(hand_back_df['Axl.X'])
    peak_indices, _ = find_peaks(hand_back_df['Accel mag'], height=1100, distance=1100)
    

    plt.figure(figsize=(12, 6))
    # Plot each axis with consistent styling
    plt.plot(hand_back_df['ReconstructedTime'], hand_back_df['Accel mag'], label='Axl. magnitude', color= 'pink' , linewidth=1.5)
    plt.plot(hand_back_df['ReconstructedTime'], hand_back_df['Axl.X'], label='Axl.X', color='#98FB98', linewidth=1.5)
    plt.plot(hand_back_df['ReconstructedTime'], hand_back_df['Axl.Y'], label='Axl.Y', color='#A569BD',  linewidth=1.5)
    plt.plot(hand_back_df['ReconstructedTime'], hand_back_df['Axl.Z'], label='Axl.Z', color='#5DADE2',linewidth=1.5)
    plt.plot(hand_back_df["ReconstructedTime"].iloc[peak_indices], hand_back_df["Accel mag"].iloc[peak_indices], "rx", label="Peaks")
    
    # Add title and axis labels
    plt.title("Acceleration Components over Time (hand_up_back)", fontsize=16, weight='bold')
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Acceleration (mg)", fontsize=13)

    # Add legend and grid
    plt.legend(title="Axes", fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Tight layout and show
    plt.tight_layout()
    plt.show()
    
    hands_forward_df = arm_df[arm_df["label"] == "hands_forward"].copy()
    hands_forward_df['Accel mag'] = np.sqrt(hands_forward_df['Axl.X']**2 + hands_forward_df['Axl.Y']**2 + hands_forward_df['Axl.Z']**2)
    gradient_x = np.gradient(hands_forward_df['Axl.X'])
    peak_indices, _ = find_peaks(hands_forward_df['Accel mag'], height=1100, distance=1100)
    

    plt.figure(figsize=(12, 6))
    # Plot each axis with consistent styling
    plt.plot(hands_forward_df['ReconstructedTime'], hands_forward_df['Accel mag'], label='Axl. magnitude', color= 'pink' , linewidth=1.5)
    plt.plot(hands_forward_df['ReconstructedTime'], hands_forward_df['Axl.X'], label='Axl.X', color='#98FB98', linewidth=1.5)
    plt.plot(hands_forward_df['ReconstructedTime'], hands_forward_df['Axl.Y'], label='Axl.Y', color='#A569BD',  linewidth=1.5)
    plt.plot(hands_forward_df['ReconstructedTime'], hands_forward_df['Axl.Z'], label='Axl.Z', color='#5DADE2',linewidth=1.5)
    plt.plot(hands_forward_df["ReconstructedTime"].iloc[peak_indices], hands_forward_df["Accel mag"].iloc[peak_indices], "rx", label="Peaks")
    
    # Add title and axis labels
    plt.title("Acceleration Components over Time (hands_forward)", fontsize=16, weight='bold')
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Acceleration (mg)", fontsize=13)

    # Add legend and grid
    plt.legend(title="Axes", fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Tight layout and show
    plt.tight_layout()
    plt.show()
    
    
    hands_up_df = arm_df[arm_df["label"] == "hands_up"].copy()
    hands_up_df['Accel mag'] = np.sqrt(hands_up_df['Axl.X']**2 + hands_up_df['Axl.Y']**2 + hands_up_df['Axl.Z']**2)
    gradient_x = np.gradient(hands_up_df['Axl.X'])
    peak_indices, _ = find_peaks(hands_up_df['Accel mag'], height=1100, distance=1100)
    
    plt.figure(figsize=(12, 6))
    # Plot each axis with consistent styling
    plt.plot(hands_up_df['ReconstructedTime'], hands_up_df['Accel mag'], label='Axl. magnitude', color= 'pink' , linewidth=1.5)
    plt.plot(hands_up_df['ReconstructedTime'], hands_up_df['Axl.X'], label='Axl.X', color='#98FB98', linewidth=1.5)
    plt.plot(hands_up_df['ReconstructedTime'], hands_up_df['Axl.Y'], label='Axl.Y', color='#A569BD',  linewidth=1.5)
    plt.plot(hands_up_df['ReconstructedTime'], hands_up_df['Axl.Z'], label='Axl.Z', color='#5DADE2',linewidth=1.5)
    plt.plot(hands_up_df["ReconstructedTime"].iloc[peak_indices], hands_up_df["Accel mag"].iloc[peak_indices], "rx", label="Peaks")
    
    # Add title and axis labels
    plt.title("Acceleration Components over Time (hands_up)", fontsize=16, weight='bold')
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Acceleration (mg)", fontsize=13)

    # Add legend and grid
    plt.legend(title="Axes", fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Tight layout and show
    plt.tight_layout()
    plt.show()