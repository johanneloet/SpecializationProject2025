import pandas as pd

from ExtractIMU_Features import ExtractIMU_Features, ExtractIMU_features_repetitions_based
from ExtractPressure_Features import ExtractPressure_Features, ExtractPressure_Features_repetitions_based
from get_paths import get_test_file_paths, get_one_foler_path
from create_feature_windows import drop_last_for_label
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
import time
from collections import Counter
"""
from Maria_code.data_analysis.ExtractIMU_Features import ExtractIMU_Features
from Maria_code.data_analysis.ExtractPressure_Features import ExtractPressure_Features
from Maria_code.data_analysis.get_paths import get_test_file_paths, get_one_foler_path
"""


"Windowing using all-or-nothing strategy"
def run_feature_extraction(df_muse_arm, 
                           df_muse_back, 
                           df_mitch_left, 
                           df_mitch_right, 
                           window_length_sec, 
                           norm_IMU, 
                           mean_fsr,
                           hdr,
                           replace_accel_hdr,
                           output_path = None,
                           feature_space = 'baseline'):
    """Runs feature extraction for a single test!
    
    Possible feature spaces:
    - 'baseline'
    - 'expanded+baseline'
    - 'expanded_only'
    - 'time_only'
    - 'time_only+exp_FSR'
    - 'freq_only'
    - 'freq_only+exp_FSR'
    - 'FSR_only'

    Args:
        df_muse_arm (pd.DataFrame): Preprocessed arm data.
        df_muse_back (pd.DataFrame): Preprocessed back data.
        df_mitch_left (pd.DataFrame): Preprocessed left foot fsr data.
        df_mitch_right (pd.dataFrame): Preprocessed right foot fsr data.
        window_length_sec (int): Length of windows in seconds.
        norm_IMU (bool): Whether to use norm on IMU axes.
        mean_fsr (bool): Whether to use the mean of the fsr insoles for feature extraction. If false, the sum will be used.
        hdr (bool): Whether to include hdr acceleration.
        replace_accel_hdr (bool): Whether to replace acceleration with hdr acceleration. hdr = True is a prerequisite.
        output_path (str, optional): Path to save feature .csv-file. Defaults to None. If left as None, features will not be saved to a .csv, and only returned as a dataframe- 
        feature_space (str, optional): Flag to determine which feature space to use.  Defaults to 'baseline'.

    Returns:
        pd.DataFrame: Dataframe containing features for all windows for a given set of muse and mitch data.
    """

    window_length_samples_muse = int(window_length_sec * 800)
    window_length_samples_mitch = int(window_length_sec * 100)

    # Get features from each IMU source
    # feat_muse_arm = ExtractIMU_Features(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
    # feat_muse_back = ExtractIMU_Features(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)

    if feature_space == 'baseline':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        print(f"Extracted IMU features for {len(feat_muse_arm)} arm repetitions")
        print(f"Extracted IMU features for {len(feat_muse_back)} back repetitions")
        #time.sleep(20)
        
        # Get features from each FSR source
        print("Getting left features")
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, fs=100, feature_space='baseline')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, fs=100, feature_space='baseline')
        print(f"Extracted FSR features for {len(feat_mitch_left)} left repetitions")
        print(f"Extracted FSR features for {len(feat_mitch_right)} right repetitions")    
        
        # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU rowgit filter-repo --path Data/ --invert-paths")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        if not (len(feat_muse_arm) == len(feat_muse_back) == len(feat_mitch_left) == len(feat_mitch_right)):
            print("STOPPING ... extracted different number of feature windowws for the different sensors. This should not be possible.")
            print("check that all windows are valid")
            print("lengths are")
            print(f"  muse_arm     : {len(feat_muse_arm)}")
            print(f"  muse_back    : {len(feat_muse_back)}")
            print(f"  mitch_left   : {len(feat_mitch_left)}")
            print(f"  mitch_right  : {len(feat_mitch_right)}")
            return None

        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
        # if window_labels_FSR < window_labels_IMU:
        #     all_features['label'] = window_labels_FSR
        # elif window_labels_FSR > window_labels_IMU:
        #     all_features['label'] = window_labels_IMU
        # else:
        #     all_features['label'] = window_labels_IMU 
        
        

    elif feature_space == 'expanded+baseline':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='expanded+baseline')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='expanded+baseline')
        
                # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
    
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
      
        
        
    
    elif feature_space == 'expanded_only':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='expanded_only')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='expanded_only')
        
        # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
        else: 
            print('mismatch between IMU and FSR labels!!')
            return None
        

        
        
    elif feature_space == 'IMU_only':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        
        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
        ], axis=1)

        all_features['label'] = window_labels_IMU
        
        
    elif feature_space == 'time_only':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, time_only=True)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, time_only=True)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='time_only')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='time_only')
        
        print(f"{feature_space}: EXTRACTED {len(feat_muse_arm)} IMU windows and {len(feat_mitch_left)} FSR windows")
        
                # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        
        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
        else: 
            print('mismatch between IMU and FSR labels!!')
            return None
        
        print("ALL FEATURE COLUMNS!!!!!")
        print(all_features.columns)
        
    elif feature_space == 'time_only+exp_FSR':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, time_only=True)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, time_only=True)
        feat_mitch_left, window_labels_FSR= ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='time_only+exp_FSR')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='time_only+exp_FSR')
        
                # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        
        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
        else: 
            print('mismatch between IMU and FSR labels!!')
            return None
        
    elif feature_space == 'freq_only':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, freq_only=True)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, freq_only=True)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='freq_only')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='freq_only')
        
        
                # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
        else: 
            print('mismatch between IMU and FSR labels!!')
            return None
        
        print("ALL FEATURE COLUMNS!!!!!")
        print(all_features.columns)
        
        
        
    elif feature_space == 'freq_only+exp_FSR':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, freq_only=True)
        feat_muse_back, _ = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr, freq_only=True)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='freq_only+exp_FSR')
        feat_mitch_right, _ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='freq_only+exp_FSR')
        
        
        # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)

            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        
        # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR == window_labels_IMU: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_IMU
        else: 
            print('mismatch between IMU and FSR labels!!')
            return None
        
        print("ALL FEATURE COLUMNS!!!!!")
        print(all_features.columns)
        
    elif feature_space == 'FSR_only':
        #feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_mitch_left, window_labels_FSR_left = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='expanded+baseline')
        feat_mitch_right, window_labels_FSR_right = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='expanded+baseline')
    
        
        
        
         # Combine features (length should now match if everything is done correctly, thereby no check. Double checking label matching below.)
        all_features = pd.concat([
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        if window_labels_FSR_left == window_labels_FSR_right: # assuming time-alignment, this should be okay
            all_features['label'] = window_labels_FSR_left
        else: 
            print('mismatch between left and right labels!!')
            return None
        
        print("ALL FEATURE COLUMNS!!!!!")
        print(all_features.columns)
        
    
    elif feature_space == 'arm_only+FSR':
        feat_muse_arm, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='expanded+baseline')
        feat_mitch_right,_ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='expanded+baseline')

        # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)
        
            if imu_count > fsr_count:
                    print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                    feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_arm, window_labels_IMU, label_value)
                    #feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)
        
        all_features = pd.concat([
            feat_muse_arm,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        all_features['label'] = window_labels_IMU
    
    elif feature_space == 'back_only+FSR':
        #feat_muse_arm = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_muse_back, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_mitch_left, window_labels_FSR = ExtractPressure_Features_repetitions_based(df_mitch_left, "left", mean_fsr, 100, feature_space='expanded+baseline')
        feat_mitch_right,_ = ExtractPressure_Features_repetitions_based(df_mitch_right, "right", mean_fsr, 100, feature_space='expanded+baseline')
        
        # Get all unique labels from both lists
        all_labels = sorted(set(window_labels_IMU) | set(window_labels_FSR)) # assume arm and back are equal and left and right are equal
        # Count how many times each label appears
        counts_IMU = Counter(window_labels_IMU)
        counts_FSR = Counter(window_labels_FSR)
    
        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = counts_IMU.get(label_value, 0)
            fsr_count = counts_FSR.get(label_value, 0)
        
            if imu_count > fsr_count:
                    print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                    feat_muse_arm, window_labels_IMU = drop_last_for_label(feat_muse_back, window_labels_IMU, label_value)
                    #feat_muse_back, _ = drop_last_for_label(feat_muse_back, window_labels_IMU.copy(), label_value)

            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                feat_mitch_left, window_labels_FSR = drop_last_for_label(feat_mitch_left, window_labels_FSR, label_value)
                feat_mitch_right, _ = drop_last_for_label(feat_mitch_right, window_labels_FSR.copy(), label_value)

        all_features = pd.concat([
            feat_muse_back,
            feat_mitch_left,
            feat_mitch_right
        ], axis=1)
        all_features['label'] = window_labels_IMU
    
    elif feature_space == 'IMU_only':
        feat_muse_arm = ExtractIMU_features_repetitions_based(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)
        feat_muse_back, window_labels_IMU = ExtractIMU_features_repetitions_based(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr, replace_accel_hdr)

        all_features = pd.concat([
            feat_muse_arm,
            feat_muse_back,
        ], axis=1)
        all_features['label'] = window_labels_IMU
    
    
    else:
        print(f"Feature space, {feature_space}, not a valid argument. Please select a valid feature space. See documentation.")
        print("STOPPING..., None will be returned")
        return None
    
    # # Label the windows using one of the IMU or FSR sources (assuming they are time-aligned)
    # window_labels, total_windows, ambiguous_windows = label_windows(df_muse_arm, window_length_samples_muse)

    # # Combine features
    # min_len = min(len(feat_muse_arm), len(feat_muse_back), len(feat_mitch_left), len(feat_mitch_right), len(window_labels))
    # # min_len = min(len(feat_muse_arm), len(feat_muse_back), len(window_labels))

    # all_features = pd.concat([
    #     feat_muse_arm[:min_len],
    #     feat_muse_back[:min_len],
    #     feat_mitch_left[:min_len],
    #     feat_mitch_right[:min_len]
    # ], axis=1)
    # all_features['label'] = window_labels[:min_len]

    # Remove windows where label was ambiguous
    all_features = all_features.dropna(subset=['label']).reset_index(drop=True)
    
    if output_path is not None:
        output_path = f"{output_path.replace('.csv', '')}_{feature_space}.csv"
        all_features.to_csv(output_path, index=False)

    return all_features


def label_windows(data, window_length_samples):
    labels = []
    num_windows = len(data) // window_length_samples
    ambiguous_count = 0
    for i in range(num_windows):
        start_idx = i * window_length_samples
        end_idx = start_idx + window_length_samples
        window_labels = data['label'][start_idx:end_idx]
        unique_labels = window_labels.unique()
        if len(unique_labels) == 1:
            labels.append(unique_labels[0])
        else:
            labels.append(None)
            ambiguous_count += 1
    return labels, num_windows, ambiguous_count


def run_feature_extraction_for_all_tests(window_length_sec=2, norm_IMU=True, mean_fsr=False, hdr=False, replace_accel_hdr=False, feature_space='baseline'):
    file_dict = get_test_file_paths()
    start = time.time()
    
    for test_id, paths in file_dict.items():
        print(f"\n--- Running feature extraction for {test_id} ---")
        try:
            df_arm = pd.read_csv(paths["arm"])
            df_back = pd.read_csv(paths["back"])
            df_left = pd.read_csv(paths["left"])
            df_right = pd.read_csv(paths["right"])

            folder = get_one_foler_path(test_id)
            print(f"Folder to save features: {folder}")
            
            
            # Drop rows where rep_id == None
            df_arm = df_arm.dropna(subset=["rep_id"])
            df_back = df_back.dropna(subset=["rep_id"])
            df_left = df_left.dropna(subset=["rep_id"])
            df_right = df_right.dropna(subset=["rep_id"])
            

            output_path = f"{folder}/{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}.csv"
#            output_path = f"{folder}/{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}.csv"

            
            all_features = run_feature_extraction(df_arm, df_back, df_left, df_right,
                                   window_length_sec, norm_IMU, mean_fsr, hdr, replace_accel_hdr,
                                   output_path=output_path,
                                   feature_space=feature_space)
#            run_feature_extraction(df_arm, df_back, None, None,
#                                   window_length_sec, norm_IMU, mean_fsr, hdr,
#                                   output_path=output_path)
            if all_features is None:
                return None

        except Exception as e:
            print(f"Failed for {test_id}: {e}")
            return None # do not allow continuing if one fails. Want to make sure everything runs correctly, as we can no longer debug from terminal

    end = time.time()
    elapsed = end - start
    print(f"\n🕒 Done! Total time uesd: {elapsed:.2f} seconds")
    
    return True


feature_spaces = [
    "baseline",
    "expanded+baseline",
    # "expanded_only",
    # "time_only",
    # "time_only+exp_FSR",
    # "freq_only",
    #"freq_only+exp_FSR",
    "FSR_only",
    "arm_only+FSR", 
    "back_only+FSR",
    "IMU_only"
]

for space in feature_spaces:
    print(f"EXTRACTING {space} FEATURES FOR ALL TESTS")
    finished_flag = run_feature_extraction_for_all_tests(window_length_sec=8, norm_IMU=False, mean_fsr=True, hdr=False, replace_accel_hdr=False, feature_space=space)
    if finished_flag == None:
        break

