from Maria_code.data_analysis.create_feature_windows import build_boundaries
from Maria_code.data_analysis.get_paths import get_test_file_paths
from Maria_code.data_analysis.resample_windows import downsample_channel
#from Maria_code.data_analysis.create_feature_windows import drop_last_for_label
from sklearn.preprocessing import StandardScaler
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def drop_last_df_for_label(windows_list, labels_list, label_value):
    for i in range(len(labels_list) - 1, -1, -1):
        if labels_list[i] == label_value:
            windows_list.pop(i)
            labels_list.pop(i)
            break
    return windows_list, labels_list


def build_downsampled_window(df,
                             start_idx,
                             end_idx,
                             fixed_len=2800,
                             target_num_samples=350,
                             keep_keywords=('axl', 'gyr', 'mag'),
                             skip_substr='hdr'):

    """
    Extract window, keep only columns containing keep_keywords,
    skip columns that contain skip_substr, downsample each channel to target_num_samples.
    """

    # slice
    window = df.iloc[start_idx:end_idx].copy()

    # check length
    if len(window) < fixed_len:
        return None, None

    # save the label before filtering
    label = window.iloc[0]['label'] if 'label' in window.columns else None

    allowed_cols = [
        col for col in window.columns
        if any(k in col.lower() for k in keep_keywords)
    ]

    # create filtered window
    window = window[allowed_cols]

    filtered_cols = [
        col for col in window.columns
        if skip_substr not in col.lower()
    ]

    window = window[filtered_cols]
    

    downsampled_window = pd.DataFrame()

    for col in window.columns:
        col_downsampled = downsample_channel(
            df_channel=window[col],
            target_num_samples=target_num_samples
        )
        downsampled_window[col] = col_downsampled

    return downsampled_window, label
    
def create_image_dfs_from_timeseries_segments():
    segments_per_test = {}
    TEST_FILES = get_test_file_paths()

    for test_id in TEST_FILES.keys():
        imu_arm_file_path = TEST_FILES[test_id]['arm']
        imu_back_file_path = TEST_FILES[test_id]['back']
        fsr_left_file_path = TEST_FILES[test_id]['left']
        fsr_right_file_path = TEST_FILES[test_id]['right']

        df_arm = pd.read_csv(imu_arm_file_path)
        df_back = pd.read_csv(imu_back_file_path)
        df_left = pd.read_csv(fsr_left_file_path)
        df_right = pd.read_csv(fsr_right_file_path)
        
        df_arm = df_arm.add_prefix("Arm_")
        df_back = df_back.add_prefix("Back_")
        df_left = df_left.add_prefix("Left_")
        df_right = df_right.add_prefix("Right_")

        # Restore 'label' & 'rep_id' (prefix messed them up)
        df_arm.rename(columns={"Arm_label": "label", "Arm_rep_id": "rep_id"}, inplace=True)
        df_back.rename(columns={"Back_label": "label", "Back_rep_id": "rep_id"}, inplace=True)
        df_left.rename(columns={"Left_label": "label", "Left_rep_id": "rep_id"}, inplace=True)
        df_right.rename(columns={"Right_label": "label", "Right_rep_id": "rep_id"}, inplace=True)
        
        # Enforce numeric ordering of FSR columns 
        
        # old_left = df_left #uncomment for comparison of before and after reordering

        # LEFT
        left_fsr_cols = [c for c in df_left.columns if "Fsr." in c]
        other_left_cols = [c for c in df_left.columns if c not in left_fsr_cols]

        left_fsr_cols_sorted = sorted(
            left_fsr_cols,
            key=lambda c: int(c.split('.')[-1])   # extract the number after the dot
        )

        df_left = df_left[other_left_cols + left_fsr_cols_sorted]

        # RIGHT
        right_fsr_cols = [c for c in df_right.columns if "Fsr." in c]
        other_right_cols = [c for c in df_right.columns if c not in right_fsr_cols]

        right_fsr_cols_sorted = sorted(
            right_fsr_cols,
            key=lambda c: int(c.split('.')[-1])
        )

        df_right = df_right[other_right_cols + right_fsr_cols_sorted]
        
        
        # sanity debug 
        # print(df_left.columns)
     
        # all_equal = (old_left['Left_Fsr.01'].values == df_left['Left_Fsr.01'].values).all()
        # print(all_equal)
    
        
        # Drop rows where rep_id is None
        df_arm = df_arm.dropna(subset=["rep_id"])
        df_back = df_back.dropna(subset=["rep_id"])
        df_left = df_left.dropna(subset=["rep_id"])
        df_right = df_right.dropna(subset=["rep_id"])
        
        # debug
        # print(df_left.head(5))
        # print(df_right.head(5))
        

        FSR_segment_boundaries = build_boundaries(df=df_left, fixed_len=350)
        IMU_segment_boundaries = build_boundaries(df=df_arm, fixed_len=2800)

        IMU_windows = []
        IMU_labels = []
        FSR_windows = []
        FSR_labels = []

        for _, row in IMU_segment_boundaries.iterrows():
            start_idx = int(row.start_idx)
            end_idx = int(row.end_idx)
            
            label = df_arm['label'].iloc[start_idx]
            rep_id = df_arm['rep_id'].iloc[start_idx]
            
            if str(label) not in str(rep_id):
                print("Rep  ID - label sanity check failed for IMU...")
                print(f"Skipping window with {label}, {rep_id}")
                continue
            
            downsampled_arm_window, arm_label = build_downsampled_window(
                df=df_arm, 
                start_idx=start_idx,
                end_idx=end_idx,
                fixed_len=2800,
                target_num_samples=350)
            
            downsampled_back_window, back_label = build_downsampled_window(
                df=df_back, 
                start_idx=start_idx,
                end_idx=end_idx,
                fixed_len=2800,
                target_num_samples=350)
            
            if downsampled_arm_window is None or downsampled_back_window is None:
                print('None window returned')
                print(label, "<- none window label")
                continue
            
            
            if arm_label == back_label:
                downsampled_window = pd.concat(
                    [downsampled_arm_window,
                    downsampled_back_window], 
                    axis=1
                )
                IMU_windows.append(downsampled_window)
                IMU_labels.append(arm_label) # assuming arm and back are the same
            else:
                print("arm and back label are different!")
                print(f"window with labels {arm_label}, {back_label} will not be added.")
                time.sleep(60)

        for _, row in FSR_segment_boundaries.iterrows():
            start_idx = int(row.start_idx)
            end_idx = int(row.end_idx)
            
            label = df_right['label'].iloc[start_idx]
            rep_id = df_right['rep_id'].iloc[start_idx]
            
            if str(label) not in str(rep_id):
                print("Rep  ID - label sanity check failed for FSR...")
                print(f"Skipping window with {label}, {rep_id}")
                continue
            
            downsampled_left_window, left_label = build_downsampled_window(
                df=df_left, 
                start_idx=start_idx,
                end_idx=end_idx,
                fixed_len=350,
                target_num_samples=350,
                keep_keywords=('fsr',))
            
            downsampled_right_window, right_label = build_downsampled_window(
                df=df_right, 
                start_idx=start_idx,
                end_idx=end_idx,
                fixed_len=350,
                target_num_samples=350,
                keep_keywords=('fsr',))
            
            
            if downsampled_left_window is None or downsampled_right_window is None:
                print('None window returned')
                continue
            
            
            if left_label == right_label:
                downsampled_window = pd.concat(
                    [downsampled_left_window,
                    downsampled_right_window], 
                    axis=1
                )
                downsampled_window['label'] = left_label # assuming labels are the same
                FSR_windows.append(downsampled_window)
                FSR_labels.append(left_label)
            else:
                print("left and right start labels are different!")
                #print(f"window with labels {left_label}, {right_label} will not be added.")

                # last label in the segment
                last_left_label  = df_left['label'].iloc[end_idx - 1]
                last_right_label = df_right['label'].iloc[end_idx - 1]

                print("last left label in segment is", last_left_label)
                print("last right label in segment is", last_right_label)

                # middle index of the segment
                mid_idx = start_idx + (end_idx - start_idx) // 2

                middle_left_label  = df_left['label'].iloc[mid_idx]
                middle_right_label = df_right['label'].iloc[mid_idx]

                print("middle left label is", middle_left_label)
                print("middle right label is", middle_right_label)
                
                if last_left_label == middle_left_label and middle_left_label == right_label:
                    print('RIGHT LABEL most likely:', right_label)
                    downsampled_window = pd.concat(
                    [downsampled_left_window,
                    downsampled_right_window], 
                    axis=1
                    )
                    downsampled_window['label'] = right_label 
                    FSR_windows.append(downsampled_window)
                    FSR_labels.append(right_label)
                    #time.sleep(20)
                elif last_right_label == middle_right_label and middle_right_label == left_label:
                    print('LEFT LABEL most likely:', left_label)
                    downsampled_window = pd.concat(
                    [downsampled_left_window,
                    downsampled_right_window], 
                    axis=1
                    )
                    downsampled_window['label'] = left_label 
                    FSR_windows.append(downsampled_window)
                    FSR_labels.append(left_label)
                else:
                    print("Could not find label consensus:(")
                    print("Skipping window")
                    time.sleep(20)
                
                #     # Create a single concatenated window just for visualization
                # viz_window = pd.concat([downsampled_left_window, downsampled_right_window], axis=1)

                # # Convert to array
                # data = viz_window.values

                # # Standard scale for visualization only
                # scaler = StandardScaler()
                # data_scaled = scaler.fit_transform(data)

                # # Convert to 0–255 grayscale
                # img = data_scaled - data_scaled.min()
                # img = img / img.max()
                # img = (img * 255).astype(np.uint8)

                # plt.imshow(img, cmap='gray', aspect='auto')
                # plt.colorbar()
                # plt.title(f"Left label={left_label}, Right label={right_label}")
                # plt.show()
                # #time.sleep(60)
            
            
        label_counts_IMU = Counter(IMU_labels)
        label_counts_FSR = Counter(FSR_labels)
            
            
        # Get all unique labels from both lists
        all_labels = sorted(set(IMU_labels) | set(IMU_labels)) 

        # Compare label counts and fix mismatches
        for label_value in all_labels:
            imu_count = label_counts_IMU.get(label_value, 0)
            fsr_count = label_counts_FSR.get(label_value, 0)
            print('LABEL COUNTS FOR', label_value)
            print("IMU", imu_count)
            print("FSR", fsr_count)
            
            #time.sleep(4)

            # I NEED SOMETHING OTHER THAN DRP LAST FOR LABEL HERE... 
            if imu_count > fsr_count:
                print(f"Label {label_value}: IMU has {imu_count}, FSR has {fsr_count} → dropping last IMU row")
                IMU_windows, IMU_labels = drop_last_df_for_label(IMU_windows, IMU_labels, label_value)


            elif fsr_count > imu_count:
                print(f"Label {label_value}: FSR has {fsr_count}, IMU has {imu_count} → dropping last FSR row")
                FSR_windows, FSR_labels = drop_last_df_for_label(FSR_windows, FSR_labels, label_value)
                    
            

        #sanity check: lengths of the lists are the same now:
        if len(IMU_windows) == len(FSR_windows):
            print("Sanity length check passed")

        else:
            # do a return here when this is turned into a function
            print("Sanity length check FAILED :((")
            time.sleep(10)
        
        if IMU_labels == FSR_labels:
            print("Label sanity check okay")
        else:
            print("Label sanity check FAILED :((")
            time.sleep(10)

        windows_combined = []
        for i in range(len(IMU_windows)):
            full_window = pd.concat([IMU_windows[i], FSR_windows[i]], axis=1)
            windows_combined.append(full_window)
        
        segments_per_test[test_id] = windows_combined

        # for win in windows_combined:
        #     # Convert to numpy (shape: window_length × num_channels)
        #     window_label = win['label'].iloc[0]
        #     win = win.drop(columns=['label'])
            
            
        #     data = win.values
            
        #     print(data)

        #     # Scale each channel
        #     # scaler = StandardScaler()
        #     # data_scaled = scaler.fit_transform(data)

        #     # Normalize to 0–255 for image
        #     img = data - data.min()
        #     img = img / img.max()
        #     img = (img * 255).astype(np.uint8)
            

        #     img = img.T

        #     plt.imshow(img, cmap='gray', aspect='auto')
        #     plt.colorbar()
        #     plt.title(f"Window as Grayscale Image {window_label}")
        #     plt.show()
    return segments_per_test
        


#segments_dict = create_image_dfs_from_timeseries_segments()

