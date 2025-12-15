import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
"""
from Maria_code.data_analysis.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from Maria_code.data_analysis.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
"""

from get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
from fsr_features import per_sample_features, aggregate_per_window, aggregate_per_rep
from resample_windows import downsample_channel
from create_feature_windows import build_boundaries

'Based on Royas code for extracting IMU features'

def ExtractPressure_Features(fsr_data, sensor_name, window_length, mean_fsr, fs, feature_space='baseline'):
    """
    Extracts features from the FSR data from one side.
    
    Valid feature spaces:
    
    """
    if not isinstance(fsr_data, pd.DataFrame):
        fsr_data = pd.read_csv(fsr_data)
    
    '''EXTRACT COLUMNS'''
    time_data   = fsr_data["ReconstructedTime"]

    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"
    fsr_data = fsr_data[fsr_columns]  # Select only the FSR columns

    # Define a list to store features for each window
    all_window_features = []

    # Calculate the number of windows
    num_samples = len(time_data)
    num_windows = num_samples // window_length
    print(f"Number of fsr  windows: {num_windows}")

    for i in range(num_windows):
        # Define the start and end index for the window
        start_idx = i * window_length
        end_idx = start_idx + window_length
        
        
        if feature_space == 'baseline':
            aggregated_fsr_features = {}
        elif feature_space == 'expanded+baseline':
            fsr_win = fsr_data[start_idx:end_idx]
            per_sample_feature_df = per_sample_features(fsr_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_rep(per_sample_feature_df, sensor_name)
        elif feature_space == 'expanded_only':
            fsr_win = fsr_data[start_idx:end_idx]
            per_sample_feature_df = per_sample_features(fsr_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_rep(per_sample_feature_df, sensor_name)
            window_features = {**aggregated_fsr_features}
        else:
            fsr_win = fsr_data[start_idx:end_idx]
            per_sample_feature_df = per_sample_features(fsr_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_rep(per_sample_feature_df, sensor_name, start_idx, end_idx)
            
        if mean_fsr == False and not feature_space == 'expanded_only':
            sum_fsr = fsr_data.sum(axis=1)

            window_fsr_sum = sum_fsr[start_idx:end_idx]

            window_features_fsr_sum_Time = get_Time_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}")
            window_features_fsr_sum_Freq = get_Freq_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}", fs)
            
            if feature_space == 'time_only':
                window_features = {**window_features_fsr_sum_Time}
            elif feature_space == 'freq_only':
                window_features == {**window_features_fsr_sum_Freq}
            else:
                window_features = {**window_features_fsr_sum_Time, 
                                **window_features_fsr_sum_Freq,
                                **aggregated_fsr_features}

        if mean_fsr == True and not feature_space == 'expanded_only':
            average_fsr = fsr_data.mean(axis=1)

            window_fsr_aver = average_fsr[start_idx:end_idx]
            

            window_features_fsr_aver_Time = get_Time_Domain_features_of_signal(window_fsr_aver, f"fsr_aver_{sensor_name}")
            window_features_fsr_aver_Freq = get_Freq_Domain_features_of_signal(window_fsr_aver, f"fsr_aver_{sensor_name}", fs)
            
            if feature_space == 'time_only':
                window_features = {**window_features_fsr_aver_Time}
            elif feature_space == 'freq_only':
                print("freq only")
                window_features ={**window_features_fsr_aver_Freq}
                print("frequency only wondow features", window_features)
            elif feature_space == 'time_only+exp_FSR':
                window_features = {**window_features_fsr_aver_Time,
                                    **aggregated_fsr_features}
            elif feature_space == 'freq_only+exp_FSR':
                window_features = {**window_features_fsr_aver_Freq,
                                    **aggregated_fsr_features}
            else:
                window_features = {**window_features_fsr_aver_Time, 
                                **window_features_fsr_aver_Freq,
                                **aggregated_fsr_features}

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df
    
    
def ExtractPressure_Features_repetitions_based(fsr_data, sensor_name, mean_fsr, fs, feature_space='baseline', target_num_samples = 350):
    """
    Extracts features from the FSR data from one side.
    
    Valid feature spaces:
    
    """
    if not isinstance(fsr_data, pd.DataFrame):
        fsr_data = pd.read_csv(fsr_data)
    
    print("FSR data cols", fsr_data.columns)
    '''EXTRACT COLUMNS'''
    time_data   = fsr_data["ReconstructedTime"]


    # Define a list to store features for each window
    all_window_features = []
    all_window_labels =[]

    boundaries = build_boundaries(fsr_data,
                                  fixed_len = target_num_samples)
    
    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"
    fsr_data_filtered = fsr_data[fsr_columns]  # Select only the FSR columns

    for _, row in boundaries.iterrows():
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)
        window_label = fsr_data.iloc[start_idx]["label"]
        window_rep_id = fsr_data.iloc[start_idx]['rep_id']
        
        fsr_win = fsr_data_filtered[start_idx:end_idx]
        fsr_win = fsr_win.copy()
        print(f"FSR sanity check [is label consistent with rep_id?]")
        print("checking....")
        if window_label in fsr_data.iloc[start_idx]['rep_id']:
            print('Sanity OKAY:)')
        else:
            print('Sanity NOT OKAY:((')
            print('rep_id', fsr_data.iloc[start_idx]['rep_id'])
            print('label:', window_label)
            time.sleep(60)
        
            
        time_win = time_data[start_idx:end_idx]
        
        # plt.plot(time_win, fsr_win)
        # plt.title(window_label)
        
        # plt.show()
        if len(fsr_win) < target_num_samples:
                print("DROPPING WINDOW THAT IS SHORTER THAN TARGET LENGTH")
                print(f"length is {len(fsr_win)}, rep_id is {fsr_data.iloc[start_idx]['rep_id']}")
                time.sleep(10)
                continue
        all_window_labels.append(window_label)
        ds_win = pd.DataFrame()
        ds_win['label'] = window_label
        ds_win['rep_id'] = window_rep_id
        for col in fsr_win.columns:
            ds_fsr_col = downsample_channel(fsr_win[col], target_num_samples=target_num_samples)
            ds_win[col] = ds_fsr_col

        
        
        if feature_space == 'baseline':
            aggregated_fsr_features = {}
        elif feature_space == 'expanded+baseline':
            per_sample_feature_df = per_sample_features(ds_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_window(per_sample_feature_df, sensor_name, start_idx, end_idx)
            #time.sleep(2)
        elif feature_space == 'expanded_only':
            per_sample_feature_df = per_sample_features(ds_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_window(per_sample_feature_df, sensor_name, start_idx, end_idx)
            window_features = {**aggregated_fsr_features}
        else:
            per_sample_feature_df = per_sample_features(ds_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_window(per_sample_feature_df, sensor_name, start_idx, end_idx)
            
        if mean_fsr == False and not feature_space == 'expanded_only':
            print("Invalid, use mean not sum")
            # sum_fsr = fsr_data_filtered.sum(axis=1)

            # window_fsr_sum = sum_fsr[start_idx:end_idx]

            # window_features_fsr_sum_Time = get_Time_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}")
            # window_features_fsr_sum_Freq = get_Freq_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}", fs)
            
            # if feature_space == 'time_only':
            #     window_features = {**window_features_fsr_sum_Time}
            # elif feature_space == 'freq_only':
            #     window_features == {**window_features_fsr_sum_Freq}
            # else:
            #     window_features = {**window_features_fsr_sum_Time, 
            #                     **window_features_fsr_sum_Freq,
            #                     **aggregated_fsr_features}

        if mean_fsr == True and not feature_space == 'expanded_only':
            average_fsr = fsr_data_filtered.mean(axis=1)

            window_fsr_aver = average_fsr[start_idx:end_idx]
            
            if len(window_fsr_aver) < target_num_samples:
                print("DROPPING WINDOW THAT IS SHORTER THAN TARGET LENGTH")
                print(f"length is {len(window_fsr_aver)}, rep_id is {fsr_data.iloc[start_idx]['rep_id']}")
                time.sleep(10)
                continue
            #all_window_labels.append(window_label)
            t_orig = np.linspace(0, 1, len(window_fsr_aver))
            #plt.plot(t_orig, window_fsr_aver, color='red')
            window_fsr_aver = downsample_channel(window_fsr_aver, target_num_samples=target_num_samples)
            t_ds   = np.linspace(0, 1, len(window_fsr_aver))
            #plt.plot(t_ds, window_fsr_aver, color='blue')
            #plt.title(fsr_data.iloc[start_idx]['rep_id'])
            #plt.show()
            
            


            window_features_fsr_aver_Time = get_Time_Domain_features_of_signal(window_fsr_aver, f"fsr_aver_{sensor_name}")
            window_features_fsr_aver_Freq = get_Freq_Domain_features_of_signal(window_fsr_aver, f"fsr_aver_{sensor_name}", fs)
            
            if feature_space == 'time_only':
                window_features = {**window_features_fsr_aver_Time}
            elif feature_space == 'freq_only':
                print("freq only")
                window_features ={**window_features_fsr_aver_Freq}
                print("frequency only window features", window_features)
            elif feature_space == 'time_only+exp_FSR':
                window_features = {**window_features_fsr_aver_Time,
                                    **aggregated_fsr_features}
            elif feature_space == 'freq_only+exp_FSR':
                window_features = {**window_features_fsr_aver_Freq,
                                    **aggregated_fsr_features}
            else:
                window_features = {**window_features_fsr_aver_Time, 
                                **window_features_fsr_aver_Freq,
                                **aggregated_fsr_features}

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df, all_window_labels

