import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

"""
from Maria_code.data_analysis.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from Maria_code.data_analysis.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
"""
'Based on the bachelor studens modification of Royas code'

from get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
from resample_windows import downsample_channel
from create_feature_windows import build_boundaries

def ExtractIMU_Features(imu_data, sensor_name, window_length, norm_IMU, fs, HDR=False, replace_acc_w_HDR = False, time_only=False, freq_only=False):
    if not isinstance(imu_data, pd.DataFrame):
        imu_data = pd.read_csv(imu_data)
        
    if not HDR and replace_acc_w_HDR:
        print("Trying to replace acceleration with HDR acceleration, but HDR flag is set to false. Set to True for desired beavior.")
        print("IMU feature extraction will continue with settings: HDR = False and replace_acc_w_HDR = False")
        print("Stop feature extraction and change flags to fix this.")

    ''' EXTRACT COLUMNS '''
    time_data   = imu_data["ReconstructedTime"]

    accel_X     = imu_data["Axl.X"]  
    accel_Y     = imu_data["Axl.Y"]  
    accel_Z     = imu_data["Axl.Z"]  

    gyro_X      = imu_data["Gyr.X"]  
    gyro_Y      = imu_data["Gyr.Y"]  
    gyro_Z      = imu_data["Gyr.Z"]  

    mag_X       = imu_data["Mag.X"]  
    mag_Y       = imu_data["Mag.Y"]  
    mag_Z       = imu_data["Mag.Z"]  

    if HDR:
        Hdr_X   = imu_data["Hdr.X"]
        Hdr_Y   = imu_data["Hdr.Y"]
        Hdr_Z   = imu_data["Hdr.Z"]

    # Define a list to store features for each window
    all_window_features = []

    # Calculate the number of windows
    num_samples = len(time_data)
    num_windows = num_samples // window_length
    print(f"Number of IMU windows: {num_windows}")

    for i in range(num_windows):
        # Define the start and end index for the window
        start_idx = i * window_length
        end_idx = start_idx + window_length
        # print(f"Getting features from window {start_idx} to {end_idx}") 
        
        # plt.plot(imu_data['ReconstructedTime'][start_idx:end_idx], imu_data["Axl.X"][start_idx:end_idx])
        # plt.plot(imu_data['ReconstructedTime'][start_idx:end_idx], imu_data["Axl.Y"][start_idx:end_idx])
        # plt.plot(imu_data['ReconstructedTime'][start_idx:end_idx], imu_data["Axl.Z"][start_idx:end_idx])
        # plt.title(f"{imu_data.iloc[start_idx]['label']}")
        # plt.show()
        if norm_IMU == True:
            # Extract acceleration and gyroscope data from the IMU dataset
            if replace_acc_w_HDR and HDR:
                norm_gyro   = np.sqrt( np.power(gyro_X, 2)  + np.power(gyro_Y, 2)  + np.power(gyro_Z, 2))
                norm_mag    = np.sqrt( np.power(mag_X, 2)   + np.power(mag_Y, 2)   + np.power(mag_Z, 2))
                norm_hdr = np.sqrt( np.power(Hdr_X, 2)   + np.power(Hdr_Y, 2)   + np.power(Hdr_Z, 2))
                
                window_gyro_Norm    = norm_gyro[start_idx:end_idx]
                window_mag_Norm     = norm_mag[start_idx:end_idx]
                window_hdr_Norm = norm_hdr[start_idx:end_idx]  
                
                window_features_gyro_Norm_Time  = get_Time_Domain_features_of_signal(window_gyro_Norm, f"gyro_Norm_{sensor_name}")
                window_features_mag_Norm_Time   = get_Time_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}")
                window_features_hdr_Norm_Time   = get_Time_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}")
                
                window_features_gyro_Norm_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Norm, f"_gyro_Norm_{sensor_name}", fs)
                window_features_mag_Norm_Freq   = get_Freq_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}", fs)
                window_features_hdr_Norm_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}", fs)
                
                if time_only:
                    window_features = {
                                **window_features_gyro_Norm_Time,
                                **window_features_gyro_Norm_Freq,
                                **window_features_mag_Norm_Time
                                }
                elif freq_only:
                    window_features = {
                        **window_features_mag_Norm_Freq,
                                **window_features_hdr_Norm_Time,
                                **window_features_hdr_Norm_Freq
                    }
                else:
                    window_features = {
                                    **window_features_gyro_Norm_Time,
                                    **window_features_gyro_Norm_Freq,
                                    **window_features_mag_Norm_Time,
                                    **window_features_mag_Norm_Freq,
                                    **window_features_hdr_Norm_Time,
                                    **window_features_hdr_Norm_Freq
                                    }
            
            else:
                norm_accel  = np.sqrt( np.power(accel_X, 2) + np.power(accel_Y, 2) + np.power(accel_Z, 2))
                norm_gyro   = np.sqrt( np.power(gyro_X, 2)  + np.power(gyro_Y, 2)  + np.power(gyro_Z, 2))
                norm_mag    = np.sqrt( np.power(mag_X, 2)   + np.power(mag_Y, 2)   + np.power(mag_Z, 2))
                if HDR:
                    norm_hdr = np.sqrt( np.power(Hdr_X, 2)   + np.power(Hdr_Y, 2)   + np.power(Hdr_Z, 2))

                # Remove gravity:
                '''
                g_constant = np.mean(norm_acceleration)
                # print(f"g constant: {g_constant}")
                gravless_norm = np.subtract(norm_acceleration, g_constant)  
                window_accel_Norm = gravless_norm[start_idx:end_idx]
                '''

                window_accel_Norm   = norm_accel[start_idx:end_idx]
                window_gyro_Norm    = norm_gyro[start_idx:end_idx]
                window_mag_Norm     = norm_mag[start_idx:end_idx]
                if HDR:
                    window_hdr_Norm = norm_hdr[start_idx:end_idx]        

                window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, f"accel_Norm_{sensor_name}")
                window_features_gyro_Norm_Time  = get_Time_Domain_features_of_signal(window_gyro_Norm, f"gyro_Norm_{sensor_name}")
                window_features_mag_Norm_Time   = get_Time_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}")
                if HDR:
                    window_features_hdr_Norm_Time   = get_Time_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}")

                window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, f"accel_Norm_{sensor_name}", fs)
                window_features_gyro_Norm_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Norm, f"_gyro_Norm_{sensor_name}", fs)
                window_features_mag_Norm_Freq   = get_Freq_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}", fs)
                if HDR:
                    window_features_hdr_Norm_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}", fs)

                ## merge all
                if not HDR:
                    window_features = {**window_features_accel_Norm_Time, 
                                    **window_features_accel_Norm_Freq,
                                    **window_features_gyro_Norm_Time,
                                    **window_features_gyro_Norm_Freq,
                                    **window_features_mag_Norm_Time,
                                    **window_features_mag_Norm_Freq,
                                    }
                if HDR:
                    window_features = {**window_features_accel_Norm_Time, 
                                    **window_features_accel_Norm_Freq,
                                    **window_features_gyro_Norm_Time,
                                    **window_features_gyro_Norm_Freq,
                                    **window_features_mag_Norm_Time,
                                    **window_features_mag_Norm_Freq,
                                    **window_features_hdr_Norm_Time,
                                    **window_features_hdr_Norm_Freq
                                    }

            

        if norm_IMU == False: 
            if HDR and replace_acc_w_HDR:
                window_gyro_X  = gyro_X[start_idx:end_idx]
                window_gyro_Y  = gyro_Y[start_idx:end_idx]
                window_gyro_Z  = gyro_Z[start_idx:end_idx]

                window_mag_X   = mag_X[start_idx:end_idx]
                window_mag_Y   = mag_Y[start_idx:end_idx]
                window_mag_Z   = mag_Z[start_idx:end_idx]

                window_hdr_X   = Hdr_X[start_idx:end_idx]
                window_hdr_Y   = Hdr_Y[start_idx:end_idx]
                window_hdr_Z   = Hdr_Z[start_idx:end_idx]
                
                window_features_gyro_X_Time  = get_Time_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}")
                window_features_gyro_Y_Time  = get_Time_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}")
                window_features_gyro_Z_Time  = get_Time_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}")

                window_features_mag_X_Time   = get_Time_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}")
                window_features_mag_Y_Time   = get_Time_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}")
                window_features_mag_Z_Time   = get_Time_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}")

                window_features_hdr_X_Time   = get_Time_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}")
                window_features_hdr_Y_Time   = get_Time_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}")
                window_features_hdr_Z_Time   = get_Time_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}")

                window_features_gyro_X_Freq  = get_Freq_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}", fs)
                window_features_gyro_Y_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}", fs)
                window_features_gyro_Z_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}", fs)

                window_features_mag_X_Freq   = get_Freq_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}", fs)
                window_features_mag_Y_Freq   = get_Freq_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}", fs)
                window_features_mag_Z_Freq   = get_Freq_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}", fs)

                window_features_hdr_X_Freq   = get_Freq_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}", fs)
                window_features_hdr_Y_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}", fs)
                window_features_hdr_Z_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}", fs)
                
                if time_only:
                    window_features = {
                        **window_features_gyro_X_Time,
                        **window_features_gyro_Y_Time,
                        **window_features_gyro_Z_Time,
                        **window_features_mag_X_Time,
                        **window_features_mag_Y_Time,
                        **window_features_mag_Z_Time,
                        **window_features_hdr_X_Time,
                        **window_features_hdr_Y_Time,
                        **window_features_hdr_Z_Time,
                    }
                elif freq_only:
                    window_features = {
                        **window_features_gyro_X_Freq,
                        **window_features_gyro_Y_Freq,
                        **window_features_gyro_Z_Freq,
                        **window_features_mag_X_Freq,
                        **window_features_mag_Y_Freq,
                        **window_features_mag_Z_Freq,
                        **window_features_hdr_X_Freq,
                        **window_features_hdr_Y_Freq,
                        **window_features_hdr_Z_Freq
                    }
                
                else:
                    window_features = {
                                        **window_features_gyro_X_Time,
                                        **window_features_gyro_Y_Time,
                                        **window_features_gyro_Z_Time,
                                        **window_features_gyro_X_Freq,
                                        **window_features_gyro_Y_Freq,
                                        **window_features_gyro_Z_Freq,
                                        **window_features_mag_X_Time,
                                        **window_features_mag_Y_Time,
                                        **window_features_mag_Z_Time,
                                        **window_features_mag_X_Freq,
                                        **window_features_mag_Y_Freq,
                                        **window_features_mag_Z_Freq,
                                        **window_features_hdr_X_Time,
                                        **window_features_hdr_Y_Time,
                                        **window_features_hdr_Z_Time,
                                        **window_features_hdr_X_Freq,
                                        **window_features_hdr_Y_Freq,
                                        **window_features_hdr_Z_Freq
                                        }
                
            
            
            else:  
                window_accel_X = accel_X[start_idx:end_idx]
                window_accel_Y = accel_Y[start_idx:end_idx]
                window_accel_Z = accel_Z[start_idx:end_idx]

                window_gyro_X  = gyro_X[start_idx:end_idx]
                window_gyro_Y  = gyro_Y[start_idx:end_idx]
                window_gyro_Z  = gyro_Z[start_idx:end_idx]

                window_mag_X   = mag_X[start_idx:end_idx]
                window_mag_Y   = mag_Y[start_idx:end_idx]
                window_mag_Z   = mag_Z[start_idx:end_idx]

                if HDR:
                    window_hdr_X   = Hdr_X[start_idx:end_idx]
                    window_hdr_Y   = Hdr_Y[start_idx:end_idx]
                    window_hdr_Z   = Hdr_Z[start_idx:end_idx]


                window_features_accel_X_Time = get_Time_Domain_features_of_signal(window_accel_X, f"accel_X_{sensor_name}")
                window_features_accel_Y_Time = get_Time_Domain_features_of_signal(window_accel_Y, f"accel_Y_{sensor_name}")
                window_features_accel_Z_Time = get_Time_Domain_features_of_signal(window_accel_Z, f"accel_Z_{sensor_name}")

                window_features_gyro_X_Time  = get_Time_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}")
                window_features_gyro_Y_Time  = get_Time_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}")
                window_features_gyro_Z_Time  = get_Time_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}")

                window_features_mag_X_Time   = get_Time_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}")
                window_features_mag_Y_Time   = get_Time_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}")
                window_features_mag_Z_Time   = get_Time_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}")

                if HDR:
                    window_features_hdr_X_Time   = get_Time_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}")
                    window_features_hdr_Y_Time   = get_Time_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}")
                    window_features_hdr_Z_Time   = get_Time_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}")

                window_features_accel_X_Freq = get_Freq_Domain_features_of_signal(window_accel_X, f"accel_X_{sensor_name}", fs)
                window_features_accel_Y_Freq = get_Freq_Domain_features_of_signal(window_accel_Y, f"accel_Y_{sensor_name}", fs)
                window_features_accel_Z_Freq = get_Freq_Domain_features_of_signal(window_accel_Z, f"accel_Z_{sensor_name}", fs)

                window_features_gyro_X_Freq  = get_Freq_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}", fs)
                window_features_gyro_Y_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}", fs)
                window_features_gyro_Z_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}", fs)

                window_features_mag_X_Freq   = get_Freq_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}", fs)
                window_features_mag_Y_Freq   = get_Freq_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}", fs)
                window_features_mag_Z_Freq   = get_Freq_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}", fs)

                if HDR:
                    window_features_hdr_X_Freq   = get_Freq_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}", fs)
                    window_features_hdr_Y_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}", fs)
                    window_features_hdr_Z_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}", fs)


                ## merge all
                if not HDR:
                    if time_only:
                        window_features = {
                            **window_features_accel_X_Time, 
                            **window_features_accel_Y_Time,
                            **window_features_accel_Z_Time,
                            **window_features_gyro_X_Time,
                            **window_features_gyro_Y_Time,
                            **window_features_gyro_Z_Time,
                            **window_features_mag_X_Time,
                            **window_features_mag_Y_Time,
                            **window_features_mag_Z_Time,
                        }
                    
                    elif freq_only:
                        window_features =   {
                            **window_features_accel_X_Freq,
                            **window_features_accel_Y_Freq,
                            **window_features_accel_Z_Freq,
                            **window_features_gyro_X_Freq,
                            **window_features_gyro_Y_Freq,
                            **window_features_gyro_Z_Freq,
                            **window_features_mag_X_Freq,
                            **window_features_mag_Y_Freq,
                            **window_features_mag_Z_Freq
                        }
                        
                        
                    else:
                        window_features = {**window_features_accel_X_Time, 
                                        **window_features_accel_Y_Time,
                                        **window_features_accel_Z_Time,
                                        **window_features_accel_X_Freq,
                                        **window_features_accel_Y_Freq,
                                        **window_features_accel_Z_Freq,
                                        **window_features_gyro_X_Time,
                                        **window_features_gyro_Y_Time,
                                        **window_features_gyro_Z_Time,
                                        **window_features_gyro_X_Freq,
                                        **window_features_gyro_Y_Freq,
                                        **window_features_gyro_Z_Freq,
                                        **window_features_mag_X_Time,
                                        **window_features_mag_Y_Time,
                                        **window_features_mag_Z_Time,
                                        **window_features_mag_X_Freq,
                                        **window_features_mag_Y_Freq,
                                        **window_features_mag_Z_Freq
                                        }
                if HDR:
                    if time_only:
                        window_features = {
                            **window_features_accel_X_Time, 
                            **window_features_accel_Y_Time,
                            **window_features_accel_Z_Time,
                            **window_features_gyro_X_Time,
                            **window_features_gyro_Y_Time,
                            **window_features_gyro_Z_Time,
                            **window_features_mag_X_Time,
                            **window_features_mag_Y_Time,
                            **window_features_mag_Z_Time,
                            **window_features_hdr_X_Time,
                            **window_features_hdr_Y_Time,
                            **window_features_hdr_Z_Time,
                        }
                    elif freq_only:
                        window_features = {
                            **window_features_accel_X_Freq, 
                            **window_features_accel_Y_Freq,
                            **window_features_accel_Z_Freq,
                            **window_features_gyro_X_Freq,
                            **window_features_gyro_Y_Freq,
                            **window_features_gyro_Z_Freq,
                            **window_features_mag_X_Freq,
                            **window_features_mag_Y_Freq,
                            **window_features_mag_Z_Freq,
                            **window_features_hdr_X_Freq,
                            **window_features_hdr_Y_Freq,
                            **window_features_hdr_Z_Freq,
                        }
                    else:
                        window_features = {**window_features_accel_X_Time, 
                                        **window_features_accel_Y_Time,
                                        **window_features_accel_Z_Time,
                                        **window_features_accel_X_Freq,
                                        **window_features_accel_Y_Freq,
                                        **window_features_accel_Z_Freq,
                                        **window_features_gyro_X_Time,
                                        **window_features_gyro_Y_Time,
                                        **window_features_gyro_Z_Time,
                                        **window_features_gyro_X_Freq,
                                        **window_features_gyro_Y_Freq,
                                        **window_features_gyro_Z_Freq,
                                        **window_features_mag_X_Time,
                                        **window_features_mag_Y_Time,
                                        **window_features_mag_Z_Time,
                                        **window_features_mag_X_Freq,
                                        **window_features_mag_Y_Freq,
                                        **window_features_mag_Z_Freq,
                                        **window_features_hdr_X_Time,
                                        **window_features_hdr_Y_Time,
                                        **window_features_hdr_Z_Time,
                                        **window_features_hdr_X_Freq,
                                        **window_features_hdr_Y_Freq,
                                        **window_features_hdr_Z_Freq
                                        }

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df



def ExtractIMU_features_repetitions_based(imu_data, sensor_name, window_length, norm_IMU, fs, HDR=False, replace_acc_w_HDR = False, time_only=False, freq_only=False, target_num_samples = 2800):
    if not isinstance(imu_data, pd.DataFrame):
        imu_data = pd.read_csv(imu_data)
        
    if not HDR and replace_acc_w_HDR:
        print("Trying to replace acceleration with HDR acceleration, but HDR flag is set to false. Set to True for desired beavior.")
        print("IMU feature extraction will continue with settings: HDR = False and replace_acc_w_HDR = False")
        print("Stop feature extraction and change flags to fix this.")

    ''' EXTRACT COLUMNS '''
    time_data   = imu_data["ReconstructedTime"]

    accel_X     = imu_data["Axl.X"]  
    accel_Y     = imu_data["Axl.Y"]  
    accel_Z     = imu_data["Axl.Z"]  

    gyro_X      = imu_data["Gyr.X"]  
    gyro_Y      = imu_data["Gyr.Y"]  
    gyro_Z      = imu_data["Gyr.Z"]  

    mag_X       = imu_data["Mag.X"]  
    mag_Y       = imu_data["Mag.Y"]  
    mag_Z       = imu_data["Mag.Z"]  

    if HDR:
        Hdr_X   = imu_data["Hdr.X"]
        Hdr_Y   = imu_data["Hdr.Y"]
        Hdr_Z   = imu_data["Hdr.Z"]

    # Define a list to store features for each window
    all_window_features = []
    all_window_labels = []

    # Calculate the number of windows
    # rep_ids = imu_data["rep_id"].fillna("none").to_numpy()
    # # Find where rep_id changes (this marks a new repetition)
    # change_points = np.where(rep_ids[:-1] != rep_ids[1:])[0] + 1
    # boundaries = np.concatenate(([0], change_points, [len(rep_ids)])) # include start and end
    
    boundaries = build_boundaries(imu_data,
                                  fixed_len = target_num_samples) # crea

    for _, row in boundaries.iterrows():
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)
        window_label = imu_data.iloc[start_idx]["label"]
        
        print(f"Getting features from window {start_idx} to {end_idx}") 
        print("Performing sanity check: [is label is consistent with repetition ID?]")
        print(".....")
        if window_label in imu_data.iloc[start_idx]['rep_id']:
            print("Sanity passed")
        else:
            print(f"Rep id is {imu_data.iloc[start_idx]['rep_id']}, while label is {window_label}")
            print('Sleeping for 60 seconds')
            time.sleep(60)
        
        # plt.plot(imu_data['ReconstructedTime'][start_idx:end_idx], imu_data["Axl.X"][start_idx:end_idx])
        # plt.plot(imu_data['ReconstructedTime'][start_idx:end_idx], imu_data["Axl.Y"][start_idx:end_idx])
        # plt.plot(imu_data['ReconstructedTime'][start_idx:end_idx], imu_data["Axl.Z"][start_idx:end_idx])
        # plt.title(f"{imu_data.iloc[start_idx]['label']}")
        # plt.show()
        if norm_IMU == True:
            # Extract acceleration and gyroscope data from the IMU dataset
            if replace_acc_w_HDR and HDR:
                norm_gyro   = np.sqrt( np.power(gyro_X, 2)  + np.power(gyro_Y, 2)  + np.power(gyro_Z, 2))
                norm_mag    = np.sqrt( np.power(mag_X, 2)   + np.power(mag_Y, 2)   + np.power(mag_Z, 2))
                norm_hdr = np.sqrt( np.power(Hdr_X, 2)   + np.power(Hdr_Y, 2)   + np.power(Hdr_Z, 2))
                
                window_gyro_Norm    = norm_gyro[start_idx:end_idx]
                window_mag_Norm     = norm_mag[start_idx:end_idx]
                window_hdr_Norm = norm_hdr[start_idx:end_idx]  
                
                window_features_gyro_Norm_Time  = get_Time_Domain_features_of_signal(window_gyro_Norm, f"gyro_Norm_{sensor_name}")
                window_features_mag_Norm_Time   = get_Time_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}")
                window_features_hdr_Norm_Time   = get_Time_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}")
                
                window_features_gyro_Norm_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Norm, f"_gyro_Norm_{sensor_name}", fs)
                window_features_mag_Norm_Freq   = get_Freq_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}", fs)
                window_features_hdr_Norm_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}", fs)
                
                if time_only:
                    window_features = {
                                **window_features_gyro_Norm_Time,
                                **window_features_gyro_Norm_Freq,
                                **window_features_mag_Norm_Time
                                }
                elif freq_only:
                    window_features = {
                        **window_features_mag_Norm_Freq,
                                **window_features_hdr_Norm_Time,
                                **window_features_hdr_Norm_Freq
                    }
                else:
                    window_features = {
                                    **window_features_gyro_Norm_Time,
                                    **window_features_gyro_Norm_Freq,
                                    **window_features_mag_Norm_Time,
                                    **window_features_mag_Norm_Freq,
                                    **window_features_hdr_Norm_Time,
                                    **window_features_hdr_Norm_Freq
                                    }
            
            else:
                norm_accel  = np.sqrt( np.power(accel_X, 2) + np.power(accel_Y, 2) + np.power(accel_Z, 2))
                norm_gyro   = np.sqrt( np.power(gyro_X, 2)  + np.power(gyro_Y, 2)  + np.power(gyro_Z, 2))
                norm_mag    = np.sqrt( np.power(mag_X, 2)   + np.power(mag_Y, 2)   + np.power(mag_Z, 2))
                if HDR:
                    norm_hdr = np.sqrt( np.power(Hdr_X, 2)   + np.power(Hdr_Y, 2)   + np.power(Hdr_Z, 2))

                # Remove gravity:
                '''
                g_constant = np.mean(norm_acceleration)
                # print(f"g constant: {g_constant}")
                gravless_norm = np.subtract(norm_acceleration, g_constant)  
                window_accel_Norm = gravless_norm[start_idx:end_idx]
                '''

                window_accel_Norm   = norm_accel[start_idx:end_idx]
                window_gyro_Norm    = norm_gyro[start_idx:end_idx]
                window_mag_Norm     = norm_mag[start_idx:end_idx]
                if HDR:
                    window_hdr_Norm = norm_hdr[start_idx:end_idx]        

                window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, f"accel_Norm_{sensor_name}")
                window_features_gyro_Norm_Time  = get_Time_Domain_features_of_signal(window_gyro_Norm, f"gyro_Norm_{sensor_name}")
                window_features_mag_Norm_Time   = get_Time_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}")
                if HDR:
                    window_features_hdr_Norm_Time   = get_Time_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}")

                window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, f"accel_Norm_{sensor_name}", fs)
                window_features_gyro_Norm_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Norm, f"_gyro_Norm_{sensor_name}", fs)
                window_features_mag_Norm_Freq   = get_Freq_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}", fs)
                if HDR:
                    window_features_hdr_Norm_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}", fs)

                ## merge all
                if not HDR:
                    window_features = {**window_features_accel_Norm_Time, 
                                    **window_features_accel_Norm_Freq,
                                    **window_features_gyro_Norm_Time,
                                    **window_features_gyro_Norm_Freq,
                                    **window_features_mag_Norm_Time,
                                    **window_features_mag_Norm_Freq,
                                    }
                if HDR:
                    window_features = {**window_features_accel_Norm_Time, 
                                    **window_features_accel_Norm_Freq,
                                    **window_features_gyro_Norm_Time,
                                    **window_features_gyro_Norm_Freq,
                                    **window_features_mag_Norm_Time,
                                    **window_features_mag_Norm_Freq,
                                    **window_features_hdr_Norm_Time,
                                    **window_features_hdr_Norm_Freq
                                    }

            

        if norm_IMU == False: 
            if HDR and replace_acc_w_HDR:
                window_gyro_X  = gyro_X[start_idx:end_idx]
                window_gyro_Y  = gyro_Y[start_idx:end_idx]
                window_gyro_Z  = gyro_Z[start_idx:end_idx]

                window_mag_X   = mag_X[start_idx:end_idx]
                window_mag_Y   = mag_Y[start_idx:end_idx]
                window_mag_Z   = mag_Z[start_idx:end_idx]

                window_hdr_X   = Hdr_X[start_idx:end_idx]
                window_hdr_Y   = Hdr_Y[start_idx:end_idx]
                window_hdr_Z   = Hdr_Z[start_idx:end_idx]
                
                window_features_gyro_X_Time  = get_Time_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}")
                window_features_gyro_Y_Time  = get_Time_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}")
                window_features_gyro_Z_Time  = get_Time_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}")

                window_features_mag_X_Time   = get_Time_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}")
                window_features_mag_Y_Time   = get_Time_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}")
                window_features_mag_Z_Time   = get_Time_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}")

                window_features_hdr_X_Time   = get_Time_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}")
                window_features_hdr_Y_Time   = get_Time_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}")
                window_features_hdr_Z_Time   = get_Time_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}")

                window_features_gyro_X_Freq  = get_Freq_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}", fs)
                window_features_gyro_Y_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}", fs)
                window_features_gyro_Z_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}", fs)

                window_features_mag_X_Freq   = get_Freq_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}", fs)
                window_features_mag_Y_Freq   = get_Freq_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}", fs)
                window_features_mag_Z_Freq   = get_Freq_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}", fs)

                window_features_hdr_X_Freq   = get_Freq_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}", fs)
                window_features_hdr_Y_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}", fs)
                window_features_hdr_Z_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}", fs)
                
                if time_only:
                    window_features = {
                        **window_features_gyro_X_Time,
                        **window_features_gyro_Y_Time,
                        **window_features_gyro_Z_Time,
                        **window_features_mag_X_Time,
                        **window_features_mag_Y_Time,
                        **window_features_mag_Z_Time,
                        **window_features_hdr_X_Time,
                        **window_features_hdr_Y_Time,
                        **window_features_hdr_Z_Time,
                    }
                elif freq_only:
                    window_features = {
                        **window_features_gyro_X_Freq,
                        **window_features_gyro_Y_Freq,
                        **window_features_gyro_Z_Freq,
                        **window_features_mag_X_Freq,
                        **window_features_mag_Y_Freq,
                        **window_features_mag_Z_Freq,
                        **window_features_hdr_X_Freq,
                        **window_features_hdr_Y_Freq,
                        **window_features_hdr_Z_Freq
                    }
                
                else:
                    window_features = {
                                        **window_features_gyro_X_Time,
                                        **window_features_gyro_Y_Time,
                                        **window_features_gyro_Z_Time,
                                        **window_features_gyro_X_Freq,
                                        **window_features_gyro_Y_Freq,
                                        **window_features_gyro_Z_Freq,
                                        **window_features_mag_X_Time,
                                        **window_features_mag_Y_Time,
                                        **window_features_mag_Z_Time,
                                        **window_features_mag_X_Freq,
                                        **window_features_mag_Y_Freq,
                                        **window_features_mag_Z_Freq,
                                        **window_features_hdr_X_Time,
                                        **window_features_hdr_Y_Time,
                                        **window_features_hdr_Z_Time,
                                        **window_features_hdr_X_Freq,
                                        **window_features_hdr_Y_Freq,
                                        **window_features_hdr_Z_Freq
                                        }
                
            
            
            else:  
                window_accel_X = accel_X[start_idx:end_idx]
                window_accel_Y = accel_Y[start_idx:end_idx]
                window_accel_Z = accel_Z[start_idx:end_idx]

                window_gyro_X  = gyro_X[start_idx:end_idx]
                window_gyro_Y  = gyro_Y[start_idx:end_idx]
                window_gyro_Z  = gyro_Z[start_idx:end_idx]

                window_mag_X   = mag_X[start_idx:end_idx]
                window_mag_Y   = mag_Y[start_idx:end_idx]
                window_mag_Z   = mag_Z[start_idx:end_idx]

                if HDR:
                    window_hdr_X   = Hdr_X[start_idx:end_idx]
                    window_hdr_Y   = Hdr_Y[start_idx:end_idx]
                    window_hdr_Z   = Hdr_Z[start_idx:end_idx]
                    
                # Code for fixing window length to a desired number of samples
                
                if len(window_accel_X) < target_num_samples:
                    # drop the window, assuming the windows are all the same length, as they should be
                    print(f"Dropping window with number of samples {len(window_accel_X)}")
                    time.sleep(10)
                    continue
                else:
                    all_window_labels.append(window_label)
                    # plt.plot(window_accel_X)
                    
                    window_accel_X = downsample_channel(window_accel_X, target_num_samples=target_num_samples)
                    window_accel_Y = downsample_channel(window_accel_Y, target_num_samples=target_num_samples)
                    window_accel_Z = downsample_channel(window_accel_Z, target_num_samples=target_num_samples)
                    
                    window_gyro_X = downsample_channel(window_gyro_X, target_num_samples=target_num_samples)
                    window_gyro_Y = downsample_channel(window_gyro_Y, target_num_samples=target_num_samples)
                    window_gyro_Z= downsample_channel(window_gyro_Z, target_num_samples=target_num_samples)
                    
                    window_mag_X = downsample_channel(window_mag_X, target_num_samples=target_num_samples)
                    window_mag_Y = downsample_channel(window_mag_Y, target_num_samples=target_num_samples)
                    window_mag_Z = downsample_channel(window_mag_Z, target_num_samples=target_num_samples)
                    
                    # plt.plot(window_accel_X)
                    # plt.show()
                    
                    if HDR:
                        window_hdr_X   = downsample_channel(window_hdr_X, target_num_samples=target_num_samples)
                        window_hdr_Y   = downsample_channel(window_hdr_Y, target_num_samples=target_num_samples)
                        window_hdr_Z   = downsample_channel(window_hdr_Z, target_num_samples=target_num_samples)
                    
                window_features_accel_X_Time = get_Time_Domain_features_of_signal(window_accel_X, f"accel_X_{sensor_name}")
                window_features_accel_Y_Time = get_Time_Domain_features_of_signal(window_accel_Y, f"accel_Y_{sensor_name}")
                window_features_accel_Z_Time = get_Time_Domain_features_of_signal(window_accel_Z, f"accel_Z_{sensor_name}")

                window_features_gyro_X_Time  = get_Time_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}")
                window_features_gyro_Y_Time  = get_Time_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}")
                window_features_gyro_Z_Time  = get_Time_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}")

                window_features_mag_X_Time   = get_Time_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}")
                window_features_mag_Y_Time   = get_Time_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}")
                window_features_mag_Z_Time   = get_Time_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}")

                if HDR:
                    window_features_hdr_X_Time   = get_Time_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}")
                    window_features_hdr_Y_Time   = get_Time_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}")
                    window_features_hdr_Z_Time   = get_Time_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}")

                window_features_accel_X_Freq = get_Freq_Domain_features_of_signal(window_accel_X, f"accel_X_{sensor_name}", fs)
                window_features_accel_Y_Freq = get_Freq_Domain_features_of_signal(window_accel_Y, f"accel_Y_{sensor_name}", fs)
                window_features_accel_Z_Freq = get_Freq_Domain_features_of_signal(window_accel_Z, f"accel_Z_{sensor_name}", fs)

                window_features_gyro_X_Freq  = get_Freq_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}", fs)
                window_features_gyro_Y_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}", fs)
                window_features_gyro_Z_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}", fs)

                window_features_mag_X_Freq   = get_Freq_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}", fs)
                window_features_mag_Y_Freq   = get_Freq_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}", fs)
                window_features_mag_Z_Freq   = get_Freq_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}", fs)

                if HDR:
                    window_features_hdr_X_Freq   = get_Freq_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}", fs)
                    window_features_hdr_Y_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}", fs)
                    window_features_hdr_Z_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}", fs)


                ## merge all
                if not HDR:
                    if time_only:
                        window_features = {
                            **window_features_accel_X_Time, 
                            **window_features_accel_Y_Time,
                            **window_features_accel_Z_Time,
                            **window_features_gyro_X_Time,
                            **window_features_gyro_Y_Time,
                            **window_features_gyro_Z_Time,
                            **window_features_mag_X_Time,
                            **window_features_mag_Y_Time,
                            **window_features_mag_Z_Time,
                        }
                    
                    elif freq_only:
                        window_features =   {
                            **window_features_accel_X_Freq,
                            **window_features_accel_Y_Freq,
                            **window_features_accel_Z_Freq,
                            **window_features_gyro_X_Freq,
                            **window_features_gyro_Y_Freq,
                            **window_features_gyro_Z_Freq,
                            **window_features_mag_X_Freq,
                            **window_features_mag_Y_Freq,
                            **window_features_mag_Z_Freq
                        }
                        
                        
                    else:
                        window_features = {**window_features_accel_X_Time, 
                                        **window_features_accel_Y_Time,
                                        **window_features_accel_Z_Time,
                                        **window_features_accel_X_Freq,
                                        **window_features_accel_Y_Freq,
                                        **window_features_accel_Z_Freq,
                                        **window_features_gyro_X_Time,
                                        **window_features_gyro_Y_Time,
                                        **window_features_gyro_Z_Time,
                                        **window_features_gyro_X_Freq,
                                        **window_features_gyro_Y_Freq,
                                        **window_features_gyro_Z_Freq,
                                        **window_features_mag_X_Time,
                                        **window_features_mag_Y_Time,
                                        **window_features_mag_Z_Time,
                                        **window_features_mag_X_Freq,
                                        **window_features_mag_Y_Freq,
                                        **window_features_mag_Z_Freq
                                        }
                if HDR:
                    if time_only:
                        window_features = {
                            **window_features_accel_X_Time, 
                            **window_features_accel_Y_Time,
                            **window_features_accel_Z_Time,
                            **window_features_gyro_X_Time,
                            **window_features_gyro_Y_Time,
                            **window_features_gyro_Z_Time,
                            **window_features_mag_X_Time,
                            **window_features_mag_Y_Time,
                            **window_features_mag_Z_Time,
                            **window_features_hdr_X_Time,
                            **window_features_hdr_Y_Time,
                            **window_features_hdr_Z_Time,
                        }
                    elif freq_only:
                        window_features = {
                            **window_features_accel_X_Freq, 
                            **window_features_accel_Y_Freq,
                            **window_features_accel_Z_Freq,
                            **window_features_gyro_X_Freq,
                            **window_features_gyro_Y_Freq,
                            **window_features_gyro_Z_Freq,
                            **window_features_mag_X_Freq,
                            **window_features_mag_Y_Freq,
                            **window_features_mag_Z_Freq,
                            **window_features_hdr_X_Freq,
                            **window_features_hdr_Y_Freq,
                            **window_features_hdr_Z_Freq,
                        }
                    else:
                        window_features = {**window_features_accel_X_Time, 
                                        **window_features_accel_Y_Time,
                                        **window_features_accel_Z_Time,
                                        **window_features_accel_X_Freq,
                                        **window_features_accel_Y_Freq,
                                        **window_features_accel_Z_Freq,
                                        **window_features_gyro_X_Time,
                                        **window_features_gyro_Y_Time,
                                        **window_features_gyro_Z_Time,
                                        **window_features_gyro_X_Freq,
                                        **window_features_gyro_Y_Freq,
                                        **window_features_gyro_Z_Freq,
                                        **window_features_mag_X_Time,
                                        **window_features_mag_Y_Time,
                                        **window_features_mag_Z_Time,
                                        **window_features_mag_X_Freq,
                                        **window_features_mag_Y_Freq,
                                        **window_features_mag_Z_Freq,
                                        **window_features_hdr_X_Time,
                                        **window_features_hdr_Y_Time,
                                        **window_features_hdr_Z_Time,
                                        **window_features_hdr_X_Freq,
                                        **window_features_hdr_Y_Freq,
                                        **window_features_hdr_Z_Freq
                                        }

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df, all_window_labels
    