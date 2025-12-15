import os

def get_test_file_paths():
    tests = {
        "test_1": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_1\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_1\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_1\mitch_B0510-new_left_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_1\mitch_B0308-old_right_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_2": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_2\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_2\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_2\mitch_B0510-new_left_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_2\mitch_B0308-old_right_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_3": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_3\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_3\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_3\mitch_B0510-new_right_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_3\mitch_B0308-old_left_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_4": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_4\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_4\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_4\mitch_B0308-old_left_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_4\mitch_B0510-new_right_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_5": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_5\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_5\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_5\mitch_B0510-new_left_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_5\mitch_B0308-old_right_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_6": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_6\Muse_E2511_RED-upper_arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_6\Muse_E2511_GREY-lower_back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_6\mitch_B0308-left_old_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_6\mitch_B0510-right_new_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_7": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_7\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_7\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_7\mitch_B0510-new_left_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_7\mitch_B0308-old_right_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_8": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_8\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_8\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_8\mitch_B0308-old_right_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_8\mitch_B0510-new_left_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_9": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_9\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_9\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_9\mitch_B0510-new_left_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_9\mitch_B0308-old_right_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_10": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_10\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_10\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_10\mitch_B0510-new_left_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_10\mitch_B0308-old_right_small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_11": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_11\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_11\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_11\mitch_B0308-old_left_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_11\mitch_B0510-new_right_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        },
        "test_12": {
            "arm": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_12\Muse_E2511_RED-upper_arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_12\Muse_E2511_GREY-lower_back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_12\mitch_B0308-left_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_12\mitch_B0510-right_big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv"
        }, 
        "test_13" : {
            "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_13\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_13\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_13\mitch_B0308-small_lef_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_13\mitch_B0510-small_righ_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
        },
         "test_14" : {
            "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_14\Muse_E2511_RED-arm_comined_ppsorted_add_rep_ids_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_14\Muse_E2511_GREY-back_combined_ppsorted_add_rep_ids_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_14\mitch_B0308-big-left_combined_ppsorted_drop_imu_add_rep_ids_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_14\mitch_B0510-big-right_combined_ppsorted_drop_imu_add_rep_ids_cleaned.csv",
        },
         "test_15": {
             "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_15\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_15\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_15\mitch_B0308-big_lef_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_15\mitch_B0510-big_righ_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
         },
         "test_16": {
            "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_16\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_16\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_16\mitch_B0510-left-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_16\mitch_B0308-right-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
         },
         "test_17": {
            "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_17\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_17\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_17\mitch_B0510-left-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_17\mitch_B0308-right-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
         }, 
        "test_18": {
             "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_18\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_18\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_18\mitch_B0510-left-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_18\mitch_B0308-right-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",    
         },
        "test_19" : {
             "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_19\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_19\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_19\mitch_B0510-left-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_19\mitch_B0308-right-small_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
         },
         "test_20" : {
             "arm" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_20\Muse_E2511_RED-arm_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "back" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_20\Muse_E2511_GREY-back_new_time_hz_labeled_no_idle_median_filter_add_rep_id_ppsorted_cleaned.csv",
            "left" : r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_20\mitch_B0308-left-big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
            "right": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_20\mitch_B0510-right-big_del_end_dupli_new_time_labeled_no_idle_median_filter_drop_imu_add_rep_id_ppsorted_cleaned.csv",
         }
         
         
    }
    return tests


def get_test_folder_paths():
    folders = {
        "test_1": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_1",
        "test_2": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_2",
        "test_3": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_3",
        "test_4": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_4",
        "test_5": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_5",
        "test_6": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_6",
        "test_7": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_7",
        "test_8": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_8",
        "test_9": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_9",
        "test_10": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_10",
        "test_11": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_11",
        "test_12": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_master_maria\test_12",
        "test_13": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_13",
        "test_14": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_14",
        "test_15": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_15",
        "test_16": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_16",
        "test_17": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_17",
        "test_18": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_18",
        "test_19": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_19",
        "test_20": r"C:\Users\Bruker\Master25_code\Master25\Data\sensor_test_files_johanne\test_20",
    }
    return folders


def get_one_test(test_number): 
    test_to_get = "test_" + str(test_number)
    return get_test_file_paths()[test_to_get]

def get_one_file(test_number, sensor):
    test_nr_to_get = "test_" + str(test_number)
    return get_test_file_paths()[test_nr_to_get][sensor]

def get_one_foler_path(test_number):
    if isinstance(test_number, int):
        test_number = "test_" + str(test_number)
    return get_test_folder_paths()[test_number]


def get_feture_paths(window_length_sec=4, norm_IMU=True, mean_fsr=False, hdr=False, feature_space='baseline'):
    folders = get_test_folder_paths()
    feature_files = {}

    for test_id, folder in folders.items():
        if mean_fsr is None:
            filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
        else:
            filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
        full_path = os.path.join(folder, filename)
        feature_files[test_id] = full_path

    return feature_files

def get_feature_paths_for_multiple_spaces(
    window_length_sec=4, 
    norm_IMU=True, 
    mean_fsr=False, 
    hdr=False, feature_spaces = [
    "baseline",
    "expanded+baseline",
    "expanded_only",
    "time_only",
    "time_only+exp_FSR",
    "freq_only",
    "freq_only+exp_FSR",
    "FSR_only",
    "arm_only", 
    "back_only"
    ]):
    folders = get_test_folder_paths()
    feature_files = {}
    
    for test_id, folder in folders.items():
        paths = []
        for feature_space in feature_spaces:
            if mean_fsr is None:
                filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
            else:
                filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
            full_path = os.path.join(folder, filename)
            paths.append(full_path)
        feature_files[test_id] = paths

    return feature_files
    
        