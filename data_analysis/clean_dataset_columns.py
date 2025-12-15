"""
Code to remove undesired columns such as 'timestamp' and 'unnamed'. Also rename the Fsr-columns to the corrects ones.
"""
from get_paths import get_test_file_paths
import pandas as pd 

FILES = get_test_file_paths()

FSR_MAPPING_LEFT =  {
    'Fsr.01' : 'Fsr.11',
    'Fsr.02' : 'Fsr.10',
    'Fsr.03' : 'Fsr.09',
    'Fsr.04' : 'Fsr.14',
    'Fsr.05' : 'Fsr.12',
    'Fsr.06' : 'Fsr.16',
    'Fsr.07' : 'Fsr.13',
    'Fsr.08' : 'Fsr.15',
    'Fsr.09' : 'Fsr.01',
    'Fsr.10': 'Fsr.02',
    'Fsr.11': 'Fsr.06',
    'Fsr.12': 'Fsr.07',
    'Fsr.13': 'Fsr.03',
    'Fsr.14': 'Fsr.05',
    'Fsr.15': 'Fsr.08',
    'Fsr.16': 'Fsr.04'
}

FSR_MAPPING_RIGHT = {
    'Fsr.01' : 'Fsr.08',
    'Fsr.02' : 'Fsr.04',
    'Fsr.03' : 'Fsr.05',
    'Fsr.04' : 'Fsr.01',
    'Fsr.05' : 'Fsr.03',
    'Fsr.06' : 'Fsr.06',
    'Fsr.07' : 'Fsr.07',
    'Fsr.08' : 'Fsr.02',
    'Fsr.09' : 'Fsr.14',
    'Fsr.10': 'Fsr.15',
    'Fsr.11': 'Fsr.16',
    'Fsr.12': 'Fsr.13',
    'Fsr.13': 'Fsr.12',
    'Fsr.14': 'Fsr.09',
    'Fsr.15': 'Fsr.11',
    'Fsr.16': 'Fsr.10'
}


def clean_and_rename_columns(arm_path, back_path, left_path, right_path):
    """
    Removes columns containing 'timestamp' or 'unnamed' (case-insensitive),
    and renames FSR columns for left/right insoles using mapping dictionaries.
    Returns the cleaned DataFrames.
    """
    # Load data
    df_arm = pd.read_csv(arm_path)
    df_back = pd.read_csv(back_path)
    df_left = pd.read_csv(left_path)
    df_right = pd.read_csv(right_path)

    # Function to drop timestamp/unnamed columns
    def _drop_unwanted(df):
        drop_cols = [c for c in df.columns if 'timestamp' in c.lower() or 'unnamed' in c.lower()]
        return df.drop(columns=drop_cols, errors='ignore')

    # Clean columns
    df_arm = _drop_unwanted(df_arm)
    df_back = _drop_unwanted(df_back)
    df_left = _drop_unwanted(df_left)
    df_right = _drop_unwanted(df_right)

    # Rename FSR columns
    df_left = df_left.rename(columns=FSR_MAPPING_LEFT)
    df_right = df_right.rename(columns=FSR_MAPPING_RIGHT)

    return df_arm, df_back, df_left, df_right

if __name__ == '__main__':
    for test_id, paths in FILES.items():
        if test_id == 'test_6': #redo test_6 only
            arm_cleaned, back_cleaned, left_cleaned, right_cleaned = clean_and_rename_columns(paths['arm'], paths['back'], paths['left'], paths['right'])
            
            arm_cleaned.to_csv(paths['arm'].replace('.csv', '_cleaned.csv'))
            back_cleaned.to_csv(paths['back'].replace('.csv', '_cleaned.csv'))
            left_cleaned.to_csv(paths['left'].replace('.csv', '_cleaned.csv'))
            right_cleaned.to_csv(paths['right'].replace('.csv', '_cleaned.csv'))