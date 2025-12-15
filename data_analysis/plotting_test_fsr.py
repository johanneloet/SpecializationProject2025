import matplotlib.pyplot as plt
import pandas as pd 

from get_paths import get_test_file_paths

FILES = get_test_file_paths()
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

for key, paths in FILES.items():
    left_before = pd.read_csv(paths['right'])
    left_after = pd.read_csv(paths['right'].replace('.csv', '_cleaned.csv'))
    
    for col in left_before.columns:
        if 'fsr' in col.lower():
            plt.plot(left_before['ReconstructedTime'], left_before[col])
            plt.plot(left_after['ReconstructedTime'], left_after[FSR_MAPPING_RIGHT[col]])
            plt.title(f'{col}mapped to {FSR_MAPPING_RIGHT[col]}')
            plt.show()