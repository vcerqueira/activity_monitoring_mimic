import pickle
import pandas as pd

pd.set_option('display.max_columns', 500)

file_names = [
    'mimic_patients_0_300.pkl',
    'mimic_patients_300_600.pkl',
    'mimic_patients_600_900.pkl',
    'mimic_patients_900_1200.pkl',
    'mimic_patients_1200_1500.pkl',
    'mimic_patients_1500_1800.pkl',
    'mimic_patients_1800_2100.pkl',
    'mimic_patients_2100_2400.pkl',
    'mimic_patients_2400_2700.pkl'
]

patients_data = {}
for file_name in file_names:
    print(file_name)
    print(len(patients_data))
    with open(file_name, 'rb') as fp:
        file_data = pickle.load(fp)
    #
    patients_data = {**patients_data, **file_data}

with open(f'mimic_patients_complete.pkl', 'wb') as fp:
    pickle.dump(patients_data, fp)