import pandas as pd
import numpy as np
import pickle
import warnings
import os
from data_models.panel import PanelData
from feature_extraction.strats import episode_dynamics_benchmark

pd.set_option('display.max_columns', 500)
warnings.simplefilter("ignore")

DATA_DIR = '../MIMIC_DB_ALL/MIMIC_SERIES/'
# DATA_DIR = '/Users/vcerqueira/Desktop/'

file_names = os.listdir(DATA_DIR)
p1, p2 = 2400, 2700
file_names = file_names[p1:p2]

df_prototype = pd.read_csv(DATA_DIR + 'mimic_2425.csv')
# df_prototype = pd.read_csv(DATA_DIR + 'mimic_125.csv')
df_prototype = df_prototype.drop('Unnamed: 0', axis=1)

data_model = PanelData(prototype=df_prototype,
                       periods=(60, 60, 30))

target_specs = \
    dict(
        hypotension=dict(above_threshold=False,
                         value_threshold=60,
                         ratio_threshold=0.9,
                         target_variable='MAP'),
        hypertension=dict(above_threshold=True,
                          value_threshold=105,
                          ratio_threshold=0.9,
                          target_variable='MAP'),
        tachycardia=dict(above_threshold=True,
                         value_threshold=100,
                         ratio_threshold=0.9,
                         target_variable='HR'),
        bradycardia=dict(above_threshold=False,
                         value_threshold=60,
                         ratio_threshold=0.9,
                         target_variable='HR'),
        tachypena=dict(above_threshold=True,
                       value_threshold=17,
                       ratio_threshold=0.9,
                       target_variable='RR'),
        bradypena=dict(above_threshold=False,
                       value_threshold=12,
                       ratio_threshold=0.9,
                       target_variable='RR'),
        hypoxia=dict(above_threshold=False,
                     value_threshold=93,
                     ratio_threshold=0.9,
                     target_variable='SPO2')
    )

patients = dict()
for j, file in enumerate(file_names):
    print(file)
    # file = 'mimic_524.csv'
    if file in patients.keys():
        continue
    #
    print(file)
    df = pd.read_csv(DATA_DIR + file)
    #
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    #
    df = df.astype(np.float64)
    #
    for col in ['MAP', 'DBP', 'SBP', 'HR']:
        df[col] = df[col].where(df[col].between(10, 200))
    #
    print('Feature engineering')
    try:
        # df = df.tail(500)
        X, Y = data_model.create_instances_all(entity=df,
                                               predictors_fun=episode_dynamics_benchmark,
                                               target_specs=target_specs)
        #
        ent_df_final = pd.concat([X, Y], axis=1)
    except (AssertionError, ValueError) as e:
        continue
    #
    patients[file] = ent_df_final
    #
    with open(f'mimic_patients_{p1}_{p2}.pkl', 'wb') as fp:
        pickle.dump(patients, fp)
