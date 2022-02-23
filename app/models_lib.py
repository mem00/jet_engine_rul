from joblib import load
import pandas as pd
import os

COL_NAMES = ['unit_nr', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
              's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
              's15', 's16', 's17', 's18', 's19', 's20', 's21']

COLS_TO_NORMALIZE = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

SENSORS_TO_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

SETTING_NAMES = COL_NAMES[2:5]


def get_and_prepare_test_data():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(ROOT_DIR + '/static/test_data/')
    fd001_test = pd.read_csv(ROOT_DIR + "/static/test_data/test_FD001.txt", sep='\s+')
    fd001_rul = pd.read_csv(ROOT_DIR + "/static/test_data/RUL_FD001.txt", sep='\s+', names=['RemainingUsefulLife'])
    fd001_rul = fd001_rul.clip(upper=125)

    fd001_test.columns = COL_NAMES

    fd001_test_pruned = fd001_test.drop(SETTING_NAMES + SENSORS_TO_DROP, axis=1)

    min_max_scaler = load(ROOT_DIR + '/static/models/min_max_scaler.joblib') 

    test = pd.DataFrame(min_max_scaler.fit_transform(fd001_test_pruned[COLS_TO_NORMALIZE]), columns=COLS_TO_NORMALIZE, index=fd001_test_pruned.index)

    test_join_df = fd001_test_pruned[fd001_test_pruned.columns.difference(COLS_TO_NORMALIZE)].join(test)
    test = test_join_df.reindex(columns = fd001_test_pruned.columns)  
    
    return test, fd001_rul

def get_error(predicted, actual):
    return abs(predicted - actual)

