import tensorflow as tf
import pandas as pd
import numpy as np
import os
from random import randrange
from sklearn.preprocessing import MinMaxScaler

COL_NAMES = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
              's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
              's15', 's16', 's17', 's18', 's19', 's20', 's21']

COLS_TO_NORMALIZE = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

SENSORS_TO_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

SETTING_NAMES = COL_NAMES[2:5]

SEQUENCE_LENGTH = 20

def gen_test_data(df, sequence_length, pad_value):
    if df.shape[0] < sequence_length:
        seq = df
        seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=sequence_length, dtype='float32', padding='pre', value=pad_value)
        print(seq)
    else:
        stop = df.shape[0]
        start = stop - sequence_length
        seq = df.iloc[start:stop,:]

    return seq

def gen_test_wrapper(test_df):
    test_sequence = []
    for id in test_df['id'].unique():
      engine_data = test_df[test_df["id"]==id]
      engine_full_sequence = engine_data.drop(labels=["id", "cycle"], axis=1)
      test_data = gen_test_data(engine_full_sequence, SEQUENCE_LENGTH, -99)

      test_sequence.append(test_data)

    return test_sequence

def get_test_sequence():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(ROOT_DIR + '/static/test_data/')
    fd001_test = pd.read_csv(ROOT_DIR + "/static/test_data/test_FD001.txt", sep='\s+')
    fd001_rul = pd.read_csv(ROOT_DIR + "/static/test_data/RUL_FD001.txt", sep='\s+', names=['RemainingUsefulLife'])
    fd001_rul = fd001_rul.clip(upper=125)

    fd001_test.columns = COL_NAMES

    fd001_test_pruned = fd001_test.drop(SETTING_NAMES + SENSORS_TO_DROP, axis=1)

    min_max_scaler = MinMaxScaler()

    test = pd.DataFrame(min_max_scaler.fit_transform(fd001_test_pruned[COLS_TO_NORMALIZE]), columns=COLS_TO_NORMALIZE, index=fd001_test_pruned.index)

    test_join_df = fd001_test_pruned[fd001_test_pruned.columns.difference(COLS_TO_NORMALIZE)].join(test)
    test = test_join_df.reindex(columns = fd001_test_pruned.columns)

    X_test = np.array(gen_test_wrapper(test))

    # pick random
    rand_index = randrange(100)
    random_engine = X_test[rand_index,:,:]
    random_engine = np.float32(random_engine)
    random_engine = random_engine.reshape(1, random_engine.shape[0], random_engine.shape[1])
    return random_engine, fd001_rul.iloc[rand_index, 0]

def predict_rul():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/models/')
    tflite_lstm_model_file = 'converted_lstm_model.tflite'

    with open(path + tflite_lstm_model_file, 'rb') as fid:
        tflite_lstm_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_lstm_model)
    interpreter.allocate_tensors()


    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    sequence, actual_rul = get_test_sequence()
    interpreter.set_tensor(input_index, sequence)
    # Invoke the interpreter.
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)

    return prediction, actual_rul
