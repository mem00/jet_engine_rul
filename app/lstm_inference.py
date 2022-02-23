import tensorflow as tf
from . import models_lib as lib
import numpy as np
import os
import math

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
    for unit_nr in test_df['unit_nr'].unique():
      engine_data = test_df[test_df["unit_nr"]==unit_nr]
      engine_full_sequence = engine_data.drop(labels=["unit_nr", "cycle"], axis=1)
      test_data = gen_test_data(engine_full_sequence, SEQUENCE_LENGTH, -99)

      test_sequence.append(test_data)

    return test_sequence

def get_test_sequence(engine_number):
    test, fd001_rul = lib.get_and_prepare_test_data()
    X_test = np.array(gen_test_wrapper(test))

    random_engine = X_test[engine_number,:,:]
    random_engine = np.float32(random_engine)
    random_engine = random_engine.reshape(1, random_engine.shape[0], random_engine.shape[1])
    return random_engine, fd001_rul.iloc[engine_number, 0]

def predict_lstm_rul(engine_number):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/models/')
    tflite_lstm_model_file = 'converted_lstm_model.tflite'

    with open(path + tflite_lstm_model_file, 'rb') as fid:
        tflite_lstm_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_lstm_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    sequence, actual_rul = get_test_sequence(engine_number=engine_number)
    interpreter.set_tensor(input_index, sequence)
    # Invoke the interpreter.
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    prediction = math.floor(prediction[0][0])

    pred_error = lib.get_error(prediction, actual_rul)
    return prediction, actual_rul, pred_error
