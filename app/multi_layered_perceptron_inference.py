import tensorflow as tf
from . import models_lib as lib
import numpy as np
import os
import math

SPECIFIC_LAGS = [1,2,3,4,5,10,20]

def add_specific_lags(df_input, list_of_lags, columns):
    df = df_input.copy()
    for i in list_of_lags:
        lagged_columns = [col + '_lag_{}'.format(i) for col in columns]
        df[lagged_columns] = df.groupby('unit_nr')[columns].shift(i)
    df.dropna(inplace=True)
    return df

def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, calculate the exponential weighted mean of desired sensors
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())
    
    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    
    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]
    
    return df

def prepare_input_data(engine_number, n=0, alpha=0.4):
    test, fd001_rul = lib.get_and_prepare_test_data()
    X_test_interim = exponential_smoothing(test, lib.COLS_TO_NORMALIZE, n, alpha)
    X_test_interim = add_specific_lags(X_test_interim, SPECIFIC_LAGS, lib.COLS_TO_NORMALIZE)
    X_test_interim = X_test_interim.groupby('unit_nr').last()
    X_test_interim = X_test_interim.drop("cycle", axis=1)
    X_test_interim = np.float32(X_test_interim)
    X_test_interim = X_test_interim[engine_number,:]
    X_test_interim = X_test_interim.reshape(1, X_test_interim.shape[0])
    return X_test_interim, fd001_rul.iloc[engine_number, 0]

def predict_mlp_rul(engine_number):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/models/')
    tflite_lagged_variable_model_file = 'lag_model.tflite'

    with open(path + tflite_lagged_variable_model_file, 'rb') as fid:
        tflite_lagged_variable_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_lagged_variable_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    input_data, actual_rul = prepare_input_data(engine_number=engine_number)

    interpreter.set_tensor(input_index, input_data)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    prediction = math.floor(prediction[0][0])

    pred_error = lib.get_error(prediction, actual_rul)
    return prediction, pred_error