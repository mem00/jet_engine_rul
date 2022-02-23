from flask import current_app as app, render_template
from .lstm_inference import predict_lstm_rul
from .multi_layered_perceptron_inference import predict_mlp_rul
from flask import render_template
from random import randrange

@app.route("/")
def predict_rul():
    rand_index = randrange(100)
    lstm_prediction, actual, lstm_pred_error = predict_lstm_rul(rand_index)
    mlp_prediction, mlp_pred_error = predict_mlp_rul(rand_index)
    return render_template("prediction.html", lstm_prediction=lstm_prediction, 
                actual=actual, lstm_pred_error=lstm_pred_error, mlp_prediction=mlp_prediction, 
                mlp_pred_error=mlp_pred_error)
