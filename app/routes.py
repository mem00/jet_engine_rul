from flask import current_app as app, render_template
from .lstm_inference import predict_rul
from flask import render_template
import math

@app.route("/")
def lstm_inference():
    prediction, actual = predict_rul()
    print(prediction[0][0], "tavo")
    return render_template("prediction.html", prediction=math.floor(prediction[0][0]), actual=actual)
