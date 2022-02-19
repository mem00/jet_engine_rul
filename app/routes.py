from flask import current_app as app, render_template
from .lstm_inference import predict_rul
from flask import render_template

@app.route("/")
def lstm_inference():
    prediction = predict_rul()
    print(prediction[0][0], "tavo")
    return render_template("prediction.html", prediction=prediction[0][0])
