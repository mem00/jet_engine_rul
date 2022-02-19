from flask import current_app as app

@app.route("/")
def hello_world():
    return "<p>RUL app!</p>"
