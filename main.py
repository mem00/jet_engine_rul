from flask import Flask

app = Flask(__name__)

server = app.server
# this is a test
@app.route("/")
def hello_world():
    return "<p>RUL app!</p>"

