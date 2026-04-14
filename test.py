print("Python is working")
from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Flask is running"

app.run(port=5001)
