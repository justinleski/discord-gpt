# app.py
from flask import Flask
from flask_cors import CORS          # pip install flask-cors
from api import api                  # the blueprint above

app = Flask(__name__)
CORS(app)                            # allow requests from Angular dev server
app.register_blueprint(api, url_prefix="/api")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
