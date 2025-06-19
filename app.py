from flask import Flask, jsonify, request
from model import encode, decode
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# app = Flask(__name__) # references this file

# Set up routes with app-route decorator
# @app.route('/predict', methods=['POST'])
# def predict():
    # function for said route defined
    # return jsonify({'class_id': 'XXX', 'class_name': 'Message'}) # TODO: probably onyl needsm essaghe respojnse
    #return render_template('index.html') # Flask kows to look in templates

@torch.no_grad()
def generate_text(prompt, n_tokens=200):
    index = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

@api.post("/generate")
def generate():
    data = request.get_json()



