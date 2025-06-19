from flask import jsonify, request, Blueprint
from model import encode, decode, GPTLanguageModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = GPTLanguageModel().to(device)
# model.load_state_dict(torch.load("discord_gpt.pt", map_location=device))
# model.eval() #FIXME:

# Set up routes with app-route decorator
# @app.route('/predict', methods=['POST'])
# def predict():
    # function for said route defined
    # return jsonify({'class_id': 'XXX', 'class_name': 'Message'}) # TODO: probably onyl needsm essaghe respojnse
    #return render_template('index.html') # Flask kows to look in templates

@torch.no_grad()
def generate_text(prompt, n_tokens=200):
    index = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(index, max_new_tokens=n_tokens)[0].toList()
    return decode(out[len(prompt):])

api = Blueprint("api", __name__)

@api.post("/generate")
def generate():
    data = request.get_json(force=True) # Flask will grab POST request's data
    prompt = data.get("prompt", "") # map into data dict with fallback
    n_tok  = int(data.get("max_tokens", 120))
    result = generate_text(prompt, n_tok)
    return jsonify({"completion": result})



