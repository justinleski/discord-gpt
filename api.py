# api.py
from flask import Blueprint, request, jsonify
import torch
from model import BigramLanguageModel , encode, decode  # helpers from model.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# build & load model once at start-up
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load("discord_gpt.pt", map_location=device))
model.eval()                                   # set to inference mode

@torch.no_grad()
def generate_text(prompt: str, n_tokens: int = 120) -> str:
    idx  = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out  = model.generate(idx, max_new_tokens=n_tokens)[0].tolist()
    full = decode(out)
    completion = full[len(prompt):]    
    return completion

api = Blueprint("api", __name__)

@api.post("/generate")
def generate():
    data   = request.get_json(force=True)    
    prompt = data.get("prompt", "")
    n_tok  = int(data.get("max_tokens", 250))
    result = generate_text(prompt, n_tok)
    return jsonify({"output": result})
