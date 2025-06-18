import torch, torch.nn as nn
from model import (
    GPTLanguageModel,               # <- same class you defined
    encode, decode, vocab_size,     # <- re-exported tokenizer helpers
    n_embd, n_head, n_layer, block_size, dropout
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path="discord_gpt.pt"):
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def generate(model, prompt:str, max_new:int=200):
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new)[0].tolist()
    return decode(out)

if __name__ == "__main__":
    m = load_checkpoint()
    print(generate(m, "hello there,", 100))
