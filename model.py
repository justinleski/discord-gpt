import torch, torch.nn as nn, torch.nn.functional as F
import argparse, pickle
from pathlib import Path

# ======== CLI ==============
#
# Chat:
# python3 model.py --mode chat --prompt "hello: "
#
# Train:
# python3 model.py --mode train
#

# Hyper parameters
batch_size = 64 # no. seq processed in parallel before optimizer
block_size = 256 # max cont a model can see in a step
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = 3000
eval_interval = 500
lr = 3e-4
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#defien for use of Flask temp
META_FILE = "meta_vocab.pkl"
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda t: ''.join(itos[i] for i in t)
vocab_size = len(stoi) 
# -----------------------------------------------------

# ==== Classes =====
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # idx and targets are both (B,T) tensor of integers
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cond = idx[:, -block_size:]
            # focus only on the last time step
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    








CKPT_FILE  = "discord_gpt.pt"
META_FILE  = "meta_vocab.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or chat with the Bigram model in a single file")
    parser.add_argument("--mode", choices=["train", "chat"],
                        help="'train' = run training loop, "
                             "'chat'  = load .pt and generate text."
                             "If omitted: train when ckpt missing, "
                             "otherwise chat.")
    parser.add_argument("--prompt", default="onigiri: ",
                        help="Seed text used in chat mode")
    args = parser.parse_args()

    # decide mode automatically when --mode not given
    mode = args.mode or ("chat" if Path(CKPT_FILE).exists() else "train")


    # -------------- train or chat ----------------------------
    if mode == "train":

        # open and read input file, map characters
        with open("dialog.txt", "r", encoding="utf-8") as f:     
            text = f.read()

        chars = sorted((set(text)))
        vocab_size = len(chars)

        # Map string to index, then vice cersa
        stoi = { ch:i for i,ch in enumerate(chars)}
        itos = { i:ch for i,ch in enumerate(chars)}

        # save vocab
        import pickle
        with open("meta_vocab.pkl", "wb") as f:
            pickle.dump({"stoi": stoi, "itos": itos}, f)


        # Make lambda fucntions to ecnode and decode; return array of chars/i
        encode = lambda s : [stoi[c] for c in s]
        decode = lambda s : ''.join([itos[i] for i in s])

        # train and test
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.7 * len(data))
        train_data = data[:n]
        val_data = data[n:]


        # Functions ----------
        def get_batch(split):
            # generate a small batch of data of inputs x and targets y
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
            x, y = x.to(device), y.to(device) # to device makes sure the data lives on the CPU/GPU depending on what is selected
            return x, y

        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out
        
        model = BigramLanguageModel().to(device)
        m = model.to(device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for iter in range(max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = get_batch('train')

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # sample output

        torch.save(model.state_dict(), "discord_gpt.pt") 
        
    # ------------------------------------------------------------------------
    else:
        # restore vocab + helpers
        with open(META_FILE, "rb") as f:
            meta  = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda t: "".join(itos[i] for i in t)

        vocab_size = len(stoi) 

        # build model skeleton and load weights
        model = BigramLanguageModel().to(device)
        model.load_state_dict(torch.load(CKPT_FILE, map_location=device))
        model.eval()

        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
        output  = model.generate(context, max_new_tokens=500)[0].tolist()
        print(decode(output))


