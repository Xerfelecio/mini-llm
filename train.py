import os
import torch
import time
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    batch_size = 8
    block_size = 64
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 10
    
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.3
    print(f"Using device: {device} (CPU-optimized but capable)")
    print(f"Settings: batch={batch_size}, context={block_size}, embd={n_embd}, layers={n_layer}, iters={max_iters}")
else:
    batch_size = 64
    block_size = 256
    max_iters = 15000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 100
    
    n_embd = 256
    n_head = 8
    n_layer = 6
    dropout = 0.1
    print(f"Using device: {device} (GPU-optimized settings)")

torch.manual_seed(1337)

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nStarting training for {max_iters} iterations...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Will evaluate every {eval_interval} steps\n")

start_time = time.time()
for step in range(max_iters):
    if step > 0 and (step % eval_interval == 0 or step == max_iters - 1):
        eval_start = time.time()
        losses = estimate_loss(model)
        eval_time = time.time() - eval_start
        elapsed = time.time() - start_time
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time {elapsed:.1f}s | eval {eval_time:.1f}s")

    elif step % 50 == 0:
        elapsed = time.time() - start_time
        print(f"step {step:4d} | training... | time {elapsed:.1f}s")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nTraining completed in {time.time() - start_time:.1f}s")

os.makedirs("checkpoints", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "config": {
        "n_embd": n_embd,
        "block_size": block_size,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
}

torch.save(checkpoint, "checkpoints/mini_llm.pt")
print("Saved checkpoint to checkpoints/mini_llm.pt")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=300, temperature=0.9)[0].tolist()

print("\n=== SAMPLE OUTPUT ===\n")
print(decode(generated))