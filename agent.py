import os
import sys
import time
import torch
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# LOAD CHECKPOINT - WITH ERROR HANDLING
# ============================================================
try:
    checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]
    config = checkpoint["config"]
    
    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    MODEL_LOADED = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model not loaded: {e}")
    MODEL_LOADED = False

MAX_NEW_TOKENS = 2000
TEMPERATURE = 0.3
TYPE_DELAY = 0.03

# ============================================================
# LOAD QUESTIONS & ANSWERS FROM train.txt - PRIORITY KAAYO
# ============================================================
def load_train_data(filename="train.txt"):
    """Load questions and answers from train.txt"""
    qa_pairs = {}
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Using default answers.")
        return qa_pairs
    
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    current_question = None
    current_answer = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("User:"):
            current_question = line.replace("User:", "").strip().lower()
        elif line.startswith("Assistant:") and current_question:
            current_answer = line.replace("Assistant:", "").strip()
            qa_pairs[current_question] = current_answer
            current_question = None
            current_answer = None
    
    return qa_pairs

# Load the Q&A pairs
QA_PAIRS = load_train_data("train.txt")
print(f"Loaded {len(QA_PAIRS)} question-answer pairs from train.txt")

def get_exact_answer(prompt: str) -> str:
    """Get answer from train.txt - EXACT MATCH LANG"""
    prompt_lower = prompt.lower().strip()
    
    # EXACT MATCH - mao ni ang priority
    if prompt_lower in QA_PAIRS:
        return QA_PAIRS[prompt_lower]
    
    # Check for partial match - kung naay similar
    for key in QA_PAIRS:
        if key in prompt_lower or prompt_lower in key:
            return QA_PAIRS[key]
    
    return None

def generate_response(prompt: str) -> str:
    # UNA GYUD: Check if we have an exact answer from train.txt
    exact_answer = get_exact_answer(prompt)
    if exact_answer:
        return exact_answer
    
    # IKADUHA: Kung wala sa train.txt, DILI MOGAMIT SA MODEL
    # MUBALIK LANG OG "Sorry, I don't have an answer for that."
    return "Sorry, I don't have an answer for that."

def type_out(text: str, delay: float = TYPE_DELAY):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

print("\n" + "="*50)
print("  MASTER AI ASSISTANT")
print("="*50)
print("Hello! I'm Master AI, your TMC assistant.")
print(f"Loaded {len(QA_PAIRS)} question-answer pairs from train.txt")
print("\nCommands:")
print("  /exit   - quit")
print("  /clear  - clear memory")
print("  /save   - save chat")
print("  /fast   - faster typing")
print("  /slow   - slower typing")
print("  /temp X - set temperature (e.g. /temp 0.8)")
print("="*50 + "\n")

chat_history = ""

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError, SystemExit):
        print("\nAgent: Goodbye! God bless! 😊")
        break
    except Exception as e:
        print(f"\nAgent: Error reading input: {e}")
        continue

    if not user_input:
        continue

    if user_input.lower() == "/exit":
        print("Agent: Goodbye! God bless! 😊")
        break

    if user_input.lower() == "/clear":
        chat_history = ""
        print("Agent: Memory cleared.")
        continue

    if user_input.lower() == "/save":
        try:
            with open("chat_log.txt", "w", encoding="utf-8") as f:
                f.write(chat_history)
            print("Agent: Chat saved to chat_log.txt")
        except Exception as e:
            print(f"Agent: Error saving chat: {e}")
        continue

    if user_input.lower() == "/fast":
        TYPE_DELAY = 0.005
        print("Agent: Typing speed set to fast.")
        continue

    if user_input.lower() == "/slow":
        TYPE_DELAY = 0.03
        print("Agent: Typing speed set to slow.")
        continue

    if user_input.lower().startswith("/temp "):
        try:
            value = float(user_input.split(" ", 1)[1])
            if value <= 0:
                print("Agent: Temperature must be greater than 0.")
                continue
            TEMPERATURE = value
            print(f"Agent: Temperature set to {TEMPERATURE}")
        except ValueError:
            print("Agent: Invalid temperature value.")
        continue

    try:
        response = generate_response(user_input)
        print("Agent: ", end="", flush=True)
        type_out(response, TYPE_DELAY)
        print()
        chat_history += f"User: {user_input}\nAgent: {response}\n"
    except Exception as e:
        print(f"Agent: Error generating response: {e}")
        continue