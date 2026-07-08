import os
import sys
import time
import torch
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
config = checkpoint["config"]

def encode(s: str):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

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

MAX_NEW_TOKENS = 2000
TEMPERATURE = 0.3
TYPE_DELAY = 0.015

VALID_QUESTIONS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what's up",
    "vision", "vision sa tmc", "vision of tmc", "what is the vision",
    "mission", "mission sa tmc", "mission of tmc", "what is the mission",
    "goal", "goal sa tmc", "goal of tmc", "what is the goal",
    "who created you", "who made you", "creator",
    "first name", "middle name", "last name",
    "what is tmc", "tmc meaning", "what does tmc mean",
    "thank you", "thanks", "thank you very much",
    "bye", "goodbye", "see you later", "take care"
]

def clean_response(text: str, prompt: str = "") -> str:
    text = text.replace("\r", "").strip()
    
    if text.startswith("Assistant:"):
        text = text.replace("Assistant:", "").strip()
    
    if "[END]" in text:
        text = text.split("[END]")[0].strip()
    
    if "User:" in text:
        text = text.split("User:")[0].strip()
    
    if "Question:" in text:
        text = text.split("Question:")[0].strip()
    if "Answer:" in text:
        text = text.split("Answer:")[0].strip()
    
    if "---" in text:
        text = text.split("---")[0].strip()
    
    prompt_lower = prompt.lower().strip()
    
    # Check if question is valid
    is_valid = False
    for valid in VALID_QUESTIONS:
        if valid in prompt_lower or prompt_lower in valid:
            is_valid = True
            break
    
    if not is_valid and len(prompt_lower) > 2:
        if "vision" in prompt_lower or "mission" in prompt_lower or "goal" in prompt_lower:
            is_valid = True
        elif "tmc" in prompt_lower:
            is_valid = True
        elif "creator" in prompt_lower or "created" in prompt_lower or "made" in prompt_lower:
            is_valid = True
        elif "name" in prompt_lower:
            is_valid = True
        elif "hello" in prompt_lower or "hi" in prompt_lower or "hey" in prompt_lower:
            is_valid = True
        elif "thank" in prompt_lower or "bye" in prompt_lower or "goodbye" in prompt_lower:
            is_valid = True
    
    if not is_valid:
        return "Sorry, I don't have an answer for that."
    
    # COMPLETE ANSWERS
    vision_complete = 'The vision of TMC is: "A Model Institution with Fully Developed Academic Technical-Vocational Education and Skill Manpower with Positive Work Attitudes Anchored in the Core Values of Leadership and Professionalism Essential in the Creation of Self Reliant Citizen."'
    mission_complete = 'The mission of TMC is: "To Build Well Trained, Competent, and Employable Professionals Who Will Meet the Demands of Local and International Workplaces."'
    goal_complete = 'The goal of TMC is: "TMC Aims at Evolving a Whole Individual as a Child of God and a Member of Democratic Society Who is Professionally Competent that Can Provide Leadership and Advance Knowledge, Well Trained in a Certain Vocation Not Only to Help Himself but to Help Others and Practical Yet Responsible and Obedient to the Laws of God and to the Laws of the Government."'
    
    # GREETINGS
    if prompt_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "what's up"]:
        if "vision" in text.lower() or "mission" in text.lower() or "goal" in text.lower() or "TMC" in text:
            if prompt_lower == "hi" or prompt_lower == "hello":
                text = "Hello! How are you? I'm TmcAi, your TMC assistant."
            elif prompt_lower == "hey":
                text = "Hey! How can I help you today?"
            elif prompt_lower == "good morning":
                text = "Good morning! How are you today?"
            elif prompt_lower == "good afternoon":
                text = "Good afternoon! How can I help you?"
            elif prompt_lower == "good evening":
                text = "Good evening! How can I help you?"
            elif prompt_lower == "how are you":
                text = "I'm doing great! How about you?"
            elif prompt_lower == "what's up":
                text = "Not much! How can I help you today?"
    
    # VISION - Force complete
    if "vision" in prompt_lower:
        if "Self Reliant Citizen" not in text:
            text = vision_complete
        elif len(text) < len(vision_complete) - 10:
            text = vision_complete
    
    # MISSION - Force complete
    if "mission" in prompt_lower:
        if "International Workplaces" not in text:
            text = mission_complete
        elif len(text) < len(mission_complete) - 10:
            text = mission_complete
    
    # GOAL - Force complete
    if "goal" in prompt_lower:
        if "Laws of the Government" not in text:
            text = goal_complete
        elif len(text) < len(goal_complete) - 10:
            text = goal_complete
    
    # CREATOR
    if "creator" in prompt_lower or "created" in prompt_lower or "made" in prompt_lower:
        if "TMC" in text and "Roxanne" not in text:
            text = "I was created by Roxanne Boiser Duman-ag."
        elif "vision" in text.lower() or "mission" in text.lower() or "goal" in text.lower():
            text = "I was created by Roxanne Boiser Duman-ag."
    
    # FIRST, MIDDLE, LAST NAME
    if "first name" in prompt_lower and "Roxanne" not in text:
        text = "Roxanne."
    if "middle name" in prompt_lower and "Boiser" not in text:
        text = "Boiser."
    if "last name" in prompt_lower and "Duman-ag" not in text:
        text = "Duman-ag."
    
    # TMC
    if "tmc" in prompt_lower and "Trinidad" not in text and "vision" not in prompt_lower and "mission" not in prompt_lower and "goal" not in prompt_lower:
        if "vision" not in prompt_lower and "mission" not in prompt_lower and "goal" not in prompt_lower:
            text = "TMC stands for Trinidad Municipal College."
    
    if len(text) < 3:
        return "Sorry, I don't have an answer for that."
    
    return text

def type_out(text: str, delay: float = TYPE_DELAY):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def generate_response(prompt: str) -> str:
    full_prompt = f"User: {prompt}\nAssistant: "
    
    encoded = encode(full_prompt)
    if not encoded:
        return "Sorry, I cannot understand that."

    context = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )[0].tolist()

    full_output = decode(generated)
    
    if "Assistant:" in full_output:
        reply = full_output.split("Assistant:")[-1].strip()
    else:
        reply = full_output.strip()
    
    reply = clean_response(reply, prompt)
    
    return reply

print("\n" + "="*50)
print("  TMC AI ASSISTANT")
print("="*50)
print("Hello! I'm TmcAi, your TMC assistant.")
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
    user_input = input("You: ").strip()

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
        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write(chat_history)
        print("Agent: Chat saved to chat_log.txt")
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

    response = generate_response(user_input)

    print("Agent: ", end="", flush=True)
    type_out(response, TYPE_DELAY)
    print()

    chat_history += f"User: {user_input}\nAgent: {response}\n"