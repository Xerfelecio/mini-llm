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

# ============================================================
# VALID QUESTIONS - ONLY THESE WILL BE ANSWERED
# ============================================================
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

# ============================================================
# EXACT ANSWERS - DILI MAG-USAB-USAB
# ============================================================
EXACT_ANSWERS = {
    "hi": "Hello! How are you? I'm Master AI, your TMC assistant.",
    "hello": "Hi there! How are you? This is Master AI.",
    "hey": "Hey! How can I help you today?",
    "good morning": "Good morning! How are you today?",
    "vision": 'The vision of TMC is: "A Model Institution with Fully Developed Academic Technical-Vocational Education and Skill Manpower with Positive Work Attitudes Anchored in the Core Values of Leadership and Professionalism Essential in the Creation of Self Reliant Citizen."',
    "vision sa tmc": 'The vision of TMC is: "A Model Institution with Fully Developed Academic Technical-Vocational Education and Skill Manpower with Positive Work Attitudes Anchored in the Core Values of Leadership and Professionalism Essential in the Creation of Self Reliant Citizen."',
    "vision of tmc": 'The vision of TMC is: "A Model Institution with Fully Developed Academic Technical-Vocational Education and Skill Manpower with Positive Work Attitudes Anchored in the Core Values of Leadership and Professionalism Essential in the Creation of Self Reliant Citizen."',
    "what is the vision": 'The vision of TMC is: "A Model Institution with Fully Developed Academic Technical-Vocational Education and Skill Manpower with Positive Work Attitudes Anchored in the Core Values of Leadership and Professionalism Essential in the Creation of Self Reliant Citizen."',
    "mission": 'The mission of TMC is: "To Build Well Trained, Competent, and Employable Professionals Who Will Meet the Demands of Local and International Workplaces."',
    "mission sa tmc": 'The mission of TMC is: "To Build Well Trained, Competent, and Employable Professionals Who Will Meet the Demands of Local and International Workplaces."',
    "mission of tmc": 'The mission of TMC is: "To Build Well Trained, Competent, and Employable Professionals Who Will Meet the Demands of Local and International Workplaces."',
    "what is the mission": 'The mission of TMC is: "To Build Well Trained, Competent, and Employable Professionals Who Will Meet the Demands of Local and International Workplaces."',
    "goal": 'The goal of TMC is: "TMC Aims at Evolving a Whole Individual as a Child of God and a Member of Democratic Society Who is Professionally Competent that Can Provide Leadership and Advance Knowledge, Well Trained in a Certain Vocation Not Only to Help Himself but to Help Others and Practical Yet Responsible and Obedient to the Laws of God and to the Laws of the Government."',
    "goal sa tmc": 'The goal of TMC is: "TMC Aims at Evolving a Whole Individual as a Child of God and a Member of Democratic Society Who is Professionally Competent that Can Provide Leadership and Advance Knowledge, Well Trained in a Certain Vocation Not Only to Help Himself but to Help Others and Practical Yet Responsible and Obedient to the Laws of God and to the Laws of the Government."',
    "goal of tmc": 'The goal of TMC is: "TMC Aims at Evolving a Whole Individual as a Child of God and a Member of Democratic Society Who is Professionally Competent that Can Provide Leadership and Advance Knowledge, Well Trained in a Certain Vocation Not Only to Help Himself but to Help Others and Practical Yet Responsible and Obedient to the Laws of God and to the Laws of the Government."',
    "what is the goal": 'The goal of TMC is: "TMC Aims at Evolving a Whole Individual as a Child of God and a Member of Democratic Society Who is Professionally Competent that Can Provide Leadership and Advance Knowledge, Well Trained in a Certain Vocation Not Only to Help Himself but to Help Others and Practical Yet Responsible and Obedient to the Laws of God and to the Laws of the Government."',
    "who created you": "I was created by Rex Joseph R Felecio.",
    "who made you": "I was created by Rex Joseph R Felecio.",
    "creator": "I was created by Rex Joseph R Felecio.",
    "first name": "Rex.",
    "middle name": "Joseph.",
    "last name": "Felecio.",
    "what is tmc": "TMC stands for Trinidad Municipal College.",
    "tmc meaning": "TMC stands for Trinidad Municipal College.",
    "thank you": "You're welcome! Always happy to help. God bless! 😊",
    "thanks": "You're welcome! Always happy to help. God bless! 😊",
    "bye": "Bye! God bless! Come back if you have any questions. 😊",
    "goodbye": "Goodbye! God bless! Come back if you have any questions. 😊",
}

def is_valid_question(prompt: str) -> bool:
    """Check if the question is valid"""
    prompt_lower = prompt.lower().strip()
    
    # Check for exact match
    if prompt_lower in VALID_QUESTIONS:
        return True
    
    # Check for partial match
    for valid in VALID_QUESTIONS:
        if valid in prompt_lower or prompt_lower in valid:
            return True
    
    # Check for keywords
    if "vision" in prompt_lower:
        return True
    if "mission" in prompt_lower:
        return True
    if "goal" in prompt_lower:
        return True
    if "creator" in prompt_lower or "created" in prompt_lower or "made" in prompt_lower:
        return True
    if "first name" in prompt_lower or "middle name" in prompt_lower or "last name" in prompt_lower:
        return True
    if "name" in prompt_lower:
        return True
    if "tmc" in prompt_lower:
        return True
    if "thank" in prompt_lower:
        return True
    if "bye" in prompt_lower or "goodbye" in prompt_lower:
        return True
    if "hi" in prompt_lower or "hello" in prompt_lower or "hey" in prompt_lower:
        return True
    if "how are you" in prompt_lower or "what's up" in prompt_lower:
        return True
    
    return False

def get_exact_answer(prompt: str) -> str:
    """Return the exact answer based on the prompt"""
    prompt_lower = prompt.lower().strip()
    
    # Check for exact match
    if prompt_lower in EXACT_ANSWERS:
        return EXACT_ANSWERS[prompt_lower]
    
    # Check for partial match
    for key in EXACT_ANSWERS:
        if key in prompt_lower or prompt_lower in key:
            return EXACT_ANSWERS[key]
    
    # Check for keywords
    if "vision" in prompt_lower:
        return EXACT_ANSWERS["vision"]
    if "mission" in prompt_lower:
        return EXACT_ANSWERS["mission"]
    if "goal" in prompt_lower:
        return EXACT_ANSWERS["goal"]
    if "creator" in prompt_lower or "created" in prompt_lower or "made" in prompt_lower:
        return EXACT_ANSWERS["who created you"]
    if "first name" in prompt_lower:
        return EXACT_ANSWERS["first name"]
    if "middle name" in prompt_lower:
        return EXACT_ANSWERS["middle name"]
    if "last name" in prompt_lower:
        return EXACT_ANSWERS["last name"]
    if "tmc" in prompt_lower:
        return EXACT_ANSWERS["what is tmc"]
    if "thank" in prompt_lower:
        return EXACT_ANSWERS["thank you"]
    if "bye" in prompt_lower or "goodbye" in prompt_lower:
        return EXACT_ANSWERS["bye"]
    if "hi" in prompt_lower or "hello" in prompt_lower or "hey" in prompt_lower:
        return EXACT_ANSWERS["hi"]
    
    return None

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
    
    # Check if question is valid
    if not is_valid_question(prompt):
        return "Sorry, I don't have an answer for that."
    
    # Check if we have an exact answer
    exact_answer = get_exact_answer(prompt)
    if exact_answer:
        return exact_answer
    
    if len(text) < 3:
        return "Sorry, I don't have an answer for that."
    
    return text

def type_out(text: str, delay: float = TYPE_DELAY):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def generate_response(prompt: str) -> str:
    # First check if question is valid
    if not is_valid_question(prompt):
        return "Sorry, I don't have an answer for that."
    
    # Check if we have an exact answer
    exact_answer = get_exact_answer(prompt)
    if exact_answer:
        return exact_answer
    
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
print("  MASTER AI ASSISTANT")
print("="*50)
print("Hello! I'm Master AI, your TMC assistant.")
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