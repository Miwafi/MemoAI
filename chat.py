import os, json, torch, shutil
from safetensors.torch import load_file
from train import *
from collections import OrderedDict
from colorama import init as colorama_init, Fore, Style, Back

colorama_init(autoreset=True)

# ================================================

TERMINAL_WIDTH = shutil.get_terminal_size().columns

def print_header():
    print()
    print(Fore.CYAN + "╔" + "═" * (TERMINAL_WIDTH - 2) + "╗")
    title = " WafiGPT "
    padding = (TERMINAL_WIDTH - 2 - len(title)) // 2
    print(Fore.CYAN + "║" + " " * padding + Fore.WHITE + Style.BRIGHT + title + Fore.CYAN + " " * (TERMINAL_WIDTH - 2 - padding - len(title)) + "║")
    print(Fore.CYAN + "╚" + "═" * (TERMINAL_WIDTH - 2) + "╝")
    print()

def print_separator(char="─", color=Fore.BLUE):
    print(color + char * TERMINAL_WIDTH + Style.RESET_ALL)

def print_section(title, color=Fore.YELLOW):
    print()
    print(color + "┌" + "─" * (TERMINAL_WIDTH - 2) + "┐")
    padding = (TERMINAL_WIDTH - 2 - len(title)) // 2
    print(color + "│" + " " * padding + Fore.WHITE + Style.BRIGHT + title + color + " " * (TERMINAL_WIDTH - 2 - padding - len(title)) + "│")
    print(color + "└" + "─" * (TERMINAL_WIDTH - 2) + "┘" + Style.RESET_ALL)
    print()

def print_info(label, value, label_color=Fore.GREEN, value_color=Fore.WHITE):
    print(f"  {label_color}{label}{Style.RESET_ALL}: {value_color}{value}{Style.RESET_ALL}")

def print_loading_spinner(message, duration=0.5):
    import time
    import sys
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    for _ in range(int(duration * 10)):
        for char in spinner:
            sys.stdout.write(f'\r{Fore.YELLOW}{char}{Style.RESET_ALL} {message}')
            sys.stdout.flush()
            time.sleep(0.05)
    sys.stdout.write('\r' + ' ' * (len(message) + 2) + '\r')
    sys.stdout.flush()

# ================================================

def sample_next_token(logits, generated_tokens, repetition_penalty, presence_penalty, temperature):
    for token in set(generated_tokens):
        if logits[token] < 0:
            logits[token] *= repetition_penalty
        else:
            logits[token] /= repetition_penalty
    vocab_size = logits.size(0)
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    mask[list(set(generated_tokens))] = True
    logits[mask] += presence_penalty
    probs = torch.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item(), probs

# ================================================

def generate_response(model, tokenizer, prompt, device, config, max_length=512, temperature=0.3, repetition_penalty=1.0, presence_penalty=-1.5):
    encoded = tokenizer(f"<|user|>{prompt}<|assistant|>", update=False)
    generated = encoded["input_ids"].unsqueeze(0).to(device)
    unknown_id = tokenizer.split_tokens.get("<|unknown|>")
    end_id = tokenizer.split_tokens.get("<|end|>")
    newline_id = tokenizer.split_tokens.get("\\n")

    print()
    print(f"{Fore.GREEN}╭{'─' * (TERMINAL_WIDTH - 2)}╮{Style.RESET_ALL}")
    print(f"{Fore.GREEN}│{Style.RESET_ALL} {Fore.WHITE + Style.BRIGHT}WafiGPT{Style.RESET_ALL} {' ' * (TERMINAL_WIDTH - 12)}{Fore.GREEN}│{Style.RESET_ALL}")
    print(f"{Fore.GREEN}╰{'─' * (TERMINAL_WIDTH - 2)}╯{Style.RESET_ALL}")
    print()
    print(f"{Fore.WHITE}", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_length):
            if generated.size(1) > config["max_seq_length"]:
                current_input = generated[:, -config["max_seq_length"] :]
                pos_offset = generated.size(1) - config["max_seq_length"]
            else:
                current_input = generated
                pos_offset = 0

            outputs = model(current_input, pos_offset=pos_offset)
            logits = outputs["logits"][0, -1, :].clone()
            gen_tokens = generated[0].tolist()
            token_id, probs = sample_next_token(logits, gen_tokens, repetition_penalty, presence_penalty, temperature)

            if token_id == unknown_id and probs.sum() > 0:
                probs[unknown_id] = 0.0
                probs = probs / probs.sum()
                token_id = torch.multinomial(probs, num_samples=1).item()

            generated = torch.cat((generated, torch.tensor([[token_id]], device=generated.device)), dim=1)
            if token_id == end_id:
                break
            token_str = tokenizer.decode([token_id])
            if token_id == newline_id:
                print()
            else:
                print(token_str, end="", flush=True)
    print(Style.RESET_ALL)
    print()

# ================================================

def load_chat_model(model_dir, device):
    print_loading_spinner("Loading configuration...")
    
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print_loading_spinner("Loading tokenizer...")
    
    with open(os.path.join(model_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        token_dict = json.load(f)
    
    tokenizer = ChatTokenizer(config)
    tokenizer.split_tokens = OrderedDict(token_dict)
    
    print_loading_spinner("Loading model weights...")
    
    model = ChatModel(config).to(device)
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()

    print_section("Model Information", Fore.CYAN)
    total_params = sum(p.numel() for p in model.parameters())
    print_info("Total Parameters", f"{total_params:,}")
    print_info("Hidden Size", config["hidden_size"])
    print_info("Number of Layers", config["block_count"])
    print_info("Number of Heads", config["num_heads"])
    print_info("Vocabulary Size", config["vocab_size"])
    print_info("Device", str(device).upper())
    print()
    
    return model, tokenizer, config

# ================================================

def find_latest_model(model_dir):
    if not os.path.exists(model_dir):
        return None
    
    model_folders = []
    for folder in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, "config.json")) and \
               os.path.exists(os.path.join(folder_path, "tokenizer.json")) and \
               os.path.exists(os.path.join(folder_path, "model.safetensors")):
                model_folders.append(folder_path)
    
    if not model_folders:
        return None
    
    model_folders.sort(key=os.path.getmtime, reverse=True)
    return model_folders[0]

# ================================================

def print_help():
    print_section("Available Commands", Fore.YELLOW)
    print(f"  {Fore.CYAN}/help{Style.RESET_ALL}     - Show this help message")
    print(f"  {Fore.CYAN}/clear{Style.RESET_ALL}    - Clear the screen")
    print(f"  {Fore.CYAN}/exit{Style.RESET_ALL}     - Exit the chat")
    print(f"  {Fore.CYAN}/quit{Style.RESET_ALL}     - Same as /exit")
    print()
    print(f"  {Fore.GREEN}Tip:{Style.RESET_ALL} Type your message and press Enter to chat with WafiGPT")
    print()

# ================================================

if __name__ == "__main__":
    print_header()
    
    print_section("Loading Model", Fore.CYAN)
    
    model_dir = find_latest_model("./model")
    if model_dir:
        print_info("Model Path", model_dir)
        print()
    else:
        print(f"{Fore.RED}Error:{Style.RESET_ALL} No valid model found in ./model directory.")
        print(f"{Fore.YELLOW}Please run the training script first to generate a model.{Style.RESET_ALL}")
        exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_chat_model(model_dir, device)
    
    print_separator("═", Fore.GREEN)
    print(f"{Fore.GREEN}  ✓ Model loaded successfully! Ready to chat.{Style.RESET_ALL}")
    print_separator("═", Fore.GREEN)
    print()
    
    print_help()
    
    while True:
        print(f"{Fore.CYAN}╭{'─' * (TERMINAL_WIDTH - 2)}╮{Style.RESET_ALL}")
        print(f"{Fore.CYAN}│{Style.RESET_ALL} {Fore.WHITE + Style.BRIGHT}You{Style.RESET_ALL} {' ' * (TERMINAL_WIDTH - 7)}{Fore.CYAN}│{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╰{'─' * (TERMINAL_WIDTH - 2)}╯{Style.RESET_ALL}")
        
        try:
            prompt = input(f"  {Fore.CYAN}▶{Style.RESET_ALL} ")
        except (KeyboardInterrupt, EOFError):
            print()
            print(f"\n{Fore.YELLOW}Goodbye! 👋{Style.RESET_ALL}")
            break
        
        if not prompt.strip():
            continue
            
        command = prompt.strip().lower()
        
        if command in ["/exit", "/quit"]:
            print()
            print(f"{Fore.YELLOW}Goodbye! 👋{Style.RESET_ALL}")
            break
        elif command == "/help":
            print_help()
            continue
        elif command == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print_header()
            continue
        
        generate_response(model, tokenizer, prompt, device, config)
        print_separator("─", Fore.BLUE)