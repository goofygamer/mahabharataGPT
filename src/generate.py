import torch
import yaml
import argparse

from model import GPTLanguageModel
from tokenizer import CharacterTokenizer
from utils import download_data

# --- Main Generation Script ---

def main(config_path, model_path, prompt, max_new_tokens):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # For reproducibility of generation
    torch.manual_seed(1337)

    # 2. Setup Device (UPDATED FOR MACOS MPS)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # 3. Re-build Tokenizer from the original data
    # This ensures we have the exact same vocabulary as during training.
    download_data(config['data_url'], config['data_path'])
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)
    vocab_size = tokenizer.vocab_size
    config['vocab_size'] = vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # 4. Initialize Model
    model = GPTLanguageModel(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout'],
        device=device
    )
    m = model.to(device)

    # 5. Load Trained Model Weights
    print(f"Loading model weights from {model_path}...")
    try:
        m.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Please ensure you have trained the model first using train.py")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    m.eval() # Set the model to evaluation mode

    # 6. Generate Text
    print("\nGenerating text...")
    # Encode the starting prompt
    start_context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate the sequence
    generated_indices = m.generate(start_context, max_new_tokens=max_new_tokens)
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_indices[0].tolist())
    
    # 7. Print the result
    print("-" * 50)
    print(generated_text)
    print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained Mahabharata GPT model.')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file used for training.')
    parser.add_argument('--model_path', type=str, default='checkpoints/mahabharata_gpt.pth',
                        help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--prompt', type=str, default='\n',
                        help='The starting prompt for text generation. Default is a newline character.')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                        help='The maximum number of new tokens to generate.')
    
    args = parser.parse_args()
    main(args.config, args.model_path, args.prompt, args.max_new_tokens)
