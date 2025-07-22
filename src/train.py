import torch
import torch.nn as nn
import yaml
import argparse
import os

from model import GPTLanguageModel
from tokenizer import CharacterTokenizer
from utils import download_data, get_batch

# --- Main Training Script ---

def main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # For reproducibility
    torch.manual_seed(1337)

    # 2. Setup Device (UPDATED FOR MACOS MPS)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # 3. Load Data and Initialize Tokenizer
    download_data(config['data_url'], config['data_path'])
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)
    vocab_size = tokenizer.vocab_size
    config['vocab_size'] = vocab_size # Update config with dynamic vocab size
    print(f"Vocabulary size: {vocab_size}")

    # 4. Prepare Data Tensors
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # 5. Initialize Model and Optimizer
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
    
    # Print the number of parameters in the model
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 6. Training Loop
    print("\nStarting training...")
    for iter in range(config['max_iters']):
        # Every once in a while evaluate the loss on train and val sets
        if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(train_data, config['block_size'], config['batch_size'], device)

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training finished.")

    # 7. Save the trained model
    os.makedirs(config['out_dir'], exist_ok=True)
    model_path = os.path.join(config['out_dir'], 'mahabharata_gpt.pth')
    torch.save(m.state_dict(), model_path)
    print(f"Model saved to {model_path}")

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    """
    Estimates the loss for training and validation sets over several batches.
    `torch.no_grad()` is used to disable gradient calculation, saving memory and compute.
    """
    out = {}
    model.eval() # Set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        data = train_data if split == 'train' else val_data
        for k in range(config['eval_iters']):
            X, Y = get_batch(data, config['block_size'], config['batch_size'], device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set the model back to training mode
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GPT model on the Mahabharata text.')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
