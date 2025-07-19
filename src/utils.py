import os
import requests
import torch

def download_data(url, data_path):
    """
    Downloads a file from a URL to a specified path if it doesn't already exist.

    Args:
        url (str): The URL of the file to download.
        data_path (str): The local path to save the file to.
    """
    # Ensuring that the directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data file not found. Downloading from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Successfully downloaded and saved to {data_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            exit(1) # Exit if data cannot be downloaded
    else:
        print(f"Data file already exists at {data_path}")

def get_batch(data, block_size, batch_size, device):
    """
    Generates a mini-batch of inputs x and targets y from the data.

    Args:
        data (torch.Tensor): The full dataset as a tensor.
        block_size (int): The context length for predictions.
        batch_size (int): The number of independent sequences per batch.
        device (str): The device to move the tensors to ('cpu' or 'cuda').

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    """
    # Generate random starting points for the batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create the input sequences (x)
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Create the target sequences (y), which are shifted by one position
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move tensors to the specified device
    x, y = x.to(device), y.to(device)
    
    return x, y
