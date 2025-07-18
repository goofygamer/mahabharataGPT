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
