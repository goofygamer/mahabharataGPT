import json

class CharacterTokenizer:
    """
    A simple character-level tokenizer.
    """
    def __init__(self, text=None):
        """
        Initializes the tokenizer.

        Args:
            text (str, optional): The text to build the vocabulary from.
        """
        self.chars = []
        self.stoi = {}
        self.itos = {}
        if text:
            self.build_vocab(text)

    def build_vocab(self, text):
        """Builds the vocabulary from the provided text."""
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.chars)

    def encode(self, s):
        """
        Encodes a string into a list of integers.

        Args:
            s (str): The input string.

        Returns:
            list[int]: The list of encoded integers.
        """
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """
        Decodes a list of integers into a string.

        Args:
            l (list[int]): The list of integers.

        Returns:
            str: The decoded string.
        """
        return ''.join([self.itos[i] for i in l])

    def save_vocab(self, vocab_path):
        """Saves the vocabulary to a file."""
        vocab_data = {
            'chars': self.chars,
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()} # JSON keys must be strings
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {vocab_path}")

    def load_vocab(self, vocab_path):
        """Loads the vocabulary from a file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.chars = vocab_data['chars']
        self.stoi = vocab_data['stoi']
        self.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        print(f"Vocabulary loaded from {vocab_path}")

