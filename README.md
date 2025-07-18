# mahabharataGPT: A From-Scratch Implementation
This repository contains a reproducible implementation of a decoder-only Generative Pre-trained Transformer (GPT) model, built from scratch in Python using PyTorch. The project is heavily inspired by and based on the educational "Let's build GPT" tutorial by Andrej Karpathy. T

The primary goal is to translate the core concepts from the tutorial into a robust, modular, and well-documented codebase suitable for serious experimentation and learning.

<p align="center">
  <a href="mahabharataGPT.pdf"><strong>View the Full Research Paper (PDF)</strong></a>
</p>

---

## Project Goal
While the original tutorial is an outstanding pedagogical tool, this project aims to build upon it with a focus on software engineering and research best practices. The key objectives are:

---

- **Modularity**: Separating concerns into logical components (model architecture, data handling, training, inference) to create a clean and maintainable codebase.

- **Reproducibility**: Ensuring that experiments are fully reproducible through comprehensive configuration management. All hyperparameters are externalized from the code.

- **Clarity**: Providing a clear and well-commented implementation that directly corresponds to the mathematical foundations of the Transformer architecture.

For a detailed mathematical breakdown of the model architecture and training process, please refer to our formal project blueprint.

---

## Core Architecture
The model is a decoder-only Transformer, built upon the following key principles:

- **Token and Positional Embeddings**: Input text is converted into a sequence of numerical vectors that represent both the token's identity and its position in the sequence.

- **Masked Multi-Head Self-Attention**: The core mechanism that allows the model to weigh the importance of different tokens in the input sequence when making a prediction, while respecting the autoregressive (causal) nature of text generation.

- **Position-wise Feed-Forward Networks**: A sub-layer within each Transformer block that provides additional non-linear processing.

- **Residual Connections & Layer Normalization**: Critical components that ensure stable training and efficient gradient flow through the deep network.

---

## How to Use
_To be developed_

---

## References 
1. Karpathy, A. (2023). Let's build GPT: from scratch, in code, spelled out. YouTube

2. Vaswani, A., et al. (2017). Attention Is All You Need. arXiv:1706.03762