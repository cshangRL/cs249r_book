#!/usr/bin/env python3
"""
TinyTalks Demo - Training a Transformer on Conversational Q&A

This demo follows the instructions from datasets/tinytalks/README.md
to train a GPT model on the TinyTalks dataset.

Usage:
    python demo.py
"""

import numpy as np
from tinytorch.core.transformer import GPT
from tinytorch.core.tokenization import CharTokenizer
from tinytorch.core.optimizers import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.tensor import Tensor


def load_dataset():
    """Load the TinyTalks training dataset."""
    with open('datasets/tinytalks/splits/train.txt', 'r') as f:
        train_text = f.read()
    return train_text


def create_batches(text, tokenizer, seq_len=64, batch_size=8):
    """
    Create training batches from text.

    For language modeling, we predict the next character at each position.
    Input: tokens[:-1], Target: tokens[1:]
    """
    # Encode the entire text
    tokens = tokenizer.encode(text)

    # Create sequences of fixed length
    sequences = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
        seq = tokens[i:i + seq_len + 1]
        if len(seq) == seq_len + 1:
            sequences.append(seq)

    # Create batches
    batches = []
    for i in range(0, len(sequences) - batch_size + 1, batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = np.array([seq[:-1] for seq in batch_seqs])
        targets = np.array([seq[1:] for seq in batch_seqs])
        batches.append((inputs, targets))

    return batches


def train_model(model, batches, optimizer, criterion, epochs=10):
    """Training loop for the GPT model."""
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for inputs, targets in batches:
            input_tensor = Tensor(inputs)
            target_tensor = Tensor(targets)

            # Forward pass
            logits = model.forward(input_tensor)

            # Reshape for loss computation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            target_flat = target_tensor.reshape(-1)

            # Compute loss and backpropagate
            loss = criterion(logits_flat, target_flat)
            total_loss += loss.data
            num_batches += 1

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    """Generate text autoregressively from a prompt."""
    tokens = tokenizer.encode(prompt)
    tokens = Tensor(np.array([tokens]))

    max_seq_len = model.max_seq_len

    for _ in range(max_new_tokens):
        # Use sliding window if sequence too long
        if tokens.shape[1] >= max_seq_len:
            tokens = Tensor(tokens.data[:, -max_seq_len + 1:])

        logits = model.forward(tokens)
        next_token_logits = logits.data[0, -1, :]

        # Apply temperature and sample
        next_token_logits = next_token_logits / temperature
        probs = np.exp(next_token_logits - np.max(next_token_logits))
        probs = probs / probs.sum()
        next_token = np.random.choice(len(probs), p=probs)

        # Append token
        new_tokens = np.concatenate([tokens.data, [[next_token]]], axis=1)
        tokens = Tensor(new_tokens)

        # Stop on double newline
        decoded_char = tokenizer.decode([next_token])
        if decoded_char == '\n':
            decoded = tokenizer.decode(tokens.data[0].astype(int).tolist())
            if decoded.count('\n') >= 2:
                break

    return tokenizer.decode(tokens.data[0].astype(int).tolist())


def main():
    print("=" * 60)
    print("TinyTalks Demo - Training a Transformer on Q&A")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading TinyTalks dataset...")
    train_text = load_dataset()
    print(f"   Loaded {len(train_text)} characters of training text")

    # Initialize tokenizer
    print("\n2. Building character tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([train_text])
    print(f"   Vocabulary size: {tokenizer.vocab_size}")

    # Initialize model
    print("\n3. Initializing GPT model...")
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=64
    )
    print(f"   Model: embed_dim=128, layers=4, heads=4")
    print(f"   Parameters: {len(model.parameters())}")

    # Initialize optimizer and loss
    print("\n4. Setting up optimizer and loss function...")
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # Create batches
    print("\n5. Creating training batches...")
    batches = create_batches(train_text, tokenizer, seq_len=64, batch_size=8)
    print(f"   Created {len(batches)} batches")

    # Training loop
    print("\n6. Training model (10 epochs)...")
    print("-" * 40)
    train_model(model, batches, optimizer, criterion, epochs=10)
    print("-" * 40)

    # Generate text
    print("\n7. Generating responses to sample questions...")
    prompts = [
        "Q: What is your name?\nA:",
        "Q: What color is the sky?\nA:",
        "Q: What is 2 plus 3?\nA:",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {repr(prompt)}")
        response = generate_text(model, tokenizer, prompt, max_new_tokens=30)
        # Extract just the answer part
        if "\nA:" in response:
            answer_start = response.find("\nA:") + 3
            answer = response[answer_start:].strip()
        else:
            answer = response
        print(f"Response: {answer[:60]}...")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNote: This is an educational demo. For better results,")
    print("see milestones/05_2017_transformer/ for optimized training.")


if __name__ == "__main__":
    main()
