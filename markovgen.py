import random
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_markov_chain(text: str, n: int = 2) -> Dict[str, List[str]]:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if len(text) < n + 1:
        raise ValueError("Text is too short for the specified n-gram size")

    model: Dict[str, List[str]] = {}
    for i in range(len(text) - n):
        key = text[i:i+n]
        next_char = text[i+n]
        model.setdefault(key, []).append(next_char)
    
    logging.info(f"Trained Markov model with {len(model)} keys using n={n}")
    return model

def generate_text(model: Dict[str, List[str]], seed: str, length: int = 100) -> str:
    n = len(next(iter(model)))  # infer n from model keys length
    if len(seed) != n:
        raise ValueError(f"Seed length must be {n}")

    output = seed
    for _ in range(length):
        key = output[-n:]
        next_chars = model.get(key)
        if not next_chars:
            logging.warning(f"No next characters found for key '{key}'. Stopping generation.")
            break
        next_char = random.choice(next_chars)
        output += next_char
    
    logging.info(f"Generated text of length {len(output)} starting with seed '{seed}'")
    return output

if __name__ == "__main__":
    try:
        with open("input.txt", "r", encoding="utf-8") as file:
            text = file.read().replace("\n", " ")
        
        n = 3  # Example: use trigrams instead of bigrams
        model = train_markov_chain(text, n=n)

        seed = "The"  # Seed length must match n
        generated = generate_text(model, seed=seed, length=200)
        print(generated)
    except Exception as e:
        logging.error(f"Error: {e}")
