[![Python application](https://github.com/U1186204/nlp_markov_text_generation/actions/workflows/python-app.yml/badge.svg)](https://github.com/U1186204/nlp_markov_text_generation/actions/workflows/python-app.yml)

# Markov Text Generation
A bare-bones implementation of a Markov chain model for natural language text generation.

## Overview

This project implements a simple yet effective Markov model for text generation. Given a seed phrase, the model can continue the text in a way that resembles natural language by analyzing patterns in a training corpus.

The core functionality allows both deterministic prediction (always choosing the most likely next word) and stochastic generation (randomly sampling from probable next words), providing flexibility for different use cases.

## How It Works

The model builds n-gram probability distributions from a training corpus. When generating text, it:

1. Takes the most recent n-1 words as context
2. Finds all possible next words that have followed this context in the corpus
3. Either:
   - Selects the most probable next word (deterministic mode)
   - Randomly samples a word based on probability distribution (stochastic mode)

When the model can't find a match for the current context, it uses "stupid backoff" - reducing the context length and trying again with a discounted probability.

## Key Features

- Configurable n-gram size
- Deterministic or stochastic text generation
- Stupid backoff for handling unseen contexts
- Automatic sentence termination at punctuation

## Example Usage

```python
import nltk
from mtg import finish_sentence

# Load corpus
corpus = nltk.word_tokenize(
    nltk.corpus.gutenberg.raw('austen-sense.txt').lower()
)

# Deterministic generation
seed = ['she', 'was', 'not']
result = finish_sentence(seed, 3, corpus, randomize=False)
print(result)
# Output: ['she', 'was', 'not', 'in', 'the', 'world', ',', 'and', 'the', 'two']

# Stochastic generation
result = finish_sentence(seed, 3, corpus, randomize=True)
print(result)
# Output varies with each run, e.g.: ['she', 'was', 'not', 'the', 'same', 'as', 'the', 'others', '.']
