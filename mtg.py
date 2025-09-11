import numpy as np
from collections import Counter, defaultdict

def finish_sentence(sentence, n, corpus, randomize=False):
    """
    Finish a sentence using a Markov model trained on a corpus.
    
    Args:
        sentence: list of tokens to start the sentence
        n: maximum n-gram size to consider
        corpus: training corpus as a tuple of tokens
        randomize: whether to sample randomly or take the most likely token
    
    Returns:
        list of tokens completing the sentence
    """
    # Build n-gram models
    models = {}
    for i in range(1, n+1):
        models[i] = build_ngram_model(corpus, i)
    
    # Initialize the result with the input sentence
    result = list(sentence)
    
    # Generate up to 10 words
    for _ in range(10 - len(result)):
        # Get the next word using stupid backoff
        next_word = predict_next_word(result, n, models, randomize)
        result.append(next_word)
        
        # Stop if we hit a terminal punctuation
        if next_word in [".", "!", "?"]:
            break
    
    return result

def build_ngram_model(corpus, n):
    """Build an n-gram model from a corpus."""
    model = defaultdict(Counter)
    
    # Count occurrences of each n-gram
    for i in range(len(corpus) - n + 1):
        ngram = corpus[i:i+n]
        prefix = ngram[:-1]
        suffix = ngram[-1]
        model[prefix][suffix] += 1
    
    return model

def predict_next_word(sentence, n, models, randomize=False, alpha=0.4):
    """Predict the next word using stupid backoff."""
    candidates = defaultdict(float)
    
    # Try different context lengths, starting with the longest
    for context_length in range(min(n-1, len(sentence)), 0, -1):
        context = tuple(sentence[-context_length:])
        discount = alpha ** (n - 1 - context_length)
        
        # If this context exists in the appropriate model
        if context in models[context_length + 1]:
            # Add scores for each possible next word
            for word, count in models[context_length + 1][context].items():
                candidates[word] += discount * count
            
            # If we found some candidates, no need to back off further
            if candidates:
                break
    
    # If no candidates were found, fall back to unigram model
    if not candidates and () in models[1]:
        discount = alpha ** (n - 1)
        for word, count in models[1][()].items():
            candidates[word] += discount * count
    
    # If still no candidates, return a period
    if not candidates:
        return "."
    
    # Select the next word
    if randomize:
        # Sample randomly according to the distribution
        total = sum(candidates.values())
        probabilities = {word: count/total for word, count in candidates.items()}
        words = list(probabilities.keys())
        probs = list(probabilities.values())
        return np.random.choice(words, p=probs)
    else:
        # Select the most likely word (with alphabetical tie-breaking)
        max_count = max(candidates.values())
        best_words = [word for word, count in candidates.items() 
                     if abs(count - max_count) < 1e-10]
        return min(best_words)