

def n_grams(corpus, n):
    tokens = [tuple(corpus[i : i + n]) for i in range(len(corpus) - n + 1)]
    n_gram_dictionary = dict(Counter(tokens))
    return n_gram_dictionary

def probability_distribution(corpus, n, context):
    return

def finish_sentence(sentence, n, corpus, randomize=False):
    return

if __name__ == "__main__":
    
    finish_sentence()
