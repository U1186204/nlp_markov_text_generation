import math


def predict_bigram(previous_word: str) -> dict[str, float]:
    if previous_word == "and":
        return {"the": 0.6, "and": 0.0, "hello": 0.2, "dinosaur": 0.2}
    if previous_word == "the":
        return {"the": 0.0, "and": 0.0, "hello": 0.4, "dinosaur": 0.6}
    if previous_word == "dinosaur":
        return {"the": 0.0, "and": 0.8, "hello": 0.2, "dinosaur": 0.0}
    if previous_word == "hello":
        return {"the": 0.2, "and": 0.4, "hello": 0.0, "dinosaur": 0.4}
    raise ValueError(f"unsupported previous word {previous_word}")


def evaluate_log(predict, corpus: list[str]) -> float:
    """Evaluate log(p(corpus)) under the model."""
    log_p_corpus = 0.0
    previous_word = corpus[0]
    for word in corpus[1:]:
        log_p_word = math.log(predict(previous_word)[word])
        log_p_corpus += log_p_word
        previous_word = word
    return log_p_corpus


if __name__ == "__main__":
    corpus = ["the", "hello", "and", "the", "dinosaur"]
    print(evaluate_log(predict_bigram, corpus))
