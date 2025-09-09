from collections import defaultdict
import math


def predict_the() -> dict[str, float]:
    # every unset key effectively has value 0.0
    distribution: dict[str, float] = defaultdict(float)
    distribution["the"] = 1
    return distribution


def predict_dist() -> dict[str, float]:
    return {"the": 0.6, "and": 0.2, "hello": 0.2, "dinosaur": 0.0}


def evaluate(predict, corpus: list[str]) -> float:
    """Evaluate p(corpus) under the model.

    This is subject to underflow.
    """
    p_corpus = 1
    for word in corpus:
        p_word = predict()[word]
        p_corpus *= p_word
    return p_corpus


def evaluate_log(predict, corpus: list[str]) -> float:
    """Evaluate log(p(corpus)) under the model."""
    log_p_corpus = 0.0
    for word in corpus:
        log_p_word = math.log(predict()[word])
        log_p_corpus += log_p_word
    return log_p_corpus


if __name__ == "__main__":
    corpus = ["the", "and", "the", "hello", "the"] * 200
    print(evaluate_log(predict_dist, corpus))
