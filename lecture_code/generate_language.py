import random
from ngram_model import predict_bigram


def generate_language(predict, previous_word: str):
    for _ in range(10):
        word_dist = predict(previous_word)
        word = random.choices(
            list(word_dist.keys()),
            weights=list(word_dist.values()),
        )[0]
        print(word, end=" ")
        previous_word = word
    print("")


def chat():
    print("CTRL-C to stop.")
    while True:
        message = input("user: ")
        last_word = message.split()[-1]
        generate_language(predict_bigram, last_word)


if __name__ == "__main__":
    # generate_language(predict_bigram)
    chat()
