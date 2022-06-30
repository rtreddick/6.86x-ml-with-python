from utils import load_stop_words

stop_words = load_stop_words()[:-1]

print(type(stop_words))
print(stop_words)