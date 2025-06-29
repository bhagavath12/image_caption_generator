import pickle

with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Is '<start>' in tokenizer?", '<start>' in tokenizer.word_index)
print("Sample vocab:", list(tokenizer.word_index.items())[:20])
