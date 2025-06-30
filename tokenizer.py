# tokenizer.py
import pickle

class Tokenizer:
    def __init__(self, text):
        self.tokens = text.strip().split()
        self.vocab = sorted(set(self.tokens))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
    
    def encode(self, text):
        return [self.token_to_id[t] for t in text.strip().split()]
    
    def decode(self, ids):
        return " ".join([self.id_to_token[i] for i in ids])

    def vocab_size(self):
        return len(self.vocab)

# === Run & Save ===
if __name__ == "__main__":
    with open("gpt_action_training.txt", "r") as f:
        data = f.read()

    tokenizer = Tokenizer(data)
    encoded = tokenizer.encode(data)

    print(f"✅ Vocab size: {tokenizer.vocab_size()}")
    print(f"✅ First 10 encoded tokens: {encoded[:10]}")

    # ✅ Save the tokenized dataset as a pickle
    with open("gpt_tokens.pkl", "wb") as f:
        pickle.dump({
            "encoded": encoded,
            "vocab_size": tokenizer.vocab_size(),
            "token_to_id": tokenizer.token_to_id,
            "id_to_token": tokenizer.id_to_token,
        }, f)

    print("✅ Saved gpt_tokens.pkl")
