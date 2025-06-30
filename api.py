# api.py
from flask import Flask, request, jsonify
import torch
import pickle
from mini_gpt_model import MiniGPT

from flask_cors import CORS  # <-- ADD THIS LINE

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # <-- ADD THIS LINE

# === Load tokenizer metadata ===
try:
    with open("vx_action_meta.pkl", "rb") as f:
        meta = pickle.load(f)
        if isinstance(meta, dict):
            stoi = meta["stoi"]
            itos = meta["itos"]
        elif isinstance(meta, tuple) and len(meta) == 2:
            stoi, itos = meta
        else:
            raise ValueError("vx_action_meta.pkl must be a dict with 'stoi' and 'itos' or a (stoi, itos) tuple.")
except Exception as e:
    raise RuntimeError(f"Failed to load vx_action_meta.pkl: {e}")

# === Load trained model ===
device = "cpu"
config = {
    'vocab_size': len(stoi),
    'block_size': 16,
    'embedding_dim': 32,
}
try:
    model = MiniGPT(config['vocab_size'], config['block_size'], config['embedding_dim']).to(device)
    model.load_state_dict(torch.load("gpt_action_model.pt", map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# === API Endpoint ===
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    context = data.get("context", [])

    if not context:
        return jsonify({"error": "No context provided"}), 400

    # Encode context
    context_ids = [stoi.get(token, 0) for token in context]
    input_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_new_tokens=20)[0].tolist()

    predicted_tokens = [itos[i] for i in output_ids]
    predicted_actions = []
    for tok in predicted_tokens:
        if "|" in tok:
            predicted_actions.append(tok.split("|")[-1])
        else:
            predicted_actions.append(tok)

    return jsonify({
        "input": context,
        "predicted_tokens": predicted_tokens,
        "predicted_actions": predicted_actions
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)