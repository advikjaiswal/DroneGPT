import numpy as np

data = np.load("labeled_actions.npz", allow_pickle=True)
X, y = data["X"], data["y"]

with open("gpt_action_training.txt", "w") as f:
    for seq, label in zip(X, y):
        line = f"{','.join([str(round(v, 2)) for v in seq])} -> {label}\n"
        f.write(line)

print("✅ Saved gpt_action_training.txt")
