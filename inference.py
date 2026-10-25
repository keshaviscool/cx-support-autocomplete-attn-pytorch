import re
import torch
from train import LanguageModel

CHECKPOINT = "checkpoints/checkpoint_1792925359.pt"
PROMPT = "please"
MAX_NEW_TOKENS = 300

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# load checkpoint
checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

vocab = checkpoint["vocab"]
word_to_id = {word: i for i, word in enumerate(vocab)}

# recreate model and restore weights
model = LanguageModel(
    vocab_size=len(vocab),
    context_len=checkpoint["context_len"],
    n_layers=3,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def tokenize(text):
    text = text.lower()
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    return text.split()


tokens = tokenize(PROMPT)
if not tokens:
    raise ValueError("PROMPT must contain at least one token")

ids = torch.tensor(
    [word_to_id.get(token, word_to_id["<unk>"]) for token in tokens],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    for _ in range(MAX_NEW_TOKENS):
        context = ids[-model.context_len:]

        logits = model(context)
        probabilities = torch.softmax(logits[-1], dim=-1)

        next_id = torch.multinomial(probabilities, num_samples=1)

        if next_id.item() == word_to_id["<eos>"]:
            break

        ids = torch.cat([ids, next_id])

print(" ".join(vocab[i] for i in ids.tolist()))