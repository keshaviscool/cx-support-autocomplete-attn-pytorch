
import torch
import torch.nn as nn
from datasets import load_dataset
from decoder import DecoderBlock
import re
import datetime

torch.manual_seed(42)
def tokenize(text):
    text = text.lower()
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    return text.split()




### LanguageModel

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, context_len=64, n_layers=2):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.dim_model = 512
        self.n_heads = 8

        self.token_embedding = nn.Embedding(vocab_size, self.dim_model)
        self.positional_embedding = nn.Embedding(context_len, self.dim_model) # okay so positonal embedding depends on context len, not all the previous tokens, context len is basically how many last tokens you want to look up

        self.blocks = nn.Sequential(
            *[DecoderBlock(self.dim_model, self.n_heads) for _ in range(n_layers)]
        )

        self.lm_head = nn.Parameter(torch.randn(self.dim_model, self.vocab_size)) ## final prediction head, which takes the last token's contextual embedding as input and outputs the next prediction of token

    def forward(self, token_ids):
        # token_ids: (T,) 
        T = token_ids.shape[0]
        positions = torch.arange(T, device=token_ids.device)

        X = self.token_embedding(token_ids)
        X = X + self.positional_embedding(positions)

        X = self.blocks(X)
        logits = X @ self.lm_head # (T, vocab_size) # for training,
        # for inference, we produce logits for last token by last_logits = X[-1] @ self.lm_head 

        return logits



# model = LanguageModel(vocab_size=len(vocab))

# x = train_ids[:64]
# logits = model(x)

# print("input:", x.shape)        # torch.Size([64])
# print("output:", logits.shape)  # torch.Size([64, len(vocab)]), corresponding to every token, we have output len(vocab) logits, which should be the next token


if __name__ == '__main__':

    ### load dataset
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )

    # print(dataset[0])
    # print(dataset[0]["response"])

    # teach the model to predict the next word, in customer assistance
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    # print(splits)
    train_replies = splits["train"]["response"]
    test_replies = splits["test"]["response"]

    # create vocab, collect unique words
    words = set()
    for reply in train_replies:
        words.update(tokenize(reply))

    vocab = ["<unk>", "<eos>"] + sorted(words - {"<unk>", "<eos>"})
    word_to_id = {word: i for i, word in enumerate(vocab)} # lookup
    id_to_word = {i: word for i, word in enumerate(vocab)} # lookup

    def encode_replies(replies): # flattening all dataset, with <eos> in between
        ids = []

        for reply in replies:
            for token in tokenize(reply):
                ids.append(word_to_id.get(token, word_to_id["<unk>"]))
            ids.append(word_to_id["<eos>"])

        return torch.tensor(ids, dtype=torch.long)


    train_ids = encode_replies(train_replies)
    test_ids = encode_replies(test_replies)

    print("vocab size: ", len(vocab))
    print("train tokens:", train_ids.numel())
    print("test tokens:", test_ids.numel())

    # training and testing data ready

    print(test_ids.shape) # shape: (T,)



    ## training 

    import torch.nn.functional as F

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = LanguageModel(
        vocab_size=len(vocab),
        context_len=64,
        n_layers=3,
    ).to(device)

    # checkpoint loading
    # checkpoint = torch.load(
    #     "checkpoint.pt",
    #     map_location=device,
    #     weights_only=True,
    # )
    # model.load_state_dict(checkpoint["model_state_dict"])

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_trainable_params:,}")

    total_steps = 5000

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=3e-6,
    )

    # create input output pairs shifted by one token
    def get_example(ids):
        T = model.context_len
        start = torch.randint(0, len(ids) - T, ()).item() # randomly pick

        chunk = ids[start : start + T + 1] # chunk will be of context len

        x = chunk[:-1].to(device)
        y = chunk[1:].to(device)

        # x = tensor([10, 20, 30, 40])  # check your email for
        # y = tensor([20, 30, 40, 50])  # your email for a

        return x, y



    # training loop
    assert len(train_ids) > model.context_len
    assert len(train_ids) > model.context_len

    for step in range(total_steps):
        model.train()

        x, y = get_example(train_ids) 

        logits = model(x)                 # (context len(64), vocab_size)
        loss = F.cross_entropy(logits, y)  # internally for every token, we have logits against each vocab token, and the ground truth at the same index of y, so y[i] can be compared to x[i] all logits and calculate the cross entropy loss

        optimizer.zero_grad(set_to_none=True) # clear prev gradients
        loss.backward() # calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient normalization
        optimizer.step() # apply the update 
        scheduler.step() # update the lr

        if step % 500 == 0:
            model.eval() # eval

            with torch.no_grad(): # dont calculate and update weights
                val_losses = []

                for _ in range(20): 
                    vx, vy = get_example(test_ids)
                    val_loss = F.cross_entropy(model(vx), vy)
                    val_losses.append(val_loss.item())

            print(
                f"step {step} | "
                f"train loss: {loss.item():.3f} | "
                f"val loss: {sum(val_losses) / len(val_losses):.3f}"
            )
            print("learning rate:", optimizer.param_groups[0]["lr"])



    ## save training weights
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "context_len": model.context_len,
        },
        f"checkpoints/checkpoint_{int(datetime.datetime.now().timestamp())}.pt",
    )

