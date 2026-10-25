1. this is a small decoder-only transformer built using pytorch, trained to autocomplete customer support replies.

2. the decoder uses 8 attention heads, a model dimension of 512, and a feed forward network with dimensions 512 -> 2048 -> 512.

3. it uses causal masking so each token can only attend to itself and previous tokens. both attention and feed forward layers have residual connections and layer normalization.

4. the data comes from the bitext customer support dataset on hugging face. only the response column is used for training.

5. text is converted to lowercase, punctuation is separated, and tokens are split on whitespace. the vocabulary is built from training data, with <unk> for unknown words and <eos> for the end of a reply.

6. train.py adds trainable token and positional embeddings, passes them through the decoder blocks, and projects the output to vocabulary logits.

7. training uses inputs and targets shifted by one token, cross entropy loss, adamw, gradient clipping, and a cosine learning rate scheduler.

8. run python train.py to train the model and generate a checkpoint.

9. set the checkpoint path and prompt in inference.py, then run python inference.py.

10. inference samples one token at a time using the last position's logits. it stops at <eos> or the maximum number of new tokens, using only the latest context window as input.