import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, dim_model, n_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.dim_model = dim_model
        self.ffn_scale = 4 # upscale to 4 times

        assert self.dim_model % self.n_heads == 0
        self.head_dim = self.dim_model // self.n_heads
        self.hidden_dim = self.dim_model * self.ffn_scale
        self.scale = 1/ self.dim_model**0.5

        self.W_Q = nn.Parameter(torch.randn(self.dim_model, self.dim_model) * self.scale)
        self.W_K = nn.Parameter(torch.randn(self.dim_model, self.dim_model) * self.scale)
        self.W_V = nn.Parameter(torch.randn(self.dim_model, self.dim_model) * self.scale)
        self.W_O = nn.Parameter(torch.randn(self.dim_model, self.dim_model) * self.scale) # for output project of multi head attention


        # norm layers have learnable parameters as well
        self.norm1 = nn.LayerNorm(self.dim_model)
        self.norm2 = nn.LayerNorm(self.dim_model)

        # feed forward nn (two linear layer)
        self.W1 = nn.Parameter(torch.randn(self.dim_model, self.hidden_dim) * self.scale)
        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim)) # it will get broadcasted to each token, also bias should be init to zero, backprop will adjust the bias accordingly

        # apply relu after upprojection
        self.W2 = nn.Parameter(torch.randn(self.hidden_dim, self.dim_model) / self.hidden_dim **0.5)
        self.b2 = nn.Parameter(torch.zeros(self.dim_model))

    def forward(self, X): # causal mask multi head attention block

        ### MUTLI HEAD ATTENTION
        n_tokens = X.shape[0] # X shape = (T, dim_model)
        # X is already positionally encoded
        Q = X @ self.W_Q # 3 * 512
        K = X @ self.W_K
        V = X @ self.W_V

        # multi head attention
        Q = Q.reshape(n_tokens, self.n_heads, self.head_dim).transpose(0,1)
        K = K.reshape(n_tokens, self.n_heads, self.head_dim).transpose(0,1)
        V = V.reshape(n_tokens, self.n_heads, self.head_dim).transpose(0,1)


        logits = (Q @ K.transpose(1, 2)) / self.head_dim ** 0.5
        # causal mask
        mask = torch.triu(
            torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=X.device),
            diagonal=1)

        logits = logits.masked_fill(mask, float("-inf"))
        attention = torch.softmax(logits, dim=-1)

        out = attention @ V # (8, T, 64)
        # again, concat back to get the orignial shape
        out = out.transpose(0,1) # (T, 8, 64)
        out = out.reshape(n_tokens, self.dim_model) #(T, dim_model)

        # output projection
        out = out @ self.W_O

        ### attention residual AND NORM
        X = self.norm1(out + X)

        ### ffn, 512 -> 2048 -> 512
        out = torch.relu(X @ self.W1 + self.b1) # up projection, (T, 2048)
        out =  out @ self.W2 + self.b2 # down projection back to 512

        ### ffn residual and NORM
        out = self.norm2(X + out)

        return out









