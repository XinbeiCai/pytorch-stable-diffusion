import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    Token + Position Embedding 模块
    """
    def __init__(self, n_vocab, n_embd, n_tokens):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens, n_embd)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        # (b, l, d)
        return x


class CLIPLayer(nn.Module):
    """
    单层 Transformer 模块，用于处理一个注意力 + 前馈结构。

    构成:
    - LayerNorm -> SelfAttention -> 残差连接
    - LayerNorm -> MLP (GELU approx using Swish) -> 残差连接
    """
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += residue
        return x


class CLIP(nn.Module):
    """
    多层堆叠的 Transformer Encoder

    构成:
    - Token + Position Embedding
    - 多层 CLIPLayer
    - 最后一层 LayerNorm
    """
    def __init__(self, n_vocab=49408, n_embd=768, n_token=77, n_heads=12, layer_num=12):
        super().__init__()
        self.embedding = CLIPEmbedding(n_vocab, n_embd, n_token)
        self.layers = nn.ModuleList([CLIPLayer(n_heads, n_embd) for i in range(layer_num)])
        self.layernorm = nn.LayerNorm(n_embd)

    # def forward(self, tokens):
    #     tokens = tokens.type(torch.long)
    #
    #     # (b, l) -> (b, l, d)
    #     state = self.embedding(tokens)
    #     for layer in self.layers:
    #         state = layer(state)
    #
    #     output = self.layernorm(state)
    #     return output
    def forward(self, tokens=None, input_embeds=None):
        """
        如果传入 tokens，就用 embedding lookup；
        如果传入 input_embeds，直接用 input_embeds。
        """
        if input_embeds is not None:
            state = input_embeds
        elif tokens is not None:
            tokens = tokens.type(torch.long)
            state = self.embedding(tokens)
        else:
            raise ValueError("Either tokens or input_embeds must be provided.")

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)
        return output
