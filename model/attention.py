from model.config import ModelConfig
from torch import nn
from torch.nn import functional as F
import torch


class EfficientMultiheadAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        self.n_head  = config.n_head

        # Wrapping the Q, K and V matrix, for all heads, into a single one
        # (B, T, C) x ( Q | K | V )
        self.QKV  = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)

        self.lin  = nn.Linear(config.n_embed, config.n_embed, bias=False)

        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.size()

        # (B, T, C) -> (B, T, 3C) -> split on dim=2 -> q,k,v have dimensions B, T, C
        q, k, v = self.QKV(x).split(self.n_embed, dim=2)

        # (B, T, n_head, head_size) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # (B, nh, T, hs) => (B, T, nh, hs) => (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.lin(y)
        y = self.drop(y)
        return y

# The following implementation of attention (in the AttentionHead and MultiheadAttention classes) is much more intuitive and explicit
# I leave it here to serve as reference, but I will use the efficient attention for the real model
class AttentionHead(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        head_size = int(config.n_embed / config.n_head)

        # Using the same dimensions for all the matrices as in the og paper
        self.Q = nn.Linear(config.n_embed, head_size, bias=config.use_bias)
        self.K = nn.Linear(config.n_embed, head_size, bias=config.use_bias)
        self.V = nn.Linear(config.n_embed, head_size, bias=config.use_bias)

        # self.dropout = nn.Dropout(config.dropout)

        # Create the causal mask
        # register_buffer binds it to the module (it will be moved to devices, and saved along with the module) but it will not be considered by autograd
        self.register_buffer('causal_mask', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.use_flash = config.flash_att

    def forward(self, x):
        B, T, C = x.shape

        # (B, T, C) x (C, hs) -> (B, T, hs)
        # Done sequentially here for clarity
        # This would be bundled in a 4D tensor in a real system for parallelization
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)


        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # (B, T, hs) @ (B, hs, T) / sqrt(hs)
            # A is (B, T, T)
            A = ( q @ k.transpose(-1, -2) ) * k.size(-1)**-0.5 # Transpose the last 2 dims

            # Apply the causal mask and softmax
            A = A.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
            A = F.softmax(A, dim=-1)

            # Apply the values 
            # (B, T, T) @ (B, T, hs) -> y (B, T, hs)

            # Omitting the Batch dimension, we get something like this:

            # [1,    0,    0   ]     [ a, b, c ]  Each row is a token in hs space  
            # [0.5,  0.5,  0   ]  @  [ e, f, g ]  The first row ([a, b, c]) is reproduced exactly during matmul because it is the only token available
            # [0.33, 0.33, 0.33]     [ h, i, j ]  Subsequent rows are always modified by the rows above
            out = A @ v

        return out # (B, T, hs)
    


class MultiHeadAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([ AttentionHead(config) for _ in range(config.n_head) ])
        self.proj  = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        # Since all heads are producing embeddings in dimension head_size
        # and since head_size is n_embed / n_head
        # this comes out to size B, T, C with C = n_embed
        out = torch.cat([ head(x) for head in self.heads ], dim=-1)
        out = self.proj(out) # Projection here shuffles the output from each head 
        out = self.dropout(out)
        return out