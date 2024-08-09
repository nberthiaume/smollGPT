from torch import nn
from model.attention import EfficientMultiheadAttention
from model.mlp import MLP
from model.config import ModelConfig

class Block(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = EfficientMultiheadAttention(config)
        self.mlp = MLP(config)
        self.ln_in = nn.LayerNorm(config.n_embed)
        self.ln_out = nn.LayerNorm(config.n_embed)

    def forward(self, x):

        # LayerNorm on the input
        x = self.ln_in(x)

        # Bypass connections
        x = x + self.attn(x)
        x = x + self.mlp(x)

        # LayerNorm on the output
        x = self.ln_out(x)
        
        return x