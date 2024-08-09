from torch import nn
from model.config import ModelConfig

class MLP(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.GELU(),
            nn.Linear(4*config.n_embed, config.n_embed),
            nn.Dropout(config.dropout)
         )

    def forward(self, x):
        out = self.net(x)
        return out
