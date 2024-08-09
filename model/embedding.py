import torch
from torch import nn
from model.config import ModelConfig

# (B, T) -> (B, T, C)
class EmbeddingsLayer(nn.Module):

    def __init__(self, config: ModelConfig):
        # Both word and positional embeddings are learned
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed) # This turns words into embedding
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embed)  # This turns position in block into embedding
        self.device = config.device

    def forward(self, idx):
        B, T = idx.shape # here we assume that we always receive (B, T) shapped input
        pos = torch.arange(0, T, dtype=torch.long, device=self.device) # What happens when the data is smaller than T????
        return self.word_embeddings(idx) + self.pos_embeddings(pos) # Combined word and position embed through addition

