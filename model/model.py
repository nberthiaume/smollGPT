from torch.nn import functional as F
from torch import nn
import torch
from model.embedding import EmbeddingsLayer
from model.block import Block
from model.config import ModelConfig


class SmollGPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingsLayer(config)
        self.blocks = nn.Sequential(*[ Block(config) for _ in range(config.n_layer) ])
        self.head   = nn.Linear(config.n_embed, config.vocab_size) # Karparthy uses weight tying here

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.blocks(x)
        logits = self.head(x)

        if y is not None:
            # Logits are (B, T, Vocab), Targets are (B, T)
            # cross_entropy takes in (n_example, vocab) as input and (n_example,) as output
            b,t,v = logits.shape
            loss = F.cross_entropy(logits.view(b*t, v), y.view(b*t)) # The cross_entropy fct has a built in softmax so no need to do it here
        else:
            loss = None

        return logits, loss


    # note: implement temperature and top_k
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=1000):
        # Expect idx (B, T) where each element is the index of the proper token

        for _ in range(max_new_tokens):
            # First crop for the appropriate context window
            b,t = idx.shape
            idx_cropped = idx[:, -self.config.block_size:] if t > self.config.block_size else idx

            # Run through the model, discard the loss
            # B, T, vocab
            logits, _ = self(idx_cropped)
            
            # What is interesting to us is the last prediction
            prediction_logits = logits[:, -1, :]

            # Turn logits to probabilities with softmax and sample from the distribution
            probs = F.softmax(prediction_logits, dim=-1)
            pick = torch.multinomial(probs, num_samples=1)
            
            # Add the new pick at the end and go again
            idx = torch.cat((idx, pick), dim=1)

        return idx

    def get_n_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
