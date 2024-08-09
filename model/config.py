from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int # The max length of token sequences the model can use. i.e. context_window
    n_embed: int    # Dimension of the embedding space
    n_layer: int    # Number of transformer blocks
    n_head: int     # Number of attention heads per block
    use_bias: bool  # Whether to use the intercept terms in layers
    device: str     # Whether to use cuda, mps, or cpu
    vocab_size: int # Size of the tokenization vocabulary
    flash_att: bool

    dropout: float  # How much dropout to use during training
