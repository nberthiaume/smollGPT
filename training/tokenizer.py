import tiktoken

def get_tokenizer():
    tokenizer = tiktoken.get_encoding("p50k_base")
    assert tokenizer.decode(tokenizer.encode("dogs are cool")) == "dogs are cool"
    vocab_size = tokenizer.n_vocab
    return tokenizer, vocab_size
