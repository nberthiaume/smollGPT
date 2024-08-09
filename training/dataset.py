import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class WikipediaDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        self.data = data

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[ 0:self.block_size ]
        y = chunk[ 1:self.block_size + 1 ] # Here is why we made the chunk_size be block_size + 1
        return { 'x': torch.tensor(x, dtype=torch.long), 'y': torch.tensor(y, dtype=torch.long) }
    
    def __len__(self):
        return self.data.shape[0]

def get_dataset(file_path: str, batch_size: int, block_size: int):
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    data = data.reshape((-1, block_size + 1))
    num_chunks = data.shape[0]
    num_training_chunks = int(0.9 * num_chunks)
    train_raw = data[:num_training_chunks]
    test_raw = data[num_training_chunks:]

    train_data = WikipediaDataset(data=train_raw, block_size=block_size)
    test_data = WikipediaDataset(data=test_raw, block_size=block_size)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader