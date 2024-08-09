import torch
import logging

from model.model import ModelConfig, SmollGPT
from training.dataset import get_dataset
from training.tokenizer import get_tokenizer
from training.telemetry import Telemetry
from training.train import train, TrainingConfig

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    telemetry_dir = './log'
    logger = logging.getLogger()
    logging.basicConfig(
        filename=f'{telemetry_dir}/training.log', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    telemetry = Telemetry(telemetry_dir=telemetry_dir, file_name='metrics')

    tokenizer, vocab_size = get_tokenizer()

    model_config = ModelConfig(
        block_size = 1024,
        n_embed    = 768,
        n_head     = 8,
        n_layer    = 8,
        use_bias   = True,
        device     = device,
        vocab_size = vocab_size,
        flash_att  = True,
        dropout    = 0.1
    )

    train_config = TrainingConfig(
        device=device,
        checkpoint_interval=500,
        checkpoint_dir='./checkpoints',
        epochs=5,
        run_validation_samples=5,
        run_validation_effective_batch_interval=100,
        effective_batch_size=256,
        mini_batch_size=64,
        learning_rate=1.2e-03
    )

    model = SmollGPT(config=model_config)
    model = model.to(device)
    if device == 'cuda':
        model = torch.compile(model)
    else:
        pass

    train_dataloader, test_dataloader = get_dataset(
        file_path='./dataset/wikipedia_ctx_1024.dat',
        batch_size=train_config.mini_batch_size,
        block_size=model_config.block_size
    )

    train(
        model=model, 
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=train_config,
        logger=logger,
        telemetry=telemetry,
        tokenizer=tokenizer
    )

if __name__ == '__main__':
    main()