import torch
import numpy as np
import math
import logging
import os

from dataclasses import dataclass
from training.telemetry import Telemetry, TelemetryDataPoint


@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int
    run_validation_effective_batch_interval: int
    checkpoint_interval: int
    checkpoint_dir: str
    run_validation_samples: int
    device: str
    mini_batch_size: int
    effective_batch_size: int

def get_cosine_decay_f(initial_learning_rate: float, final_learning_rate: float, decay_until_iter_number: int, warmup_steps: int):
    def cosine_decay(i: int):
        if i < warmup_steps:
            return (i/warmup_steps) * initial_learning_rate
        elif i > decay_until_iter_number:
            return final_learning_rate
        elif i == 0:
            return initial_learning_rate
        elif 0 < i <= decay_until_iter_number:
            coeff = 0.5 * (1 + np.cos(math.pi * ((i - warmup_steps)/decay_until_iter_number)))
            lr = final_learning_rate + coeff * (initial_learning_rate - final_learning_rate)
            return lr
    return cosine_decay

def measure_loss_on_test_data(model, test_data, config: TrainingConfig) -> float:
    batch_size  = config.run_validation_samples

    def get_samples(data):
        random_indices = np.random.choice(len(data), batch_size, replace=False)
        xs = []; ys = []
        for idx in random_indices:
            sample = data[idx]
            xs.append(sample['x']), ys.append(sample['y'])

        x = torch.stack(xs); y = torch.stack(ys)
        return x.to(config.device), y.to(config.device)

    x_test, y_test = get_samples(test_data)


    _, validation_loss = model(x_test, y_test)
    
    return validation_loss

def generate_something(model, tokenizer, device, prompt='I am smollGPT, a language model built for learning, and I like to'):
    prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    return tokenizer.decode(model.generate(prompt, 20)[0].tolist())

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, filename)

def train(
        model,
        train_dataloader,
        test_dataloader,
        config: TrainingConfig,
        logger: logging.Logger,
        telemetry: Telemetry,
        tokenizer
    ):
    assert config.effective_batch_size % config.mini_batch_size == 0, 'mini_batch_size must perfectly divide effective_batch_size'
    mini_batches_per_effective_batch = config.effective_batch_size / config.mini_batch_size

    # Ensure the checkpoint location exists and is empty
    if os.path.exists(config.checkpoint_dir):
        files_in_checkpoint_dir = os.listdir(config.checkpoint_dir)
        assert len(files_in_checkpoint_dir) == 0, "Checkpoint directory is not empty."
    else:
        os.makedirs(config.checkpoint_dir)

    # Assess the amount of training tokens available and batches to run
    examples_in_training_data     = train_dataloader.dataset.data.shape[0]
    tokens_per_example            = train_dataloader.dataset.data.shape[1] - 1
    tokens_in_training_data       = tokens_per_example * examples_in_training_data
    tokens_per_effective_batch    = config.effective_batch_size * tokens_per_example
    n_params = model.get_n_params()

    logger.info(f"{tokens_in_training_data / 1e9 :.2f} Billion tokens in training data.")
    logger.info(f"{tokens_per_effective_batch / 1e3 :.2f} K tokens per effective batch.")
    logger.info(f"Training on {config.device}.")
    logger.info(f"Model has { n_params/ 1e6 :.2f} M parameters")
    logger.info(f"Model has { tokens_in_training_data / n_params :.2f} tokens per parameters")


    torch.set_float32_matmul_precision('high')

    # Using AdamW as in the GPT-3 paper
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-08)

    cosine_decay_f = get_cosine_decay_f(
        initial_learning_rate=config.learning_rate, 
        final_learning_rate=0.1*config.learning_rate,
        decay_until_iter_number=tokens_in_training_data / tokens_per_effective_batch,
        warmup_steps=(tokens_in_training_data / tokens_per_effective_batch) / 5, # 2% of epoch
    )

    tokens_processed = 0
    effective_batch_number = 0
    
    for epoch in range(config.epochs):
        logger.info(f"Begin epoch {epoch}")

        minibatch_counter = 0
        loss_counter = 0
        latest_validation_loss = measure_loss_on_test_data(model=model, test_data=test_dataloader.dataset, config=config)
        for i, minibatch in enumerate(train_dataloader):

            # Load data onto GPU
            x = minibatch['x'].to(config.device); y = minibatch['y'].to(config.device)

            # Run the forward pass
            if config.device == 'cuda':
                # Pytorch magic - makes use of BFloat16 whenever possible
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)

            loss = loss / mini_batches_per_effective_batch
            loss_counter += loss.detach()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Usage of gradient clipping like GPT-3
            minibatch_counter += 1
            if minibatch_counter == mini_batches_per_effective_batch:

                # Update parameters and reset gradient accumulation
                optimizer.step()
                optimizer.zero_grad()
                minibatch_counter = 0

                # Update learning rate
                new_lr = cosine_decay_f(effective_batch_number)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                # Once in a while check the validation loss
                if effective_batch_number % config.run_validation_effective_batch_interval == 0:
                    with torch.no_grad():
                        model.eval()
                        latest_validation_loss = measure_loss_on_test_data(model=model, test_data=test_dataloader.dataset, config=config)
                        generation1 = generate_something(model=model, tokenizer=tokenizer, device=config.device, prompt='Kill two birds with')
                        generation2 = generate_something(model=model, tokenizer=tokenizer, device=config.device, prompt='The capital of Italy is')
                        generation3 = generate_something(model=model, tokenizer=tokenizer, device=config.device)

                    model.train()
                    logger.info(f"Validation loss measured at: {latest_validation_loss :.2f}")
                    logger.info(f"Sample from model: {generation1}")
                    logger.info(f"Sample from model: {generation2}")
                    logger.info(f"Sample from model: {generation3}")

                if effective_batch_number % config.checkpoint_interval == 0:
                    save_checkpoint(
                        model=model, 
                        optimizer=optimizer, 
                        epoch=epoch, 
                        loss=loss, 
                        filename=f"{config.checkpoint_dir}/checkpoint_{effective_batch_number}.pt"
                    )

                # Handle some telemetry
                logger.debug("Updated weights")
                tokens_processed += minibatch['x'].shape[0] * minibatch['x'].shape[1] * mini_batches_per_effective_batch
                logger.info(f"Effective batch: {effective_batch_number} | Training loss: {loss_counter.item()}")
                telemetry.log(TelemetryDataPoint(
                    epoch=epoch,
                    tokens_processed=tokens_processed,
                    effective_batch_number=effective_batch_number,
                    training_loss=loss_counter.item(),
                    validation_loss=latest_validation_loss.item(),
                    learning_rate=optimizer.param_groups[0]['lr'],
                    gradient_norm=norm
                ))
                effective_batch_number += 1
                loss_counter = 0

            # TODO Figure out distributed proccesing
            # TODO Make everything float16???
            
        #  === Post epoch calculations ===
        # Measure loss for epoch
        # Keep track of the loss 
        # print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_valdiation_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")


