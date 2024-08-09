import csv
import time
from dataclasses import dataclass

@dataclass
class TelemetryDataPoint:
    epoch: int
    tokens_processed: int
    effective_batch_number: int
    training_loss: float
    validation_loss: float
    learning_rate: float
    gradient_norm: float

class Telemetry:
    def __init__(self, telemetry_dir: str, file_name: str, buffer_size=100):
        self.file = f"{telemetry_dir}/{file_name}.csv"
        self._write_header()
        self.buffer_size = buffer_size
        self.buffer = []

    def _write_header(self):
        with open(self.file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'epoch', 'tokens_processed', 'effective_batch_number', 'training_loss', 'validation_loss', 'learning_rate', 'gradient_norm'])

    def log(self, d: TelemetryDataPoint):
        self.buffer.append([
            time.time(),
            d.epoch,
            d.tokens_processed,
            d.effective_batch_number,
            d.training_loss,
            d.validation_loss,
            d.learning_rate,
            d.gradient_norm.item()
        ])

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        with open(self.file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer = []
