from typing import Any, Optional, Sequence
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import os
from transformers import AutoTokenizer

class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, model_name):
        super().__init__(write_interval)
        self.output_dir = output_dir
    
    def write_on_epoch_end(self, trainer, pl_module, predictions: Sequence[Any], batch_indices: Sequence[Any] | None) -> None:
        predictions = [tt for t in predictions for tt in t]
        batch_indices = [ttt for t in batch_indices for tt in t for ttt in tt]
        torch.save({"indices": batch_indices, "predictions": predictions}, os.path.join(self.output_dir, f'alltask_{trainer.global_rank}.pt'))