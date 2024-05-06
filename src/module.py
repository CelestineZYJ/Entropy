from typing import Any
import lightning.pytorch as pl
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
import math
import functools
import torch



def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_ratio: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_ratio: float = 0.0
):
    lr_lambda = functools.partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_ratio=min_ratio,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class SFTModule(pl.LightningModule):
    def __init__(
        self,
        model_name:str,
        optimizer_lr:float=1e-5,
        optimizer_lr_min_decay:float=1e-1,
        warmup_steps:int=600,
        warmup_total_steps:int=10000,
        do_sample:bool=True,
        temperature:float=0.5,
        num_return_sequences:int=1,
        max_new_tokens:int=100,
        min_new_tokens:int=1,
        load_pretrained:bool=True
    ):
        super().__init__()
        self.model_name = model_name
        if load_pretrained:
            # self.model = GPTNeoXForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision="step3000")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        else:
            # self.model = GPTNeoXForCausalLM.from_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="step3000"))
            self.model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True), trust_remote_code=True)
        self.optimizer_lr = optimizer_lr
        self.optimizer_lr_min_decay = optimizer_lr_min_decay
        self.warmup_steps = warmup_steps
        self.warmup_total_steps = warmup_total_steps
        self.generation_config = {
            'do_sample': do_sample,
            'temperature': temperature,
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'pad_token_id': self.model.config.eos_token_id,
        }
        self.tokenizer = None

    def forward(self, batch):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def split_old_new_tokens(self, outputs, input_ids):
        num_return_sequences = self.generation_config['num_return_sequences']
        seq_length = input_ids.shape[1]
        generated_ids = outputs[:, seq_length:]
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision="step3000")
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        outputs = [
            {"input_text": input_texts[i], "generated_text": generated_texts[i * num_return_sequences: (i + 1) * num_return_sequences]}
            for i in range(len(input_texts))
        ]
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.generation_config
        )
        outputs = self.split_old_new_tokens(outputs, input_ids)
        return outputs#, batch['batch_indices']
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW([p for p in self.trainer.model.parameters() if p.requires_grad], lr=self.optimizer_lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, self.warmup_total_steps, min_ratio=self.optimizer_lr_min_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }