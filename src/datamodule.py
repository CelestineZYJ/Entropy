from functools import partial
import json
import os
import lightning.pytorch as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from typing import List, Optional, Tuple, Dict
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import glob
import numpy.random as random



class TextProcessor(object):
    def __init__(
        self,
        data:List[Dict], 
        tokenizer:Optional[PreTrainedTokenizerBase],
        keys:List[Tuple[str, bool]],
        add_eos_token:bool=True,
        concat_samples:bool=True
    ):
        
        self.data = data
        self.keys = keys
        self.tokenizer = tokenizer
        self.tokens = []
        self.mask = []
        self.add_eos_token = add_eos_token
        self.concat_samples = concat_samples
        self.tokenize()
    
    def tokenize(self,):
        tokens = []
        mask = []
        for example in self.data:
            example_tokens = []
            example_mask = []
            for key, is_target in self.keys:
                encodings = self.tokenizer.encode(example[key])
                example_tokens.extend(encodings)
                example_mask.extend([int(is_target)] * len(encodings))
            if self.add_eos_token:
                example_tokens.append(self.tokenizer.eos_token_id)
                example_mask.append(1)
            if self.concat_samples:
                tokens.extend(example_tokens)
                mask.extend(example_mask)
            else:
                tokens.append(example_tokens)
                mask.append(example_mask)
                    
        self.tokens = tokens
        self.mask = mask


class TextDataset(Dataset):
    def __init__(self, tokens, mask, seq_len:int=512, concat_samples:bool=True, randomize_max_position:int=-1):
        self.tokens = tokens
        self.mask = mask
        self.seq_len = seq_len
        self.concat_samples = concat_samples
        self.randomize_max_position = randomize_max_position
    
    @classmethod
    def from_processor(cls, processor, seq_len:int=512, randomize_max_position:int=-1):
        return cls(processor.tokens, processor.mask, seq_len=seq_len, concat_samples=processor.concat_samples, randomize_max_position=randomize_max_position)
    
    def __len__(self):
        if self.concat_samples:
            return (len(self.tokens) - 1) // self.seq_len + 1
        else:
            return len(self.tokens)
    
    def __getitem__(self, idx):
        if self.concat_samples:
            start = idx * self.seq_len
            end = (idx + 1) * self.seq_len
            return {
                "input_ids": self.tokens[start:end],
                "labels": [t if m == 1 else -100 for t, m in zip(self.tokens[start:end], self.mask[start:end])],
                "idx": idx
            }
        else:
            seq_length = min(self.seq_len, len(self.tokens[idx]))
            if self.randomize_max_position > 0:
                position_offset = random.randint(0, self.randomize_max_position - seq_length)
                return {
                    "input_ids": self.tokens[idx][:seq_length],
                    "labels": [t if m == 1 else -100 for t, m in zip(self.tokens[idx], self.mask[idx])][:seq_length],
                    "idx": idx,
                    "position_ids": [i + position_offset for i in range(seq_length)]
                }
            else:
                return {
                    "input_ids": self.tokens[idx][:seq_length],
                    "labels": [t if m == 1 else -100 for t, m in zip(self.tokens[idx], self.mask[idx])][:seq_length],
                    "idx": idx
                }


def collate_fn(batch, left_pad=False, pad_token_id:int=0):

    max_len = max([len(x["input_ids"]) for x in batch])
    input_ids = torch.ones((len(batch), max_len), dtype=torch.long) * pad_token_id
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros((len(batch), max_len), dtype=torch.long)
    position_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    
    if left_pad:
        for i, x in enumerate(batch):
            input_ids[i, -len(x["input_ids"]):] = torch.tensor(x["input_ids"], dtype=torch.long)
            attention_mask[i, -len(x["input_ids"]):] = 1
            labels[i, -len(x["input_ids"]):] = torch.tensor(x["labels"], dtype=torch.long)
            if "position_ids" in x:
                position_ids[i, -len(x["input_ids"]):] = torch.tensor(x["position_ids"], dtype=torch.long)
    else:
        for i, x in enumerate(batch):
            input_ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"], dtype=torch.long)
            attention_mask[i, :len(x["input_ids"])] = 1
            labels[i, :len(x["input_ids"])] = torch.tensor(x["labels"], dtype=torch.long)
            if "position_ids" in x:
                position_ids[i, :len(x["position_ids"])] = torch.tensor(x["position_ids"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "batch_indices": [x["idx"] for x in batch],
        "position_ids": position_ids if position_ids.sum() > 0 else None
    }


def parse_keys(keys):
    keys = keys.split(",")
    keys = [k.split(":") for k in keys]
    keys = [(k, True if v == "1" else False) for k, v in keys]
    return keys


class SFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        keys:str="text:1",
        data_dir: str = "./lmft",
        batch_size:int=8,
        seq_len:int=1024,
        tokenizer_name:str="microsoft/phi-2", # EleutherAI/pythia-2.8b  EleutherAI/pythia-1b-deduped
        concat_samples:bool=True,
        randomize_max_position:int=-1,
        num_workers:int=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train = None
        self.val = None
        self.test = None
        self.predict = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.keys = parse_keys(keys)
        self.concat_samples = concat_samples
        self.randomize_max_position = randomize_max_position

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        current_dir = 'negation_10/' # negation_100 aRandomb111_multiGroup_10k_10
        if stage == "fit":
            train_file = os.path.join(self.data_dir, current_dir+"train.json")
            with open(train_file, "rt") as f:
                train = [json.loads(l) for l in f]
            train_file = os.path.join(self.data_dir, current_dir+"rest.json")
            with open(train_file, "rt") as f:
                train += [json.loads(l) for l in f]
            val_file = os.path.join(self.data_dir, current_dir+"val.json")
            with open(val_file, "rt") as f:
                val = [json.loads(l) for l in f]

            train_processor = TextProcessor(train, self.tokenizer, keys=self.keys, concat_samples=self.concat_samples)
            val_processor = TextProcessor(val, self.tokenizer, keys=self.keys, concat_samples=self.concat_samples)
            self.train = TextDataset.from_processor(train_processor, seq_len=self.seq_len, randomize_max_position=self.randomize_max_position)
            self.val = TextDataset.from_processor(val_processor, seq_len=self.seq_len, randomize_max_position=self.randomize_max_position)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            with open(os.path.join(self.data_dir, current_dir+"test.json"), "rt") as f:
                predict = [json.loads(l) for l in f]
            predict_processor = TextProcessor(predict, self.tokenizer, add_eos_token=False, keys=self.keys, concat_samples=self.concat_samples)
            self.predict = TextDataset.from_processor(predict_processor, seq_len=self.seq_len, randomize_max_position=-1)

    def train_dataloader(self):
        return DataLoader(self.train, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=partial(collate_fn, left_pad=False, pad_token_id=self.tokenizer.pad_token_id))

    def val_dataloader(self):
        return DataLoader(self.val, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=partial(collate_fn, left_pad=False, pad_token_id=self.tokenizer.pad_token_id))

    def test_dataloader(self):
        return DataLoader(self.test, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=partial(collate_fn, left_pad=False, pad_token_id=self.tokenizer.pad_token_id))

    def predict_dataloader(self):
        return DataLoader(self.predict, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=partial(collate_fn, left_pad=True, pad_token_id=self.tokenizer.pad_token_id))