import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint
import torch
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Mapping, Union
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from palm_rlhf_pytorch import PaLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,7'
train_strategy = 'syn_time_10'
load_previous_checkpoint=False
localpath = "/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/NousResearch/Llama-2-7b-chat-hf-lr3e-05-seq1600-ratio0.03/final_"+train_strategy+'100'

import datasets

from accelerate import Accelerator, DistributedType
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    GPTNeoXForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    MistralForCausalLM,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import warnings

warnings.filterwarnings('ignore')

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Set environment variables
os.environ["HF_DATASETS_CACHE"] = "data/.hf_dataset_cache"
os.environ["WANDB_PROJECT"] = "SIU"

# Define parameters
# BASE_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_MODEL_PATH='NousResearch/Llama-2-7b-chat-hf'
# BASE_MODEL_PATH='facebook/opt-1.3b'
# BASE_MODEL_PATH='EleutherAI/pythia-1.4b-deduped'
SEQ_LENGTH = 100 # 1600
# LR = 6.5e-6
LR = 1e-5
WEIGHT_DECAY = 1e-2
PER_DEVICE_BSZ = 1
GRAD_ACC = 1
N_EPOCH = 50
RATIO = 0.03
DATA_ROOT = "data"

# Define output directory
EXP_ID = f"{BASE_MODEL_PATH}-lr{LR}-seq{SEQ_LENGTH}-ratio{RATIO}"
OUTPUT_DIR = f"models/ckpts/{EXP_ID}"
os.environ["WANDB_NAME"] = EXP_ID
print(f"Saving to {OUTPUT_DIR}")

os.environ["HF_DATASETS_CACHE"] = "data/.hf_dataset_cache"
os.environ["WANDB_PROJECT"] = "SIU"

# Set sys.argv to simulate command line arguments
sys.argv = [
    "nl_train.py",
    "--model_name_or_path", BASE_MODEL_PATH,
    "--with_tracking",
    "--output_dir", OUTPUT_DIR,
    "--report_to", "wandb",
    "--seed", "42",
    "--preprocessing_num_workers", "32",
    "--per_device_train_batch_size", str(PER_DEVICE_BSZ),
    "--gradient_accumulation_steps", str(GRAD_ACC),
    "--num_train_epochs", str(N_EPOCH),
    "--checkpointing_steps", "epoch",
    "--lr_scheduler_type", "linear",
    "--learning_rate", str(LR),
    "--weight_decay", str(WEIGHT_DECAY),
    "--lr_scheduler_warmup_ratio", str(RATIO),
    "--block_size", str(SEQ_LENGTH),
    "--train_file", os.path.join(DATA_ROOT, train_strategy+"/train.json"),
    "--validation_file", os.path.join(DATA_ROOT, train_strategy+"/val.json")
]

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv, txt or a json file containing the test data.",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--variable_seq_lengths",
        action="store_true",
        help="If passed, will use variable sequence lengths (up to max_seq_length) for training.",
    )
    parser.add_argument(
        "--finished_model",
        type=str,
        help="Path to finished model",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--lr_scheduler_warmup_ratio", type=float, default=0.01, help="Ratio for warmup steps in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    args = parser.parse_args()
    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args

def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    
    # if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
    #     from accelerate import FullyShardedDataParallelPlugin
    #     fsdp_plugin = FullyShardedDataParallelPlugin(
    #         activation_checkpointing=True,
    #     )
    #     accelerator_log_kwargs["fsdp_plugin"] = fsdp_plugin

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    # data_files = {"train": args.train_file, "validation": args.validation_file, "test": args.test_file}
    data_files = {"train": args.train_file, "validation": args.validation_file}
    # accelerator.print(data_files)
    accelerator.print("Start loading dataset")
    raw_datasets = load_dataset('json', data_files=data_files)
    
    accelerator.print(raw_datasets.keys())
    accelerator.print("Finish loading dataset")

      # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        accelerator.print('Finish loading config')
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:   
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
        
    elif args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token_id = 0

    if args.model_name_or_path:
        accelerator.print("Start loading pretrianed")
        if BASE_MODEL_PATH == 'EleutherAI/pythia-1b-deduped'or BASE_MODEL_PATH == 'EleutherAI/pythia-1.4b-deduped': 
            model = transformers.GPTNeoXForCausalLM.from_pretrained(
            # model = transformers.LlamaForCausalLM.from_pretrained(
            # model = MistralForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16
                # this is important, without this bfloat16, the code go wrong.
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
            # model = transformers.LlamaForCausalLM.from_pretrained(
            # model = MistralForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16
                # this is important, without this bfloat16, the code go wrong.
            )
        accelerator.print("Finish loading pretrianed")
    else:
        logger.info("Training new model from scratch")
        # model = transformers.LlamaForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        model = transformers.AutoModelForCausalLM.from_pretrained(config, trust_remote_code=args.trust_remote_code)
    if load_previous_checkpoint:
        config = AutoConfig.from_pretrained(
            localpath
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            localpath
        )
        accelerator.print('Finish loading config')
        accelerator.print("Start loading pretrained")
            
        model = transformers.AutoModelForCausalLM.from_pretrained(
            localpath,
            config=config
        )
        accelerator.print("Finish loading pretrained")
        
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     # model.resize_token_embeddings(len(tokenizer))
    #     model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    #     accelerator.print(f"Embedding size updated to {model.get_input_embeddings().weight.shape[0]}")
    #     # config.vocab_size = model.get_input_embeddings().weight.shape[0]
    
        # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    
    # prompt_template = "Classify the text into background, purpose, method, finding, or other.\nText: %s\nType: "
#     Instruction: You are a researcher. You can come up with new hypotheses
# based on your existing knowledge. Hypotheses are given against the
# following background. You should be as detailed as possible

    prompt_template = ""
    target_template = "{context}"
    if train_strategy == 'tempReason' or train_strategy == 'onlineAdapt':
        prompt_template = "{prefix}{midfix}"
        target_template = "{context}"
    def tokenize_function(examples):
        # input = examples['molecule']
        # target = examples['caption']
        tokens = []
        loss_masks = []  # 1 for assistant, 0 for others
        # txt = input_template % input
        # prompt_ = prompt_template % txt
        if train_strategy == 'tempReason' or train_strategy == 'onlineAdapt':
            prompt_ = prompt_template.format(prefix=examples['prefix'],midfix=examples["midfix"])
            target = target_template.format(context=examples['context'])
        else:
            target = target_template.format(context=examples['text'])
        # p_token = tokenizer(prompt_).input_ids
        # loss_masks.extend([0] * len(p_token))
        # tokens.extend(p_token)
        c_token = tokenizer(target).input_ids[1:] + [tokenizer.eos_token_id]
        # c_token = tokenizer(target, max_length=100,truncation=True).input_ids[1:] + [tokenizer.eos_token_id]
        loss_masks.extend([1] * len(c_token))
        # print()
        # print(examples["idx"])
        # print(examples["paper_id"])
        # input()
        # print(p_token)
        # print(c_token)
        # input()
        # print(type(p_token))
        # print(type(c_token))
        tokens.extend(c_token)
        # print(tokens)
        return  {
            "input_ids": tokens,
            "loss_masks": loss_masks,
        }
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
    def conv_collator(
        batch
    ) -> Dict:
        IGNORE_TOKEN_ID = -100  # -100 is the default value for ignore_index in CrossEntropyLoss
        def round_to_multiple_of(x: int, y: int) -> int:
            return ((x + y - 1) // y) * y
        
        batch_seq_length = args.block_size
        if args.variable_seq_lengths:
            batch_seq_length = max(len(x["input_ids"]) for x in batch)
            batch_seq_length = min(args.block_size, round_to_multiple_of(batch_seq_length, 16))
            # batch_seq_length = min(args.block_size, batch_seq_length)
        
        # pad data to seq_len, create attention mask
        batch_size = len(batch)
        attention_mask = torch.ones((batch_size, batch_seq_length), dtype=torch.long)
        
        # Need to use pad instead of IGNORE_TOKEN_ID since latter will be sent to model and cause index error
        input_ids = torch.full_like(attention_mask, tokenizer.pad_token_id)
        loss_masks = torch.ones_like(attention_mask)  # 1 for assistant, 0 for others
        
        # other_infos = defaultdict(list)
        
        for batch_idx, tokenized_conv in enumerate(batch):
            
            cur_input_ids = tokenized_conv["input_ids"]
            cur_loss_masks = tokenized_conv["loss_masks"]
            cur_seq_len = len(cur_input_ids)
            assert len(cur_loss_masks) == cur_seq_len
            
            # Truncate if necessary
            # if cur_seq_len > batch_seq_length:
            #     cur_input_ids = cur_input_ids[:batch_seq_length]
            #     cur_loss_masks = cur_loss_masks[:batch_seq_length]
            #     cur_seq_len = batch_seq_length

            # assert cur_seq_len <= batch_seq_length
            attention_mask[batch_idx, :-cur_seq_len] = 0
            input_ids[batch_idx, -cur_seq_len:] = torch.from_numpy(np.array(cur_input_ids))
            loss_masks[batch_idx, -cur_seq_len:] = torch.from_numpy(np.array(cur_loss_masks))
            
            # for col_name in column_names:
            #     other_infos[col_name].append(tokenized_conv[col_name])

        # Use loss_masks to find labels (when loss_mask == 0 -> labels == -100, otherwise labels == input_ids)
        # labels will be shifted by transformers so no need to shift here
        labels = input_ids.clone()
        labels[loss_masks == 0] = IGNORE_TOKEN_ID
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # "other_infos": other_infos
        }
    
    
    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]
    # test_dataset = lm_datasets["test"]
    
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=conv_collator, batch_size=args.per_device_train_batch_size
    )
    val_dataloader = DataLoader(
        valid_dataset, collate_fn=conv_collator, batch_size=args.per_device_eval_batch_size
    )
    # test_dataloader = DataLoader(
    #     test_dataset, collate_fn=conv_collator, batch_size=args.per_device_eval_batch_size
    # )
    
    if accelerator.distributed_type == DistributedType.FSDP:
        model = accelerator.prepare(model)        

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = args.max_train_steps * args.lr_scheduler_warmup_ratio
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        # num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_warmup_steps=num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if accelerator.distributed_type == DistributedType.FSDP:
        optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler, val_dataloader#, test_dataloader
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, val_dataloader#, test_dataloader
        )
        
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("hypothesis", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]


    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    def extract_sentence(tokens):
        # Filter out '!' and join the tokens to form a sentence
        sentence = ' '.join(token for token in tokens if token != '!' and token.strip() != '')
        # Remove leading and trailing whitespace
        sentence = sentence.strip()
        return sentence
    

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # outputs = model(**batch)
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                outputs=model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                # loss = loss.cpu()
                # optimizer.param_groups[0]['lr'] = 1e-6

                accelerator.backward(loss)
                optimizer.step()
                
                lr_scheduler.step()
                
                # import pdb; pdb.set_trace()
                inputs= batch["input_ids"].cpu().numpy()[0]
                # text = extract_sentence([tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs])
                
                # batch['input_ids']=batch['input_ids'].cuda()
                # output = model.generate(inputs = batch['input_ids'], max_new_tokens=50, do_sample=True, top_k=5)
                # generated_sents = tokenizer.batch_decode(output)
                # import pdb; pdb.set_trace()
                # print('*'*100)
                # print(text)
                # print('$'*100)
                # print(generated_sents)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    
            if completed_steps >= args.max_train_steps:
                break
            if args.with_tracking:
                if step % 300 == 0:
                    accelerator.log(
                        {
                            "train_loss": total_loss.item() / (step+1),
                            # "epoch": epoch,
                            # "step": step,
                        },
                        step=completed_steps,
                    )
        logger.info(f"epoch {epoch}: train_loss: {total_loss.item()}")

        model.eval()
        losses = []
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    # "step": completed_steps,
                },
                step=completed_steps,
            )


        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if 1:
        # if (epoch+1) % 10 == 0 or epoch > args.num_train_epochs-2:
            if args.output_dir is not None:
                output_dir_model = args.output_dir + '/final_'+train_strategy
                # accelerator.print(f"save checkpoint: {args.output_dir}")
                accelerator.print(f"save checkpoint: {output_dir_model}")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                # if 1:#args.push_to_hub and epoch < args.num_train_epochs - 1:
                #     accelerator.wait_for_everyone()
                #     unwrapped_model = accelerator.unwrap_model(model)
                #     unwrapped_model.save_pretrained(
                #         output_dir_model, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                #     )
                #     if accelerator.is_main_process:
                #         tokenizer.save_pretrained(args.output_dir)
                #         repo.push_to_hub(
                #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                #         )

                if accelerator.distributed_type == DistributedType.FSDP:
                    unwrapped_model.save_pretrained(
                        # args.output_dir, 
                        output_dir_model,
                        is_main_process=accelerator.is_main_process, 
                        save_function=accelerator.save,
                        safe_serialization=False,
                        # state_dict=state,
                        state_dict=accelerator.get_state_dict(model, unwrap=False),
                    )

                else:
                    unwrapped_model.save_pretrained(
                        # args.output_dir, 
                        output_dir_model,
                        is_main_process=accelerator.is_main_process, 
                        save_function=accelerator.save,
                        safe_serialization=False,
                        state_dict=accelerator.get_state_dict(model),
                        # state_dict=accelerator.get_state_dict(unwrapped_model, unwrap=False),
                    )
                if accelerator.is_main_process:
                    # accelerator.save_model(unwrapped_model, output_dir_model)
                    tokenizer.save_pretrained(
                        output_dir_model
                        # args.output_dir
                        )
    # if args.output_dir is not None:
    #     output_dir_model = args.output_dir + '/final_pretrain'
    #     # accelerator.print(f"save checkpoint: {args.output_dir}")
    #     accelerator.print(f"save checkpoint: {output_dir_model}")
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)

    #     # if 1:#args.push_to_hub and epoch < args.num_train_epochs - 1:
    #     #     accelerator.wait_for_everyone()
    #     #     unwrapped_model = accelerator.unwrap_model(model)
    #     #     unwrapped_model.save_pretrained(
    #     #         output_dir_model, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #     #     )
    #     #     if accelerator.is_main_process:
    #     #         tokenizer.save_pretrained(args.output_dir)
    #     #         repo.push_to_hub(
    #     #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
    #     #         )

    #     if accelerator.distributed_type == DistributedType.FSDP:
    #         unwrapped_model.save_pretrained(
    #             # args.output_dir, 
    #             output_dir_model,
    #             is_main_process=accelerator.is_main_process, 
    #             save_function=accelerator.save,
    #             safe_serialization=False,
    #             # state_dict=state,
    #             state_dict=accelerator.get_state_dict(model, unwrap=False),
    #         )

    #     else:
    #         unwrapped_model.save_pretrained(
    #             # args.output_dir, 
    #             output_dir_model,
    #             is_main_process=accelerator.is_main_process, 
    #             save_function=accelerator.save,
    #             safe_serialization=False,
    #             state_dict=accelerator.get_state_dict(model),
    #             # state_dict=accelerator.get_state_dict(unwrapped_model, unwrap=False),
    #         )
    #     if accelerator.is_main_process:
    #         # accelerator.save_model(unwrapped_model, output_dir_model)
    #         tokenizer.save_pretrained(
    #             output_dir_model
    #             # args.output_dir
    #             )


if __name__ == "__main__":
    main()
