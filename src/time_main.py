import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from lightning.pytorch.cli import LightningCLI
import torch
# from module import SFTModule
from module_time import SFTModule
from datamodule_time import SFTDataModule

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

precision = "high"
torch.set_float32_matmul_precision(precision)

def cli_main():
    # import ipdb; ipdb.set_trace()
    cli = LightningCLI(SFTModule, SFTDataModule)


if __name__ == '__main__':
    cli_main()
    