from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import numpy as np
import os
import torch

# 设置随机种子，保证实验可复现
def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
	

def main():
    seed_everything(0)
	
    args = parameter_parser()
    # tab_printer(args)

    trainer = Trainer(args)


if __name__ == "__main__":
    main()