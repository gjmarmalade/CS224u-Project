import os

import torch
import sys
sys.path.append('../')
from utils import compress_seq3
from sys_config import BASE_DIR

checkpoint = "seq3_whole_dataset_use_minor_info_lm_prior_3_65000"
seed = 1
device = "cuda"
verbose = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

datasets = {
    "gigaword": os.path.join(BASE_DIR,
                             "evaluation/gigaword/input_min8.txt"),
    "DUC2003": os.path.join(BASE_DIR, "evaluation/DUC2003/input.txt"),
    "DUC2004": os.path.join(BASE_DIR, "evaluation/DUC2004/input.txt"),
}

for name, src_file in datasets.items():

    if name == "gigaword":
        length = None
    else:
        length = 17

    out_file = os.path.join(BASE_DIR,
                            f"evaluation/hyps/{name}_{checkpoint}_preds.txt")

    compress_seq3(checkpoint, src_file, out_file, device, mode="results")

