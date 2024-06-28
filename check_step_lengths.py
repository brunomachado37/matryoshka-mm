import os
import torch
from tqdm import tqdm
import json

dataset_path = "/gpfsscratch/rech/uli/ujf38zl/DROID/train"

length_list = []

for folder in tqdm(os.listdir(dataset_path)):
    if not '.' in folder:
        with open(f"{dataset_path}/{folder}/step_0", "rb") as file:       
            step = torch.load(file)
        
        input_ids = f"{step['language_instruction']} Next Action: {step['action']}"
        length_list.append(len(input_ids.split()))
        del step

with open(f"{dataset_path}/metadata.json", "r") as file: 
    meta = json.load(file)

meta["text_lengths"] = length_list

with open(f"{dataset_path}/metadata.json", "w") as file:
    json.dump(meta, file)
