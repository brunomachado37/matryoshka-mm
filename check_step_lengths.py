import os
import pickle
from tqdm import tqdm
import json
import pickle as pkl

dataset_path = "/gpfsscratch/rech/uli/uuv83ah/droid_pick/train"

length_list = []

for folder in tqdm(os.listdir(dataset_path)):
    if not '.' in folder:
        with open(f"{dataset_path}/{folder}/step_0", "rb") as file:       
            step = pkl.load(file)
        
        input_ids = f"{step['language_instruction']} Next Action: {step['action']}"
        length_list.append(len(input_ids.split()))
        del step

with open(f"{dataset_path}/metadata.json", "r") as file: 
    meta = json.load(file)

meta["text_lengths"] = length_list

with open(f"{dataset_path}/metadata.json", "w") as file:
    json.dump(meta, file)
