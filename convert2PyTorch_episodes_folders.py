import tensorflow_datasets as tfds
import torch
from tqdm import tqdm
import os
import numpy as np
from discretize_actions import discretize

DATASET_PATH = "/gpfsdswork/dataset/DROID"
SAVE_PATH = "/gpfsscratch/rech/uli/ujf38zl/DROID/"

dataset = tfds.load("droid", data_dir=DATASET_PATH, split="train")
episode_count = 0

for episode in tqdm(dataset):
    for step_id, step in enumerate(episode["steps"]):
        instruction = step["language_instruction"].numpy().decode('utf-8')

        if not instruction:
            break

        if step_id == 0:
            episode_path = f"{SAVE_PATH}/episode_{episode_count}"
            os.mkdir(episode_path)
            episode_count += 1
        
        new_step = {}

        new_step['language_instruction'] = instruction
        new_step['image'] = torch.tensor(step['observation']['exterior_image_1_left'].numpy())
        new_step['action'] = discretize(np.concatenate((np.expand_dims(step['action'].numpy(), axis=0), np.expand_dims(np.expand_dims(step['is_terminal'].numpy(), axis=0), axis=0)), axis=1).squeeze())

        with open(f"{episode_path}/step_{step_id}", "wb") as file:
            torch.save(new_step, file)
