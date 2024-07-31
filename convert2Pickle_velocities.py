import tensorflow_datasets as tfds
import torch
from tqdm import tqdm
import os
import numpy as np
from data.discretize_actions import discretize_velocities
import random
from PIL import Image
import pickle as pkl

DATASET_PATH = "/gpfsdswork/dataset/DROID"
SAVE_PATH = "/gpfsscratch/rech/uli/uuv83ah/droid_pickle/"

dataset = tfds.load("droid", data_dir=DATASET_PATH, split="train")
episode_count = 0

for episode in tqdm(dataset):
    for step_id, step in enumerate(episode["steps"]):
        if step_id == 0:
            instructions = [step["language_instruction"].numpy().decode('utf-8'), step["language_instruction_2"].numpy().decode('utf-8'), step["language_instruction_3"].numpy().decode('utf-8')]
            instructions = [instruction for instruction in instructions if instruction]

            if len(instructions) == 0:
                break

            episode_path = f"{SAVE_PATH}/episode_{episode_count}"
            os.mkdir(episode_path)
            episode_count += 1
        
        new_step = {}

        new_step['language_instruction'] = random.choice(instructions)
        new_step['image'] = Image.fromarray(step['observation'][f'exterior_image_{random.randint(1, 2)}_left'].numpy())
        new_step['action'] = discretize_velocities(np.concatenate((np.expand_dims(step['action_dict']['cartesian_velocity'].numpy(), axis=0),
                                                        np.expand_dims(1 - step['action_dict']['gripper_position'].numpy(), axis=0),
                                                        np.expand_dims(np.expand_dims(step['is_terminal'].numpy(), axis=0), axis=0)), axis=1).squeeze())

        with open(f"{episode_path}/step_{step_id}", "wb") as file:
            pkl.dump(new_step, file)
