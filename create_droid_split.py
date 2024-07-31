import os
from tqdm import tqdm
import pickle as pkl
from distutils.dir_util import copy_tree


VERB = "pick"

dataset_path = "/gpfsscratch/rech/uli/uuv83ah/droid_pickle/train"
save_path = f"/gpfsscratch/rech/uli/uuv83ah/droid_{VERB}/train"

if not os.path.exists(save_path):
	os.makedirs(save_path)

episode_count = 0

for folder in tqdm(os.listdir(dataset_path)):
    if 'episode' in folder:
        with open(f"{dataset_path}/{folder}/step_0", "rb") as file:
            step = pkl.load(file)

        if VERB in step['language_instruction'].lower():
            copy_tree(f"{dataset_path}/{folder}", f"{save_path}/episode_{episode_count}")
            episode_count += 1

        del step


