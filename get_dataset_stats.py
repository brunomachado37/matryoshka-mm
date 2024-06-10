import os
import json

dataset_path = ""

number_of_steps_per_episode = [len(files) for _, _, files in sorted(os.walk(dataset_path), key = lambda x: int(x[0].split("_")[-1]) if x[0].split("_")[-1].isdigit() else -1)][1:]
number_of_episodes = len(number_of_steps_per_episode)
number_of_steps = sum(number_of_steps_per_episode)
accumulated_sum = [sum(number_of_steps_per_episode[:i+1]) for i in range(len(number_of_steps_per_episode))]

meta = {"number_of_episodes": number_of_episodes, "number_of_steps": number_of_steps, "number_of_steps_per_episode": number_of_steps_per_episode, "accumulated_sum": accumulated_sum}

with open(f"{dataset_path}/metadata.json", "w") as outfile: 
    json.dump(meta, outfile)