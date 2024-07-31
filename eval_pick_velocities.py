import tensorflow_datasets as tfds
import torch
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import transformers

from llava.model import LlavaLlamaForCausalLM
from data_utils import DataArguments, make_supervised_data_module, expand2square, format_input_prompt, preprocess_droid
from discretize_actions import discretize_velocities


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


DATASET_PATH = "../datasets/"
BATCH_SIZE = 1
DUMMY_DATASET_PATH = "../datasets/droid_torch_dc"
NUM_EPISODES = 5000


def compute_metrics(outputs, labels, collated_batch):  
    action_index = torch.transpose(torch.nonzero(labels != -100, as_tuple=False).squeeze(-1), 0, 1)
    target = labels[action_index[0], action_index[1]].unsqueeze(1)

    logits = outputs[:, -labels.shape[1]:, :]                                   # It only works because the image is known to be in the beginning of the sequence
    action_index[1, :] -= 1                                                     # Adjust for the shift
    logits = logits[action_index[0], action_index[1], :]

    action_logits = logits[:,31744:32000]           # Just look at valid actions
    probs = action_logits.softmax(dim=-1)
    conf, predictions = probs.topk(1)
    predictions += 31744                            # Map it back to the vocabulary space

    acc = float(torch.sum(predictions == target) / len(predictions))
    mae = float(torch.sum(torch.abs(predictions - target)) / len(predictions))
    mse = float(torch.sum(torch.square(predictions - target)) / len(predictions))

    all_conf, all_predictions = logits.topk(1)
    # print(all_predictions.squeeze(-1))
    # print(target.squeeze(-1))
    # print(float(torch.sum(target == all_predictions) / (all_predictions.shape[0] * all_predictions.shape[1])))
    # print(acc)

    return acc, mse, mae


model = LlavaLlamaForCausalLM.from_pretrained("models/pick/test_02-dev-M3-pick", torch_dtype=torch.bfloat16).to("cuda:0")
model.eval()

vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.float32)

data_args = DataArguments(data_path=f"{DUMMY_DATASET_PATH}/train", 
                          eval_path=f"{DUMMY_DATASET_PATH}/eval",
                          image_aspect_ratio="pad",
                          image_processor=vision_tower.image_processor,
                          matryoshka_vis_token_scale=36)

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer",
                                                       model_max_length=2048,
                                                       padding_side="right",
                                                       use_fast=False)

data_module = make_supervised_data_module(tokenizer, data_args)

dataset = tfds.load("droid", data_dir=DATASET_PATH, split="train")
episode_count = 0

acc, mse, mae, loss, batch = [], [], [], [], []

for episode in tqdm(dataset):
    if episode_count >= NUM_EPISODES:
        break

    for step_id, step in enumerate(episode["steps"]):
        if step_id == 0:
            instructions = [step["language_instruction"].numpy().decode('utf-8'), step["language_instruction_2"].numpy().decode('utf-8'), step["language_instruction_3"].numpy().decode('utf-8')]
            instructions = [instruction for instruction in instructions if instruction]

            if len(instructions) == 0:
                break

            language_instruction = random.choice(instructions)

            if "pick" not in language_instruction.lower():
                break

            episode_count += 1
        
        new_step = {}

        new_step['language_instruction'] = language_instruction
        new_step['image'] = Image.fromarray(step['observation'][f'exterior_image_{random.randint(1, 2)}_left'].numpy())
        new_step['action'] = discretize_velocities(np.concatenate((np.expand_dims(step['action_dict']['cartesian_velocity'].numpy(), axis=0),
                                                        np.expand_dims(1 - step['action_dict']['gripper_position'].numpy(), axis=0),
                                                        np.expand_dims(np.expand_dims(step['is_terminal'].numpy(), axis=0), axis=0)), axis=1).squeeze())

        processor = data_args.image_processor

        if data_args.image_aspect_ratio == 'pad':
            image = expand2square(new_step['image'], tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(new_step['image'], return_tensors='pt')['pixel_values'][0]

        input_prompt = format_input_prompt(new_step['language_instruction'], new_step['action'])
        
        data_dict = preprocess_droid(input_prompt, tokenizer)
        data_dict['image'] = image

        batch.append(data_dict)

        if len(batch) == BATCH_SIZE:
            collated_batch = {k: v.to("cuda:0") if type(v) == torch.Tensor else v for k, v in data_module['data_collator'](batch).items()}
            collated_batch["images"] = collated_batch["images"].to("cuda:0", dtype=torch.bfloat16)
            batch = []

            with torch.no_grad():
                preds = model(**collated_batch)

            loss.append(preds.loss.item())
            batch_acc, batch_mse, batch_mae = compute_metrics(preds.logits, collated_batch["labels"], collated_batch)

            acc.append(batch_acc)
            mse.append(batch_mse)
            mae.append(batch_mae)

            del collated_batch, preds

    if step_id == 0:
        continue

    print(f"Episode {episode_count} | {step_id} steps | Average Accuracy: {sum(acc[-step_id:])/step_id*100:.1f}% | Average MSE: {sum(mse[-step_id:])/step_id:.2f} | Average MAE: {sum(mae[-step_id:])/step_id:.4f} | Average Loss: {sum(loss[-step_id:])/step_id}")


print(f"\n\n{episode_count} episodes with a total of {len(acc)} steps evaluated")
print(f"Average Accuracy: {sum(acc)/len(acc)*100:.1f}%")
print(f"Average MSE: {sum(mse)/len(mse):.2f}")
print(f"Average MAE: {sum(mae)/len(mae):.4f}")
print(f"Average Loss: {sum(loss)/len(loss)}")