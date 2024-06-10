import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import DroidLazySupervisedDataset, DataCollatorForSupervisedDataset, DataArguments


BATCH_SIZE = 16
DATASET_PATH = "../datasets/droid_torch_dc"
MODEL_PATH = "./models/full_dataset/test_scale_36/checkpoint-145000"


def compute_metrics(outputs, labels):  
    action_index = torch.transpose(torch.nonzero(labels != -100, as_tuple=False).squeeze(-1), 0, 1)
    target = labels[action_index[0], action_index[1]].unsqueeze(1)

    logits = outputs[:, -labels.shape[1]:, :]                                   # It only works because the image is known to be in the beginning of the sequence
    action_index[1, :] -= 1
    logits = logits[action_index[0], action_index[1], :]

    logits = logits[:,31744:32000]          # Just look at valid actions

    probs = logits.softmax(dim=-1)
    conf, predictions = probs.topk(1)

    predictions += 31744                    # Map it back to the vocabulary space

    acc = float(torch.sum(predictions == target) / len(predictions))
    mae = float(torch.sum(torch.abs(predictions - target)) / len(predictions))
    mse = float(torch.sum(torch.square(predictions - target)) / len(predictions))

    return acc, mse, mae


data_args = DataArguments(image_aspect_ratio="pad")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=2048, padding_side="right", use_fast=False)

model = LlavaLlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda:0")
model.eval()
        
vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.bfloat16, device="cuda:0")
data_args.image_processor = vision_tower.image_processor            

datasets = {"eval": DroidLazySupervisedDataset(f"{DATASET_PATH}/eval", tokenizer=tokenizer, data_args=data_args)}
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

for mode, dataset in datasets.items():
    acc, mse, mae, loss = [], [], [], []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        if i+BATCH_SIZE > len(dataset):
            batch = [dataset[idx] for idx in range(i, len(dataset))]
        else:
            batch = [dataset[idx] for idx in range(i, i+BATCH_SIZE)]
        
        collated_batch = data_collator(batch)
        collated_batch = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in collated_batch.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                preds = model(**collated_batch)

        loss.append(preds.loss.item())
        batch_acc, batch_mse, batch_mae = compute_metrics(preds.logits, collated_batch["labels"])

        acc.append(batch_acc)
        mse.append(batch_mse)
        mae.append(batch_mae)

        del batch, collated_batch, preds

        print(f"{mode} Average Accuracy: {sum(acc)/len(acc)*100:.1f}%")
        print(f"{mode} Average MSE: {sum(mse)/len(mse):.2f}")
        print(f"{mode} Average MAE: {sum(mae)/len(mae):.4f}")
        print(f"{mode} Average Loss: {sum(loss)/len(loss)}")
