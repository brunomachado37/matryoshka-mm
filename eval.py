import torch
import transformers
from tqdm import tqdm
import random

from llava.model import LlavaLlamaForCausalLM
from data_utils import DataArguments, make_supervised_data_module


BATCH_SIZE = 2
DATASET_PATH = "../datasets/droid_torch_dc"
NUM_BATCHES = 20
SET = "train"


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
    print(all_predictions.squeeze(-1))
    print(target)
    print(float(torch.sum(target == all_predictions) / (all_predictions.shape[0] * all_predictions.shape[1])))
    print(acc)

    return acc, mse, mae


model = LlavaLlamaForCausalLM.from_pretrained("models/pick/test_02-dev-M3-pick", torch_dtype=torch.bfloat16).to("cuda:0")
model.eval()

vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.float32)

data_args = DataArguments(data_path=f"{DATASET_PATH}/train", 
                          eval_path=f"{DATASET_PATH}/eval",
                          image_aspect_ratio="pad",
                          image_processor=vision_tower.image_processor,
                          matryoshka_vis_token_scale=36)

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer",
                                                       model_max_length=2048,
                                                       padding_side="right",
                                                       use_fast=False)

data_module = make_supervised_data_module(tokenizer, data_args)

acc, mse, mae, loss = [], [], [], []

for i in tqdm(range(0, NUM_BATCHES)):
    batch = []

    while len(batch) < BATCH_SIZE:
        step_id = random.randint(0, len(data_module[f'{SET}_dataset'])-1)

        input_ids = data_module[f'{SET}_dataset'][step_id]["input_ids"]
        input_ids = input_ids[input_ids > 0]

        language_instruction = tokenizer.decode(input_ids, skip_special_tokens=True) 

        if "pick" in language_instruction.lower():
            batch.append(data_module[f'{SET}_dataset'][step_id])

    collated_batch = {k: v.to("cuda:0") if type(v) == torch.Tensor else v for k, v in data_module['data_collator'](batch).items()}

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            preds = model(**collated_batch)

    loss.append(preds.loss.item())
    batch_acc, batch_mse, batch_mae = compute_metrics(preds.logits, collated_batch["labels"], collated_batch)

    acc.append(batch_acc)
    mse.append(batch_mse)
    mae.append(batch_mae)

    del batch, collated_batch, preds


print(f"{NUM_BATCHES * BATCH_SIZE} steps evaluated on {SET} set")
print(f"Average Accuracy: {sum(acc)/len(acc)*100:.1f}%")
print(f"Average MSE: {sum(mse)/len(mse):.2f}")
print(f"Average MAE: {sum(mae)/len(mae):.4f}")
print(f"Average Loss: {sum(loss)/len(loss)}")
