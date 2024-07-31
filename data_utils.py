import copy
from dataclasses import dataclass
import json
import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token


def expand2square(pil_img, background_color):
    '''Expand PIL image to a square using the provided background color'''
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def format_input_prompt(instruction, action):
    return f"{DEFAULT_IMAGE_TOKEN}\n{instruction}. Next Action: {action}"


def preprocess_droid(prompt, tokenizer):
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')

    targets = copy.deepcopy(input_ids)
    targets[:-8] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def make_supervised_data_module(tokenizer, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = DroidLazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    eval_dataset = DroidLazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.eval_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, matryoshka_vis_token_scale=data_args.matryoshka_vis_token_scale)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


@dataclass
class DataArguments:
    data_path: str = None
    eval_path: str = None
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: str = None
    image_aspect_ratio: str = 'square'
    image_processor: transformers.CLIPProcessor = None
    matryoshka_vis_token_scale: int = None


class DroidLazySupervisedDataset(Dataset):
    """PyTorch DROID dataset."""

    def __init__(self, data_path, tokenizer, data_args):
        super(DroidLazySupervisedDataset, self).__init__()
        with open(f"{data_path}/metadata.json", "r") as meta:
            metadata = json.load(meta)

        self.number_of_steps = metadata["number_of_steps"]
        self.accumulated_sum = metadata["accumulated_sum"]

        self.length_list = []
        for i in range(len(metadata["text_lengths"])):
            self.length_list += [metadata["text_lengths"][i]] * metadata["number_of_steps_per_episode"][i]

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.dataset_path = data_path


    def __len__(self):
        return self.number_of_steps

    @property
    def modality_lengths(self):
        return self.length_list

    def __getitem__(self, i):
        episode_id = next(j for j, v in enumerate(self.accumulated_sum) if v > i)
        step_id = i - self.accumulated_sum[episode_id-1] if episode_id > 0 else i

        with open(f"{self.dataset_path}/episode_{episode_id}/step_{step_id}", "rb") as file:       
            step = torch.load(file)

        processor = self.data_args.image_processor
        image = Image.fromarray(step["image"].numpy())

        if self.data_args.image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_prompt = format_input_prompt(step['language_instruction'], step['action'])
        
        data_dict = preprocess_droid(input_prompt, self.tokenizer)
        data_dict['image'] = image

        return data_dict
    


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    matryoshka_vis_token_scale: int = None

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if self.matryoshka_vis_token_scale != None:
            batch['matryoshka_vis_token_scale'] = self.matryoshka_vis_token_scale

        return batch