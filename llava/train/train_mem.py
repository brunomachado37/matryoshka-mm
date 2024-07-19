from llava.train.train import train
import os

os.environ["WANDB_PROJECT"] = "M3_DROID_pick_01"
os.environ["WANDB_SILENT"] = "true"
os.environ['WANDB_MODE'] = 'offline'
# os.environ["WANDB_DISABLED"] = "true"

os.environ["HF_HOME"] = "/gpfswork/rech/uli/uuv83ah/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/gpfswork/rech/uli/uuv83ah/.cache/huggingface/hub"

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
