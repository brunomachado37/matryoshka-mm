from llava.train.train import train
import os

os.environ["WANDB_PROJECT"] = "M3_DROID_01"
os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
