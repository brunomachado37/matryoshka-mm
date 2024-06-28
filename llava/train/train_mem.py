from llava.train.train import train
import os

os.environ["WANDB_PROJECT"] = "M3_DROID_02"
os.environ["WANDB_SILENT"] = "true"
os.environ['WANDB_MODE'] = 'offline'
# os.environ["WANDB_DISABLED"] = "true"

os.environ["HF_HOME"] = "/gpfswork/rech/uli/ujf38zl/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/gpfswork/rech/uli/ujf38zl/.cache/huggingface"

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
