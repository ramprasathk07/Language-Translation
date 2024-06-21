from dataset import config
from pathlib import Path

def get_weights_file_path(config,epoch:str):
    model_folder = config['model_folder']
    model_basename= config['model_filename']

    model_filename = f"{model_basename}_{epoch}.pt"

    return str(Path('.')/model_folder/model_filename)

