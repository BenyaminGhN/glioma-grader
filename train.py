import os
import click
from pydoc import locate
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from src.data_preparation import DataLoader
from src.utils import create_callbacks
from src.model_building import ModelBuilder

np.random.seed(2)

@click.command()
@click.option('--data_dir', default=None, help='data directory path')
def main(data_dir: Path):
    # get config file
    data_cfpath = Path('configs/data.yml')
    prep_cfpath = Path('configs/preprocessing.yml')
    model_cfpath = Path('configs/model.yml')
    data_config = OmegaConf.load(data_cfpath)
    prep_config = OmegaConf.load(prep_cfpath)
    model_config = OmegaConf.load(model_cfpath)

    if data_dir is None:
        data_dir = data_config.root_dir

    # create data generators
    data_loader = DataLoader(config = data_config, 
                             prep_config = prep_config,
                             data_dir = data_dir)
    
    train_seq, n_iter_train, val_seq, n_iter_val = data_loader.create_train_val_generator()
    class_weights = data_loader.get_class_weights()
    
    # # define model for training
    model = ModelBuilder(config = model_config)

    ##train the models individually 
    ##TODO 

if __name__ == "__main__":
    main()
