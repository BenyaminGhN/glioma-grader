from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click
from pydoc import locate
from pathlib import Path
import tensorflow as tf
from omegaconf import OmegaConf
from pickle import load

tfk = tf.keras
tfkl = tfk.layers

from src.data_ingestion import prepare_df_from_dir
from src.utils import load_dicom_series, is_abnormal
from src.data_preparation import DataLoader
from src.model_building import ModelBuilder
from src.inference_engine import InferenceEngine


@click.command()
@click.option(
    "--data-dir",
    type = Path,
    default = None
    help = "path to data directory, which consists of folders of Dicom files, each one corresponding to a Dicom series.",
    )
@click.option(
    "--predictions-fpath", 
    type=Path,
    default = None,
    help = "path to final predictions csv file",
    )
def main(data_dir: Path, predictions_fpath: Path):
    # get config file
    data_cfpath = Path('configs/data.yml')
    prep_cfpath = Path('configs/preprocessing.yml')
    model_cfpath = Path('configs/model.yml')
    inference_cfpath = Path('configs/inference.yml')
    data_config = OmegaConf.load(data_cfpath)
    prep_config = OmegaConf.load(prep_cfpath)
    model_config = OmegaConf.load(model_cfpath)
    inference_config = OmegaConf.load(inference_cfpath)

    if data_dir is None:
        data_dir = Path(data_config.root_dir)

    # create data generators
    data_loader = DataLoader(config = data_config, 
                             prep_config = prep_config,
                             data_dir = data_dir)
    
    test_seq, _ = data_loader.create_test_generator()

    model = ModelBuilder(config = model_config)

    inference_engine = InferenceEngine(model)
    preds_df = inference_engine.predict(test_seq)

    # save the predictions
    if predictions_fpath is None:
        predictions_fpath = inference_config.results_csv_path
    preds_df.to_csv(predictions_fpath, index=False)

if __name__ == "__main__":
    main()
    