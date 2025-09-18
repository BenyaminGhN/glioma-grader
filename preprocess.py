from typing import Optional
import numpy as np
import pandas as pd
import click
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger

from src.preprocessing import Preprocessor


@click.command()
@click.option(
    "--csv-fpath",
    type = Path,
    help = "info csv path.",
    default = None,
    )
@click.option(
    "--working-dir",
    type = Path,
    help = "working directory for preprocessing",
    default = None,
    )
def main(csv_fpath: Path, working_dir: Path):

    # get config file
    data_cfpath = Path('configs/data.yml')
    preprocessing_cfpath = Path('configs/preprocessing.yml')
    data_config = OmegaConf.load(data_cfpath)
    prep_config = OmegaConf.load(preprocessing_cfpath)

    if csv_fpath is None:
        csv_fpath = Path(data_config.data_info_fpath)
    info_df = pd.read_csv(csv_fpath)
    preprocessor = Preprocessor(
        config = prep_config,
    )
    pp_dir = preprocessor.run_from_df(info_df)
    logger.info(f"The Preprocessed Dir: {pp_dir}")

if __name__ == "__main__":
    main()