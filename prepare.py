from typing import Optional
import numpy as np
import pandas as pd
import click
from pathlib import Path
from omegaconf import OmegaConf

from src.data_ingestion import prepare_df_from_dir
# from src.utils import load_dicom_series

@click.command()
@click.option(
    "--data-dir",
    type=Path,
    help="path to data directory, which consists of folders of Dicom files," \
         "each one corresponding to a Dicom series.",
    default=None,
    )
@click.option(
    "--out-fpath",
    type=Path,
    help="output csv file-name",
    default=None,
    )
def main(data_dir: Path, out_fpath: Path):

    # get config file
    config_file_path = Path('configs/data.yml')
    config = OmegaConf.load(config_file_path)
    if data_dir is None:
        data_dir = config.data_dir
    if out_fpath is None:
        out_fpath = config.data_info_fpath
    print(data_dir)
    info_df = prepare_df_from_dir(data_dir=Path(data_dir))
    info_df.to_csv(out_fpath, index=False)


if __name__ == "__main__":
    main()