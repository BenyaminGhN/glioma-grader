import numpy as np
import pandas as pd
import SimpleITK as sitk
import h5py
import re
import shutil
from typing import Dict, Tuple, Optional, List, Union
import pydicom as dcm
import nibabel as nib
from nipype.interfaces.dcm2nii import Dcm2niix
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend


def load_dicom_series(series_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_path))
    # reader.SetFileNames(dicom_names)
    # image = reader.Execute()
    image = sitk.ReadImage(dicom_names, sitk.sitkFloat32)
    return image

def load_nifti(data_path):
    return nib.load(data_path).get_fdata().astype("float32")

# def convert2nifti(series_path, dst_path="fname.nii.gz"):
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(str(series_path))
#     reader.SetFileNames(dicom_names)
#     image = reader.Execute()

#     # Added a call to PermuteAxes to change the axes of the data
#     # image = sitk.PermuteAxes(image, [2, 1, 0])

#     sitk.WriteImage(image, str(dst_path))
#     return dst_path

def prepare_cross_validation_splits(dataframe: pd.DataFrame, 
                                   patient_labels: Dict[str, str],
                                   n_splits: int = 5,
                                   random_seed: int = 42) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold
    
    df_copy = dataframe.copy()
    
    # Create stratified splits at patient level
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    pids = list(patient_labels.keys())
    labels = list(patient_labels.values())
    
    splits = skf.split(np.arange(len(pids)), labels)
    
    for i, (train_splits, val_splits) in enumerate(splits):
        val_pids = np.array(pids)[val_splits]
        df_copy[f"Split{i+1}"] = "train"
        
        for pid in val_pids:
            df_copy.loc[df_copy["PatientID"] == pid, f"Split{i+1}"] = "validation"
    
    return df_copy

def create_data_sequences(train_df: pd.DataFrame, 
                         val_df: pd.DataFrame,
                         model_type: str,
                         batch_size: int = 32,
                         target_size: Tuple[int, int] = (240, 240),
                         brats_data_dir: Optional[Path] = None,
                         glioma_data_dir: Optional[Path] = None,
                         random_seed: int = 123) -> Tuple[AugmentedImageSequence, AugmentedImageSequence, int, int]:
    """
    Create training and validation data sequences.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        model_type: Type of model ("normal_abnormal" or "lgg_hgg")
        batch_size: Batch size for training
        target_size: Target image size
        brats_data_dir: Path to BraTS data
        glioma_data_dir: Path to glioma data
        random_seed: Random seed for reproducibility
        
    Returns:
        train_seq, val_seq, n_iter_val, n_iter_train
    """
    if model_type == "normal_abnormal":
        class_names = ['normal', 'abnormal']
    elif model_type == "lgg_hgg":
        class_names = ['lgg', 'hgg']
    else:
        class_names = ['normal', 'lgg', 'hgg']
    
    # Training sequence
    train_steps = int(np.ceil(len(train_df) / batch_size))
    train_seq = AugmentedImageSequence(
        dataset_csv_file=train_df,
        x_col="Path",
        y_col="NumLabels",
        class_names=class_names,
        source_image_dir=Path("."),  # Will be determined in load_image
        is_binary=False,
        model_type=model_type,
        batch_size=batch_size,
        target_size=target_size,
        add_channel=False,
        crop_ratio=[0.2, 0.2],
        augmentation=True,
        steps=train_steps,
        shuffle_on_epoch_end=True,
        random_state=random_seed,
        brats_data_dir=brats_data_dir,
        glioma_data_dir=glioma_data_dir,
    )
    
    # Validation sequence  
    val_steps = int(np.ceil(len(val_df) / batch_size))
    val_seq = AugmentedImageSequence(
        dataset_csv_file=val_df,
        x_col="Path",
        y_col="NumLabels", 
        class_names=class_names,
        source_image_dir=Path("."),  # Will be determined in load_image
        is_binary=False,
        model_type=model_type,
        batch_size=batch_size,
        target_size=target_size,
        add_channel=False,
        crop_ratio=[0.2, 0.2],
        augmentation=False,
        steps=val_steps,
        shuffle_on_epoch_end=False,
        random_state=random_seed,
        brats_data_dir=brats_data_dir,
        glioma_data_dir=glioma_data_dir,
    )
    
    n_iter_train = len(train_df) // batch_size
    n_iter_val = len(val_df) // batch_size
    
    return train_seq, val_seq, n_iter_val, n_iter_train

def check_considerations(cons: str):
    cons_parts = cons.split(',')
    return_statement_dict = {
        'status': 'OK',
        'flair': '',
        't1c': ''
    }

    # check for descriptions
    for p in cons_parts:
        desc = re.findall(r'\((.*?)\)', p)
        if len(desc) != 0:
            if "delete" in desc[0]:
                return_statement_dict['status'] = 'DELETE'
                break
        elif p:
            # check for modality considerations
            particles = p.split('-')
            return_statement_dict[particles[0].lower()] = particles[1:]
    return return_statement_dict

def convert2nifti(series_path, dst_path):
    converter = Dcm2niix()
    converter.inputs.source_dir = series_path
    converter.inputs.compression = 5
    converter.inputs.output_dir = dst_path.parent
    converter.inputs.out_filename = dst_path.name.split('.')[0]
    converter.inputs.single_file = True
    converter.cmdline
    converter.run() # doctest: +SKIP

def store_as_h5(img_arr, dst_path, meta_data = [], file_name = ''):
    """ stores an array of images to HDF5.
        Args:
            img_arr (ndarray/list[ndarray]): images array, (N, h, w, c) to be stored,
            dst_path (Path): path of the file directory,
            meta_data (list): list of any meta information to be stroed alongside the data,
                default to []
            file_name (str): final file name, default to '' ( None )
        Returns: 
            final_path (Path): the path of the stored file
    """
    num_images = len(img_arr)
    if file_name != '':
        fname = file_name
    else:
        fname = f"{num_images}.h5"

    # create a new HDF5 file
    final_path = dst_path / fname
    file = h5py.File(final_path, "w")

    # create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(img_arr),
        # h5py.h5t.STD_U8BE,
        compression="gzip",
        data=img_arr
    )

    # storing meta data
    meta_set = file.create_dataset(
        "meta", np.shape(meta_data),
        # h5py.h5t.STD_U8BE,
        data=meta_data
    )

    file.close()

    return final_path

def load_h5(data_path):
    """ load the HDF5 file.
    Args:
        data_path (Path): the full path of the .h5 file

    Returns:
        images: images array, (N, h, w, c) to be stored
        meta: associated meta data,
    """
    # open the HDF5 file
    file = h5py.File(data_path, "r+")

    images = np.array(file["/images"]).astype("float32")
    meta = np.array(file["/meta"])

    return images, meta

def is_abnormal(lst, ptn=[1]):
    size_ptn = len(ptn)
    size_lst = len(lst)
    flag = False
    for i in range(size_lst-size_ptn+1):
        if lst[i:i+size_ptn] == ptn:
            flag = True
            break
    return flag

def is_abnormal2(lst, thresh=1):
      num_true = sum(lst)
      return num_true >= thresh

