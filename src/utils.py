import numpy as np
import pandas as pd
import SimpleITK as sitk
import h5py
import re
import shutil
import pydicom as dcm
import nibabel as nib
from nipype.interfaces.dcm2nii import Dcm2niix
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
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

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

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

def create_callbacks(checkpoint_path, history_output_path, log_dir):
    check1 = ModelCheckpoint(checkpoint_path,
                             monitor = 'val_loss',
                             verbose = 1, 
                             save_best_only = True, 
                             save_weights_only = True, 
                             mode = 'min')

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode="min", min_lr=1e-8)
    history_logger = CSVLogger(history_output_path, separator=",", append=True)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    return [check1, lr_reduction, history_logger, tensor_board]

