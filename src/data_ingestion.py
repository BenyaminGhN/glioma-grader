import numpy as np
import pandas as pd
from pathlib import Path
import pydicom as dcm
import SimpleITK as sitk
import shutil
from tqdm import tqdm

from src.utils import check_considerations, convert2nifti

def prepare_dirs(case_path, 
                dst_path,
                data_considerations: dict = {}):
    
    ## checking for data considerations
    ret = {
        'status': 'OK',
        'flair': '',
        't1c': ''
    }

    case_fname = case_path.name
    if len(data_considerations) != 0:
        data_version = str(case_path.parent).split("/")[-2]
        if data_version in list(data_considerations.keys()):
            if case_fname in list(data_considerations[data_version].keys()):
                data_version_cons = data_considerations[data_version]
                case_cons = data_version_cons[case_fname]
                ret = check_considerations(case_cons)

    assert ret['status'] != 'DELETE', "Data Considerations Interupts! Type: DELETE"

    slices_dsc = [dcm.read_file(fpath)['SeriesDescription'].value
                  for fpath in case_path.rglob('*.dcm')]
    
    case_modalities, modalities_counts = np.unique(slices_dsc, return_counts=True)

    t1_descs = [name for name in case_modalities if 't1' in name.lower()]
    flair_descs = [name for name in case_modalities if 'flair' in name.lower()]

    # checking if there is any modality corruption
    assert (len(flair_descs) != 0) and (len(t1_descs) != 0), "Modality Corruptions!"

    t1_slices_siuids = [dcm.read_file(fpath)['SeriesInstanceUID'].value
                        for fpath in case_path.rglob('*.dcm')
                        if dcm.read_file(fpath)['SeriesDescription'].value in t1_descs]
    flair_slices_siuids = [dcm.read_file(fpath)['SeriesInstanceUID'].value
                        for fpath in case_path.rglob('*.dcm')
                        if dcm.read_file(fpath)['SeriesDescription'].value in flair_descs]

    t1_siuids, t1_scounts = np.unique(t1_slices_siuids, return_counts=True)
    flair_siuids, flair_scounts = np.unique(flair_slices_siuids, return_counts=True)

    ##TODO checking for the nmuber of series-uids and modalities identifications

    selected_t1_metric = None
    selected_flair_metric = None

    # checking for t1 modality
    if ret['t1c'] != '':
        if ret['t1c'][0] == 'NumberOfSlices':
            siuids_idx = np.where(t1_scounts == int(ret['t1c'][1]))[0][0]
            selected_t1_metric = ['SeriesInstanceUID', t1_siuids[siuids_idx]]
        elif ret['t1c'][0] == 'SeriesInstanceUID':
            selected_t1_metric = ['SeriesInstanceUID', ret['t1c'][1]]

    if selected_t1_metric is None:
        selected_t1_metric = ['SeriesInstanceUID', t1_siuids[0]]

    if ret['flair'] != '':
        if ret['flair'][0] == 'NumberOfSlices':
            siuids_idx = np.where(flair_scounts == int(ret['flair'][1]))[0][0]
            selected_flair_metric = ['SeriesInstanceUID', flair_siuids[siuids_idx]]
        elif ret['flair'][0] == 'SeriesInstanceUID':
            selected_flair_metric = ['SeriesInstanceUID', ret['flair'][1]]

    if selected_flair_metric is None:
        selected_flair_metric = ['SeriesInstanceUID', flair_siuids[0]]

    dst_path_t1 = dst_path / 't1'
    dst_path_t1.mkdir(parents=True, exist_ok=True)
    dst_path_flair = dst_path / 'flair'
    dst_path_flair.mkdir(parents=True, exist_ok=True)

    t1_series_descs = []
    flair_series_descs = []
    for dcm_fpath in case_path.rglob('*.dcm'):
        dcm_obj = dcm.read_file(dcm_fpath)
        t1_dcm_tag = selected_t1_metric[0]
        t1_dcm_value = selected_t1_metric[1]
        if str(t1_dcm_value) == str(dcm_obj[t1_dcm_tag].value):
            t1_series_descs.append(dcm_obj['SeriesDescription'].value)
            if not (dst_path_t1 / dcm_fpath.name).exists():
                shutil.copy(dcm_fpath, dst_path_t1 / dcm_fpath.name)
                # shutil.copy(dcm_fpath, dst_path_t1)

        flair_dcm_tag = selected_flair_metric[0]
        flair_dcm_value = selected_flair_metric[1]
        if str(flair_dcm_value) == str(dcm_obj[flair_dcm_tag].value):
            flair_series_descs.append(dcm_obj['SeriesDescription'].value)
            if not (dst_path_flair / dcm_fpath.name).exists():
                shutil.copy(dcm_fpath, dst_path_flair / dcm_fpath.name)
                # shutil.copy(dcm_fpath, dst_path_flair)

    ## preparing the nii files for brainles processings if needed
    fpath_t1_nii = dst_path / "nifti" / f"{case_fname}_0000.nii.gz"
    fpath_t1_nii.parent.mkdir(parents=True, exist_ok=True)
    if not fpath_t1_nii.exists():
        convert2nifti(series_path=dst_path_t1,
                    dst_path=fpath_t1_nii,
                    # reorient_nifti=True
                    )

    fpath_flair_nii = dst_path / "nifti" / f"{case_fname}_0001.nii.gz"
    fpath_flair_nii.parent.mkdir(parents=True, exist_ok=True)
    if not fpath_flair_nii.exists():
        convert2nifti(series_path=dst_path_flair,
                    dst_path=fpath_flair_nii,
                    # reorient_nifti=True
                    )

    # shutil.rmtree(dst_path_t1)
    # shutil.rmtree(dst_path_flair)
    return ret
      
def prepare_df_from_dir(data_dir: Path):
    series_paths = [p for p in data_dir.glob('./*') if p.is_dir()]
    series_paths.sort()
    info_dict = {
        "PatientID": [],
        "SeriesInstanceUID": [],
        "Modality": [],
        "Path": [],
    }
    for sp in tqdm(series_paths):
        ## prepare the directory 
        sp_dst_path = data_dir / sp.name
        status = prepare_dirs(sp, sp_dst_path)
        sequences = ['t1', 'flair']
        for seq in sequences:
            seq_path = sp / seq
            dcm_fpaths = sorted(seq_path.rglob('./*.dcm'))
            dcm_obj = dcm.dcmread(dcm_fpaths[0])
            pid = str(dcm_obj['PatientID'].value)
            series_desc = str(dcm_obj['SeriesDescription'].value)
            series_uid = str(dcm_obj['SeriesInstanceUID'].value)
            series_modality = ''
            if 't1' in series_desc.lower():
                series_modality = 't1'
            elif 'flair' in series_desc.lower():
                series_modality = 'flair'
            else:
                series_modality = 'UNKNOWN'

            info_dict["PatientID"].append(pid)
            info_dict["SeriesInstanceUID"].append(series_uid)
            info_dict["Modality"].append(series_modality)
            info_dict["Path"].append(str(seq_path))

    return pd.DataFrame(info_dict)