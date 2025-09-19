import numpy as np
import pandas as pd
import tempfile
from tqdm import tqdm
import cv2
from pathlib import Path
import shutil
import SimpleITK as sitk
from utils.HD_BET.run import run_hd_bet
from src.utils import load_dicom_series, store_as_h5, load_h5, load_nifti
from utils.HD_BET.model_loader import BrainExtractor
from utils.DeepSeg.inference import TumorSegmentor
import ants
import torch
from omegaconf import OmegaConf

import pydicom as dcm
from auxiliary.normalization.percentile_normalizer import PercentileNormalizer
from auxiliary.turbopath import turbopath

from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import Modality
from brainles_preprocessing.preprocessor import Preprocessor as BrlesPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator


class Preprocessor():
    def __init__(self, config: OmegaConf):

        # Create temporary storage
        self.temp_working_dir = config.working_dir
        if self.temp_working_dir is None:
            storage = tempfile.TemporaryDirectory()
            self.temp_working_dir = Path(storage.name)
        else:
            self.temp_working_dir = Path(self.temp_working_dir)
            self.temp_working_dir.mkdir(parents=True, exist_ok=True)

        if config.preprocessing_dir is None:
            self.preprocessing_dir = Path("data/preprocessed_dir")
        else:
            self.preprocessing_dir = Path(config.preprocessing_dir)
        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)    

        self.to_correct_bias = config.actions.to_correct_bias
        self.to_register = config.actions.to_register
        self.to_extract_brain = config.actions.to_extract_brain
        self.to_segment = config.actions.to_segment
        self.to_resize = config.actions.to_resize
        self.excluded_modalities = config.actions.excluded_modalities
        self.to_normalize = config.actions.to_normalize
        self.to_postprocess = config.actions.to_postprocess
        self.if_brainles_exists = config.actions.if_brainles_exists

        if self.to_segment:
            self.deep_seg = TumorSegmentor()
        if self.to_extract_brain & (not self.if_brainles_exists):
            self.brain_extractor = BrainExtractor()

    def correct_bias(self, image):
        img_mask = sitk.OtsuThreshold(image)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        image = corrector.Execute(image, img_mask)
        return image

    def register(self, fixed_image, moving_image):
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
            # estimateLearningRate=registration_method.Once,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(initial_transform)

        final_transform = registration_method.Execute(fixed_image, moving_image)

        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )

        return moving_resampled

    def ants_register(self, fixed_image, moving_image, out_fpath, type_of_transform="Rigid"):
        # # read / write images
        fixed_arr = ants.image_read(fixed_image)
        moving_arr = ants.image_read(moving_image)

        # registration
        result = ants.registration(fixed_arr, moving_arr, type_of_transform = type_of_transform,
                                #    aff_iterations=(1000, 500, 100, 2)
                                    random_seed=123,
                                   )
        ants.image_write(result["warpedmovout"], out_fpath)
        return sitk.ReadImage(out_fpath, sitk.sitkFloat32)
    
    def extract_brain(self,
                      input_image_path: str,
                      masked_image_path: str,
                      # brain_mask_path: str,
                      log_file_path: str = None,
                      # TODO convert mode to enum
                      mode: str = "fast", # "accurate",
                      device: int | str = "cpu", # 0 as gpu
                      do_tta: bool = True,
                      ):
        # GPU + accurate + TTA
        """skullstrips images with HD-BET generates a skullstripped file and mask"""
        run_hd_bet(
            mri_fnames=[input_image_path],
            output_fnames=[masked_image_path],
            # device=0,
            # TODO consider postprocessing
            # postprocess=False,
            mode=mode,
            device=device,
            postprocess=True,
            do_tta=do_tta,
            keep_mask=True,
            overwrite=True,
        )

        hdbet_mask_path = (
            Path(masked_image_path)
            # Path(masked_image_path).parent
            # / f"{name_extractor(masked_image_path)}_mask.nii.gz"
        )
        # if hdbet_mask_path.resolve() != Path(brain_mask_path).resolve():
        #     copyfile(
        #         src=hdbet_mask_path,
        #         dst=brain_mask_path,
        #     )

        return hdbet_mask_path

    def extract_brain_manual(self, img_arr):
        masks = []
        for img in img_arr:
            gray = np.copy(img)
            gray = ((gray - gray.min()) / (gray.max()-gray.min()))*255
            gray = gray.astype('uint8')

            ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_OTSU)

            ret, markers = cv2.connectedComponents(thresh)
            #Get the area taken by each component. Ignore label 0 since this is the background.
            marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
            #Get label of largest component by area
            largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above
            #Get pixels which correspond to the brain
            brain_mask = markers==largest_component

            final_mask = np.uint8(brain_mask)
            kernel = np.ones((8,8),np.uint8)
            closing = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            masks.append(closing)
        return np.array(masks)

    def resize(self):
        pass

    def normalize(self, image):
        return (image - np.min(image))/(np.max(image)-np.min(image)+1e-10)

    def run(self,
            t1_fpath,
            flair_fpath,
            center_modality = None,
            be_dir = "",
            check_for_registeration = True):

        self.target_fpaths = {
            "t1": t1_fpath,
            "flair": flair_fpath,
        }

        self.itk_images = {
            "t1" : load_dicom_series(str(self.target_fpaths["t1"])),
            "flair" : load_dicom_series(str(self.target_fpaths["flair"])),
        }

        if self.to_correct_bias:
            for k in self.itk_images.keys():
                self.itk_images[k] = self.correct_bias(self.itk_images[k])

        self.arr_images = {
            "t1" : sitk.GetArrayFromImage(self.itk_images["t1"]),
            "flair" : sitk.GetArrayFromImage(self.itk_images["flair"]),
        }

        if not be_dir:
            be_dir = self.temp_working_dir / f"brain_extracion" 
        be_dir.mkdir(parents=True, exist_ok=True)

        self.meta_info = {}
        if center_modality is None:
            imgs_size = []
            for k in self.itk_images.keys():
                imgs_size.append(self.itk_images[k].GetSize()[-1])
            center_modality = list(self.itk_images.keys())[np.argmin(imgs_size)]

        if check_for_registeration:
            imgs_size = []
            for k in self.itk_images.keys():
                # imgs_size.append(self.itk_images[k].GetSize())
                imgs_size.append(self.itk_images[k].GetSpacing())
            for s in imgs_size:
                if np.any([[i != j] for i, j in zip(imgs_size[0], s)]):
                    self.to_register = True
                    only_rigid = False

        only_rigid = True
        self.meta_info["center_modality"] = center_modality
        self.meta_info["is_registered"] = False
        if self.to_register:
            self.meta_info["is_registered"] = True
            moving_modalities = [k for k in self.itk_images.keys() if k != center_modality]
            for k in self.itk_images.keys():
                sitk.WriteImage(self.itk_images[k], str(be_dir / f"{k}.nii.gz"))
            for moving_modality in moving_modalities:
                fixed_path = str(be_dir / f"{center_modality}.nii.gz")
                moving_path = str(be_dir / f"{moving_modality}.nii.gz")
                out_path = str(be_dir / f"{moving_modality}_registered.nii.gz")
                if only_rigid:
                    type_of_transform = "Rigid"
                else:
                    type_of_transform = "Affine"
                self.itk_images[moving_modality] = self.ants_register(fixed_path,
                                                                    moving_path,
                                                                    out_path,
                                                                    type_of_transform)

        for k in self.itk_images.keys():
            self.arr_images[k] = sitk.GetArrayFromImage(self.itk_images[k])

        # for k in self.itk_images.keys():
        #     sitk.WriteImage(self.itk_images[k], str(be_dir / f"{k}.nii.gz"))
            
        # t1_nii_paths.append(str(be_dir / "t1.nii.gz"))

        if self.to_extract_brain:
            manual = False
            if not manual:
                seg_results = self.brain_extractor.predict([self.itk_images["t1"]],
                                                    "t1_be.nii.gz",
                                                    )
                brain_mask_arr = sitk.GetArrayFromImage(seg_results[0])
            else:
                brain_mask_arr = self.extract_brain_manual(self.arr_images["t2"])

            for k in self.arr_images.keys():
                img_arr = self.arr_images[k]
                self.arr_images[k] = np.array([img * mask for img, mask in zip(img_arr, brain_mask_arr)])

        if self.to_resize:
            pass

        if self.to_normalize:
            for k in self.arr_images.keys():
                self.arr_images[k] = np.array([self.normalize(img) for img in self.arr_images[k]])

        preprocessed_arrs = self.arr_images
        return preprocessed_arrs
    
    def run_brainlens(self,
                    t1_fpath,
                    flair_fpath,
                    be_dir
                    ):

        # brainles_dir = be_dir / f"{self.temp_working_dir.name}_brainles"
        be_dir.mkdir(parents=True, exist_ok=True)
        norm_bet_dir = be_dir / "normalized_bet"
        norm_bet_dir.mkdir(parents=True, exist_ok=True)

        # normalizer
        percentile_normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )

        # define modalities
        center = Modality(
                modality_name="t1",
                input_path=t1_fpath,
                normalized_bet_output_path = norm_bet_dir / f"{be_dir.name}_t1_bet_normalized.nii.gz",
                atlas_correction=True,
                normalizer=percentile_normalizer,
        )

        moving_modalities = [
            Modality(
            modality_name="flair",
            input_path=flair_fpath,
            normalized_bet_output_path = norm_bet_dir / f"{be_dir.name}_fla_bet_normalized.nii.gz",
            atlas_correction=True,
            normalizer=percentile_normalizer,
            ),
        ]

        preprocessor = BrlesPreprocessor(
        center_modality = center,
        moving_modalities = moving_modalities,
        registrator = ANTsRegistrator(),
        brain_extractor=HDBetExtractor(),
        # atlas_image_path="/Users/benyaminghn/Researches/glioma/data/atlas_image/spgr.nii",
        # optional: we provide a temporary directory as a sandbox for the preprocessin
        # temp_folder="temporary_directory",
        limit_cuda_visible_devices="0",
        )

        preprocessor.run(
            save_dir_coregistration = be_dir / "co-registration",
            save_dir_atlas_registration = be_dir / "atlas-registration",
            save_dir_atlas_correction = be_dir / "atlas-correction",
            save_dir_brain_extraction = be_dir / "brain-extraction",
        )

        t1_final_preprocessed_path = be_dir / "brain-extraction" / "atlas_bet_t1.nii.gz"
        flair_final_preprocessed_path = be_dir / "brain-extraction" / "brain_masked" / "brain_masked__flair.nii.gz"
        final_preprocessed_path = {
            't1': t1_final_preprocessed_path,
            'flair': flair_final_preprocessed_path
        }
        return final_preprocessed_path
    
    def run_from_df(self,
                data_df = [],
                center_modality = None,
                check_for_registeration = True):
        
        self.corrupted_files = []
        self.corrupted_pids = []
        pids = np.unique(data_df["PatientID"].values)
        self.final_paths = []
        for pid in tqdm(pids):
            if len(data_df[data_df["PatientID"]==pid].values) != 2:
                self.corrupted_pids.append(pid)
                continue

            self.dicom_fpaths = {
                "t1": data_df[(data_df["PatientID"]==pid) &
                              (data_df["Modality"]=="t1")]["Path"].values[0],
                "flair": data_df[(data_df["PatientID"]==pid) &
                                 (data_df["Modality"]=="flair")]["Path"].values[0],
            }

            case_path = Path(self.dicom_fpaths["t1"]).parent
            case_name = case_path.name
            self.nifti_fpaths = {
                "t1": case_path / "nifti" / f"{case_name}_0000.nii.gz",
                "flair": case_path / "nifti" / f"{case_name}_0001.nii.gz",
            }

            be_dir = self.temp_working_dir / f"{pid}" 
            final_path = self.preprocessing_dir / f"{pid}" 
            center_modality = "t1"
            check_for_registeration = True

            if self.if_brainles_exists:
                self.preprocessed_fpaths = self.run_brainlens(
                    t1_fpath = self.nifti_fpaths["t1"],
                    flair_fpath = self.nifti_fpaths["flair"],
                    be_dir = be_dir
                )

                t1_preprocessed_fname = f"{str(self.nifti_fpaths['t1'].name).split('.')[0]}_preprocessed.nii.gz"
                flair_preprocessed_fname = f"{str(self.nifti_fpaths['flair'].name).split('.')[0]}_preprocessed.nii.gz"
                t1_dst_path = self.preprocessing_dir / case_name / t1_preprocessed_fname
                flair_dst_path = self.preprocessing_dir / case_name / flair_preprocessed_fname
                t1_dst_path.parent.mkdir(parents=True, exist_ok=True)
                flair_dst_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.move(src = self.preprocessed_fpaths["t1"],
                            dst = t1_dst_path
                )
                shutil.move(src = self.preprocessed_fpaths["flair"],
                            dst = flair_dst_path
                )

                shutil.rmtree(be_dir)
                self.preprocessed_arrs = {
                    "t1": np.moveaxis(load_nifti(self.preprocessed_fpaths["t1"]), -1, 0),
                    "flair": np.moveaxis(load_nifti(self.preprocessed_fpaths["flair"]), -1, 0)
                }

            else:
                self.preprocessed_arrs = self.run(
                    t1_fpath=self.target_fpaths["t1"],
                    flair_fpath=self.target_fpaths["flair"],
                    center_modality=center_modality,
                    be_dir = be_dir,
                    check_for_registeration = check_for_registeration
                )

            if self.to_postprocess:
                hdf5_fpaths = []
                n_slices = self.preprocessed_arrs["t1"].shape[0]
                for idx in range(n_slices - 1):
                    if idx < 3:
                        continue
                    
                    # if not self.to_segment:
                    t1_slice = self.arr_images["t1"][idx]
                    fla_slice = self.arr_images["flair"][idx]

                    # final_path = preprocessed_mbe_dir / f"p{pid}_{idx}.h5"
                    # storing as single file for each of the slice instances
                    stacked_arr = np.dstack([t1_slice, fla_slice])
                    # else:
                    #     fla_slice = self.arr_images["flair"][idx]
                    #     seg_slice = self.seg_arrs[idx]

                    #     # final_path = preprocessed_mbe_dir / f"p{pid}_{idx}.h5"
                    #     # storing as single file for each of the slice instances
                    #     stacked_arr = np.dstack([t2_slice, fla_slice, seg_slice])

                    if np.isnan(stacked_arr).any():
                        self.corrupted_files.append(f"p{pid}_{idx}.h5")
                        continue

                    self.final_paths.append(Path(final_path) / f"p{pid}_{idx}.h5")
                    hdf5_fpaths.append(store_as_h5(stacked_arr,
                                                dst_path = final_path,
                                                file_name = f"p{pid}_{idx}.h5",
                                                meta_data = []))

        if self.to_segment:
            torch.cuda.empty_cache()
            from utils.DeepSeg.inference import TumorSegmentor
            self.deep_seg = TumorSegmentor()
            for slice_path in tqdm(self.final_paths):
                img_arr = load_h5(slice_path)[0]
                seg_arr = self.deep_seg.predict(img_arr[:, :, 1])
                if self.to_resize:
                    # Intensity normalization (zero mean and unit variance)
                    width = img_arr.shape[0]
                    height = img_arr.shape[1]
                    img = cv2.resize(seg_arr, (width, height), interpolation = cv2.INTER_NEAREST)
                    seg_arr = img.astype("float32")

                t1_slice = img_arr[:, :, 0]
                fla_slice = img_arr[:, :, 1]
                seg_slice = seg_arr

                # final_path = preprocessed_mbe_dir / f"p{pid}_{idx}.h5"
                # storing as single file for each of the slice instances
                stacked_arr = np.dstack([t1_slice, fla_slice, seg_slice])
                hdf5_fpaths.append(store_as_h5(stacked_arr,
                                            dst_path = Path(slice_path.parent),
                                            file_name = slice_path.name,
                                            meta_data = []))

        return str(be_dir.parent)