import torch
import numpy as np
import SimpleITK as sitk
from utils.HD_BET.data_loading import load_and_preprocess, save_segmentation_nifti, prepare_segmentation
from utils.HD_BET.predict_case import predict_case_3D_net
import imp
from utils.HD_BET.utils import postprocess_prediction, SetNetworkToVal, get_params_fname, maybe_download_parameters
import os
import utils.HD_BET


class BrainExtractor():
    def __init__(self, mode="fast",
                 config_file=os.path.join(utils.HD_BET.__path__[0], "config.py"),
                 device=0):
        
        self.device = device

        self.list_of_param_files = []
        if mode == 'fast':
            params_file = get_params_fname(0)
            maybe_download_parameters(0)

            self.list_of_param_files.append(params_file)
        elif mode == 'accurate':
            for i in range(5):
                params_file = get_params_fname(i)
                maybe_download_parameters(i)

                self.list_of_param_files.append(params_file)
        else:
            raise ValueError("Unknown value for mode: %s. Expected: fast or accurate" % mode)

        assert all([os.path.isfile(i) for i in self.list_of_param_files]), "Could not find parameter files"

        self.cf = imp.load_source('cf', config_file)
        self.cf = self.cf.config()

        self.net, _ = self.cf.get_network(self.cf.val_use_train_mode, None)
        if self.device == "cpu":
            self.net = self.net.cpu()
        else:
            self.net.cuda(self.device)

        self.params = []
        for p in self.list_of_param_files:
            self.params.append(torch.load(p, map_location=lambda storage, loc: storage))

    @staticmethod
    def apply_bet(img, bet, out_fname):
        img_itk = sitk.ReadImage(img)
        img_npy = sitk.GetArrayFromImage(img_itk)
        img_bet = sitk.GetArrayFromImage(sitk.ReadImage(bet))
        img_npy[img_bet == 0] = 0
        out = sitk.GetImageFromArray(img_npy)
        out.CopyInformation(img_itk)
        sitk.WriteImage(out, out_fname)

    def predict(self, mri_fnames, output_fnames, postprocess=False,
                do_tta=True, keep_mask=True, overwrite=True, bet=False):
        
        if not isinstance(mri_fnames, (list, tuple)):
            mri_fnames = [mri_fnames]

        if not isinstance(output_fnames, (list, tuple)):
            output_fnames = [output_fnames]

        # assert len(mri_fnames) == len(output_fnames), "mri_fnames and output_fnames must have the same length"
        seg_results = []
        for in_fname, out_fname in zip(mri_fnames, output_fnames):
            mask_fname = out_fname[:-7] + "_mask.nii.gz"
            if overwrite or (not (os.path.isfile(mask_fname) and keep_mask) or not os.path.isfile(out_fname)):
                # print("File:", in_fname)
                # print("preprocessing...")
                try:
                    data, data_dict = load_and_preprocess(in_fname)
                except RuntimeError:
                    print("\nERROR\nCould not read file", in_fname, "\n")
                    continue
                except AssertionError as e:
                    print(e)
                    continue

                softmax_preds = []

                # print("prediction (CNN id)...")
                for i, p in enumerate(self.params):
                    # print(i)
                    self.net.load_state_dict(p) ## this line should be in the init also??
                    self.net.eval()
                    self.net.apply(SetNetworkToVal(False, False))
                    _, _, softmax_pred, _ = predict_case_3D_net(self.net, data, do_tta, self.cf.val_num_repeats,
                                                                self.cf.val_batch_size, self.cf.net_input_must_be_divisible_by,
                                                                self.cf.val_min_size, self.device, self.cf.da_mirror_axes)
                    softmax_preds.append(softmax_pred[None])

                seg = np.argmax(np.vstack(softmax_preds).mean(0), 0)

                if postprocess:
                    seg = postprocess_prediction(seg)

                prep_seg = prepare_segmentation(seg, data_dict)
                seg_results.append(prep_seg)
                # print("exporting segmentation...")
                # save_segmentation_nifti(seg, data_dict, mask_fname)
                # if bet:
                #     self.apply_bet(in_fname, mask_fname, out_fname)

                # if not keep_mask:
                #     os.remove(mask_fname)

        return seg_results
