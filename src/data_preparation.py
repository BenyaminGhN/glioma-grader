import os 
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from omegaconf import DictConfig

import tensorflow as tf
from tensorflow.keras.utils import Sequence
# from imgaug import augmenters as iaa
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

from src.preprocessing import Preprocessor
from src.utils import load_h5
from src.data_ingestion import prepare_df_from_dir


class DataLoader():
    def __init__(self, config: DictConfig, data_dir):
        self.config_ = config
        self.config = config.data_pipeline
        self.data_dir = data_dir
        self.class_weights = None
        
        np.random.seed(self.config_.seed)
        self.preprocessing_dir = self.config_.preprocessing_dir
        self.df_info = prepare_df_from_dir(data_dir=self.data_dir)

    def create_dataset(self):
        ## create dataframe out of directory
        preprocessor = Preprocessor(
            temp_working_dir=self.config_.working_dir,
            to_correct_bias=self.config_.preprocessing.to_correct_bias,
            to_register=self.config_.preprocessing.to_register,
            to_extract_brain=self.config_.preprocessing.to_extract_brain,
            to_segment=self.config_.preprocessing.to_segment,
            to_resize=self.config_.preprocessing.to_resize,
            to_normalize=self.config_.preprocessing.to_normalize,
            to_postprocess=self.config_.preprocessing.to_postprocess,
            excluded_modalities=self.config_.preprocessing.excluded_modalities,
        )
        self.preprocessing_dir = preprocessor.run_from_df(self.df_info)
        return self.preprocessing_dir

    def get_class_weights(self):
        return self.class_weights

    def create_train_val_generator(self):
        csv_filename = self.config_.target_csv
        train_df, val_df = self._get_train_val_df(os.path.join(self.data_dir, csv_filename))

        # Train
        train_seq = self._create_tf_seq(
            train_df, augmentation=self.config.to_augment, shuffle=self.config.shuffle
        )
        n_iter_train = len(train_df) // self.config.batch_size

        # Validation
        val_seq = self._create_tf_seq(val_df)
        n_iter_val = len(val_df) // self.config.batch_size

        return train_seq, n_iter_train, val_seq, n_iter_val
    

    def create_test_generator(self):
        csv_filename = self.config_.target_csv
        csv_file_path = os.path.join(self.data_dir, csv_filename)

        if Path(csv_file_path).exists():
            test_df = self._get_test_df(str(csv_file_path))
        else:
            self.create_dataset()
            slices_list = Path(self.preprocessing_dir).rglob("./*/*.h5")
            labels = {"PatientID": [], "Path": [], "NumLabels": []}
            for slice_path in slices_list:
                labels["PatientID"].append(str(slice_path.parent.name))
                labels["Path"].append(str(slice_path))
                labels["NumLabels"].append(0)
            test_df = pd.DataFrame(labels)

        # Evaluation
        test_seq = self._create_tf_seq(test_df, shuffle_on_epoch_end=False)
        n_iter_test = len(test_df) // self.config.batch_size

        return test_seq, n_iter_test
    
    def _get_test_df(self, path):
        df = pd.read_csv(path)
        # df['NumLabels'] = df['Label'].apply(lambda x: 1 if x=='abnormal' else 0)
        df['NumLabels'] = df['Label'].apply(lambda x: 1 if x==1 else 0)

        return df

    def _get_train_val_df(self, path):

        df = pd.read_csv(path)
        df['NumLabels'] = df['Label'].apply(lambda x: 1 if x=='abnormal' else 0)
        n_data = len(df)

        val_split = self.config.val_split
        val_ind = np.random.randint(0, n_data, int(n_data * val_split))

        val_df = df.iloc[val_ind]
        train_df = df.drop(val_df.index)

        do_oversampling = self.config_.augmentation.oversampling
        if do_oversampling:
            abnormal_train = train_df[train_df['Label']=='abnormal']
            for i in range(3):
                train_df = train_df.append(abnormal_train, ignore_index=True)
            
        train_df = shuffle(train_df, random_state=self.config_.seed)
        val_df = shuffle(val_df, random_state=self.config_.seed)

        if self.config.is_binary:
            y_ = train_df['NumLabels'].values
            y_r = 1-y_
            gt_arr = np.vstack([y_r, y_]).T
        else:    
            gt_arr = train_df['NumLabels'].values
        self.class_weights = self.__calculating_class_weights(gt_arr)
        return train_df, val_df

    @staticmethod
    def __calculating_class_weights(y_true):
        if len(np.shape(y_true)) == 1:
            y_true = y_true.reshape((-1, 1))
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', classes=np.unique(y_true[:, i]), y=y_true[:, i])
        return weights

    def _create_tf_seq(self, df, augmentation=False, 
                       shuffle=False, shuffle_on_epoch_end=True):
        
        if shuffle:
            permutation = np.random.RandomState(self.config_.seed).permutation(len(df))
            dataset_csv_file = df.iloc[permutation].reset_index(drop=True)
        else:
            dataset_csv_file = df
        image_source_dir = self.data_dir
        target_size = self.config_.preprocessing.target_size
        class_names = self.config.class_names
        is_binary = self.config.is_binary
        batch_size = self.config.batch_size
        add_channel = self.config_.preprocessing.add_channel
        crop_ratio = self.config_.preprocessing.crop_ratio
        steps = int(np.ceil(len(df) / batch_size))

        x_col = "Path"
        y_col = "NumLabels"
        df_seq = AugmentedImageSequence(
            dataset_csv_file=dataset_csv_file,
            x_col=x_col,
            y_col=y_col,
            class_names=class_names,
            source_image_dir=image_source_dir,
            is_binary=is_binary,
            batch_size=batch_size,
            target_size=target_size,
            add_channel = add_channel,
            crop_ratio=crop_ratio, 
            augmentation=augmentation,
            steps=steps,
            shuffle_on_epoch_end=shuffle_on_epoch_end,
            random_state=self.config_.seed,
        )

        return df_seq
        
class AugmentedImageSequence(Sequence):
    def __init__(
        self,
        dataset_csv_file,
        x_col,
        y_col,
        class_names,
        source_image_dir,
        is_binary=False,
        batch_size=16,
        target_size=(224, 224),
        add_channel=True,
        crop_ratio=[0.0, 0.0], 
        augmentation=True,
        verbose=0,
        steps=None,
        shuffle_on_epoch_end=True,
        random_state=None,
    ):

        self.dataset_df = dataset_csv_file
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.add_channel = add_channel
        self.crop_ratio = crop_ratio
        self.augmentation = augmentation
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.x_col = x_col
        self.y_col = y_col
        self.is_binary = is_binary
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def get_dataset_df(self):
        return self.dataset_df
    
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)

        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_file):
        # image_path = os.path.join(str(self.source_image_dir), image_file)
        # image_array = self.preprocess_img(image_path)
        # print(image_file)
        image_array = load_h5(image_file)[0]
        w = self.target_size[1]
        h = self.target_size[0]
        image_array = cv2.resize(image_array, (w, h))
        image_array = (image_array - image_array.min()) / (image_array.max()-image_array.min()+1e-7)
        return image_array

    def transform_batch_images(self, batch_x):
        # if self.augmentation:
        #     sometimes = lambda aug: iaa.Sometimes(0.25, aug)
        #     augmenter = iaa.Sequential(
        #         [
        #             iaa.Fliplr(0.25),
        #             iaa.Flipud(0.10),
        #             iaa.geometric.Affine(
        #                 rotate=(-45, 45), order=1, mode="constant", fit_output=False
        #             ),
        #             sometimes(
        #                 iaa.Crop(px=(0, 25), keep_size=True, sample_independently=False)
        #             ),
        #         ],
        #         random_order=True,
        #     )
        #     batch_x = augmenter.augment_images(batch_x)
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError(
                """
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """
            )
        return self.y[: self.steps * self.batch_size]

    def get_x_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError(
                """
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """
            )
        return self.x_path[: self.steps * self.batch_size]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1.0, random_state=self.random_state)
        self.x_path, self.y = (df[self.x_col].values, 
                                df[self.y_col].values.astype('float32'))
        
        if self.is_binary:
            y_ = self.y
            y_r = 1-y_
            self.y = np.vstack([y_r, y_]).T

        # self.y = np.where(self.y=='abnormal', 1, 0)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

    def preprocess_img(self, path):
        preprocessor = Preprocessor()
        img = preprocessor.read_dcm(path)
        preprocessed_img = preprocessor.preprocess_image(img)
        return preprocessed_img
