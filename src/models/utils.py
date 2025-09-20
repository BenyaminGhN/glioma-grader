import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend

from src.utils import load_h5

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

def get_patient_prediction_vector(pid: str, 
                                 target_df: pd.DataFrame, 
                                 thresh: float = 0.5, 
                                 model = None,
                                 padding_factor: str = "max", 
                                 to_return_total_slices: bool = False, 
                                 to_predict: bool = False, 
                                 preds_column: str = "Prediction",
                                 brats_data_dir: Optional[Path] = None,
                                 glioma_data_dir: Optional[Path] = None) -> Tuple[List, List, List]:
    """Get prediction vector for a patient across all slices."""
    patient_df = target_df[target_df['PatientID'] == pid]
    num_slices = len(patient_df)
    
    source_data_dir = Path("/".join(patient_df['Path'].values[0].split("/")[:-1]))
    num_slices_list = [int(str(Path(patient_df['Path'].values[i]).stem).split("_")[-1]) for i in range(num_slices)]
    num_slices_list = sorted(num_slices_list)
    
    # Determine dataset directory
    if "brats" in str(pid).lower() and brats_data_dir is not None:
        dataset_dir = brats_data_dir / "data"
    elif glioma_data_dir is not None:
        dataset_dir = glioma_data_dir
    else:
        raise ValueError("Data directory not provided")
    
    start_slice = num_slices_list[0]
    end_slice = num_slices_list[-1]
    slices_preds = []
    slices_labels = []
    slices_ignored = []
    
    for i in range(start_slice, end_slice + 1):
        slice_fname = f"{pid}_{i}.h5"
        slice_path = dataset_dir / slice_fname
        source_slice_path = source_data_dir / slice_fname
        
        if patient_df[patient_df['Path'] == str(source_slice_path)].empty:
            if not to_return_total_slices:
                slices_ignored.append(i)
                continue
            else:
                if to_predict and model is not None:
                    # Predict on missing slice
                    img_arr = load_h5(slice_path)[0]
                    # Apply preprocessing similar to load_image
                    slice_pred = model.predict(np.expand_dims(img_arr, 0))[0]
                    slice_label = int(slice_pred > thresh)
                    slices_preds.append(slice_pred)
                    slices_labels.append(slice_label)
                else:
                    slices_preds.append(0.0)
                    slices_labels.append(0)
        else:
            if to_predict and model is not None:
                img_arr = load_h5(slice_path)[0]
                slice_pred = model.predict(np.expand_dims(img_arr, 0))[0]
            else:
                slice_pred = patient_df[patient_df['Path'] == str(source_slice_path)][preds_column].values[0]
                
            slice_label = patient_df[patient_df['Path'] == str(source_slice_path)]["NumLabels"].values[0]
            slices_preds.append(float(slice_pred))
            slices_labels.append(int(slice_label))

    return slices_labels, slices_preds, slices_ignored


class DataProcessor:
    """Handles data processing and feature extraction between model stages."""
    
    @staticmethod
    def extract_features_from_predictions(predictions: np.ndarray, 
                                        confidence_scores: np.ndarray = None) -> np.ndarray:
        """
        Extract features from model predictions for next stage.
        
        Args:
            predictions: Model predictions
            confidence_scores: Optional confidence scores
            
        Returns:
            Feature array for next model stage
        """
        features = []
        
        # Add prediction probabilities
        if predictions.ndim > 1:
            features.extend(predictions.flatten())
        else:
            features.extend(predictions)
            
        # Add confidence metrics if available
        if confidence_scores is not None:
            features.extend([
                np.mean(confidence_scores),
                np.std(confidence_scores),
                np.max(confidence_scores),
                np.min(confidence_scores)
            ])
            
        # Add statistical features
        features.extend([
            np.mean(predictions),
            np.std(predictions),
            np.max(predictions),
            np.min(predictions)
        ])
        
        return np.array(features).reshape(1, -1)
    
    @    staticmethod
    def create_ml_dataset(pids: List[str],
                         target_df: pd.DataFrame,
                         slice_preds_list: List[List[float]],
                         slice_labels_list: List[List[int]], 
                         patient_labels: List[str],
                         padding_factor: int = 154,
                         split_idx: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/validation datasets for ML aggregator.
        
        Args:
            pids: Patient IDs
            target_df: DataFrame with split information
            slice_preds_list: List of slice predictions per patient
            slice_labels_list: List of slice labels per patient
            patient_labels: List of patient-level labels
            padding_factor: Number of features to pad/truncate to
            split_idx: Split index for train/val separation
            
        Returns:
            X_train, X_val, y_train, y_val arrays
        """
        X_train = []
        X_val = []
        y_train = []
        y_val = []
        
        for i, pid in enumerate(pids):
            split = target_df[target_df["PatientID"] == pid][f"Split{split_idx}"].values[0]
            slices_preds = slice_preds_list[i].copy()
            patient_label = patient_labels[i]
            num_slices = len(slices_preds)
            
            # Pad or truncate to desired length
            if padding_factor > num_slices:
                num_missing_slices = int(padding_factor - num_slices)
                for nslice in range(num_missing_slices):
                    if nslice % 2 == 0:
                        slices_preds.insert(-1, 0.0)
                    else:
                        slices_preds.insert(0, 0.0)
            elif padding_factor < num_slices:
                num_additional_slices = int(num_slices - padding_factor)
                for nslice in range(num_additional_slices):
                    if nslice % 2 == 0:
                        del slices_preds[-1]
                    else:
                        del slices_preds[0]
            
            # Assign to train or validation
            if split == "train":
                X_train.append(slices_preds)
                y_train.append(1 if patient_label == "hgg" else 0)
            else:
                X_val.append(slices_preds)
                y_val.append(1 if patient_label == "hgg" else 0)
                
        return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

# Utility functions for evaluation and visualization
def get_confusion_matrix(labels: np.ndarray, preds: np.ndarray, thresh: float = 0.5) -> Tuple[List[int], List[np.ndarray]]:
    """Calculate confusion matrix components."""
    preds_binary = np.array(preds > thresh, dtype=int)
    
    tp_indices = (preds_binary == 1) & (labels == 1)
    tn_indices = (preds_binary == 0) & (labels == 0)
    fp_indices = (preds_binary == 1) & (labels == 0)
    fn_indices = (preds_binary == 0) & (labels == 1)
    
    num_tp = sum(tp_indices.astype(int))
    num_tn = sum(tn_indices.astype(int))
    num_fp = sum(fp_indices.astype(int))
    num_fn = sum(fn_indices.astype(int))
    
    cm = [num_tp, num_tn, num_fp, num_fn]
    indices = [tp_indices, tn_indices, fp_indices, fn_indices]
    
    return cm, indices

def get_filtered_df(df, preds_column="Prediction", thresh=0.5,
                   n_slices=10, method="slice_based", model_type="lgg_hgg"):
    pids = sorted(set(df["PatientID"].values))
    filtered_df = pd.DataFrame([], columns=df.columns)
    for pid in pids:
        patient_df = df[df["PatientID"]==pid]
        if method == "slice_based":
            preds = patient_df[preds_column].values
            labels = np.where(preds>=thresh, 1, 0)
            if np.sum(labels) < n_slices:
                sindices = np.argsort(preds)[::-1]
                filtered_df = pd.concat([filtered_df, patient_df.iloc[sindices[:n_slices], :]])
            else:
                pos_indices = labels == 1
                filtered_df = pd.concat([filtered_df, patient_df.iloc[pos_indices, :]])
        elif method == "patient_based":
            filtered_df.pd.concat([filtered_df, patient_df])

    if model_type == 'normal_abnormal':
        filtered_df['NumLabels'] = filtered_df['Label'].apply(lambda x: 0 if x.lower() == 'normal' else 1)
    else:
        filtered_df['NumLabels'] = filtered_df['Label'].apply(lambda x: 1 if x.lower() == 'hgg' else 0)
 
    return filtered_df

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
