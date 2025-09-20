import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import argparse

# Import our model components
from src.model_building import ModelBuilder
from src.models.utils import (get_filtered_df, get_patient_prediction_vector,
                              create_ml_vector)

class InferenceEngine:
    """Manages the complete inference pipeline of abnormality detection → glioma classification → aggregation."""
    
    def __init__(self, model: ModelBuilder):
        """
        Initialize the model pipeline.
        
        Args:
            model: model builder
        """
        self.abn_detector = model.get_abn_detector()
        self.glioma_classifier = model.get_glioma_classifier()
        self.aggregator = model.get_aggregator()
        
    def predict_patient(self, patient_df, pid) -> Dict[str, np.ndarray]:
        """
        Run complete pipeline prediction for a single patient.
        
        Args:
            patient_df: DataFrame containing patient slice information
            pid: Patient ID
            
        Returns:
            Dictionary containing predictions from each stage
        """
        results = {}
        
        # Get patient slice-level predictions from abnormality detector
        abn_labels, abn_preds, _ = get_patient_prediction_vector(
            pid, patient_df, model = self.abn_detector,
            to_predict = True, to_return_total_slices = True,
            preds_column = "abnormality_predictions"
        )
        results['abnormality_predictions'] = abn_preds
        results['abnormality_labels'] = abn_labels
        
        # Filter slices for glioma classification
        filtered_df = get_filtered_df(patient_df, 
                                    preds_column = "abnormality_predictions",
                                    model_type = "lgg_hgg")
        
        if len(filtered_df) > 0:
            # Get glioma classification predictions
            glioma_labels, glioma_preds, _ = get_patient_prediction_vector(
                pid, filtered_df, model = self.glioma_classifier,
                to_predict = True, to_return_total_slices = True,
            )
            results['glioma_predictions'] = glioma_preds
            results['glioma_labels'] = glioma_labels
            
            slices_vector = create_ml_vector(slice_preds = glioma_preds)
            # Final prediction using ML aggregator
            if self.aggregator is not None:
                final_pred = self.aggregator.predict(np.array([slices_vector]))
                results['final_prediction'] = final_pred[0]
                
                if hasattr(self.aggregator, 'predict_proba'):
                    final_proba = self.aggregator.predict_proba(np.array([slices_vector]))
                    results['final_probability'] = final_proba[0]
        else:
            # No abnormal slices found
            results['glioma_predictions'] = []
            results['glioma_labels'] = []
            results['final_prediction'] = 0  # Normal case
            results['final_probability'] = [1.0, 0.0]  # [normal, abnormal]
        
        return results
    
    def predict(self, data_seq) -> pd.DataFrame:
        """
        Run pipeline prediction on a batch of patients.
        
        Args:
            dataframe: DataFrame containing patient information
            
        Returns:
            DataFrame with predictions added
        """
        data_df = data_seq.get_dataset_df()
        results_df = data_df.copy()
        pids = sorted(set(data_df['PatientID'].values))
        
        final_predictions = []
        final_probabilities = []
        
        for pid in pids:
            patient_df = data_df[data_df['PatientID'] == pid]
            results = self.predict_patient(patient_df, pid)
            
            final_predictions.append(results.get('final_prediction', 0))
            final_probabilities.append(results.get('final_probability', [1.0, 0.0]))
        
        # Add predictions to dataframe (replicate for all slices of each patient)
        for i, pid in enumerate(pids):
            mask = results_df['PatientID'] == pid
            results_df.loc[mask, 'final_prediction'] = final_predictions[i]
            results_df.loc[mask, 'final_probability'] = final_probabilities[i][1]  # Probability of positive class
            
        return results_df
    
