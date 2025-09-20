"""
Model Building Pipeline for Multi-Stage Medical Image Classification

This file demonstrates how to assemble the 3-stage model pipeline:
1. Abnormality Detection (CNN)
2. Glioma Classification (CNN) 
3. ML Aggregation (Classical ML)
"""

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
from src.models.blocks import DenseNet121, EfficientNetB2
from src.models.abn_detector import AbnormalityDetector  
from src.models.glioma_classifier import GliomaClassifier
from src.models.aggregator import MLAggregator
from src.utils import prepare_cross_validation_splits
from src.models.utils import (
    DataProcessor,
    get_train_val_df, get_filtered_df,
    get_patient_prediction_vector
)

class ModelBuilder:
    """Main class for building and training the complete model pipeline."""
    
    def __init__(self, config: DictConfig, **kawrgs):

        self.config = config
        
        # Initialize models
        self.abn_detector = None
        self.glioma_classifier = None 
        self.aggregator = None
        self.pipeline = None

        # essential parameters
        self.class_weights = kawrgs.get('class_weights', None)
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the combined dataset."""
        # Load the combined dataframe (assumes preprocessing is done)
        df_path = Path(self.config.data.combined_df_path)
        df = pd.read_csv(df_path)
        
        print(f"Loaded dataset with {len(df)} samples")
        print(f"Datasets: {df['Dataset'].value_counts().to_dict()}")
        print(f"Labels: {df['Label'].value_counts().to_dict()}")
        
        return df
    
    def prepare_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare cross-validation splits ensuring no patient leakage."""
        # Create patient-level labels for stratification
        patient_labels = {}
        patient_ids = sorted(set(df['PatientID'].values))
        
        for pid in patient_ids:
            patient_df = df[df['PatientID'] == pid]
            labels = patient_df['Label'].values
            
            if "hgg" in labels:
                patient_labels[pid] = "hgg"
            elif "lgg" in labels:
                patient_labels[pid] = "lgg" 
            else:
                patient_labels[pid] = "normal"
        
        print(f"Patient-level distribution: {pd.Series(patient_labels.values()).value_counts().to_dict()}")
        
        # Create stratified splits
        df_with_splits = prepare_cross_validation_splits(
            df, patient_labels, 
            n_splits=self.config.training.n_splits,
            random_seed=self.config.training.random_seed
        )
        
        return df_with_splits
    
    def build_abnormality_detector(self) -> AbnormalityDetector:
        """Build and train abnormality detection model."""
        
        # Build model
        self.detector = AbnormalityDetector(
            self.config,
            class_weights=self.class_weights
        )
        self.abn_detector = self.detector.build_model()
        
        print(f"Model architecture: {self.config.models.abnormality_detector.architecture}")
        print(f"Model parameters: {self.abn_detector.count_params():,}")
        
        # # Train model
        # if self.config.training.train_models:
        #     history = detector.train(
        #         train_seq, val_seq,
        #         epochs=self.config.training.epochs,
        #         batch_size=self.config.training.batch_size
        #     )
            
        #     # Save model
        #     model_path = self.output_dir / f"abn_detector_s{split_idx}.h5"
        #     detector.save_model(str(model_path))
        #     print(f"Saved abnormality detector to {model_path}")
            
        #     # Save training history
        #     history_path = self.output_dir / f"abn_detector_s{split_idx}_history.pkl"
        #     with open(history_path, 'wb') as f:
        #         pickle.dump(history.history, f)
        # else:
        #     # Load pre-trained model
        #     model_path = self.output_dir / f"abn_detector_s{split_idx}.h5"
        #     if model_path.exists():
        #         detector.load_model(str(model_path))
        #         print(f"Loaded pre-trained abnormality detector from {model_path}")
        #     else:
        #         raise FileNotFoundError(f"Pre-trained model not found: {model_path}")
        
        return self.abn_detector
    
    def build_glioma_classifier(self) -> GliomaClassifier:
        """Build and train glioma classification model."""
        # Build model
        classifier = GliomaClassifier(
            self.config,
            class_weights=self.class_weights
        )
        self.glioma_classifier = classifier.build_model()
    
    def build_ml_aggregator(self) -> MLAggregator:
        """Build and train ML aggregator."""
        
        self.aggregator = MLAggregator(
            model_name=self.config.aggregator.model_name,
        )
        return self.aggregator
    
    def get_abn_detector(self):
        return self.abn_detector
    
    def get_glioma_classifier(self):
        return self.glioma_classifier
    
    def get_aggregator(self):
        return self.aggregator