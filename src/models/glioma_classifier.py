import tensorflow as tf
import numpy as np
from omegaconf.dictconfig import DictConfig
from typing import Optional, Dict, Any
from pydoc import locate

from src.models.utils import create_callbacks
from src.utils import f1

tfk = tf.keras

class GliomaClassifier:
    """
    Glioma classification model (LGG vs. HGG).
    Second stage in the pipeline - classifies gliomas as LGG (Low Grade) or HGG (High Grade).
    """
    
    def __init__(self, config: DictConfig,
                    class_weights = None):
        """
        Initialize abnormality detector.
        
        Args:
            config: Configuration object containing model parameters
            model_type: Type of model architecture ('densenet121' or 'efficientnetb2')
        """
        self.config = config
        self.model_name = self.config.glioma_classifier.model_name
        self.model = None
        self.history = None
        self.class_weights = class_weights

        self.callbacks = create_callbacks(
            checkpoint_path = self.config.glioma_classifier.chkp_path, 
            history_output_path = self.config.glioma_classifier.hist_outpath, 
            log_dir = self.config.glioma_classifierlog_dir
        )
    
    def build_model(self, class_weights: Optional[Dict[int, float]] = None) -> tfk.Model:
        """
        Build the glioma classification model.
        
        Args:
            class_weights: Optional class weights for handling imbalanced data
            
        Returns:
            Compiled Keras model
        """
        if class_weights is None:
            class_weights = self.class_weights
        self.model = locate(self.config.glioma_classifier.model)(config=self.config).get_compiled_model(class_weights) 
        return self.model
    
    def train(self, 
              train_seq,
              val_seq,
              epochs: int = 50,
              batch_size: int = 32,
              callbacks: Optional[list] = None,
              **kwargs) -> tf.keras.callbacks.History:
        """
        Train the glioma classification model.
        
        Args:
            train_seq: Training Sequence
            val_seq: Validation Sequence
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: List of Keras callbacks
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
 
        # Default callbacks
        if callbacks is None:
            callbacks = self.callbacks
        
        # Train the model
        self.history = self.model.fit(
            train_seq,
            val_seq,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        return self.history
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Predict glioma grade probabilities.
        
        Args:
            x: Input images
            batch_size: Prediction batch size
            
        Returns:
            Glioma grade probabilities (probability of HGG)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        predictions = self.model.predict(x, batch_size=batch_size)
        return predictions
    
    def predict_classes(self, x: np.ndarray, threshold: float = 0.5, batch_size: int = 32) -> np.ndarray:
        """
        Predict glioma grade classes.
        
        Args:
            x: Input images
            threshold: Classification threshold
            batch_size: Prediction batch size
            
        Returns:
            Predicted classes (0=LGG, 1=HGG)
        """
        probabilities = self.predict(x, batch_size=batch_size)
        return (probabilities > threshold).astype(int)
    
    def evaluate(self, data_seq, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test images
            y_test: Test labels
            batch_size: Evaluation batch size
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        results = self.model.evaluate(data_seq, batch_size=batch_size, verbose=0)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, results))
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = tfk.models.load_model(filepath, custom_objects={'f1': self._get_f1_metric()})
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def fine_tune(self, unfreeze_layers: int = 20, learning_rate: float = 1e-7):
        """
        Fine-tune the model by unfreezing top layers.
        
        Args:
            unfreeze_layers: Number of top layers to unfreeze
            learning_rate: Learning rate for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Find the base model layer
        base_model = None
        for layer in self.model.layers:
            if 'densenet' in layer.name.lower() or 'efficientnet' in layer.name.lower():
                base_model = layer
                break
                
        if base_model is not None:
            base_model.trainable = True
            # Fine-tune from this layer onwards
            fine_tune_at = len(base_model.layers) - unfreeze_layers
            
            # Freeze all the layers before fine_tune_at
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
                
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
                loss=self.model.loss,
                metrics=self.model.metrics
            )
    
    @staticmethod
    def _get_f1_metric():
        """Get F1 metric for model loading."""
        return f1