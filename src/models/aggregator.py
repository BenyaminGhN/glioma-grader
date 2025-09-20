import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional, Tuple, List
import joblib
import warnings
warnings.filterwarnings('ignore')


class MLAggregator:
    """
    Machine Learning aggregator for combining CNN model outputs.
    Third stage in the pipeline - combines abnormality detection and glioma classification
    outputs to make final predictions using classical ML algorithms.
    """
    
    def __init__(self, model_name: str = 'random_forest', random_state: int = 42):
        """
        Initialize ML aggregator.
        
        Args:
            model_type: Type of ML model to use 
                       ('random_forest', 'gradient_boosting', 'logistic_regression', 
                        'svm', 'neural_network', 'ensemble')
            random_state: Random state for reproducibility
        """
        self.model_name = model_name.lower()
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
        # Validate model type
        valid_models = ['random_forest', 'gradient_boosting', 'logistic_regression', 
                       'svm', 'neural_network', 'ensemble']
        if self.model_name not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")
    
    def _get_base_model(self) -> Any:
        """Get the base ML model based on model_type."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                gamma='scale'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state,
                alpha=0.001
            ),
            'ensemble': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                    ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000))
                ],
                voting='soft'
            )
        }
        return models[self.model_type]
    
    def build_model(self, use_scaling: bool = True) -> Pipeline:
        """
        Build the ML aggregator model.
        
        Args:
            use_scaling: Whether to include feature scaling in the pipeline
            
        Returns:
            Scikit-learn pipeline with the ML model
        """
        base_model = self._get_base_model()
        
        if use_scaling:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', base_model)
            ])
        else:
            self.model = Pipeline([
                ('classifier', base_model)
            ])
            
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MLAggregator':
        """
        Train the ML aggregator model.
        
        Args:
            X: Feature matrix from CNN outputs
            y: Target labels
            **kwargs: Additional arguments for model fitting
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.build_model()
            
        # Store feature names if provided
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix from CNN outputs
            
        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix from CNN outputs
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        results = {}
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        results['accuracy'] = report['accuracy']
        results['precision'] = report['weighted avg']['precision']
        results['recall'] = report['weighted avg']['recall']
        results['f1_score'] = report['weighted avg']['f1-score']
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # ROC AUC (for binary classification)
        if len(np.unique(y_test)) == 2:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            results['average_precision'] = average_precision_score(y_test, y_pred_proba[:, 1])
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            self.build_model()
            
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores.tolist()
        }
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, param_grid: Dict = None, cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        if self.model is None:
            self.build_model()
            
        # Default parameter grids for different models
        if param_grid is None:
            param_grids = {
                'random_forest': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [5, 10, 15, None],
                    'classifier__min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'classifier__n_estimators': [50, 100, 150],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                },
                'logistic_regression': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2']
                },
                'svm': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__gamma': ['scale', 'auto', 0.1, 1]
                },
                'neural_network': {
                    'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'classifier__alpha': [0.001, 0.01, 0.1]
                }
            }
            param_grid = param_grids.get(self.model_type, {})
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        if filepath is None:
            filepath = self.config.aggregator.model_path
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str = None):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if filepath is None:
            filepath = self.config.aggregator.model_path
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data.get('feature_names')
        self.random_state = model_data.get('random_state', 42)
        self.is_fitted = True