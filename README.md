# Multi-Stage Deep Learning Framework for Glioma Grading (LGG/HGG)

This repository contains the implementation for our paper: **"MRI-Based Classification of Low- and High-Grade Gliomas Using Deep Learning, Radiomics, and Their Combination"**

## Overview

This work presents a novel three-stage deep learning pipeline for automated glioma grading from MRI images, combining the strengths of convolutional neural networks (CNNs) and classical machine learning approaches.

## Architecture

The repository implements a multi-stage classification pipeline:

1. **Stage 1: Abnormality Detection** - CNN model to distinguish normal vs abnormal brain slices
2. **Stage 2: Glioma Classification** - CNN model to classify gliomas as Low-Grade Glioma (LGG) vs High-Grade Glioma (HGG)  
3. **Stage 3: ML Aggregation** - Classical ML model that aggregates slice-level predictions to make final patient-level classifications

### Repository Structure

```
├── configs/                    # Configuration files (YAML format)
├── data/                      # Data directory
│   ├── preprocessed/           # Preprocessed data storage
│   └── raw_data/            # Raw MRI data (see Data Structure below)
├── runs/                     # Training runs and logs
├── src/                      # Source code
│   ├── models/              # Model architecture implementations
│   │   ├── __init__.py
│   │   ├── abn_detector.py   # Abnormality detection model
│   │   ├── aggregator.py     # ML aggregation model
│   │   ├── blocks.py         # Neural network building blocks
│   │   ├── glioma_classifier.py # Glioma classification model
│   │   └── utils.py          # Data pipeline and utility functions
│   ├── data_ingestion.py    # Data ingestion to create data directories
│   ├── data_preparation.py  # Data preparation for training and inference
│   ├── inference_engine.py  # Inference engine for complete pipeline
│   ├── interpretation.py    # Model interpretability (Grad-CAM, XAI)
│   ├── model_building.py    # Model building orchestration
│   ├── preprocessing.py     # Data Preprocessing Utilities
│   └── utils.py             # Functional Utilities
├── utils/                    # Preprocessing utilities ()
├── inference.py              # Get predictions script
├── prepare.py                # Data preparation script
├── preprocess.py             # Data preprocessing script
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
└── LICENSE
```

## Data Structure

The raw data should be organized as follows:

```
data/raw_data/
├── [PatientID]/              # e.g., A0B267EC7/
│   ├── flair/               # FLAIR MRI sequences
│   ├── nifti/               # NIfTI format files  
│   ├── t1/                  # T1-weighted images
│   └── [numeric_files...]   # raw_dicom files (not necessary if you've prepared the above folders)
└── [Additional patients...]
```

Each patient directory should contain:
- **flair/**: FLAIR (Fluid Attenuated Inversion Recovery) MRI sequences
- **nifti/**: NIfTI format brain imaging files
- **t1/**: T1-weighted MRI images
- Numeric files representing different scan parameters or processed volumes

### Data Pipeline Setup

**Important**: Simply place your MRI data in the `data/` directory with each patient in a separate folder. The `prepare.py` script will automatically create the complete data pipeline from your organized patient folders.

The preparation script handles:
- Automatic detection of patient directories
- Extraction and conversion of MRI sequences
- Creation of training/validation splits while maintaining patient separation
- Generation of metadata CSV files
- Preprocessing pipeline setup

This automated approach ensures that patient data remains properly isolated across different splits, preventing data leakage during cross-validation.

## Training Strategy

### Individual Model Training

Each model in the pipeline is trained independently:

1. **Abnormality Detection Model**: 
   - Trained on normal vs abnormal brain slices
   - Uses data from both healthy controls and glioma patients
   - Implements class balancing techniques for imbalanced datasets

2. **Glioma Classification Model**:
   - Trained specifically on LGG vs HGG classification
   - Uses only abnormal slices (filtered from Stage 1)
   - Employs transfer learning from pretrained CNNs

3. **ML Aggregator**:
   - Trained on patient-level features extracted from CNN predictions
   - Uses classical ML algorithms (Random Forest, SVM, etc.)
   - Combines slice-level predictions into final patient diagnosis

### Training Configuration

The training process is controlled through YAML configuration files in the `configs/` directory. Key parameters include:
- Model architectures (DenseNet, EfficientNet, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Cross-validation splits

## Usage

### Configuration-Driven Approach

All parameters and settings can be specified in the YAML configuration files instead of hardcoding them in bash scripts. This includes:
- **data.yml**: Input/output directories, file patterns, data preparations, augmentations 
- **preprocessing.yml**: Preprocessing params/steps: Image dimentions, ...
- **model.yml**: Architectures, hyperparameters, training
- **inference.yml**: infernece directories, strategies (integerated into model.yml)
- **training.yml**: Training params (integerated into model.yml) #TODO

Simply modify the `configs/x.yml` files to customize all aspects of the data pipeline and model training without editing any source code.

### 1. Data Preparation
```bash
# Prepare data and create CSV files for preprocessing
python prepare.py --data-dir /path/to/patient/data --out-fpath /path/to/data-info.csv
```

### 2. Preprocessing
```bash
# Preprocess raw MRI data
python preprocess.py --csv-fpath /path/to/data-info.csv --working-dir path/to/preprocessing/working/dir
```

### 3. Model Training

> TODO

```bash 
# Train individual models (each trained separately)
python train.py --config configs/config.yaml --model abnormality_detector
python train.py --config configs/config.yaml --model glioma_classifier  
python train.py --config configs/config.yaml --model ml_aggregator
```

### 4. Model Evaluation

> TODO

```bash
# Evaluate trained models
python evaluate.py --config configs/config.yaml
```

### 5. Model Interpretation

> TODO

```bash
# Generate explanations using Grad-CAM and other XAI methods
python explain.py --config configs/config.yaml
```

## Inference Pipeline

The complete three-stage pipeline is implemented in `inference.py`. This provides a unified interface that:

1. **Loads all three trained models** (abnormality detector, glioma classifier, ML aggregator)
2. **Processes input MRI data** through the complete pipeline
3. **Handles data flow** between stages:
   - Stage 1 output filters data for Stage 2
   - Both CNN outputs feed into Stage 3 ML aggregator
4. **Returns comprehensive predictions** including:
   - Slice-level abnormality predictions
   - Slice-level glioma classifications  
   - Final patient-level diagnosis
   - Confidence scores and interpretability maps

### Inference Usage
```bash
# Run complete pipeline inference
python inference.py --data_dir /path/to/patient/data --predictions-fpath /path/to/results.csv
```

The inference engine automatically handles:
- Data preprocessing and normalization
- Model loading and prediction coordination
- Result aggregation and visualization
- Generation of explanation heatmaps

## Key Features

- **Modular Design**: Each component can be trained and evaluated independently
- **Flexible Architecture**: Support for multiple CNN backbones (DenseNet, EfficientNet, ResNet)
- **Robust Pipeline**: Handles missing data and provides confidence estimates
- **Interpretability**: Integrated Grad-CAM and other XAI techniques
- **Cross-Validation**: Built-in support for k-fold cross-validation
- **Configuration-Driven**: All parameters controlled through YAML configs

## Requirements

See `requirements.txt` for detailed dependencies. Key libraries include:
- TensorFlow/Keras for deep learning models
- scikit-learn for ML aggregation
- SimpleITK for medical image processing
- OpenCV for image preprocessing
- imgaug for data augmentation

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{paper,
  title={[Paper Title]},
  author={[Authors]},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]}
}
```

## License

This project is licensed under the MIT - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- [Name] - [Email]
- [Collaborator] - [Email]

## Acknowledgments

- [Data providers]
- [Collaborating institutions]
