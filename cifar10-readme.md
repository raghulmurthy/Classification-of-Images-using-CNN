# CIFAR-10 Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 different categories.

## Dataset Overview

The CIFAR-10 dataset consists of:
- 60,000 32x32 color images
- 10 different classes
- 6,000 images per class
- 50,000 training images
- 10,000 test images

Classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Model Architecture

The CNN model is built using a flexible architecture with hyperparameter tuning:

### Final Model Architecture
After hyperparameter tuning, the best model consists of:

Convolutional Layers:
- Conv2D: 64 filters, (30, 30) output shape
- Conv2D: 32 filters, (30, 30) output shape
- MaxPooling2D: (15, 15) output shape

Dense Layers:
- Flatten: 7200 units
- Dense: 128 units
- Dense: 32 units
- Dropout: 0.3 rate
- Dense: 10 units (output layer)

Total Parameters: 946,442 (3.61 MB)

### Model Performance

Training Results (15 epochs):
- Final Training Accuracy: 96.80%
- Final Training Loss: 0.0929
- Final Validation Accuracy: 62.97%
- Final Validation Loss: 2.4517

Test Performance:
- Test Accuracy: 62.97%
- Test Loss: 2.4517

Note: The gap between training and validation performance indicates overfitting, suggesting room for improvement through regularization techniques.

## Hyperparameter Tuning

Using Keras-Tuner with RandomSearch:
- Trials: 10
- Epochs per trial: 5
- Executions per trial: 1
- Optimization metric: validation accuracy

Results:
- Best validation accuracy: 67.23%
- Total tuning time: 8m 48s
- Best hyperparameters were used to build the final model

Note: The hyperparameter search explored various combinations of:
- Number of convolutional layers
- Number of filters
- Number of dense layers
- Units per layer
- Dropout rate
- Learning rate

## Data Preprocessing

1. Data Loading:
```python
(trn_img, trn_label), (test_img, test_label) = datasets.cifar10.load_data()
```

2. Train-Test Split:
```python
trn_img, test_img, trn_label, test_label = train_test_split(
    trn_img, trn_label, test_size=0.2, random_state=0
)
```

3. Normalization:
```python
trn_img = trn_img/255
test_img = test_img/255
```

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Keras-tuner

## Setup and Usage

1. Install required packages:
```bash
pip install tensorflow numpy matplotlib scikit-learn keras-tuner
```

2. Import necessary libraries:
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
```

3. Run hyperparameter tuning:
```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='cifar10_tunning'
)
```

## Model Training

The model is compiled with:
- Optimizer: Adam (with tunable learning rate)
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

## Visualization

The project includes visualization of:
- Sample images from each class
- Training history
- Model architecture
- Predictions on test data

## Future Improvements

1. Data Augmentation techniques
2. Implementation of more advanced architectures:
   - ResNet
   - VGG
   - DenseNet
3. Ensemble methods
4. Advanced regularization techniques