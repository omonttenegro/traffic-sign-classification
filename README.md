# Traffic Sign Classification

A computer vision project developed for **CS50’s Introduction to Artificial Intelligence with Python**, focused on classifying traffic signs using a convolutional neural network.

## Overview

This project trains a neural network to recognize traffic signs from images across **43 categories**.

The system:

- loads and preprocesses image data
- resizes images to a fixed input size
- splits the dataset into training and testing sets
- trains a convolutional neural network with TensorFlow
- evaluates model performance
- optionally saves the trained model to a `.h5` file

## Main Features

- image loading with OpenCV
- preprocessing and resizing
- multi-class classification
- train/test split with scikit-learn
- convolutional neural network built with TensorFlow/Keras
- optional trained model export

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **scikit-learn**
- **NumPy**

## Files

```text
traffic-sign-classification/
├── README.md
├── requirements.txt
└── traffic.py
```

## Requirements

The project uses the following Python packages:

- `opencv-python`
- `scikit-learn`
- `tensorflow`

Install them with:

```bash
pip install -r requirements.txt
```

## How to Run

Run the script with the dataset directory as argument:

```bash
python traffic.py data_directory
```

To save the trained model:

```bash
python traffic.py data_directory model.h5
```

## Dataset

The dataset is **not included in this repository** due to its size.

To run the project, download the traffic sign image dataset separately and place it in a directory structured with one folder per category, numbered from `0` to `42`.

Example:

```text
data_directory/
├── 0/
├── 1/
├── 2/
├── ...
└── 42/
```

## Model Architecture

The neural network includes:

- two convolutional layers
- max-pooling layers
- a flatten layer
- a dense hidden layer
- dropout for regularization
- a softmax output layer with 43 classes

## Author

**André Montenegro**
