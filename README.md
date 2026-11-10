
# PA04: Tom and Jerry Image Classification

## 1. Overview

This project investigates how training dataset size and class imbalance affect image classification performance.

Five binary classification models were trained to classify Tom and Jerry images using Google Teachable Machine.

## 2. Research Questions

- RQ1: How does training dataset size affect classification accuracy?
- RQ2: How does class imbalance affect classification performance?

## 3. Dataset

Dataset: Tom and Jerry Image Classification

Source:
https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

The Kaggle dataset page lists the license as CC0 (Public Domain).

The original images are not redistributed in this repository.
Please download the dataset from Kaggle and use the provided dataset split files to reproduce the experiments.

The selected dataset consists of:
- Tom: 1,930 images
- Jerry: 1,240 images

For each class, the first 800 images sorted by frame number were selected as the training pool.

From the remaining images, 200 odd-numbered frames per class were randomly selected for testing using random seed 42.

The fixed test dataset contains:
- Tom: 200 images
- Jerry: 200 images

## 4. Experimental Setup

| Model | Tom | Jerry | Total |
|---|---:|---:|---:|
| Model 1 | 200 | 200 | 400 |
| Model 2 | 800 | 800 | 1600 |
| Model 3 | 200 | 800 | 1000 |
| Model 4 | 800 | 200 | 1000 |
| Model 5 | 500 | 500 | 1000 |

Training settings:
- Platform: Google Teachable Machine
- Epochs: 50
- Batch size: 16
- Learning rate: 0.001

All models were exported in Keras format.

## 5. Environment

Recommended environment:
- Windows
- Python 3.11
- TensorFlow 2.15.0

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 6. Evaluation

Download the original dataset from Kaggle.

Prepare the fixed test images under:

```text
dataset/test/tom/
dataset/test/jerry/
```

Each folder must contain 200 images.

Run the evaluation:

```powershell
python evaluate_models.py
```

The script evaluates the five trained models using the same test dataset.

Evaluation metrics:
- Accuracy
- Macro F1-score
- Precision
- Recall
- Confusion Matrix

## 7. Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Model 1 | 76.00% | 75.98% |
| Model 2 | 79.50% | 79.40% |
| Model 3 | 71.00% | 69.79% |
| Model 4 | 70.50% | 68.93% |
| Model 5 | 76.75% | 76.74% |

Model 2 achieved the highest overall accuracy.

Class imbalance reduced overall accuracy and minority-class recall in this experiment.

Detailed predictions, class-level metrics, and confusion matrices are stored in the results folder.

## 8. Limitations

The images were extracted from videos, so consecutive frames may contain similar scenes.

The sequential dataset split may introduce scene-related bias and does not guarantee independence between training and test data.

## 9. Model Files

Each model folder contains:
- keras_model.h5
- labels.txt

The original training images are not redistributed in this repository.
