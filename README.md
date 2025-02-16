---

# Custom Object Detection with Novel Bounding Box Metric

This repository provides the implementation for training a custom object detection model using YOLOv5. 

## Table of Contents

1. [Repository Setup](#1-repository-setup)
2. [Installation](#2-installation)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training the Model](#4-training-the-model)
5. [Custom Metric Definition](#5-custom-metric-definition)
<!-- 6. [Model Evaluation](#6-model-evaluation) -->
6. [YOLOv5 Setup and Architecture](#6-yolov5-setup-and-architecture)

---

## 1. Repository Setup

To begin, clone the repository to your local machine:

```bash
git clone https://github.com/Moshintha20/Custom-Object-Detection-and-Novel-Bounding-Box-Metric-with-YOLOv5.git
```

Navigate into the project folder:

```bash
cd Custom-Object-Detection-and-Novel-Bounding-Box-Metric-with-YOLOv5
```

---

## 2. Installation

Install the required dependencies by running the following command inside the project directory:

```bash
pip install -r requirements.txt
```

This will install all necessary packages.

---

## 3. Dataset Preparation
### Original Kaggle Dataset

The dataset used for training the model is the **Dog and Cat Detection** dataset, available on Kaggle. You can access and download the original dataset from the following link:

[Dog and Cat Detection Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection/data)


### Downloading the Dataset

The `dataloader.py` script is used to download the Kaggle dataset, resize images, and organize the data into directories suitable for YOLOv5. The script also generates a `data.yaml` file, which includes the dataset configuration.

Run the following command to start the dataset preparation:

```bash
python dataloader.py --output /path/to/download_dataset_dir
```

Replace `/path/to/download_dataset_dir` with the actual path where you want the dataset to be saved.

### Dataset Structure

Once executed, the following structure will be created:
- The images will be resized.
- The dataset will be split into training, validation, and testing sets.
- A `data.yaml` file will be generated containing the necessary dataset configuration for YOLOv5.

---

## 4. Training the Model

### Training Command

Once the dataset is prepared, you can train the YOLOv5 model using the `train_yolo.py` script. This script utilizes the custom bounding box similarity metric for training the model.

Use the following command to begin training:

```bash
python train_yolo.py --data /path/to/data.yaml --weights /full/path/to/yolov5/yolov5su.pt --batch-size 8 --epochs 10
```

- Replace `/path/to/data.yaml` with the path to your `data.yaml` file.
- Adjust the `--batch-size` and `--epochs` parameters as per your requirements.

---

## 5. Custom Metric Definition

This repository defines a custom bounding box similarity metric for evaluating object detection performance. The metric is integrated into the training process via the `train_yolo.py` script, but it can be further customized by modifying the metric-related code within this script.

---
<!-- 
## 6. Model Evaluation

After training the model, you can evaluate its performance on the test set using the `evaluate.py` script. To evaluate the model, use the following command:

```bash
python evaluate.py --weights /path/to/trained_model.pt --data /path/to/data.yaml
```

- Replace `/path/to/trained_model.pt` with the path to your trained model weights.
- Replace `/path/to/data.yaml` with the path to the `data.yaml` file.

---
-->
## 6. YOLOv5 Setup and Architecture

The object detection system in this repository is built using the YOLOv5 architecture. The architecture has been modified to support the integration of a custom bounding box similarity metric. YOLOv5 provides a fast, reliable, and scalable solution for object detection, and this repository enhances it with customized performance metrics.
