---

# Custom Object Detection with Novel Bounding Box Metric

This repository provides the implementation for training a custom object detection model using YOLOv5. 

## Table of Contents

1. [Repository Setup](#1-repository-setup)
2. [Installation](#2-installation)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training the Model](#4-training-the-model)
5. [Custom Bounding Box Similarity Metric](#5-custom-bounding-box-similarity-metric)
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

## 5. Custom Bounding Box Similarity Metric

This repository introduces a **Hybrid IoU (HIoU)** similarity metric, which extends the traditional **Intersection over Union (IoU)** by incorporating additional geometric properties. **HIoU combines Fused IoU (FIoU) and Complete IoU (CIoU)** using a dynamic weighting approach to account for both **shape and position differences**.  

### **FIoU (Fused IoU)**  
FIoU refines IoU by adding a **corner-based penalty term**:  

$$
\text{FIoU} = \text{IoU} - \frac{l_2}{c^2}
$$  

where:  
- \( l_2 \) represents the **sum of squared differences** between the bounding box corners.  
- \( c^2 \) is the **squared diagonal** of the smallest enclosing box.  

---

### **CIoU (Complete IoU)**  
CIoU improves IoU by **penalizing center distance and aspect ratio differences**:  

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2}{c^2} - \alpha v
$$  

where:  
- \( \rho^2 \) is the **squared center distance**.  
- \( v \) represents the **aspect ratio difference**.  
- \( \alpha \) is a weight factor to adjust the aspect ratio penalty dynamically.  

---

### **HIoU (Hybrid IoU) – Our Custom Method**  
HIoU **dynamically adjusts** the influence of **FIoU and CIoU** based on IoU values:  

$$
\text{HIoU} = w_{\text{FIoU}} \cdot \text{FIoU} + w_{\text{CIoU}} \cdot \text{CIoU}
$$  

where the **weights are defined using a sigmoid function** to smoothly transition between FIoU (when IoU is low) and CIoU (when IoU is high):  

$$
w_{\text{FIoU}} = \sigma(k_1 (1 - \text{IoU}))
$$  

$$
w_{\text{CIoU}} = 1 - \sigma(k_2 (1 - \text{IoU}))
$$  

where \( \sigma(x) \) is the **sigmoid function**.  

This approach ensures that:  
- **When boxes have low overlap** → More emphasis is placed on **corner alignment (FIoU)**.  
- **When boxes are well-matched** → **Aspect ratio and center alignment (CIoU) become more important**.  

Although HIoU introduces **additional computations**, it provides a **structured approach** to bounding box similarity by incorporating multiple geometric factors beyond simple overlap.  


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

The object detection system in this repository is built using the YOLOv5 architecture. We started with the original YOLOv5 model and replaced its bounding box loss function with our custom one. In our approach, we added a new loss called Hybrid IoU (HIoU) that combines FIoU and CIoU. FIoU adds a penalty based on differences in the box corners, while CIoU adds penalties for the distance between box centers and differences in aspect ratio. We use a dynamic weighting system to balance these two losses depending on the overlap between boxes, so our model learns to adjust based on more geometric details.
