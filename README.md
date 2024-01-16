<div align="center">
  <center><h1> 🍃 Leaffliction - Leaf Disease Recognition using Computer Vision</h1></center>
  </div>
  
## Overview
Leaffliction is a deep learning project aimed at identifying diseases in leaves through computer vision techniques. The dataset is from PlantVillage. It leverages image processing, augmentation, and Convolutional Neural Networks using Python libraries such as FastAI, OpenCV, and PyTorch.

## Features
- **Data Preprocessing**: Includes image augmentation and transformation for dataset enhancement.
- **Model Training**: Utilizes FastAI with a VGG19 model for training on augmented datasets.
- **Disease Classification**: Employs classification techniques to identify various leaf diseases.
- **Image Analysis Tools**: Features tools for image analysis and visualization.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, use the following command with the path to your training dataset:
```bash
python train.py /path/to/dataset
```

### Predicting Diseases
For disease prediction on new leaf images, use:

```bash
python predict.py /path/to/image_or_directory
```

### Image Augmentation
To augment images in your dataset, execute:

```bash
python Augmentation.py /path/to/images
```

### Image Transformation
For applying various transformations to your images:

```bash
python Transformation.py /path/to/image_or_directory
```

### Data Visualization
To visualize the distribution of classes in your dataset:

```bash
python data_visualization.py /path/to/dataset
```




## Acknowledgments
Made in collaboration with [Walter](https://github.com/WalterLuong)
