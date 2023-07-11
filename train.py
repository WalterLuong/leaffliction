from fastai.vision.all import ImageDataLoaders, accuracy, vision_learner, models
from pathlib import Path
import argparse


# Plant selection and dataset paths
plant = 'Apple'
dataset_train_path = Path('data/images', plant)


# Load dataset
data = ImageDataLoaders.from_folder(
    dataset_train_path, valid_pct=0.2, size=224, num_workers=4, bs=4)


# Model definition (VGG16/VGG19)
learn = vision_learner(data, models.vgg19_bn, metrics=accuracy)
learn.model


# Train model
learn.fit(2)
learn.export(f'{plant}_vgg19.pkl')


# Evaluate model
learn.show_results()
