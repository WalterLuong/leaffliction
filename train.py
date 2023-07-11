from fastai.vision.all import ImageDataLoaders, accuracy, vision_learner, models
from pathlib import Path


# Plant selection and dataset paths
plant = 'Grape'
dataset_train_path = Path('data/images', plant)


# Load dataset
data = ImageDataLoaders.from_folder(
    dataset_train_path, valid_pct=0.2, size=224, num_workers=4, bs=4)


# Model definition (VGG16)
learn = vision_learner(data, models.vgg16_bn, metrics=accuracy)
learn.model


# Train model
learn.fit(2)
learn.save('stage-1')
learn.export('export.pkl')


# Evaluate model
learn.show_results()
