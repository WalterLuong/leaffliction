from fastai.vision.all import *
from pathlib import Path

plant = 'Grape'
dataset_train_path = Path('data/images', plant)

data = ImageDataLoaders.from_folder(
    dataset_train_path, valid_pct=0.2, size=224, num_workers=4, bs=4)

learn = vision_learner(data, models.vgg16_bn, metrics=accuracy)
learn.model
learn.fit(2)
learn.save('stage-1')
learn.export('export.pkl')
learn.show_results()
