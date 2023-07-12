from fastai.vision.all import ImageDataLoaders, accuracy, vision_learner, models
from pathlib import Path
import argparse
import os
import sys
import warnings
from zipfile import ZipFile
import logging

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


def train_model(dataset_train_path: Path) -> None:
    plant = dataset_train_path.name
    try:
        assert plant in ['Apple', 'Grape'], 'Invalid path.'
    except AssertionError as e:
        logging.error(e)
        sys.exit(1)

    # MODIFY IMAGES HERE
    data = ImageDataLoaders.from_folder(
        dataset_train_path, valid_pct=0.2, size=224, num_workers=4, bs=4)

    learn = vision_learner(data, models.vgg16_bn, metrics=accuracy)

    # learn.fit(2)

    if not os.path.isdir(Path('data', 'models', plant)):
        os.makedirs(Path('data', 'models', plant))
    learn.path = Path('data', 'models', plant)
    if os.path.isfile(Path(learn.path, f'{plant}_vgg16.pkl')):
        version = 1
        while os.path.isfile(Path(learn.path, f'{plant}_vgg16_v{version}.pkl')):
            version += 1
        model_to_save = f'{plant}_vgg16_v{version}.pkl'
        learn.export(Path(model_to_save))

    with ZipFile(Path(learn.path, f'{plant}_vgg16_v{version}.zip'), 'w') as zipObj:
        # ZIP IMAGES HERE
        logging.info(f'Zipping {plant}_vgg16_v{version}.pkl')
        zipObj.write(Path(learn.path, f'{plant}_vgg16_v{version}.pkl'))
        logging.info('Done')

    learn.show_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        default='data/images/Apple/',
        help='Path to image dataset to train the model on'
    )
    args = parser.parse_args()

    try:
        assert os.path.isdir(args.path), 'Invalid path.'
        assert len(os.listdir(args.path)) > 0, 'Directory is empty.'
        assert len(os.listdir(Path(args.path, os.listdir(
            args.path)[0]))) > 0, 'Subdirectories are empty.'
        assert len(list(Path(args.path).glob('**/*.JPG'))
                   ) > 0, 'No image files found.'
    except AssertionError as e:
        logging.error(e)
        sys.exit(1)

    try:
        train_model(Path(args.path))
    except Exception as e:
        logging.error(e)
        sys.exit(1)
