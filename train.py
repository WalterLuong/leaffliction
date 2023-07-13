from fastai.vision.all import *
from pathlib import Path
import argparse
from zipfile import ZipFile
import os
import sys
import warnings
import logging
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train_model(dataset_train_path: Path) -> None:
    dataset_name = dataset_train_path.name

    # MODIFY IMAGES HERE
    data = ImageDataLoaders.from_folder(
        dataset_train_path, valid_pct=0.2, size=224, num_workers=4, bs=4, batch_tfms=aug_transforms(mult=2))

    learn = vision_learner(data, models.vgg19_bn, metrics=accuracy)

    learn.fit(1)

    if not os.path.isdir(Path('data', 'models', dataset_name)):
        os.makedirs(Path('data', 'models', dataset_name))
    learn.path = Path('data', 'models', dataset_name)
    version = 1
    model_to_save = ''
    if os.path.isfile(Path(learn.path, f'{dataset_name}_vgg19_v{version}.pkl')):
        version = 1
        while os.path.isfile(Path(learn.path, f'{dataset_name}_vgg19_v{version}.pkl')):
            version += 1
        model_to_save = f'{dataset_name}_vgg19_v{version}.pkl'
        learn.export(Path(model_to_save))
    else:
        model_to_save = f'{dataset_name}_vgg19_v{version}.pkl'
        learn.export(Path(model_to_save))

    with ZipFile(Path(learn.path, f'{dataset_name}_vgg19_v{version}.zip'), 'w') as zipObj:
        # ZIP IMAGES HERE
        logger.info(f'Zipping {dataset_name}_vgg19_v{version}.pkl')
        print((Path(learn.path, f'{dataset_name}_vgg19_v{version}.pkl')))
        zipObj.write(Path(learn.path, f'{dataset_name}_vgg19_v{version}.pkl'))
        logger.info('Done')

    learn.show_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        default='augmented_directory/',
        help='Path to image dataset to train the model on'
    )
    args = parser.parse_args()

    try:
        if not os.path.isdir(args.path):
            raise NotADirectoryError('Path is not a directory.')
        if not len(os.listdir(args.path)) > 0 or not os.path.isdir(Path(args.path, os.listdir(args.path)[0])):
            raise FileNotFoundError('Subdirectories not found.')
        if not len(list(Path(args.path).glob('**/*.JPG'))) > 0:
            raise FileNotFoundError('No image files found.')

        train_model(Path(args.path))
    except Exception as e:
        logger.error(e)
        sys.exit(1)
