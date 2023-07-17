from fastai.vision.all import ImageDataLoaders, vision_learner, \
    models, accuracy, aug_transforms
from pathlib import Path
import argparse
from zipfile import ZipFile
import subprocess
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
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] \
        in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train_model(dataset_train_path: Path) -> str:
    dataset_name = dataset_train_path.name

    try:
        logger.info(f'Loading {dataset_name} dataset')
        data = ImageDataLoaders.from_folder(
            dataset_train_path,
            valid_pct=0.2,
            size=224,
            num_workers=4,
            bs=4,
            batch_tfms=aug_transforms(mult=2)
        )
        logger.info('Loading dataset done')
        logger.info('Training model')
        learn = vision_learner(data, models.vgg19_bn, metrics=accuracy)
        learn.fit(1)
        logger.info('Training done')
    except Exception as e:
        logger.error('Error while training model')
        raise e

    try:
        logger.info('Saving model')
        learn.path = Path('data', 'models', dataset_name)
        if not os.path.isdir(learn.path):
            os.makedirs(Path(learn.path))
        version = 1
        while os.path.isfile(
            Path(
                learn.path,
                f'{dataset_name}_vgg19_v{version}.pkl')
        ):
            version += 1
        model_to_save = f'{dataset_name}_vgg19_v{version}.pkl'
        learn.export(Path(model_to_save))
        logger.info('Saving done')
        return Path(model_to_save)
    except Exception as e:
        logger.error('Error while saving model')
        raise e


def zip_model(
    dataset_path: Path,
    model_to_save: Path
) -> None:
    try:
        model_path = Path('data', 'models', dataset_path, model_to_save)
        with ZipFile(model_path.with_suffix('.zip'), 'w') as zipObj:
            logger.info(f'Zipping {model_to_save}')
            zipObj.write(model_path, model_to_save.name)
            logger.info(f'Zipping {dataset_path}')
            for folderName, subfolders, filenames in os.walk(dataset_path):
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath)
            logger.info('Zipping done')

        logger.info(
            f'Moving {model_to_save.with_suffix(".zip")} \
                to zipped_files folder')
        zipped_files_path = Path('data', 'zipped_files')
        if not os.path.isdir(zipped_files_path):
            os.makedirs(zipped_files_path)
        os.rename(Path(model_path.with_suffix('.zip')),
                  Path(
                      zipped_files_path,
                      f'{model_to_save.with_suffix(".zip")}')
                  )
        logger.info('Moving done')
    except Exception as e:
        logger.error('Error while zipping model')
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        default='data/images/',
        help='Path to image dataset to train the model on'
    )
    args = parser.parse_args()

    try:
        if not os.path.isdir(args.path):
            raise NotADirectoryError('Path is not a directory.')
        if not len(os.listdir(args.path)) > 0 or \
                not os.path.isdir(Path(args.path, os.listdir(args.path)[0])):
            raise FileNotFoundError('Subdirectories not found.')
        if not len(list(Path(args.path).glob('**/*.JPG'))) > 0:
            raise FileNotFoundError('No image files found.')

        logger.info('Augmenting images')
        subprocess.run(['python3', 'Augmentation.py', args.path])
        logger.info('Image augmentation done')
        model = train_model(Path('augmented_directory'))
        zip_model(Path('augmented_directory'), model)
    except Exception as e:
        logger.error(e)
        sys.exit(1)
