from sklearn.metrics import classification_report, confusion_matrix
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import sys
import warnings
import logging
from Augmentation import Augmentation, cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] \
        in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


y_true = []
y_pred = []


def unzip_model(
    model_path: Path = Path(
        'data/zipped_files/vgg19_SOTA.zip')
) -> Path:
    if not model_path.exists():
        raise FileNotFoundError('Model does not exist.')

    try:
        logger.info(f'Unzipping {model_path.name}')
        extract_dir = Path(model_path.parent, model_path.stem)
        if not os.path.isdir(extract_dir):
            os.mkdir(extract_dir)
        with ZipFile(model_path, 'r') as zipObj:
            zipObj.extractall(extract_dir)
        logger.info('Success')
        return (Path(extract_dir, Path(model_path.stem).with_suffix('.pkl')))
    except Exception as e:
        logger.error('Error unzipping model.')
        raise e


def print_prediction(
    image_path: Path,
    pred_class: str,
    pred_idx: int,
    outputs: torch.Tensor,
    truth: str
) -> None:
    print('---------------------------')
    print(f'Image name: {image_path.stem}')
    print('Prediction: ', pred_class)
    print('Probability: ', outputs[pred_idx].item())
    print('Truth: ', truth)
    print('---------------------------')


def plot_confusion_matrix(y_true: list, y_pred: list) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,
                    s=cm[i, j],
                    va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def predict_image(
    model_path: Path = Path('data/zipped_files/vgg19_SOTA/vgg19_SOTA.pkl'),
    image_path: Path = Path('data/valid/')
) -> None:
    model = torch.load(model_path)
    if image_path.is_dir():
        images = list(image_path.glob('**/*.JPG'))
        if not len(images) > 0:
            raise FileNotFoundError('No image files found.')
        logger.info('Predicting images in directory')

        for index, image in enumerate(images):
            true_class = image.parent.name
            pred_class, pred_idx, outputs = model.predict(image)
            y_true.append(true_class)
            y_pred.append(pred_class)
            tqdm.write(f"Progress: {index+1}/{len(images)}")
            print_prediction(image, pred_class, pred_idx, outputs, true_class)
        print(classification_report(y_true, y_pred))
        plot_confusion_matrix(y_true, y_pred)

    else:
        print(image_path)
        if not image_path.suffix == '.JPG':
            raise FileNotFoundError('File is not a JPG')
        logger.info('Predicting one image')

        pred_class, pred_idx, outputs = model.predict(image_path)
        print_prediction(image_path, pred_class, pred_idx, outputs, '')
        aug = Augmentation()
        img = cv2.imread(str(image_path))
        aug_img = aug.contrast(img)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(aug_img)
        plt.title('Augmented Image')
        plt.suptitle(f'Class predicted: {pred_class}')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        default=Path('data/valid/'),
        help='Path to image/dataset to test the model on'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=Path('data/zipped_files/vgg19_SOTA.zip'),
        help='Path to model to use'
    )
    args = parser.parse_args()

    try:
        if not Path(args.path).exists():
            raise FileNotFoundError('Path does not exist.')

        unzipped_model_path = Path(
            'data',
            'zipped_files',
            Path(args.model).stem,
            Path(args.model).stem+'.pkl'
        )
        model_path = unzip_model(
            Path(args.model)) \
            if not unzipped_model_path.exists() \
            else unzipped_model_path
        predict_image(model_path, Path(args.path))
    except Exception as e:
        logger.error(e)
        sys.exit(1)
