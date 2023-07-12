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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


y_true = []
y_pred = []


def unzip_model(
    model_path: Path = Path(
        'data/zipped_files/augmented_directory_vgg16_v1.zip')
) -> Path:
    with ZipFile(model_path, 'r') as zipObj:
        logger.info(f'Unzipping {model_path.name}')
        zipObj.extractall(Path(model_path).parent)
        logger.info('Done')


def print_prediction(
    pred_class: str,
    pred_idx: int,
    outputs: torch.Tensor,
    truth: str
) -> None:
    print('---------------------------')
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


def predict_image(image_path: Path) -> None:
    model = torch.load('data/zipped_files/augmented_directory_vgg16_v1.pkl')
    if image_path.is_dir():
        images = list(image_path.glob('**/*.JPG'))
        try:
            assert len(images) > 0, 'No images found'
        except AssertionError as e:
            logger.error(e)
            sys.exit(1)

        for index, image in enumerate(images):
            pred_class, pred_idx, outputs = model.predict(image)
            y_true.append(image.parent.name)
            y_pred.append(pred_class)
            tqdm.write(f"Progress: {index}/{len(images)}")
            print_prediction(pred_class, pred_idx, outputs, image.parent.name)
        print(classification_report(y_true, y_pred))
        plot_confusion_matrix(y_true, y_pred)

    else:
        try:
            assert image_path.suffix == '.JPG', 'Not a JPG file'
        except AssertionError as e:
            logger.error(e)
            sys.exit(1)

        pred_class, pred_idx, outputs = model.predict(image_path)
        y_true.append(image_path.parent.name)
        y_pred.append(pred_class)
        print_prediction(pred_class, pred_idx, outputs, image_path.parent.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        default=Path('data/valid/Apple'),
        help='Path to image/dataset to test the model on'
    )
    args = parser.parse_args()

    try:
        assert Path(args.path).exists(), 'Path does not exist'
        unzip_model()
        predict_image(Path(args.path))
    except Exception as e:
        logger.error(e)
        sys.exit(1)
