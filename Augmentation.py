import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
import glob
from tqdm import tqdm
import sys


class Augmentation:
    '''
    This class is used to augment the images in the dataset.
    Eight (8) different types of augmentation are used:
    1. Translation
    2. Shear
    3. Flip
    4. Rotate
    5. Crop
    6. Skew
    7. Distortion
    8. Blur
    '''

    def __init__(self):
        pass

    def translation(self, img: np.ndarray) -> np.ndarray:
        '''
        Translate the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be translated.

        Returns:
            translated_img (np.ndarray): The translated image.
        '''
        rows, cols, dim = img.shape
        M = np.float32([[1, 0, np.random.randint(-100, 100)],
                        [0, 1, np.random.randint(-100, 100)],
                        [0, 0, 1]])
        translated_img = cv2.warpPerspective(img, M, (cols, rows))
        return translated_img

    def shear(self, img, axis=0) -> np.ndarray:
        '''
        Shear the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be sheared.
            axis (int, optional): The axis along which the image is to be sheared. Defaults to 0.

        Returns:
            sheared_img (np.ndarray): The sheared image.
        '''
        rows, cols, dim = img.shape
        if axis == 0:
            M = np.float32([[1, np.random.uniform(0.1, 0.5), 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        else:
            M = np.float32([[1, 0, 0],
                            [np.random.uniform(0.1, 0.5), 1, 0],
                            [0, 0, 1]])
        sheared_img = cv2.warpPerspective(img, M, (cols, rows))
        return sheared_img

    def flip(self, img: np.ndarray, axis: int = 0) -> np.ndarray:
        '''
        Flip the image along the x or y axis.

        Args:
            img (np.ndarray): The image to be flipped.
            axis (int, optional): The axis along which the image is to be flipped. Defaults to 0.

        Returns:
            reflected_img (np.ndarray): The flipped image.
        '''
        rows, cols, dim = img.shape
        if axis == 0:
            M = np.float32([[-1, 0, cols],
                            [0, 1, 0],
                            [0, 0, 1]])
        else:
            M = np.float32([[1, 0, 0],
                            [0, -1, rows],
                            [0, 0, 1]])
        reflected_img = cv2.warpPerspective(img, M, (cols, rows))
        return reflected_img

    def rotate(self, img: np.ndarray) -> np.ndarray:
        '''
        Rotate the image by a random angle.

        Args:
            img (np.ndarray): The image to be rotated.

        Returns:
            rotated_img (np.ndarray): The rotated image.
        '''
        rows, cols, dim = img.shape
        M = cv2.getRotationMatrix2D(
            (cols/2, rows/2), np.random.randint(-180, 180), 1)
        rotated_img = cv2.warpAffine(img, M, (cols, rows))
        return rotated_img

    def crop(self, img: np.ndarray) -> np.ndarray:
        '''
        Crop the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be cropped.

        Returns:
            cropped_img (np.ndarray): The cropped image.
        '''
        rows, cols, dim = img.shape
        x = np.random.randint(0, cols-100)
        y = np.random.randint(0, rows-100)
        cropped_img = img[y:y+100, x:x+100]
        return cropped_img

    def skew(self, img: np.ndarray) -> np.ndarray:
        '''
        Skew the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be skewed.

        Returns:
            skewed_img (np.ndarray): The skewed image.
        '''
        rows, cols, dim = img.shape
        pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
        pts2 = np.float32([[np.random.randint(0, 100), np.random.randint(0, 100)], [cols-np.random.randint(
            0, 100), np.random.randint(0, 100)], [np.random.randint(0, 100), rows-np.random.randint(0, 100)]])
        M = cv2.getAffineTransform(pts1, pts2)
        skewed_img = cv2.warpAffine(img, M, (cols, rows))
        return skewed_img

    def distortion(self, img: np.ndarray) -> np.ndarray:
        '''
        Distort the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be distorted.

        Returns:
            distorted_img (np.ndarray): The distorted image.
        '''
        rows, cols, dim = img.shape
        pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        pts2 = np.float32([[np.random.randint(0, 100), np.random.randint(0, 100)], [cols-np.random.randint(
            0, 100), np.random.randint(0, 100)], [np.random.randint(0, 100), rows-np.random.randint(0, 100)], [cols-np.random.randint(0, 100), rows-np.random.randint(0, 100)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        distorted_img = cv2.warpPerspective(img, M, (cols, rows))
        return distorted_img

    def blur(self, img: np.ndarray) -> np.ndarray:
        '''
        Blur the image.

        Args:
            img (np.ndarray): The image to be blurred.

        Returns:
            blurred_img (np.ndarray): The blurred image.
        '''
        blurred_img = cv2.blur(img, (5, 5), 0)
        return blurred_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment image(s) in the dataset. Augmentation applied: Translation, Flip, Rotate, Blur, Crop, Distortion. Images are saved in the "augmented_directory" folder.')
    parser.add_argument('-p', '--path', type=str, default='data/images/Apple/apple_healthy/image (1).JPG',
                        help='Path to the image/directory to be augmented.')
    args = parser.parse_args()

    try:
        assert os.path.isdir(args.path) or os.path.isfile(
            args.path), 'Invalid path.'
    except AssertionError as e:
        print(e)
        sys.exit(1)

    if os.path.isdir(args.path):
        images = Path(args.path).glob('**/*.JPG')
        for image in tqdm(images, desc=f'Augmenting images from {args.path}', total=len(os.listdir(args.path))):
            aug = Augmentation()
            img = cv2.imread(str(image))
            save_path = Path('augmented_directory',
                             image.parent.parent.stem, image.parent.stem)
            os.makedirs(save_path, exist_ok=True)

            plt.imsave(Path(save_path, image.name), img)

            translated_img = aug.translation(img)
            plt.imsave(
                Path(save_path, f'{image.stem}_Translation.JPG'), translated_img)

            flipped_img = aug.flip(img, axis=np.random.randint(0, 2))
            plt.imsave(Path(save_path, f'{image.stem}_Flip.JPG'), flipped_img)

            rotated_img = aug.rotate(img)
            plt.imsave(
                Path(save_path, f'{image.stem}_Rotate.JPG'), rotated_img)

            blurred_img = aug.blur(img)
            plt.imsave(Path(save_path, f'{image.stem}_Blur.JPG'), blurred_img)

            cropped_img = aug.crop(img)
            plt.imsave(Path(save_path, f'{image.stem}_Crop.JPG'), cropped_img)

            distorted_img = aug.distortion(img)
            plt.imsave(Path(save_path, f'{image.stem}_Distortion.JPG'),
                       distorted_img)
    else:
        aug = Augmentation()
        img = cv2.imread(args.path)
        img_name = Path(args.path).stem

        translated_img = aug.translation(img)
        plt.imsave(img_name + '_Translation.JPG', translated_img)

        flipped_img = aug.flip(img, axis=np.random.randint(0, 2))
        plt.imsave(img_name + '_Flip.JPG', flipped_img)

        rotated_img = aug.rotate(img)
        plt.imsave(img_name + '_Rotate.JPG', rotated_img)

        blurred_img = aug.blur(img)
        plt.imsave(img_name + '_Blur.JPG', blurred_img)

        cropped_img = aug.crop(img)
        plt.imsave(img_name + '_Crop.JPG', cropped_img)

        distorted_img = aug.distortion(img)
        plt.imsave(img_name + '_Distortion.JPG', distorted_img)

        plt.figure(figsize=(20, 20))
        plt.subplot(1, 7, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 7, 2)
        plt.imshow(translated_img)
        plt.title('Translated Image')
        plt.axis('off')
        plt.subplot(1, 7, 3)
        plt.imshow(flipped_img)
        plt.title('Flipped Image')
        plt.axis('off')
        plt.subplot(1, 7, 4)
        plt.imshow(rotated_img)
        plt.title('Rotated Image')
        plt.axis('off')
        plt.subplot(1, 7, 5)
        plt.imshow(blurred_img)
        plt.title('Blurred Image')
        plt.axis('off')
        plt.subplot(1, 7, 6)
        plt.imshow(cropped_img)
        plt.title('Cropped Image')
        plt.axis('off')
        plt.subplot(1, 7, 7)
        plt.imshow(distorted_img)
        plt.title('Distorted Image')
        plt.axis('off')
        plt.show()
