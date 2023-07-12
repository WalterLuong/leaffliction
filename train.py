import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.optimizers import *
from keras.losses import *
from keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import numpy as np
import tensorflow
import keras
import sklearn
import os
import sys
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model on a dataset')
    parser.add_argument('--dataset', type=str, default='augmented_directory/Apple/', help='Path to dataset')