"""
クラスタリング_fastai
"""

from fastai.vision import *
from torchvision import datasets, transforms
from torch import nn
import torch
import PIL
from tqdm import tqdm
from dlcliche.image import *

#多分必要なライブラリ
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
#%matplotlib notebook

#arcfaceでの学習
from metrics import *

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label
