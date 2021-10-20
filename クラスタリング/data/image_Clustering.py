# coding: utf-8

from fastai import *
from fastai.vision import *
from torch.nn import modules
from torchvision import datasets, transforms
from torch import nn
import torch
import PIL
from tqdm import tqdm
from dlcliche.image import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import cv2
import os
from progressbar import ProgressBar 
import shutil

import XFaceNet

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label


class Image_Clustering:
    def __init__(self, n_clusters=7, video_file='IMG_2140.MOV', image_file_temp='img_%s.png', input_video=True):
        self.n_clusters = n_clusters            # The number of cluster
        self.video_file = video_file            # Input video file name
        self.image_file_temp = image_file_temp  # Image file name template
        self.input_video = input_video          # If input data is a video


    def main(self):
        self.label_images()
        self.classify_images()


    def label_images(self):
        print('Label images...')

        # Load a model
        model = VGG16(weights='imagenet', include_top=False)
    
        # Get images
        images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.png', '.jpg']]
        assert(len(images)>0)
        
        X = []
        pb = ProgressBar(max_value=len(images))
        for i in range(len(images)):
            # Extract image features
            feat = self.__feature_extraction(model, TARGET_IMAGES_DIR+images[i])
            X.append(feat)
            pb.update(i)  # Update progressbar

        # Clutering images by k-means++
        X = np.array(X)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
        print('')
        print('labels:')
        print(kmeans.labels_)
        print('')
        
        # Merge images and labels
        df = pd.DataFrame({'image': images, 'label': kmeans.labels_})
        df.to_csv(DATA_DIR+IMAGE_LABEL_FILE, index=False)

    def learner_ArcFace(train_data):
        learn = cnn_learner(train_data, models.resnet18, metrics=accuracy)
        learn.model = XFaceNet(learn.model, train_data, ArcMarginProduct, m=0.5)
        learn.callback_fns.append(partial(LabelCatcher))
        return learn

    def body_feature_model(model):
        """
        Returns a model that output flattened features directly from CNN body.
        """
        try:
            body, head = list(model.org_model.children()) # For XXNet defined in this notebook
        except:
            body, head = list(model.children()) # For original pytorch model
        return nn.Sequential(body, head[:-1])


    def get_embeddings(embedding_model, data_loader, label_catcher=None, return_y=False):
        """
        Calculate embeddings for all samples in a data_loader.
        
        Args:
            label_catcher: LearnerCallback for keeping last batch labels.
            return_y: Also returns labels, for working with training set.
        """
        embs, ys = [], []
        for X, y in data_loader:
            # For each batch (X, y),
            #   Set labels (y) if label_catcher's there.
            if label_catcher:
                label_catcher.on_batch_begin(X, y, train=False)
            #   Get embeddings for this batch, store in embs.
            with torch.no_grad():
                # Note that model's output is not softmax'ed.
                out = embedding_model(X).cpu().detach().numpy()
                out = out.reshape((len(out), -1))
                embs.append(out)
            ys.append(y)
        # Putting all embeddings in shape (number of samples, length of one sample embeddings)
        embs = np.concatenate(embs) # Maybe in (10000, 10)
        ys   = np.concatenate(ys)
        if return_y:
            return embs, ys
        return embs

    def __feature_extraction(self, model, img_path):
        img = image.load_img(img_path, target_size=(224, 224))  # resize
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # add a dimention of samples
        x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels

        feat = model.predict(x)  # Get image features
        feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array

        return feat


    def classify_images(self):
        print('Classify images...')

        # Get labels and images
        df = pd.read_csv(DATA_DIR+IMAGE_LABEL_FILE)
        labels = list(set(df['label'].values))
        
        # Delete images which were clustered before
        if os.path.exists(CLUSTERED_IMAGES_DIR):
            shutil.rmtree(CLUSTERED_IMAGES_DIR)

        for label in labels:
            print('Copy and paste label %s images.' % label)

            # Make directories named each label
            new_dir = CLUSTERED_IMAGES_DIR + str(label) + '/'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Copy images to the directories
            clustered_images = df[df['label']==label]['image'].values
            for ci in clustered_images:
                src = TARGET_IMAGES_DIR + ci
                dst = CLUSTERED_IMAGES_DIR + str(label) + '/' + ci
                shutil.copyfile(src, dst)

        
if __name__ == "__main__":
    Image_Clustering().main()