from fastai import *
from fastai.vision import *
from torch.nn import modules
from torchvision import datasets, transforms
from torch import nn
import torch
import PIL
from tqdm import tqdm
from dlcliche.image import *

#多分必要なライブラリ
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
#arcfaceでの学習
from metrics import *

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

# T-sneで根拠を可視化
def show_2D_tSNE(latent_vecs, target, title='t-SNE viz'):
    latent_vecs = latent_vecs
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap='jet')
    plt.colorbar()
    plt.show()
    plt.savefig(FILE_NAME+title+".jpg")

DATA = Path('data')
data = prepare_full_MNIST_databunch(DATA, get_transforms(do_flip=False))


#arcfaceでの学習
class LabelCatcher(LearnerCallback):
    last_labels = None

    def __init__(self, learn:Learner):
        super().__init__(learn)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        LabelCatcher.last_labels = last_target
        return {'last_input': last_input, 'last_target': last_target} 


class XFaceNet(nn.Module):
    def __init__(self, org_model, data, xface_product=ArcMarginProduct, m=0.5):
        super().__init__()
        self.org_model = org_model
        self.feature_model = body_feature_model(org_model)
        self.metric_fc = xface_product(512, data.c, m=m).cuda()
    
    def forward(self, x):
        x = self.feature_model(x)
        x = self.metric_fc(x, LabelCatcher.last_labels)
        return x


def learner_ArcFace(train_data):
    learn = cnn_learner(train_data, models.vgg19_bn, metrics=accuracy)
    learn.model = XFaceNet(learn.model, train_data, ArcMarginProduct, m=0.5)
    learn.callback_fns.append(partial(LabelCatcher))
    return learn

learn = learner_ArcFace(data)
#modelの保存
learn.load(MODEL_PATH+MODEL_NAME_CNN_ARCFACE)

embs = get_embeddings(body_feature_model(learn.model), data.valid_dl)
show_2D_tSNE(embs, [int(y) for y in data.valid_ds.y], title='ArcFace (t-SNE)')
show_3D_tSNE(embs,[int(y) for y in data.valid_ds.y],title='ArcFace (t-SNE) 3D')