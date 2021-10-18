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

#学習回数
TRAING_NUM = 5
#T-sneによる可視化の保存場所
FILE_NAME = "../image/"
#modelの保存場所
MODEL_PATH = "../model/"
MODEL_NAME_CNN = "cnn_model"
MODEL_NAME_CNN_ARCFACE = "cnn_arcface_model"


def prepare_full_MNIST_databunch(data_folder, tfms):
    """
    Prepare dataset as images under:
        data_folder/images/('train' or 'valid')/(class)
    where filenames are:
        img(class)_(count index).png
    """
    train_ds = datasets.MNIST(data_folder, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    valid_ds = datasets.MNIST(data_folder, train=False,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))

    def have_already_been_done():
        return (data_folder/'images').is_dir()
    def build_images_folder(data_root, X, labels, dest_folder):
        images = data_folder/'images'
        for i, (x, y) in tqdm.tqdm(enumerate(zip(X, labels))):
            folder = images/dest_folder/f'{y}'
            ensure_folder(folder)
            x = x.numpy()
            image = np.stack([x for ch in range(3)], axis=-1)
            PIL.Image.fromarray(image).save(folder/f'img{y}_{i:06d}.png')

    if not have_already_been_done():
        build_images_folder(data_root=DATA, X=train_ds.train_data,
                            labels=train_ds.train_labels, dest_folder='train')
        build_images_folder(data_root=DATA, X=valid_ds.test_data, 
                            labels=valid_ds.test_labels, dest_folder='valid')

    return ImageDataBunch.from_folder(data_folder/'images', ds_tfms=tfms)


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


def show_3D_tSNE(latent_vecs, target, title='3D t-SNE viz'):
    latent_vecs = latent_vecs
    tsne = TSNE(n_components=3, random_state=0).fit_transform(latent_vecs)
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=target, cmap='jet')
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.show()
    plt.savefig(FILE_NAME+title+".jpg")


def show_as_PCA(latent_vecs, target, title='PCA viz'):
    latent_vecs = latent_vecs
    #latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap='jet')
    plt.colorbar()
    plt.show()

#dataを入れる。
DATA = Path('data')
data = prepare_full_MNIST_databunch(DATA, get_transforms(do_flip=False))

#まず可視化、何も使わずに可視化を行なっている。
raw_x = np.array([a.data.numpy() for a in data.valid_ds.x])
raw_y = np.array([int(y.obj) for y in data.valid_ds.y])
raw_x = raw_x.reshape((len(raw_x), -1))

if False: # for saving time
    LIMIT = 1000
    chosen_idxes = np.random.choice(list(range(len(raw_x))), LIMIT)
    raw_x = raw_x[chosen_idxes]
    raw_y = raw_y[chosen_idxes]

show_2D_tSNE(raw_x, raw_y, 'Raw sample distributions (t-SNE)')

#cnn 普通のcnnでの学習。比較対象
def learner_conventional(train_data):
    learn = cnn_learner(train_data, models.resnet18, metrics=accuracy)
    learn.fit(1)
    learn.unfreeze()
    learn.fit(TRAING_NUM)
    return learn

learn = learner_conventional(data)
#modelの保存
learn.save(MODEL_PATH+MODEL_NAME_CNN)
embs = get_embeddings(body_feature_model(learn.model), data.valid_dl)
show_2D_tSNE(embs, [int(y) for y in data.valid_ds.y], title='Simply trained　CNN (t-SNE)')
show_3D_tSNE(embs,[int(y) for y in data.valid_ds.y],title='Simply trained　CNN (t-SNE) 3D')

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
    learn = cnn_learner(train_data, models.resnet18, metrics=accuracy)
    learn.model = XFaceNet(learn.model, train_data, ArcMarginProduct, m=0.5)
    learn.callback_fns.append(partial(LabelCatcher))
    learn.fit(1)
    learn.unfreeze()
    learn.fit(TRAING_NUM)
    return learn

learn = learner_ArcFace(data)
#modelの保存
learn.save(MODEL_PATH+MODEL_NAME_CNN_ARCFACE)
embs = get_embeddings(body_feature_model(learn.model), data.valid_dl)
show_2D_tSNE(embs, [int(y) for y in data.valid_ds.y], title='ArcFace (t-SNE)')
show_3D_tSNE(embs,[int(y) for y in data.valid_ds.y],title='ArcFace (t-SNE) 3D')