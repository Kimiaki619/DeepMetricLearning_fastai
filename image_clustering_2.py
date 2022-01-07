# coding: utf-8
import tensorflow as tf
from tqdm import tqdm
from dlcliche.image import *

from fastai.vision import *
from torchvision import datasets, transforms
from torch import nn
import PIL

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing, cluster
import numpy as np
import pandas as pd
import sys
import cv2
import os
from progressbar import ProgressBar 
import shutil
import pathlib

#たくぼがつくったものグラフ化するライブラリ
import tsne

TARGET_IMAGES_DIR = 'clustring/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = 'clustring/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label
IMAGE_CLASTER_FILE = 'label.csv'					#ラベルのcsvの名前をここに書く

#IMAGE_PATH = [4,1,5]
IMAGE_PATH = [4,0,6,2,1,3,5]
#IMAGE_PATH = []

class Image_Clustering:
	def __init__(self, embs, n_clusters=7, video_file='IMG_2140.MOV', image_file_temp='img_%s.jpg', image_size=224,t_SNE_OK=True,image_path=[4,0,6,2,1,3,5]):
		self.n_clusters = n_clusters            # The number of cluster
		self.video_file = video_file            # Input video file name
		self.image_file_temp = image_file_temp  # Image file name template
		self.t_SNE_OK = t_SNE_OK
		self.image_size = image_size
		self.embs = embs
		self.image_path = image_path

	def main(self):
		self.label_images()
		self.classify_images()

	def label_images(self):
		print('Label images...')	
		# Get images
		#クラスタリングした画像のパス

		path_l=[]
		image_path = [4,0,6,2,1,3,5]
		for i in self.image_path:
			path = pathlib.Path((TARGET_IMAGES_DIR+str(i)+"/")).glob('*.jpg')      
			for p in path:
				path_l.append(p)
		#path = pathlib.Path(TARGET_IMAGES_DIR+str(i)).glob('*.jpg')
		#path_l = [p for p in path]

		#クラスタリングする画像
		#images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.jpg']]
		images = []

		for i in self.image_path:
			for p in os.listdir(TARGET_IMAGES_DIR+str(i)):
				images.append(p)

		assert(len(images)>0)
		
		X = self.embs

		# Clutering images by k-means++
		
		#X = np.array(X)
		Tsne = tsne.tSNE(X,path_l,n_jobs=50)
		#X = preprocessing.normalize(X)

        #この関数にtsneの値が入っている
		if self.t_SNE_OK == True:
			Tsne = tsne.tSNE(X,path_l,n_jobs=50)
			X = Tsne.t_sne
			X = np.array(X)
			print("T-SNEはTrueです")
		else:
			X = preprocessing.normalize(X)

		kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
		print('')
		print('labels:')
		print(kmeans.labels_)
		print('')
		
		# Merge images and labels
		print(len(images))
		df = pd.DataFrame({'image': images, 'label': kmeans.labels_})
		df.to_csv(IMAGE_LABEL_FILE, index=False)

		#ラベルのcsv
		df_label = pd.read_csv(IMAGE_CLASTER_FILE,index_col=0)
		df_cla = pd.read_csv(IMAGE_LABEL_FILE,index_col=1)
		#ラベルの列を変える
		df_cla = df_cla.values
		df_cla = [df_cla[f][0] for f in range(len(df_cla))]
		df_label = df_label.reindex(df_cla)
		df_label_index = df_label.values
		df_label_index = [df_label_index[i][0] for i in range(len(df_label_index))]
        
		#df_label_index = [df_label_index[f][1] for f in range(len(df_label_index))]
		
		#print(df_cla)
        #df_cla = df_cla.reindex(index=df_label)
		#df_cla = df_cla.values
		#df_cla = [df_cla[f][0] for f in range(len(df_cla))]
		A_R_I = metrics.normalized_mutual_info_score(df_label_index,kmeans.labels_)
		print(df_label_index)
		print(kmeans.labels_)
		print("---------------------------")
		print("精度："+str(A_R_I))
        
		#ここで画像つきのtsneを出力
		Tsne.graph_image()
		#ここではkmeansのクラス分けを見る
		Tsne.graph_clstering(kmeans.labels_)
		Tsne.graph_clstering_3d(kmeans.labels_)
		#ここでは元々のラベルを見る
		Tsne.graph_clstering(df_label_index)
		Tsne.graph_clstering_3d(df_label_index)


	def classify_images(self):
		print('Classify images...')

		# Get labels and images
		df = pd.read_csv(IMAGE_LABEL_FILE)
		labels = list(set(df['label'].values))
		
		# Delete images which were clustered before
		if os.path.exists(CLUSTERED_IMAGES_DIR):
			shutil.rmtree(CLUSTERED_IMAGES_DIR)
		
		images = []
		targetlabel = []
		image_path = [4,0,6,2,1,3,5]
		#for i in [4,1,5]:
		for i in self.image_path:
			for p in os.listdir(TARGET_IMAGES_DIR+str(i)):
				targetlabel.append(i)
				images.append(p)

		for label in labels:
			print('Copy and paste label %s images.' % label)

			# Make directories named each label
			new_dir = CLUSTERED_IMAGES_DIR + str(label) + '/'
			if not os.path.exists(new_dir):
				os.makedirs(new_dir)

			# Copy images to the directories
			clustered_images = df[df['label']==label]['image'].values
			for ci in clustered_images:
				n = images.index(ci)
				src = TARGET_IMAGES_DIR+str(targetlabel[n])+"/" + ci
				dst = CLUSTERED_IMAGES_DIR + str(label) + '/' + ci
				shutil.copyfile(src, dst)

		
if __name__ == "__main__":
	Image_Clustering().main()


