import requests
from bs4 import BeautifulSoup
import cv2
import pathlib
import numpy as np
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import axes3d, Axes3D 


class tSNE():
	def __init__(self,image_path_x,path_l,n_jobs):
		self.image_path_x = image_path_x
		self.path_l = path_l
		self.t_sne = self.t_SNE(n_jobs)

	def imscatter(self,x, y, image_list, ax=None, zoom=1):
	    if ax is None:
	        ax = plt.gca()
	    im_list = [OffsetImage(plt.imread(str(p)), zoom=zoom) for p in image_list]
	    x, y = np.atleast_1d(x, y)
	    artists = []
	    for x0, y0, im in zip(x, y, im_list):
	        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
	        artists.append(ax.add_artist(ab))
	    ax.update_datalim(np.column_stack([x, y]))
	    ax.autoscale()
	    return artists

	def graph_clstering(self,cals):
		colors = ["c", "g", "b", "r", "y", "m", "k", "orange","pink"]
		color = [colors[i] for i in range(len(set(cals)))]
		#plt.figure(figsize = (30, 30))
		for x,i in zip(self.t_sne,cals):
			plt.scatter(x[0],x[1],color=colors[i],s=10,label=str(i))            
		#plt.legend()
		plt.show()

	def graph_clstering_3d(self,cals):
		image = preprocessing.normalize(self.image_path_x)
		colors = ["c", "g", "b", "r", "y", "m", "k", "orange","pink"]
		color = [colors[i] for i in range(len(set(cals)))]
		tsne = TSNE(n_components=3, random_state=0).fit_transform(self.image_path_x)
		fig = plt.figure(figsize=(13,10))
		ax = fig.add_subplot(111, projection='3d')
		scatter = ax.scatter3D(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=cals, cmap='jet')
		#ax.set_title(title)
		plt.colorbar(scatter)
		plt.show()
		
        
	def graph_image(self):
		#こっちでもいいかも
		# perplexity: 20
		fig, ax = plt.subplots(figsize=(30,30))
		self.imscatter(self.t_sne[:,0], self.t_sne[:,1], self.path_l, ax=ax, zoom=0.4)
		plt.show()

	def t_SNE(self,n_jobs):
		image = preprocessing.normalize(self.image_path_x)
		tsne =TSNE(n_components=2, random_state=0) # 20が一番いい感じでした
		kills_reduced = tsne.fit_transform(self.image_path_x)
		return kills_reduced

