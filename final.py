import numpy as np
import pandas as pd
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.cluster import AffinityPropagation

# for everything else
import matplotlib.pyplot as plt
from random import randint
import pickle

filepath = []

for dirname, _, filenames in os.walk('visualsimilarity/bottoms_resized_png/'):
    for filename in filenames:
        filepath.append(os.path.join(dirname, filename))



x = np.genfromtxt('data.csv', delimiter=',')

affprop = AffinityPropagation(affinity="euclidean", damping=0.5).fit(np.array(x))

cluster_map_a = pd.DataFrame()
cluster_map_a['data_index'] = filepath
cluster_map_a['cluster'] = affprop.labels_


def view_cluster(file, filedf):
    plt.figure(figsize=(25, 25));
    # gets the list of filenames for a cluster
    i = filedf.data_index[filedf.data_index == file].index.tolist()
    filecluster = filedf.cluster[i[0]]
    files = []
    for j in range(filedf.shape[0]):
        if filedf.cluster[j] == filecluster:
            files.append(filedf.data_index[j])

    if len(files) > 10:
        files = files[:10]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    print(files)



if __name__ == '__main__':
    k = str(input())
    view_cluster(k, cluster_map_a)
