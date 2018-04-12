# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:17:02 2018

@author: USER_
"""

# ------------------------------------------------------
#                         K-MEANS
# ------------------------------------------------------

#geracao do dataset
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 150, n_features = 2, centers = 3, 
                  cluster_std = 0.5, random_state = 0)

#gráfica de dispersao dos pontos
import matplotlib.pyplot as plt
plt.title('Gráfico de dispersao')
plt.scatter(x[:, 0], x[:, 1], c = 'blue', marker = 'o')
plt.grid()
plt.show()

#fase de treinamento
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300,
                tol = 1e-4, random_state = 0)
kmeans.fit(x, y)
y_pred = kmeans.predict(x)

#gráfica do modelo
plt.title('Gráfica K-Means')
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], c = 'lightgreen', 
            marker = 's', label = 'cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], c = 'blue', 
            marker = 's', label = 'cluster 2')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], c = 'orange', 
            marker = 's', label = 'cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c = 'red', marker = '*', label = 'centroides', s = 200)
plt.legend(loc = 'upper right')
plt.show()

#calculo da distorcao do modelo
print('Distorcao: ', round(kmeans.inertia_, 3))

#                 -------- METODO DE AVALIACAO ELBOW --------

#funcao para desempenho x #clusters
def elbow(x, y):
    distortions = []
    for i in range(2, 11): #min: 2 clusters | #max: n_samples - 1
        kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10,
                        max_iter = 300, random_state = 0)
        kmeans.fit(x, y)
        distortions.append(kmeans.inertia_)
    return distortions

distortions = elbow(x, y)

#gráfica de desempenho x #clusters
plt.title('Gráfica de desempenho x #clusters')
plt.xlabel('#clusters')
plt.ylabel('distortions [SSE]')
plt.plot(range(1, 11), distortions, marker = 'o')
plt.show()


#              --------- METODO DE AVALIACAO SILHOUETTE --------

#gráfica silhouette
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

#funcao silhouette x 1 cluster
def silhouette_cluster(x, y_pred):
    lbl_clusters = np.unique(y_pred)
    n_clusters = lbl_clusters.shape[0]
    silhouettes = silhouette_samples(x, labels = y_pred, metric = 'euclidean')
    
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    for i, c in enumerate(lbl_clusters):
        c_silhouettes = silhouettes[y_pred == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(i/n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes, 
                 height = 1.0, color = color)
        yticks.append((y_ax_lower + y_ax_upper)/2)
        y_ax_lower += len(c_silhouettes)
    
    silhouettes_avg = np.mean(silhouettes)
    plt.title('Silhouette - desempenho clusterizacao')
    plt.axvline(silhouettes_avg, color = 'red', linestyle = '--')
    plt.yticks(yticks, lbl_clusters + 1)
    plt.xlabel('Shilhouette coefficient')
    plt.ylabel('cluster')
    plt.show()
    return round(silhouettes_avg, 3)        

#                    --- silhouette para k = 3 clusters ---

#treinamento k-means [#clusters = 3]
kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300,
                tol = 1e-4, random_state = 0)
kmeans.fit(x, y)
y_pred = kmeans.predict(x)

#desempenho - métrica silhouette
avg_silhouette3 = silhouette_cluster(x, y_pred)

#                     --- silhouette para k = 2 clusters ---

#treinamento k-means [#clusters = 2]
kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init = 10, max_iter = 300,
                tol = 1e-4, random_state = 0)
kmeans.fit(x, y)
y_pred = kmeans.predict(x)

#desempenho - métrica silhouette
avg_silhouette2 = silhouette_cluster(x, y_pred)

# --funcao para avaliacao iterativa do silhouette [verificar #clusters ideal] --

def silhouettes(x, y):
    silh_metrics = []
    for i in range(2, 11): #min: 2 clusters | #max: n_samples - 1
        kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, 
                        max_iter = 300, tol = 1e-4, random_state = 0)
        kmeans.fit(x, y)
        y_pred = kmeans.predict(x)        
        silh_cluster = np.mean(silhouette_samples(x, labels = y_pred, 
                                                  metric = 'euclidean'))
        silh_metrics.append(silh_cluster)
    return silh_metrics

silhouette_coefs = silhouettes(x, y)

#gráfica de desempenho silhouette x #clusters
plt.title('Gráfica de desempenho silhouette')
plt.plot(range(2, 11), silhouette_coefs, marker = 'o')
plt.xlabel('#clusters')
plt.ylabel('silhouette coefficient')
plt.show()


