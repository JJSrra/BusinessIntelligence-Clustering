#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import time

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

from math import floor

def ApplyKMeans(dataset, n_clusters, n_init):
    print("---KMeans---")
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
    t1 = time.time()
    cluster_predict = k_means.fit_predict(dataset) 
    t2 = time.time() - t1
    print("\tTime: {:.2f}s".format(t2))
    metric_CH = metrics.calinski_harabaz_score(dataset, cluster_predict)
    print("\tCalinski-Harabaz Index: {:.3f}".format(metric_CH))
    return cluster_predict

def ClusteringAlgorithms(dataset):
    k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)
    mbkm = MiniBatchKMeans(init='k-means++', n_clusters=4, n_init=5)
    
    clustering_algorithms = [
        ("K-Means", k_means),
        ("MiniBatchK-Means", mbkm)
    ]
    
    '''
    ("Mean-shift", mean_shift),
    ("DBSCAN", dbscan),
    ("Birch", birch),
    ("SpectralClustering", spectral),
    ("Ward", ward)
    '''
    
    for name, algorithm in clustering_algorithms:
        print("{:20s}".format(name), end='')
        t1 = time.time()
        cluster_predict = algorithm.fit_predict(dataset)
        t2 = time.time() - t1
        k = len(set(cluster_predict))
        print("| k: {:3.0f}, ".format(k),end='')
        print("{:0.2f} segundos, ".format(t2),end='')
        if (k>1 and (name is not "Ward")):
            metric_CH = metrics.calinski_harabaz_score(dataset, cluster_predict)
            metric_SC = metrics.silhouette_score(dataset, cluster_predict, metric='euclidean', sample_size=floor(0.1*len(dataset)), random_state=123456)
        else:
            metric_CH = 0
            metric_SC = 0
        print("CH Index: {:8.3f}, ".format(metric_CH),end='')
        print("SC: {:.5f}".format(metric_SC))