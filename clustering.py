#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import time

from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, DBSCAN, Birch, SpectralClustering, AgglomerativeClustering, estimate_bandwidth
from sklearn import metrics

from math import floor

def ClusteringAlgorithms(dataset):
    # K-Means
    k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)

    # MiniBatch K-Means
    mbkm = MiniBatchKMeans(init='k-means++', n_clusters=4, n_init=5)

    # Estimate bandwidth for Mean Shift Algorithm
    bandwidth = estimate_bandwidth(dataset, quantile=0.2, n_samples=500)
    # Mean Shift
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=10)

    # Birch
    birch = Birch(threshold=0.1, n_clusters=4)

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=4)

    # Ward
    ward = AgglomerativeClustering(n_clusters=4, linkage='ward')

    clustering_algorithms = [
        ("K-Means", k_means),
        ("MiniBatch K-Means", mbkm),
        ("Mean Shift", mean_shift),
        ("DBSCAN", dbscan),
        ("Birch", birch),
        ("Spectral Clustering", spectral),
        ("Ward", ward)
    ]

    for name, algorithm in clustering_algorithms:
        print("{:20s}".format(name), end='')
        t1 = time.time()
        cluster_predict = algorithm.fit_predict(dataset)
        t2 = time.time() - t1
        k = len(set(cluster_predict))
        print("| k: {:3.0f}, ".format(k),end='')
        print("{:0.2f} segundos, ".format(t2),end='')
        if (k>1) and (name is not "Ward"):
            metric_CH = metrics.calinski_harabaz_score(dataset, cluster_predict)
            metric_SC = metrics.silhouette_score(dataset, cluster_predict, metric='euclidean', sample_size=floor(0.1*len(dataset)), random_state=123456)
        else:
            metric_CH = 0
            metric_SC = 0
        print("CH Index: {:8.3f}, ".format(metric_CH),end='')
        print("SC: {:.5f}".format(metric_SC))
