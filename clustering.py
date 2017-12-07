#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import time
import seaborn as sns
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, DBSCAN, Birch, SpectralClustering, AgglomerativeClustering, estimate_bandwidth
from sklearn import metrics
from sklearn import preprocessing

from math import floor

def ClusteringAlgorithms(dataset, samples, dataset_name):
    # Selection of number of samples given and normalization of the dataset
    samples_dataset = dataset.sample(samples)
    normalized_dataset = preprocessing.normalize(samples_dataset, norm='l2')

    # K-Means
    k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)

    # MiniBatch K-Means
    mbkm = MiniBatchKMeans(init='k-means++', n_clusters=4, n_init=5)

    # Estimate bandwidth for Mean Shift Algorithm
    bandwidth = estimate_bandwidth(normalized_dataset, quantile=0.2, n_samples=500)
    # Mean Shift
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # DBSCAN
    dbscan = DBSCAN(eps=0.1)

    # Birch
    birch = Birch(threshold=0.1, n_clusters=4)

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=4)

    # Ward
    ward = AgglomerativeClustering(n_clusters=100, linkage='ward')

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
        cluster_predict = algorithm.fit_predict(normalized_dataset)
        t2 = time.time() - t1
        k = len(set(cluster_predict))
        print("| k: {:3.0f}, ".format(k),end='')
        print("{:0.2f} segundos, ".format(t2),end='')
        if (k>1) and (name is not "Ward"):
            metric_CH = metrics.calinski_harabaz_score(normalized_dataset, cluster_predict)
            metric_SC = metrics.silhouette_score(normalized_dataset, cluster_predict, metric='euclidean', sample_size=floor(0.1*len(normalized_dataset)), random_state=123456)
        else:
            metric_CH = 0
            metric_SC = 0
        print("CH Index: {:9.3f}, ".format(metric_CH),end='')
        print("SC: {:.5f}".format(metric_SC))

        # Assignment gets turned into DataFrame
        column_name = name + " clusters"
        clusters = pd.DataFrame(cluster_predict,index=samples_dataset.index,columns=[column_name])

        # Clusters column gets added to dataset
        modified_dataset = pd.concat([dataset, clusters], axis=1)

        # And now scatter matrix is saved in workspace directory
        sns.set()
        variables = list(modified_dataset)
        variables.remove(column_name)
        sns_plot = sns.pairplot(modified_dataset, vars=variables, hue=column_name, palette='Paired', plot_kws={"s": 25}, diag_kind="hist")
        sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
        sns_plot.savefig(name+"-"+dataset_name+".png")
