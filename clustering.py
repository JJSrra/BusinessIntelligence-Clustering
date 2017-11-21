#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import time

from sklearn.cluster import KMeans
from sklearn import metrics

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
