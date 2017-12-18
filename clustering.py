#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import time
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans, DBSCAN, Birch, SpectralClustering, AgglomerativeClustering
from sklearn import metrics
from sklearn import preprocessing

import matplotlib.pyplot as plt

from math import floor

# Function that preprocesses the given dataset, and then applies five clustering
# algorithms calculating their execution time, Calinski-Harabaz and Silhouette
# scores and saves their scatter matrix in a 'plots' directory in png format,
# using the given dataset name combined with the algorithm name as file name.
# If the dataset's size is under 100 samples, it also saves a heatmap
# for each algorithm.
def ClusteringAlgorithms(dataset, dataset_name):

    # Normalization of the dataset
    normalized_dataset = preprocessing.normalize(dataset, norm='l2')

    # K-Means
    k_means = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=101010)

    # DBSCAN
    dbscan = DBSCAN(eps=0.1)

    # Birch
    birch = Birch(threshold=0.1, n_clusters=4)

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=4, random_state=101010)

    # Ward
    hierarchical_clusters = 100
    half_dataset_size = floor(dataset.shape[0]/2)
    if hierarchical_clusters > half_dataset_size:
        hierarchical_clusters = half_dataset_size

    ward = AgglomerativeClustering(n_clusters=hierarchical_clusters, linkage='ward')

    clustering_algorithms = [
        ("K-Means", k_means),
        ("DBSCAN", dbscan),
        ("Birch", birch),
        ("Spectral Clustering", spectral),
        ("Ward", ward)
    ]

    print("-----------------------{}-----------------------".format(dataset_name))

    for name, algorithm in clustering_algorithms:
        #print("{:20s}".format(name), end='')
        print("{} & ".format(name), end='')
        t1 = time.time()
        cluster_predict = algorithm.fit_predict(normalized_dataset)
        t2 = time.time() - t1
        k = len(set(cluster_predict))
        print("{} & ".format(k), end='')
        print("{:.3f} & ".format(t2), end='')
        #print("| k: {:3.0f}, ".format(k),end='')
        #print("{:0.3f} seconds, ".format(t2),end='')
        if (k>1) and (name is not "Ward"):
            metric_CH = metrics.calinski_harabaz_score(normalized_dataset, cluster_predict)
            metric_SC = metrics.silhouette_score(normalized_dataset, cluster_predict, metric='euclidean', sample_size=floor(0.1*len(normalized_dataset)), random_state=123456)
        else:
            metric_CH = 0
            metric_SC = 0
        #print("CH Index: {:9.3f}, ".format(metric_CH),end='')
        #print("SC: {:.5f}".format(metric_SC))
        print("{:.3f} & ".format(metric_CH), end='')
        print("{:.5f} & ".format(metric_SC), end='')

        # Assignment gets turned into DataFrame
        column_name = 'cluster'
        clusters = pd.DataFrame(cluster_predict,index=dataset.index,columns=[column_name])

        # Clusters column gets added to dataset
        modified_dataset = pd.concat([dataset, clusters], axis=1)

        # Filter those clusters with outliers samples (clusters that represent less than 3% of the dataset)
        if len(modified_dataset) > 100:
            min_size = floor(modified_dataset.shape[0]*0.01)
        else:
            min_size = 2
        filtered_dataset = modified_dataset[modified_dataset.groupby('cluster').cluster.transform(len) > min_size]
        new_k = len(set(filtered_dataset[column_name]))
        #print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k,new_k,min_size,len(modified_dataset),len(filtered_dataset)))
        print("{} & ".format(new_k), end='')
        print("{} ".format(len(modified_dataset)-len(filtered_dataset)) + r"\\")

        # Define directory path
        script_dir = os.path.dirname(__file__)

        # Now the scatter matrix is generated with the appended dataset
        sns.set()
        variables = list(filtered_dataset)
        variables.remove(column_name)
        sns_plot = sns.pairplot(filtered_dataset, vars=variables, hue=column_name, palette='Paired', plot_kws={"s": 25}, diag_kind="hist")
        sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);

        # Directory is created if does not already exist
        plot_dir = os.path.join(script_dir, 'plots/')
        plot_name = name+"-"+dataset_name+"-ScatterMatrix.png"

        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        # File plot is saved in 'plots' directory
        sns_plot.savefig(plot_dir + plot_name)
        sns_plot.fig.clear()

        # And the heatmap comparing variables values in each cluster
        # Directory is created if does not already exist
        heatmap_dir = os.path.join(script_dir, 'heatmaps/')
        heatmap_name = name+"-"+dataset_name+"-Heatmap.png"

        if not os.path.isdir(heatmap_dir):
            os.makedirs(heatmap_dir)

        # List of clusters that have been defined by the algorithm
        clusters_list = list(set(filtered_dataset[column_name]))

        # Creation of an empty DataFrame that will be filled with a row for each cluster,
        # each column equivalent to the mean value of the variable for the
        # examples in the cluster
        mean_dataframe = pd.DataFrame()

        for cluster in clusters_list:
            cluster_dataframe = filtered_dataset[filtered_dataset[column_name] == cluster]
            mean_array = dict(np.mean(cluster_dataframe[variables],axis=0))
            aux_mean_dataframe = pd.DataFrame(mean_array,index=[str(cluster)])
            mean_dataframe = pd.concat([mean_dataframe,aux_mean_dataframe])

        # Normalization of the DataFrame, for an accurate representation in a heatmap
        mean_dataframe_normalized = preprocessing.normalize(mean_dataframe, norm='l2')
        mean_dataframe_normalized = pd.DataFrame(mean_dataframe_normalized, index=clusters_list, columns=list(mean_dataframe))

        # LaTeX table that represents the mean value for each variable and cluster
        print(mean_dataframe.to_latex(bold_rows=True, column_format=str("*{{{}}}".format(mean_dataframe.shape[1]+1)+r"{c}")))

        # Heatmap is defined and saved in the formerly defined directory
        my_cmap = sns.cubehelix_palette(start=2.5, rot=0.1, light=0.75, as_cmap=True)
        heatmap = sns.heatmap(data=mean_dataframe_normalized, yticklabels=list(set(filtered_dataset[column_name])),
                cmap=my_cmap, annot=True, linewidths=0.5)

        plt.ylabel('Clusters')
        heatmap_fig = heatmap.get_figure()
        heatmap_fig.savefig(heatmap_dir + heatmap_name)
        heatmap.clear()
        heatmap_fig.clear()

        # If the algorithm is agglomerative (as Ward is in this case study) an additional
        # dendrogram is generated in 'dendrograms' folder
        if name == "Ward":
            dendrogram_dir = os.path.join(script_dir, 'dendrograms/')
            dendrogram_name = name+"-"+dataset_name+"-Dendrogram.png"

            if not os.path.isdir(dendrogram_dir):
                os.makedirs(dendrogram_dir)

            dendrogram = sns.clustermap(mean_dataframe_normalized, method='ward', col_cluster=False, annot=True, linewidths=0.5, figsize=(20,10), cmap=my_cmap)
            dendrogram.savefig(dendrogram_dir + dendrogram_name)
