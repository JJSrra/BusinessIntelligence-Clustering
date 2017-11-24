#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import pandas as pd

from sklearn import preprocessing

import clustering

accidents = pd.read_csv('accidentes_2013.csv')

# Select "crash" type accidents
accidents_sample = accidents[accidents['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]

# Select accidents between 6AM and 12PM
#accidents_sample = accidents.loc[(accidents['HORA']>=6) & (accidents['HORA']<=12)]

# Select non-deadly accidents
#accidents_sample = accidents.loc[accidents['TOT_MUERTOS']==0]

# Select interesting variables for clustering
#selected = ['HORA', 'DIASEMANA', 'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
selected = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

# Subset with 'selected' variables
accidents_subset = accidents_sample[selected]

# Normalizing the subset
subset_normal = preprocessing.normalize(accidents_subset, norm='l2')

clustering.ClusteringAlgorithms(subset_normal)

'''
# Call of KMeans function in clustering.py file
kmeans_predict = clustering.ApplyKMeans(subset_normal, 4, 5)

# assignment gets turned into DataFrame
kmeans_clusters = pd.DataFrame(kmeans_predict,index=accidents_subset.index,columns=['kmeans_clusters'])
'''

'''
#y se añade como columna a X
X_kmeans = pd.concat([accidents_subset, clusters], axis=1)

print("---------- Preparando el scatter matrix...")
import seaborn as sns
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
sns_plot.savefig("kmeans.png")
print("")
'''    