#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import pandas as pd

import clustering

accidents = pd.read_csv('accidentes_2013.csv')

# Select accidents in highways and freeways
highway_accidents_sample = accidents[accidents['TIPO_VIA'].str.contains("AUTO")]

# Select accidents between 6AM and 12PM
morning_accidents_sample = accidents.loc[(accidents['HORA']>=6) & (accidents['HORA']<=12)]

# Select accidents in summertime (June, July and August)
summer_accidents_sample = accidents.loc[(accidents['MES']>=6) & (accidents['MES']<=8)]

# Select interesting variables for clustering
selected = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

# Subset with 'selected' variables
highway_accidents_subset = highway_accidents_sample[selected]
morning_accidents_subset = morning_accidents_sample[selected]
summer_accidents_subset = summer_accidents_sample[selected]

# Clustering Algorithms for each of the selected subsets
clustering.ClusteringAlgorithms(highway_accidents_subset, 5000, "Highway accidents")
#clustering.ClusteringAlgorithms(morning_accidents_subset, 5000, "Morning accidents")
#clustering.ClusteringAlgorithms(summer_accidents_subset, 5000, "Summer accidents")
