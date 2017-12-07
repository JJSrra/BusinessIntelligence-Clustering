#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanjo Sierra
"""

import pandas as pd

import clustering

accidents = pd.read_csv('accidentes_2013.csv')

# Select "crash" type accidents
crash_accidents_sample = accidents[accidents['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]

# Select accidents between 6AM and 12PM
morning_accidents_sample = accidents.loc[(accidents['HORA']>=6) & (accidents['HORA']<=12)]

# Select non-deadly accidents
#accidents_sample = accidents.loc[accidents['TOT_MUERTOS']==0]

# Select interesting variables for clustering
selected = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

# Subset with 'selected' variables
crash_accidents_subset = crash_accidents_sample[selected]
morning_accidents_subset = morning_accidents_sample[selected]

# Clustering Algorithms for each of the selected subsets
#clustering.ClusteringAlgorithms(crash_accidents_subset, 5000, "Crash accidents")
clustering.ClusteringAlgorithms(morning_accidents_subset, 5000, "Morning accidents")
