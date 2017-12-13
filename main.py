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

# Select accidents between between 0:00 AM and 6:00 AM
early_morning_accidents_sample = accidents.loc[(accidents['HORA']>=0) & (accidents['HORA']<=6)]

# Select accidents that happened in Andalucía, when the road was wet and the vehicle overturned
wet_overturned_accidents_sample = accidents.loc[(accidents.COMUNIDAD_AUTONOMA.str.contains("Andalucía")) &
                       (accidents.SUPERFICIE_CALZADA.str.contains("MOJADA")) &
                       (accidents.TIPO_ACCIDENTE.str.contains("Vuelco"))]

# Select interesting variables for clustering
selected = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

# Subset with 'selected' variables
highway_accidents_subset = highway_accidents_sample[selected]
early_morning_accidents_subset = early_morning_accidents_sample[selected]
wet_overturned_accidents_subset = wet_overturned_accidents_sample[selected]

# Clustering Algorithms for each of the selected subsets
#clustering.ClusteringAlgorithms(highway_accidents_subset, 5000, "Highway accidents")
clustering.ClusteringAlgorithms(early_morning_accidents_subset, 5000, "Morning accidents")
#clustering.ClusteringAlgorithms(wet_overturned_accidents_subset, 5000, "Summer accidents")
