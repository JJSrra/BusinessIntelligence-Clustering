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
#accidents_sample = accidents.loc[(accidents['HORA']>=6) & (accidents['HORA']<=12)]

# Select non-deadly accidents
#accidents_sample = accidents.loc[accidents['TOT_MUERTOS']==0]

# Select interesting variables for clustering
#selected = ['HORA', 'DIASEMANA', 'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
crash_selected = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

# Subset with 'selected' variables
crash_accidents_subset = crash_accidents_sample[selected]

clustering.ClusteringAlgorithms(crash_accidents_subset, 5000, "Crash accidents")
