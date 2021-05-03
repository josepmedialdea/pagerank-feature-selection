from prfs.feature_selection import PageRankFeatureSelector
import pandas as pd
import numpy as np

dataset = pd.read_csv('datasets/dice1.csv')
print(dataset.describe())

features = dataset.loc[:, 'd1':'d_sum']
labels = dataset.loc[:, 'Target']

fs = PageRankFeatureSelector(graph='feature_label', weight=0)
fs.fit(features, labels)
print(fs.get_feature_scores())
print(fs.select(4))
fs.show_graph()
