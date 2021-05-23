from prfs.feature_selection import PageRankFeatureSelector
import pandas as pd
import numpy as np

dataset = pd.read_csv('datasets/price-prediction.csv')
print(dataset.describe())

features = dataset.loc[:, 'bedrooms':'sqft_lot']
labels = dataset.loc[:, 'price']

fs = PageRankFeatureSelector(alpha='sparse_correlation', beta='sparse_uncorrelation')
fs.fit(features, labels)
print(fs.get_feature_scores())
print(fs.select(4))
fs.show_graph()
