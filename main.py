from prfs.feature_selection import PageRankFeatureSelector
import pandas as pd

dataset = pd.read_csv('datasets/correlation_distance_dataset.csv')

features = dataset.loc[:, 'F1':'F5']
labels = dataset.loc[:, 'L1':'L3']

fs = PageRankFeatureSelector()
fs.run(features, labels)
print(fs.get_feature_scores())
print(fs.select(4))
