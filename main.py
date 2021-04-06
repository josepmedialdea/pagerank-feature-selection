from prfs.feature_selection import PageRankFeatureSelector
import pandas as pd

dataset = pd.read_csv('datasets/heart.csv')

features = dataset.loc[:, 'age':'thal']
labels = dataset.loc[:, 'target']

fs = PageRankFeatureSelector()
fs.run(features, labels)

print(fs.select(5))
fs.show_feature_graph()
