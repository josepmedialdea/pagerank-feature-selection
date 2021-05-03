from prfs.graphs import FeatureGraph, FeatureLabelGraph
from prfs.functions import correlation, uncorrelation

class PageRankFeatureSelector():

    def __init__(self, graph='feature', alpha='correlation', beta='uncorrelation', weight=1):
        self.graph_type = graph
        
        if alpha == 'correlation':
            self.alpha = correlation
        elif alpha == 'uncorrelation':
            self.alpha = uncorrelation
        
        if beta == 'correlation':
            self.beta = correlation
        elif beta == 'uncorrelation':
            self.beta = uncorrelation

        self.weight = weight

        self.fit_called = False

    def fit(self, features, labels):
        if self.graph_type == 'feature':
            self.graph = FeatureGraph(features, labels, self.alpha, self.beta, self.weight)
        elif self.graph_type == 'feature_label':
            self.graph = FeatureLabelGraph(features, labels, self.alpha, self.beta, self.weight)

        self.fit_called = True

    def get_feature_scores(self):
        if not self.fit_called:
            raise Exception(
                'fit(features, labels) needs to be called before calling get_feature_scores()')

        return self.graph.get_feature_scores()

    def select(self, n):
        if not self.fit_called:
            raise Exception(
                'fit(features, labels) needs to be called before calling select(n)')

        return self.graph.select(n)

    def show_graph(self):
        if not self.fit_called:
            raise Exception(
                'fit(features, labels) needs to be called before calling show_graph()')

        self.graph.show()
