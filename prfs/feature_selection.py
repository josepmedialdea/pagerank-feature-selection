from prfs.graphs import FeatureGraph, FeatureLabelGraph
from prfs.functions import correlation, uncorrelation, sparse_correlation, sparse_uncorrelation, accuracy

class PageRankFeatureSelector():

    def __init__(self, graph='feature', alpha='correlation', beta='uncorrelation', weight=1):
        self.graph_type = graph
        
        if alpha == 'correlation':
            self.alpha = correlation
        elif alpha == 'uncorrelation':
            self.alpha = uncorrelation
        elif alpha == 'sparse_correlation':
            self.alpha = sparse_correlation
        elif alpha == 'sparse_uncorrelation':
            self.alpha = sparse_uncorrelation
        elif alpha == 'spearman_correlation':
            self.alpha = sparse_correlation
        elif alpha == 'spearman_uncorrelation':
            self.alpha = sparse_uncorrelation
        elif alpha == 'accuracy':
            self.alpha = accuracy
        else:
            raise Exception(f'No alpha function named {alpha}')
        
        if beta == 'correlation':
            self.beta = correlation
        elif beta == 'uncorrelation':
            self.beta = uncorrelation
        elif beta == 'sparse_correlation':
            self.beta = sparse_correlation
        elif beta == 'sparse_uncorrelation':
            self.beta = sparse_uncorrelation
        elif beta == 'spearman_correlation':
            self.beta = sparse_correlation
        elif beta == 'spearman_uncorrelation':
            self.beta = sparse_uncorrelation
        elif beta == 'accuracy':
            self.beta = accuracy
        else:
            raise Exception(f'No beta function named {beta}')

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

    def ranking(self):
        if not self.fit_called:
            raise Exception(
                'fit(features, labels) needs to be called before calling get_feature_scores()')

        ordered_features = self.select(len(self.graph.feature_names))
        rank = 1
        ranking = 'RANKING\n------\n'
        for feature in ordered_features:
            ranking += f'{rank}: {feature}\n'
            rank += 1
        return ranking

    def show_graph(self):
        if not self.fit_called:
            raise Exception(
                'fit(features, labels) needs to be called before calling show_graph()')

        self.graph.show()
