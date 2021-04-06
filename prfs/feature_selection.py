from prfs.feature_graphs import CorrelationDistanceGraph, MutualInformationGraph
from prfs.pagerank import get_pagerank_scores


class PageRankFeatureSelector():

    def __init__(self, method='correlation_distance'):
        self.method = method
        self.run_called = False

    def run(self, features, labels):
        if self.run_called:
            raise Exception(
                'run() needs to be called before calling get_feature_scores()')

        feature_names = []
        for feature_name in features.columns:
            feature_names.append(feature_name)
        self.feature_names = feature_names

        X = features.to_numpy()
        y = labels.to_numpy()

        if self.method == 'correlation_distance':
            self.feature_graph_adjacency_matrix = CorrelationDistanceGraph(
                X, y).adjacency_matrix
            self.feature_pagerank_scores = get_pagerank_scores(
                self.feature_graph_adjacency_matrix)
        elif self.method == 'mutual_information':
            self.feature_graph_adjacency_matrix = MutualInformationGraph(
                X, y).adjacency_matrix
            self.feature_pagerank_scores = get_pagerank_scores(
                self.feature_graph_adjacency_matrix)

        self.run_called = True

    def get_feature_scores(self):
        if not self.run_called:
            raise Exception(
                'run() needs to be called before calling get_feature_scores()')

        feature_score_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_score_dict[feature_name] = self.feature_pagerank_scores[i]

        return feature_score_dict

    def select(self, n):
        if n > len(self.feature_names):
            raise Exception('n > number of features')

        feature_score_dict = self.get_feature_scores()
        feature_names_copy = self.feature_names.copy()
        sorted_feature_names = sorted(
            feature_names_copy, key=lambda f: feature_score_dict[f], reverse=True)
        n_best_features = sorted_feature_names[:n]
        return n_best_features

    def show_feature_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.from_numpy_array(
            self.feature_graph_adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.circular_layout(G)

        labels = {}
        for i, label in enumerate(self.feature_names):
            labels[i] = label + f'\n{self.feature_pagerank_scores[i]:.2f}'

        nx.draw_networkx_nodes(G, pos, node_color='r',
                               node_size=self.feature_pagerank_scores*1000+1000)
        nx.draw_networkx_labels(G, pos, labels, alpha=0.8)
        nx.draw_networkx_edges(
            G, pos, node_size=self.feature_pagerank_scores*1000+1000)

        plt.title('Feature graph with PageRank scores')
        plt.axis('off')
        plt.show()
