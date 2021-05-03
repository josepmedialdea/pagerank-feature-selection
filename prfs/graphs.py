from abc import ABC, abstractmethod
import numpy as np
from .pagerank import calculate_pagerank_scores


class FeatureSelectionGraph(ABC):

    def get_feature_scores(self):
        feature_score_dict = {}

        for i, feature_name in enumerate(self.feature_names):
            feature_score_dict[feature_name] = self.pagerank_scores[i]

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
    
    @abstractmethod
    def show(self):
        pass


class FeatureGraph(FeatureSelectionGraph):

    def __init__(self, features, labels, alpha, beta, weight=1):
        feature_names = []
        for feature_name in features.columns:
            feature_names.append(feature_name)
        self.feature_names = feature_names
        self.label_name = labels.name

        features_np = features.to_numpy()
        labels_np = labels.to_numpy()

        _, n_features = features_np.shape
        
        self.adjacency_matrix = np.zeros(
            n_features*n_features).reshape(n_features, n_features)

        for feature_i in range(n_features):
            for feature_j in range(n_features):
                if feature_i != feature_j:
                    feature_i_values = features_np[:, feature_i]
                    feature_j_values = features_np[:, feature_j]

                    self.adjacency_matrix[feature_i, feature_j] = alpha(
                        feature_j_values, labels_np) + weight*beta(feature_i_values, feature_j_values)

        self.pagerank_scores = calculate_pagerank_scores(self.adjacency_matrix)

    def show(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.from_numpy_array(
            self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.circular_layout(G)

        labels = {}
        for i, label in enumerate(self.feature_names):
            labels[i] = label + f'\n{self.pagerank_scores[i]:.4f}'

        nx.draw_networkx_nodes(G, pos, node_color='r',
                               node_size=self.pagerank_scores*1000+1000)
        nx.draw_networkx_labels(G, pos, labels, alpha=0.8)
        nx.draw_networkx_edges(
            G, pos, node_size=self.pagerank_scores*1000+1000)

        plt.title('Feature graph with PageRank scores')
        plt.axis('off')
        plt.show()


class FeatureLabelGraph(FeatureSelectionGraph):

    def __init__(self, features, labels, alpha, beta, weight=1):
        feature_names = []
        for feature_name in features.columns:
            feature_names.append(feature_name)
        self.feature_names = feature_names
        self.label_name = labels.name

        features_np = features.to_numpy()
        labels_np = labels.to_numpy()

        _, n_features = features_np.shape
        
        self.adjacency_matrix = np.zeros(
            (n_features + 1)*(n_features + 1)).reshape(n_features + 1, n_features + 1)

        for feature_i in range(n_features):
            for feature_j in range(n_features):
                if feature_i != feature_j:
                    feature_i_values = features_np[:, feature_i]
                    feature_j_values = features_np[:, feature_j]

                    self.adjacency_matrix[feature_i, feature_j] = weight * \
                        beta(feature_i_values, feature_j_values)

        for feature in range(n_features):
            feature_values = features_np[:, feature]

            self.adjacency_matrix[feature, n_features] = 1
            self.adjacency_matrix[n_features, feature] = alpha(
                feature_values, labels_np)

        self.pagerank_scores = calculate_pagerank_scores(self.adjacency_matrix)

    def show(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.from_numpy_array(
            self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.circular_layout(G)

        labels = {}
        for i, label in enumerate(self.feature_names):
            labels[i] = label + f'\n{self.pagerank_scores[i]:.4f}'
        labels[len(labels)] = f'Target \n{self.pagerank_scores[len(labels)]:.4f}'

        nx.draw_networkx_nodes(G, pos, node_color='r',
                               node_size=self.pagerank_scores*1000+1000)
        nx.draw_networkx_labels(G, pos, labels, alpha=0.8)
        nx.draw_networkx_edges(
            G, pos, node_size=self.pagerank_scores*1000+1000)

        plt.title('Feature graph with PageRank scores')
        plt.axis('off')
        plt.show()
