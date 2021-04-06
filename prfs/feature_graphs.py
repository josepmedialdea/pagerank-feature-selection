import numpy as np
from sklearn.feature_selection import mutual_info_classif


class CorrelationDistanceGraph():

    def __init__(self, X, y):
        _, n_features = X.shape
        if y.ndim == 1:
            n_labels = 1
            y = y.reshape(-1, 1)
        else:
            _, n_labels = y.shape

        self.adjacency_matrix = np.zeros(
            n_features*n_features).reshape(n_features, n_features)

        correlation_distance_matrix = np.zeros(
            n_features*n_labels).reshape(n_features, n_labels)

        for feature in range(n_features):
            for label in range(n_labels):
                feature_values = X[:, feature]
                label_values = y[:, label]

                covariance = np.cov(feature_values, label_values)[0, 1]

                feature_variance = np.var(feature_values, ddof=1)
                label_variance = np.var(label_values, ddof=1)

                correlation_distance_matrix[feature, label] = 1 - \
                    covariance/(np.sqrt(feature_variance*label_variance))

        for feature_i in range(n_features):
            for feature_j in range(n_features):
                euclidean_distance = 0

                for label in range(n_labels):
                    euclidean_distance += np.square(
                        correlation_distance_matrix[feature_i, label] - correlation_distance_matrix[feature_j, label])

                euclidean_distance = np.sqrt(euclidean_distance)

                self.adjacency_matrix[feature_i,
                                      feature_j] = euclidean_distance
                self.adjacency_matrix[feature_j,
                                      feature_i] = euclidean_distance


class MutualInformationGraph():

    def __init__(self, X, y):
        _, n_features = X.shape
        if y.ndim == 1:
            n_labels = 1
            y = y.reshape(-1, 1)
        else:
            _, n_labels = y.shape

        self.adjacency_matrix = np.zeros(
            n_features*n_features).reshape(n_features, n_features)

        mutual_information_matrix = np.zeros(
            n_features*n_labels).reshape(n_features, n_labels)

        for feature in range(n_features):
            for label in range(n_labels):
                feature_values = X[:, feature].reshape(-1, 1)
                label_values = y[:, label]

                mutual_information_matrix[feature, label] = mutual_info_classif(
                    feature_values, label_values)[0]

        for feature_i in range(n_features):
            for feature_j in range(n_features):
                euclidean_distance = 0

                for label in range(n_labels):
                    euclidean_distance += np.square(
                        mutual_information_matrix[feature_i, label] - mutual_information_matrix[feature_j, label])

                euclidean_distance = np.sqrt(euclidean_distance)

                self.adjacency_matrix[feature_i,
                                      feature_j] = euclidean_distance
                self.adjacency_matrix[feature_j,
                                      feature_i] = euclidean_distance
