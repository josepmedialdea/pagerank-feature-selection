import numpy as np
from scipy.stats import spearmanr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif


def correlation(features, labels, i, j):
    _, n_features = features.shape

    if j == n_features:
        x1 = features[:, i]
        x2 = labels
    else:
        x1 = features[:, i]
        x2 = features[:, j]

    covariance = np.cov(x1, x2)[0, 1]

    x1_variance = np.var(x1, ddof=1)
    x2_variance = np.var(x2, ddof=1)

    return np.abs(covariance/(np.sqrt(x1_variance*x2_variance)))


def uncorrelation(features, labels, i, j):
    return 1 - correlation(features, labels, i, j)


def spearman_correlation(features, labels, i, j):
    _, n_features = features.shape

    if j == n_features:
        x1 = features[:, i]
        x2 = labels
    else:
        x1 = features[:, i]
        x2 = features[:, j]

    scorr, p_value = spearmanr(x1, x2)
    return np.abs(scorr)


def spearman_uncorrelation(features, labels, i, j):
    return 1 - spearman_correlation(features, labels, i, j)


def accuracy(features, labels, i, j):
    _, n_features = features.shape

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.33)

    clf = GaussianNB()

    clf.fit(X_train[:, i].reshape(-1, 1), y_train)

    accuracy_only_fi = clf.score(X_train[:, i].reshape(-1, 1), y_train)

    if j == n_features:
        return accuracy_only_fi

    clf.fit(X_train[:, [i, j]], y_train)

    accuracy_fi_fj = clf.score(X_train[:, [i, j]], y_train)

    return np.maximum(accuracy_fi_fj - accuracy_only_fi, 0)


def mutual_information(features, labels, i, j):
    _, n_features = features.shape

    if not (j == n_features):
        raise Exception(
            'mutual_information must be used as the alpha function (i.e. only between feature-target relations)')

    mi = mutual_info_classif(features[:, i].reshape(-1, 1), labels)

    return mi
