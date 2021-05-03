import numpy as np


def calculate_pagerank_scores(graph, d=0.85, epsilon=0.000001):
    n, _ = graph.shape
    row_normalized_matrix = np.copy(graph)
    sink_nodes = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(row_normalized_matrix):
        row_sum = np.sum(row)
        if row_sum == 0:
            sink_nodes[i] = 1.0
        else:
            row /= row_sum
    page_rank = np.ones(n)/n
    while True:
        previous_page_rank = page_rank
        page_rank = (np.dot(row_normalized_matrix.T, previous_page_rank) + (np.ones(n)/n)
                     * np.dot(sink_nodes, previous_page_rank))*d + np.ones(n)*(1 - d)/n
        page_rank /= np.sum(page_rank)
        difference = np.sum(np.abs(page_rank - previous_page_rank))
        if difference < epsilon:
            return page_rank
