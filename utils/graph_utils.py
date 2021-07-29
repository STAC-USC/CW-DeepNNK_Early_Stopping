__author__ = "shekkizh, davidbonet"

import numpy as np
from utils.non_neg_qpsolver import non_negative_qpsolver
from utils.ann_utils import FaissNeighborSearch as ANN


def majority_vote_classifier(D, y_neighbor, y_node):
    W = D / np.sum(D, axis=1, keepdims=True)
    predicted_label = np.sum(np.expand_dims(W, axis=2) * y_neighbor, axis=1)
    error = 1 - np.equal(
        np.argmax(predicted_label, axis=1), np.argmax(y_node, axis=1)
    ).astype(float)
    return error


def weighted_classifier(D, y_neighbor, y_node):
    shape = D.shape
    W = D / np.sum(D, axis=1, keepdims=True)
    predicted_label = np.sum(np.expand_dims(W, axis=2) * y_neighbor, axis=1)
    error = 1 - predicted_label[range(shape[0]), np.argmax(y_node, axis=1)]
    return error


def lp_distance(pointA, pointB, p):
    """
    Function to calculate the lp distance between two points
    :param p: the norm type to  calculate
    :return: distance
    """
    dist = (np.sum(np.abs(pointA - pointB) ** p)) ** (1.0 / p)

    return dist


def create_distance_matrix(X, p=2):
    """
    Create distance matrix
    :param X:
    :param metric:
    :return:
    """
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            W[i, j] = lp_distance(X[i, :], X[j, :], p)
    W = W + W.T
    return W


def nnk_loo(
    activations,
    labels,
    interpol_queries=1.0,
    knn_param=25,
    kernel="gaussian",
    ignore_identical=True,
    edge_threshold=1e-10,
):
    """
    Function to compute NNK LOO label interpolation error
    :param activations: Layer/Channel activations of the dataset
    :param labels: Labels of the dataset
    :param interpol_queries: Fraction of training set samples to use as queries in the LOO procedure
    :param knn_param: Max number of neighbors to use
    :param kernel: Kernel to compute similarity matrix
    :param ignore_identical: Ignore data points at distance 0 to query for graph construction (identical, no new information)
    :param edge_threshold: Threshold value for edge weights
    :return: Average classification error
    """
    assert len(activations) == len(labels)
    queries = activations[:interpol_queries]
    query_labels = labels[:interpol_queries]
    num_classes = labels.shape[1]
    y_train = np.zeros((len(queries), knn_param, num_classes), dtype=float)
    W = np.zeros((len(queries), knn_param), dtype=float)

    # Initialize ANN
    d = activations.shape[1]
    neighbor_search = ANN(d, knn_param + 1, use_gpu=False)
    neighbor_search.add_to_database(x=activations)
    D, I = neighbor_search.search_neighbors(q=queries)
    I = I[:, 1:]  # Remove self
    D = D[:, 1:]  # Remove self

    for ii in range(len(queries)):
        if ii % 1000 == 0:
            print(f"\tQuery {ii}/{len(queries)}...", flush=True)
        not_identical = np.nonzero(D[ii])[0]
        if ignore_identical and len(not_identical) > 0:
            I_dif = I[ii, not_identical]
        else:
            I_dif = I[ii, :]
        X_knn = activations[I_dif]
        y_knn = labels[I_dif]
        query_and_knn = np.concatenate((queries[ii : ii + 1, :], X_knn), axis=0)

        if kernel == "cosine":
            query_and_knn_normalized = (
                query_and_knn / np.linalg.norm(query_and_knn, axis=1)[:, None]
            )
            G = 0.5 + np.dot(query_and_knn_normalized, query_and_knn_normalized.T) / 2.0
        elif kernel == "gaussian":
            D_m = create_distance_matrix(X=query_and_knn, p=2)
            sigma = D_m[0, -1] / 3
            G = np.exp(-(D_m ** 2) / (2 * sigma ** 2))
        else:
            raise Exception("Unknown kernel: " + kernel)

        G_i = G[1 : len(query_and_knn), 1 : len(query_and_knn)]
        g_i = G[1 : len(query_and_knn), 0]

        x_opt, check = non_negative_qpsolver(G_i, g_i, g_i, edge_threshold)
        if ignore_identical and len(not_identical) > 0:
            W[ii, not_identical] = x_opt
            y_train[ii, not_identical, :] = y_knn
        else:
            W[ii, :] = x_opt
            y_train[ii, :, :] = y_knn
    return np.mean(majority_vote_classifier(W, y_train, query_labels))
