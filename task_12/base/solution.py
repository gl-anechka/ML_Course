import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    x = np.asarray(x)
    labels = np.asarray(labels)

    unique_labels, inverse_labels, cluster_sizes = np.unique(
        labels,
        return_inverse=True,
        return_counts=True
    )

    n_objects = x.shape[0]
    n_clusters = unique_labels.size

    # если только один кластер, силуэт равен 0
    if n_clusters == 1:
        return 0.0

    distances = sklearn.metrics.pairwise_distances(x)

    # s_values[i] — среднее расстояние от объекта i до своего кластера
    s_values = np.zeros(n_objects, dtype=np.float64)

    # d_values[i] — минимальное среднее расстояние от объекта i до чужого кластера
    big_value = distances.max() + 1.0
    d_values = np.full(n_objects, big_value, dtype=np.float64)

    own_cluster_sizes = cluster_sizes[inverse_labels]

    for cluster_index in range(n_clusters):
        cluster_mask = inverse_labels == cluster_index
        cluster_size = cluster_sizes[cluster_index]

        # сумма расстояний от каждого объекта до объектов текущего кластера
        distance_sum_to_cluster = distances[:, cluster_mask].sum(axis=1)

        # среднее расстояние от каждого объекта до текущего кластера
        mean_distance_to_cluster = distance_sum_to_cluster / cluster_size

        # для объектов текущего кластера считаем s_i
        if cluster_size > 1:
            s_values[cluster_mask] = (
                distance_sum_to_cluster[cluster_mask] / (cluster_size - 1)
            )

        # для объектов не из текущего кластера обновляем d_i
        not_cluster_mask = ~cluster_mask
        d_values[not_cluster_mask] = np.minimum(
            d_values[not_cluster_mask],
            mean_distance_to_cluster[not_cluster_mask]
        )

    denominator = np.maximum(s_values, d_values)

    silhouette_values = np.divide(
        d_values - s_values,
        denominator,
        out=np.zeros(n_objects, dtype=np.float64),
        where=denominator > 0
    )

    # для одиночных объектов, считаем силуэт 0
    silhouette_values[own_cluster_sizes == 1] = 0.0

    return silhouette_values.mean()


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    # same_true_class[i, j] = True, если i и j в одном исходном классе
    same_true_class = true_labels[:, None] == true_labels[None, :]

    # same_predicted_cluster[i, j] = True, если i и j в одном кластере
    same_predicted_cluster = predicted_labels[:, None] == predicted_labels[None, :]

    # same_class_and_cluster[i, j] = True, если и то и другое
    same_class_and_cluster = same_true_class & same_predicted_cluster

    # intersection_sizes[i] = |L(i) ∩ C(i)|
    intersection_sizes = same_class_and_cluster.sum(axis=1).astype(np.float64)

    # predicted_cluster_sizes[i] = |C(i)|
    predicted_cluster_sizes = same_predicted_cluster.sum(axis=1).astype(np.float64)

    # true_class_sizes[i] = |L(i)|
    true_class_sizes = same_true_class.sum(axis=1).astype(np.float64)

    # precision_i = |L(i) ∩ C(i)| / |C(i)|
    precision = np.divide(
        intersection_sizes,
        predicted_cluster_sizes,
        out=np.zeros_like(intersection_sizes, dtype=np.float64),
        where=predicted_cluster_sizes > 0
    )

    # recall_i = |L(i) ∩ C(i)| / |L(i)|
    recall = np.divide(
        intersection_sizes,
        true_class_sizes,
        out=np.zeros_like(intersection_sizes, dtype=np.float64),
        where=true_class_sizes > 0
    )

    # усредняем precision и recall
    mean_precision = precision.mean()
    mean_recall = recall.mean()

    # F = 2 * precision * recall / (precision + recall)
    score = np.divide(
        2 * mean_precision * mean_recall,
        mean_precision + mean_recall,
        out=np.array(0.0),
        where=(mean_precision + mean_recall) > 0
    ).item()

    return score
