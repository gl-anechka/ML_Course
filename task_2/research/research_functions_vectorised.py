import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if x.size < 2: return -1
    first = x[:-1]
    second = x[1:]

    res = first * second
    valid_res = res[(first % 3 == 0) | (second % 3 == 0)]

    return np.max(valid_res) if valid_res.size > 1 else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.sum(image * weights, axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_new = np.repeat(x[:, 0], x[:, 1])
    y_new = np.repeat(y[:, 0], y[:, 1])
    if x_new.shape != y_new.shape:
        return -1
    else:
        return np.dot(x_new, y_new)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ans = np.dot(X, Y.T) / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    return np.nan_to_num(ans, nan=1)