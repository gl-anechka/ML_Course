from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return Counter(x) == Counter(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if (len(x) < 2): return -1
    res = []
    for i in range(len(x) - 1):
        if x[i] % 3 == 0 or x[i+1] % 3 == 0:
            res.append(x[i] * x[i+1])
    return max(res) if res else -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = []
    for i in range(len(image)):
        row_res = []
        for j in range(len(image[0])):
            summ = 0.0
            for k in range(len(image[0][0])):
                summ += image[i][j][k] * weights[k]
            row_res.append(summ)
        res.append(row_res)
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    len_x = sum(count for _, count in x)
    len_y = sum(count for _, count in y)
    if len_x != len_y: return -1
    
    res = 0
    i = j = 0
    x_count = x[i][1] if x else 0
    y_count = y[j][1] if y else 0
    x_val = x[i][0] if x else 0
    y_val = y[j][0] if y else 0
    
    for _ in range(len_x):
        res += x_val * y_val
        x_count -= 1
        y_count -= 1

        if x_count == 0:
            i += 1
            if i < len(x):
                x_val, x_count = x[i]
        if y_count == 0:
            j += 1
            if j < len(y):
                y_val, y_count = y[j]
    return res


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res = []
    for i in range(len(X)):
        res.append([])
        x_normalized=sum(tmp**2 for tmp in X[i])**0.5
        for j in range(len(Y)):
            y_normalized = sum(tmp**2 for tmp in Y[j])**0.5
            if x_normalized==0 or y_normalized==0:
                res[i].append(float(1))
            else:
                res[i].append(sum(X[i][k] * Y[j][k] for k in range(len(X[0])))/(x_normalized*y_normalized))
    return res