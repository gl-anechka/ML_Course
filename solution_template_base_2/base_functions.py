from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    rows = X[::4]

    res = []
    for i in rows:
        col = i[120:500:5]
        res.append(col)

    return res
    


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    if not X or not X[0]:
        return -1

    diag = min(len(X), len(X[0]))
    res = []
    
    for i in range(diag):
        if X[i][i] >= 0:
            res.append(X[i][i])

    return sum(res) if res else -1


def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    X_new = deepcopy(X)

    n = len(X)
    m = len(X[0])
    mean = []
    for j in range(m):
        summ = 0
        for i in range(n):
            summ += X[i][j]
        mean.append(summ / n)

    for j in range(m):
        upper = 1.5 * mean[j]
        lower = mean[j] / 4
        for i in range(n):
            if X_new[i][j] > upper or X_new[i][j] < lower:
                X_new[i][j] = -1
    
    return X_new

