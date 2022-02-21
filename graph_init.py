import numpy as np


def init():
    map_dict = {  # 仿真图上的坐标
        0: [65, 863], 1: [124, 769], 2: [130, 269],
        3: [162, 94], 4: [186, 857], 5: [197, 444],
        6: [247, 614], 7: [298, 840], 8: [377, 357],
        9: [407, 840], 10: [406, 589], 11: [407, 444],
        12: [309, 241], 13: [482, 261], 14: [536, 592],
        15: [637, 844], 16: [639, 703], 17: [656, 444],
        18: [658, 90], 19: [936, 284], 20: [963, 840],
        21: [963, 615], 22: [963, 448], 23: [1118, 615],
        24: [407, 89], 25: [1272, 93], 26: [1141, 92],
        27: [1271, 613], 28: [1272, 445], 29: [837, 708],
        30: [835, 555], 31: [79, 443], 32: [1143, 218],
        33: [1273, 837]
    }
    matrix = np.zeros((34, 34)) + np.inf

    def set_matrix(i, j, value):
        matrix[i][j] = matrix[j][i] = value

    set_matrix(0, 1, 110)
    set_matrix(4, 1, 110)
    set_matrix(31, 1, 330)
    set_matrix(31, 5, 120)
    set_matrix(5, 6, 180)
    set_matrix(2, 5, 190)
    set_matrix(5, 11, 210)
    set_matrix(7, 9, 110)
    set_matrix(9, 10, 250)
    set_matrix(11, 10, 150)
    set_matrix(11, 8, 90)
    set_matrix(12, 8, 130)
    set_matrix(12, 24, 180)
    set_matrix(3, 24, 250)
    set_matrix(13, 8, 140)
    set_matrix(13, 24, 190)
    set_matrix(9, 15, 230)
    set_matrix(16, 15, 140)
    set_matrix(16, 14, 150)
    set_matrix(10, 14, 130)
    set_matrix(16, 29, 200)
    set_matrix(30, 29, 150)
    set_matrix(30, 14, 300)
    set_matrix(11, 17, 250)
    set_matrix(18, 17, 350)
    set_matrix(18, 24, 250)
    set_matrix(20, 15, 330)
    set_matrix(20, 33, 310)
    set_matrix(20, 21, 230)
    set_matrix(23, 21, 150)
    set_matrix(23, 27, 150)
    set_matrix(33, 27, 220)
    set_matrix(28, 27, 170)
    set_matrix(17, 22, 310)
    set_matrix(28, 22, 310)
    set_matrix(18, 26, 480)
    set_matrix(25, 26, 130)
    set_matrix(25, 28, 350)
    set_matrix(32, 26, 130)
    set_matrix(19, 32, 220)
    return map_dict, matrix
