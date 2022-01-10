import itertools

import numpy as np


def form_x_y_polygons(polygons):
    x_polygs, y_polygs = [], []
    for polyg in polygons:
        x_polygs.append(list(polyg[:, 0])+[None])
        y_polygs.append(list(polyg[:, 1])+[None])

    x_polygs = list(itertools.chain(*x_polygs))
    y_polygs = list(itertools.chain(*y_polygs))
    return x_polygs, y_polygs


def is_segments_intersect(seg_1, seg_2):
    v1, v2 = seg_1
    u1, u2 = seg_2

    M = np.array([v1 - v2, u2 - u1]).T
    if np.linalg.matrix_rank(M) < 2:
        return False

    a, b = np.linalg.inv(M).dot(u2 - v2)

    return (0 < a < 1) and (0 < b < 1)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y