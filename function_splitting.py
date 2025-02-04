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


def crop_axis(polygons, range_x, range_y):
    x1, x2, = range_x
    y1, y2 = range_y

    new_polygons = [get_new_polygon(polyg, x1, x2, y1, y2) for polyg in polygons]
    return new_polygons


def check_polygon_in(polyg, x1, x2, y1, y2):
    x_in = np.logical_or(polyg[:, 0] <= x1, polyg[:, 0] >= x2)
    y_in = np.logical_or(polyg[:, 1] <= y1, polyg[:, 1] >= y2)
    return x_in, y_in


def get_new_polygon(polyg, x1, x2, y1, y2):
    A, B, C, D = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    x_in, y_in = check_polygon_in(polyg, x1, x2, y1, y2)

    if sum(x_in + y_in) == 0:  # full polygon inside area
        return polyg

    new_polyg = []
    for vector in zip(polyg, np.vstack((polyg, polyg[0]))[1:]):
        outside_vector = [is_outside(vec, x1, x2, y1, y2) for vec in vector]
        crossed = False

        if sum(outside_vector) == 2:
            continue
        elif sum(outside_vector) == 1:
            for line in zip([A, B, C, D], [B, C, D, A]):
                crossed = is_segments_intersect(np.array(vector), np.array(line))
                if crossed:
                    intersected_line = line_intersection(vector, line)

                    if outside_vector[0]:
                        new_polyg.append(intersected_line)
                        new_polyg.append(vector[1])
                    else:
                        new_polyg.append(vector[0])
                        new_polyg.append(intersected_line)
        else:
            new_polyg.append(vector[0])
            new_polyg.append(vector[1])

    new_polyg_deduplicated = deduplicate_rows(np.array(new_polyg))
    new_polyg = check_add_vertex(new_polyg_deduplicated, x1, x2, y1, y2)
    return new_polyg


def deduplicate_rows(arr):
    new_arr = np.append(arr, [arr[0]], axis=0)[1:]
    is_leave = np.equal(arr, new_arr).sum(axis=1) < arr.shape[1]
    reduplicated = arr[is_leave].round(4)
    reduplicated = np.append(reduplicated, [reduplicated[0]], axis=0)
    return reduplicated


def check_add_vertex(polyg, x1, x2, y1, y2):
    # if 22.32 in polyg.round(2):
    #     print('\nvertex ', polyg)

    x_in, y_in = check_polygon_in(polyg, x1, x2, y1, y2)

    # case x_y
    y_in_new = np.append(y_in, False)[1:]
    get_ind = np.where(x_in.astype(int) + y_in_new.astype(int) == 2)[0]

    if get_ind:
        ind_vertex = get_ind[0] + 1
        polyg = np.concatenate(
            [polyg[:ind_vertex], [[polyg[ind_vertex - 1, 0], polyg[ind_vertex, 1]]], polyg[ind_vertex:]])

    # case y_x
    x_in_new = np.append(x_in, False)[1:]
    get_ind = np.where(y_in.astype(int) + x_in_new.astype(int) == 2)[0]

    # if 22.32 in polyg.round(2):
    #     print('get_ind ', polyg[-2][1])
    #     print(x_in_new, y_in)

    if get_ind:
        ind_vertex = get_ind[0] + 1
        polyg = np.concatenate(
            [polyg[:ind_vertex], [[polyg[ind_vertex, 0], polyg[ind_vertex - 1, 1]]], polyg[ind_vertex:]])

    return polyg


def is_outside(vec, x1, x2, y1, y2):
    return vec[0] < x1 or vec[0] > x2 or vec[1] < y1 or vec[1] > y2

