import numpy as np


def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B


def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]


def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])


def skew_line(p1, p2, center_polygon, t):
    p1_ = (1 - t) * p1 + t * p2

    p1_1 = (1 - t) * p1_ + t * center_polygon
    p2_1 = (1 - t) * p2 + t * center_polygon

    points = np.array([
        p1,
        p1_1,
        p2_1,
        p1_,
        p2
    ])
    path = evaluate_bezier(points, 10)
    return path


def is_edge(p1, p2, x1, x2, y1, y2):
    p1_edge = p1[0] in [x1, x2] or p1[1] in [y1, y2]
    p2_edge = p2[0] in [x1, x2] or p2[1] in [y1, y2]
    return p1_edge and p2_edge


def skew_polygon(polygon, *kwargs):
    polygon = np.round(polygon, 4)
    written_lines, written_path, x1, x2, y1, y2 = kwargs
    new_polygon = []
    center_polygon = polygon.mean(axis=0)

    # print('written_path')

    for p1, p2 in zip(polygon[:-1], polygon[1:]):
        if len(written_lines) > 0 and 4 in np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1):
            #
            # if 22.74 in np.array([p1, p2]).round(2):
            #     print('\nrepeated')
            #     print(p1, p2)

            is_lin = np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1) == 4
            new_polygon.append(np.array(written_path)[is_lin][0])

            # if 22.74 in np.array([p1, p2]).round(2):
            #     print(np.array(written_path)[is_lin][0])

            continue
        else:
            if is_edge(p1, p2, x1, x2, y1, y2):
                path = np.array([p1, p2])
            else:
                t = np.random.uniform(0.3, 0.5, 1)[0]
                path = skew_line(p1, p2, center_polygon, t)

            new_polygon.append(path)

            written_lines.append([p1, p2])
            written_lines.append([p2, p1])

            written_path.append(path)
            written_path.append(path[::-1, :])

            # if 22.74 in np.array([p1, p2]).round(2):
            #     print('\n this is the first time ')
            #     print([p1, p2])
            #     is_lin = np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1) == 4
            #     print(np.array(written_path)[is_lin][0])

    return np.concatenate(new_polygon), written_lines, written_path


def skew_graph(polygons, x1, x2, y1, y2):
    new_polygons, written_lines, written_path = [], [], []
    for polygon in polygons:

        new_polygon, written_lines, written_path = skew_polygon(polygon, written_lines, written_path, x1, x2, y1, y2)
        new_polygons.append(new_polygon)

    return np.array(new_polygons)
