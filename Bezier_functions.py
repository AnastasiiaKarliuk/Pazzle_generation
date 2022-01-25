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


def skew_polygon(polygon):
    new_polygon = []
    center_polygon = polygon.mean(axis=0)

    save_percent = 0.5
    t = 1 - save_percent

    for p1, p2 in zip(polygon[:-1], polygon[1:]):
        p1_ = (1 - t) * p1 + t * p2
        p1_1 = (1 - t) * p1_ + t * center_polygon
        p2_1 = (1 - t) * p2 + t * center_polygon

        points = np.array([
            p1,
            p1_1,
            p2_1,
            # center_polygon,
            p2
        ])
        path = evaluate_bezier(points, 20)
        new_polygon.append(path)
    return np.concatenate(new_polygon)


def skew_graph(polygons):
    new_polygons = [skew_polygon(polygon) for polygon in polygons]
    return np.array(new_polygons)
