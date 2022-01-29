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


def get_points(a, b, a1, b1, center_polygon, t):
    a_c = t * a + (1 - t) * center_polygon
    a1_c = t * a1 + (1 - t) * center_polygon
    b1_c = t * b1 + (1-t) * center_polygon
    b_c = t * b + (1 - t) * center_polygon
    return a_c, a1_c, b1_c, b_c


def randomize_points_2c(*args):
    a1, b1, a_c1, a1_c1, b1_c1, b_c1, a_c2, a1_c2, b1_c2, b_c2 = args
    return [a1, a_c1, b1_c1, b1_c2, b_c2]


def randomize_points_1c(*args):
    a1, b1, a_c1, a1_c1, b1_c1, b_c1 = args
    return [a1, a_c1, b_c1, b1]


def vertexes_2d(a, b, a1, b1, centers_polygon):
    range_t = 0.7, 0.7

    t = np.random.uniform(range_t[0], range_t[1], 1)[0]
    a_c1, a1_c1, b1_c1, b_c1 = get_points(a, b, a1, b1, centers_polygon[0], t)

    t = np.random.uniform(range_t[0], range_t[1], 1)[0]
    a_c2, a1_c2, b1_c2, b_c2 = get_points(a, b, a1, b1, centers_polygon[1], t)

    vertexes = randomize_points_2c(a1, b1, a_c1, a1_c1, b1_c1, b_c1, a_c2, a1_c2, b1_c2, b_c2)
    vertexes = [a] + vertexes + [b]
    return vertexes


def vertexes_1d(a, b, a1, b1, centers_polygon):
    range_t = 0.7, 0.7

    t = np.random.uniform(range_t[0], range_t[1], 1)[0]
    a_c1, a1_c1, b1_c1, b_c1 = get_points(a, b, a1, b1, centers_polygon[0], t)

    vertexes = randomize_points_1c(a1, b1, a_c1, a1_c1, b1_c1, b_c1)
    vertexes = [a] + vertexes + [b]
    return vertexes


def skew_line(a, b, centers_polygon):
    range_t = 0.8, 0.8
    t = np.random.uniform(range_t[0], range_t[1], 1)[0]

    a1 = t * a + (1-t) * b
    b1 = t * b + (1-t) * a

    if len(centers_polygon) == 2:
        points = vertexes_2d(a, b, a1, b1, centers_polygon)
    else:
        points = vertexes_1d(a, b, a1, b1, centers_polygon)

    points = np.array(points)
    path = evaluate_bezier(points, 10)
    return path


def is_edge(p1, p2, x1, x2, y1, y2):
    p1_edge = p1[0] in [x1, x2] or p1[1] in [y1, y2]
    p2_edge = p2[0] in [x1, x2] or p2[1] in [y1, y2]
    return p1_edge and p2_edge


def skew_polygon(polygon, written_lines, written_path,
                 x1, x2, y1, y2,
                 list_cenetrs, written_centers):

    polygon = np.round(polygon, 4)
    new_polygon = []

    for p1, p2 in zip(polygon[:-1], polygon[1:]):
        if len(written_lines) > 0 and 4 in np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1):

            is_lin = np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1) == 4
            new_polygon.append(np.array(written_path)[is_lin][0])

            continue
        else:
            if is_edge(p1, p2, x1, x2, y1, y2):
                path = np.array([p1, p2])
            else:
                is_lin = np.equal(written_centers, [p2, p1]).sum(axis=1).sum(axis=1) == 4
                index_el = list(is_lin).index(True)
                path = skew_line(p1, p2, list_cenetrs[index_el])

            new_polygon.append(path)

            written_lines.append([p1, p2])
            written_lines.append([p2, p1])

            written_path.append(path)
            written_path.append(path[::-1, :])

    return np.concatenate(new_polygon), written_lines, written_path


def skew_graph(polygons, x1, x2, y1, y2, list_cenetrs, written_centers):
    new_polygons, written_lines, written_path = [], [], []
    for polygon in polygons:
        new_polygon, written_lines, written_path = skew_polygon(polygon, written_lines, written_path,
                                                                x1, x2, y1, y2, list_cenetrs, written_centers)
        new_polygons.append(new_polygon)

    return np.array(new_polygons)


# ------- collect centers of each line ----


def culculate_cenetrs(polygons):
    written_lines, list_cenetrs = [], []
    for polygon in polygons:
        written_lines, list_cenetrs = add_centers(polygon, written_lines, list_cenetrs)
    return written_lines, list_cenetrs


def add_centers(polygon, written_lines, list_centers):
    polygon = np.round(polygon, 4)
    center_polygon = polygon.mean(axis=0)

    for p1, p2 in zip(polygon[:-1], polygon[1:]):
        if len(written_lines) > 0 and 4 in np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1):
            is_lin = np.equal(written_lines, [p1, p2]).sum(axis=1).sum(axis=1) == 4
            index_el = list(is_lin).index(True)
            list_centers[index_el].append(center_polygon)

            is_lin = np.equal(written_lines, [p2, p1]).sum(axis=1).sum(axis=1) == 4
            index_el = list(is_lin).index(True)
            list_centers[index_el].append(center_polygon)
        else:
            written_lines.append([p1, p2])
            written_lines.append([p2, p1])

            list_centers.append([center_polygon])
            list_centers.append([center_polygon])

    return written_lines, list_centers
