import igl
import numpy as np

from .geometry import plane_through_3_points, rotate_points, rotation_matrix_from_vectors, \
    plane_normal_vector, find_boundary_loop


def init_unfold(vertices, faces, id_vertex):
    init_points_ids = np.array(faces[id_vertex], dtype=np.int64)
    x1 = vertices[init_points_ids[0]][0]
    y1 = vertices[init_points_ids[0]][1]
    z1 = vertices[init_points_ids[0]][2]
    x2 = vertices[init_points_ids[1]][0]
    y2 = vertices[init_points_ids[1]][1]
    z2 = vertices[init_points_ids[1]][2]
    x3 = vertices[init_points_ids[2]][0]
    y3 = vertices[init_points_ids[2]][1]
    z3 = vertices[init_points_ids[2]][2]
    points = np.array([
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3]
    ])
    plan = plane_through_3_points(x1, y1, z1, x2, y2, z2, x3, y3, z3)
    points = rotate_points(points, rotation_matrix_from_vectors(plane_normal_vector(plan), np.array([0, 0, 1])))
    points -= points[0]
    x, y, _ = points.T
    init_points_pos = np.ascontiguousarray(np.asarray([x, y], dtype=np.double).T)
    return init_points_ids, init_points_pos, plan


def unfold(vertices, faces, init_points_ids, init_points_pos, method="LSCM"):
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    b = np.ascontiguousarray(init_points_ids, dtype=np.int64)
    bc = np.ascontiguousarray(init_points_pos, dtype=np.float64)

    if method == "LSCM":
        res = igl.lscm(v, f, b, bc)
        if isinstance(res, tuple):
            unwrap = res[0]
        else:
            unwrap = res
    elif method == "ARAP":
        # ARAP needs an initial guess. Use LSCM.
        res_lscm = igl.lscm(v, f, b, bc)
        unwrap_init = res_lscm[0] if isinstance(res_lscm, tuple) else res_lscm

        # Precompute ARAP
        # igl.arap_precomputation(V, F, dim, b)
        # We use dim=2 for parameterization
        arap_data = igl.ARAPData()
        # ARAPEnergyType: 0: Spokes, 1: Spokes-and-rims, 2: Elements (default)
        igl.arap_precomputation(v, f, 2, b, arap_data)

        # Solve
        unwrap = igl.arap_solve(bc, arap_data, unwrap_init)
    else:
        raise ValueError(f"Unknown method: {method}")

    if unwrap is None or not len(unwrap):
        raise Exception("Impossible to unfold")
    return unwrap


def get_all_bounds(faces):
    res = igl.boundary_facets(faces)
    if isinstance(res, tuple):
        boundary_facets = res[0]
    else:
        boundary_facets = res

    adjacency_list = {}
    for edge in boundary_facets:
        u, v = int(edge[0]), int(edge[1])
        if u not in adjacency_list:
            adjacency_list[u] = []
        if v not in adjacency_list:
            adjacency_list[v] = []
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    all_boundary_loops = []
    visited = set()
    for edge in boundary_facets:
        u = int(edge[0])
        if u not in visited:
            loop, adjacency_list = find_boundary_loop(u, adjacency_list)
            all_boundary_loops.append(loop)
            visited.update(loop)
        v = int(edge[1])
        if v not in visited:
            loop, adjacency_list = find_boundary_loop(v, adjacency_list)
            all_boundary_loops.append(loop)
            visited.update(loop)
    return all_boundary_loops
