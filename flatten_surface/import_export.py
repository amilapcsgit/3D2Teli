import numpy as np
import svgwrite
import trimesh
import ezdxf


def _load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 1:
            mesh = mesh.dump(concatenate=True)
        elif len(mesh.geometry) == 1:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError("Empty 3D file")
    return mesh


def _has_boundary_edges(faces):
    if len(faces) == 0:
        return False
    edges = np.vstack((
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ))
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return np.any(counts == 1)


def _extract_open_patches_from_watertight(mesh, min_faces=200):
    directions = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    thresholds = (0.00, 0.05, 0.10, 0.20, 0.30)

    candidates = []
    seen = set()
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    for threshold in thresholds:
        for direction in directions:
            selected_faces = np.where(face_normals @ direction > threshold)[0]
            if len(selected_faces) < min_faces:
                continue

            submesh = mesh.submesh([selected_faces], append=True, repair=False)
            for part in submesh.split(only_watertight=False):
                if len(part.faces) < min_faces:
                    continue

                part_faces = np.asarray(part.faces, dtype=np.int64)
                if not _has_boundary_edges(part_faces):
                    continue

                signature = (
                    int(len(part.faces)),
                    round(float(part.area), 3),
                    tuple(np.round(np.asarray(part.bounds).reshape(-1), 2)),
                )
                if signature in seen:
                    continue

                seen.add(signature)
                candidates.append(part)

    candidates.sort(key=lambda m: m.area, reverse=True)
    return candidates[:16]


def load(path, prefer_open_surface=True):
    mesh = _load_mesh(path)

    if prefer_open_surface and mesh.is_watertight:
        extracted_patches = _extract_open_patches_from_watertight(mesh)
        if extracted_patches:
            mesh = extracted_patches[0]

    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64),
    )


def load_with_components(path):
    mesh = _load_mesh(path)
    components = mesh.split(only_watertight=False)
    components.sort(key=lambda m: m.area, reverse=True)

    open_components = [m for m in components if _has_boundary_edges(np.asarray(m.faces, dtype=np.int64))]
    if open_components:
        return open_components

    if mesh.is_watertight:
        patches = _extract_open_patches_from_watertight(mesh)
        if patches:
            return patches

    return components


def mesh_to_data(mesh):
    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64),
    )


def get_unit_scale(unit_name):
    """Returns scale factor to convert given unit to mm."""
    scales = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "inch": 25.4,
    }
    return scales.get(unit_name.lower(), 1.0)


def export_svg(unwrap, bounds, path_svg, scale=1.0):
    contours = [unwrap[bound] * scale for bound in bounds]

    min_x = float("inf")
    min_y = float("inf")

    for contour in contours:
        x, y = contour.T
        min_x = min(min_x, np.min(x))
        min_y = min(min_y, np.min(y))

    for i, contour in enumerate(contours):
        x, y = contour.T
        x -= min_x
        y -= min_y
        contours[i] = np.array([x, y]).T

    dwg = svgwrite.Drawing(path_svg, profile="tiny")
    for contour in contours:
        path_data = "M" + " L".join(f"{x},{y}" for x, y in contour) + " Z"
        dwg.add(dwg.path(d=path_data, fill="none", stroke="black"))
    dwg.save()


def export_dxf(unwrap, bounds, path_dxf, scale=1.0):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    contours = [unwrap[bound] * scale for bound in bounds]

    for contour in contours:
        points = [(float(p[0]), float(p[1])) for p in contour]
        if len(points) > 0:
            points.append(points[0])
        msp.add_lwpolyline(points)

    doc.saveas(path_dxf)
