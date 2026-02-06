import numpy as np
import svgwrite
import trimesh
import ezdxf

def load(path):
    # trimesh can load obj, 3ds, stl etc.
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            # Take the first geometry if it's a scene
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError("Empty 3D file")

    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64)
    )

def load_with_components(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 1:
            # Merge all geometries if multiple?
            # Or just take the largest?
            mesh = mesh.dump(concatenate=True)
        elif len(mesh.geometry) == 1:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError("Empty 3D file")

    components = mesh.split(only_watertight=False)
    # Sort by area descending
    components.sort(key=lambda m: m.area, reverse=True)

    return components

def mesh_to_data(mesh):
    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64)
    )

def get_unit_scale(unit_name):
    """Returns scale factor to convert given unit to mm."""
    scales = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "inch": 25.4
    }
    return scales.get(unit_name.lower(), 1.0)

def export_svg(unwrap, bounds, path_svg, scale=1.0):
    contours = [unwrap[bound] * scale for bound in bounds]

    min_x = float("inf")
    min_y = float("inf")

    for i, contour in enumerate(contours):
        x, y = contour.T
        min_x = min(min_x, np.min(x))
        min_y = min(min_y, np.min(y))

    # Center/offset to positive coordinates for SVG
    for i, contour in enumerate(contours):
        x, y = contour.T
        x -= min_x
        y -= min_y
        contours[i] = np.array([x, y]).T

    dwg = svgwrite.Drawing(path_svg, profile='tiny')
    for contour in contours:
        path_data = "M" + " L".join(f"{x},{y}" for x, y in contour) + " Z"
        dwg.add(dwg.path(d=path_data, fill='none', stroke='black'))
    dwg.save()

def export_dxf(unwrap, bounds, path_dxf, scale=1.0):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    contours = [unwrap[bound] * scale for bound in bounds]

    for contour in contours:
        # Convert 2D points to 3D points (z=0) for ezdxf
        points = [(p[0], p[1], 0) for p in contour]
        # Close the loop
        if len(points) > 0:
            points.append(points[0])
        msp.add_lwpolyline(points)

    doc.saveas(path_dxf)
