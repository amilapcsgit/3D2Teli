import os
import tkinter as tk
from tkinter import filedialog

import numpy as np

from .display import plot
from .igl_api import init_unfold, unfold, get_all_bounds
from .import_export import load, export_svg, export_dxf, get_unit_scale
from .score import compute_deformation


def _polygon_length_2d(points_2d):
    if len(points_2d) < 2:
        return 0.0
    deltas = np.diff(points_2d, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _polygon_length_3d(points_3d):
    if len(points_3d) < 2:
        return 0.0
    deltas = np.diff(points_3d, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _compute_metrics(vertices, faces, unwrap, bounds, scale):
    tri3d = vertices[faces] * scale
    tri2d = unwrap[faces] * scale

    # Triangle area = 0.5 * |cross(e1, e2)|
    area3d = 0.5 * np.sum(np.linalg.norm(np.cross(tri3d[:, 1] - tri3d[:, 0], tri3d[:, 2] - tri3d[:, 0]), axis=1))
    cross2d = (
        (tri2d[:, 1, 0] - tri2d[:, 0, 0]) * (tri2d[:, 2, 1] - tri2d[:, 0, 1])
        - (tri2d[:, 1, 1] - tri2d[:, 0, 1]) * (tri2d[:, 2, 0] - tri2d[:, 0, 0])
    )
    area2d = 0.5 * float(np.sum(np.abs(cross2d)))

    perimeter_3d = 0.0
    perimeter_2d = 0.0
    for bound in bounds:
        if len(bound) < 2:
            continue
        idx = np.asarray(bound, dtype=np.int64)
        loop3d = vertices[idx] * scale
        loop2d = unwrap[idx] * scale

        loop3d_closed = np.vstack([loop3d, loop3d[:1]])
        loop2d_closed = np.vstack([loop2d, loop2d[:1]])

        perimeter_3d += _polygon_length_3d(loop3d_closed)
        perimeter_2d += _polygon_length_2d(loop2d_closed)

    area_error_pct = abs(area2d - area3d) / max(area3d, 1e-12) * 100.0
    perimeter_error_pct = abs(perimeter_2d - perimeter_3d) / max(perimeter_3d, 1e-12) * 100.0

    return {
        "area_3d_mm2": float(area3d),
        "area_2d_mm2": float(area2d),
        "area_error_pct": float(area_error_pct),
        "perimeter_3d_mm": float(perimeter_3d),
        "perimeter_2d_mm": float(perimeter_2d),
        "perimeter_error_pct": float(perimeter_error_pct),
    }


def _loop_area_2d(points_2d):
    if len(points_2d) < 3:
        return 0.0
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _select_bounds_for_export(bounds, unwrap, mode="outer_only", hole_area_ratio=0.02):
    if not bounds:
        return bounds

    if mode == "all_loops":
        return bounds

    loop_areas = []
    for i, bound in enumerate(bounds):
        pts = unwrap[np.asarray(bound, dtype=np.int64)]
        loop_areas.append((i, abs(_loop_area_2d(pts))))

    if not loop_areas:
        return bounds

    outer_idx, outer_area = max(loop_areas, key=lambda it: it[1])
    if mode == "outer_only":
        return [bounds[outer_idx]]

    # Keep outer + only significant holes.
    selected = [bounds[outer_idx]]
    min_hole_area = max(outer_area * hole_area_ratio, 1e-12)
    for i, area in loop_areas:
        if i == outer_idx:
            continue
        if area >= min_hole_area:
            selected.append(bounds[i])
    return selected


def flatten_mesh(
    vertices,
    faces,
    path_output,
    path_png=None,
    vertice_init_id=0,
    show_plot=True,
    input_unit="mm",
    method="LSCM",
    boundary_mode="outer_only",
    seam_allowance_mm=0.0,
    align_to_x=False,
    label_text=None,
):
    bounds = get_all_bounds(faces)
    if not bounds:
        raise ValueError(
            "Could not find open boundary loops to export. "
            "The selected mesh is likely a closed solid. "
            "Try selecting a single surface patch."
        )

    init_points_ids, init_points_pos, plan = init_unfold(vertices, faces, vertice_init_id)
    unwrap = unfold(vertices, faces, init_points_ids, init_points_pos, method=method)
    export_bounds = _select_bounds_for_export(bounds, unwrap, mode=boundary_mode)
    strain_percent = compute_deformation(vertices, faces, unwrap, verbose=True)

    if show_plot:
        plot(vertices, faces, unwrap, init_points_pos, plan, init_points_ids, export_bounds, strain_percent, path_png)

    scale = get_unit_scale(input_unit)
    if path_output.lower().endswith(".dxf"):
        export_dxf(
            unwrap,
            export_bounds,
            path_output,
            scale=scale,
            seam_allowance_mm=seam_allowance_mm,
            align_to_x=align_to_x,
            label_text=label_text,
        )
    else:
        export_svg(unwrap, export_bounds, path_output, scale=scale)

    metrics = _compute_metrics(vertices, faces, unwrap, export_bounds, scale)
    return {
        "bounds_found": len(bounds),
        "bounds_exported": len(export_bounds),
        "metrics": metrics,
        "method": method,
        "boundary_mode": boundary_mode,
        "seam_allowance_mm": float(seam_allowance_mm),
        "strain_percent_per_face": strain_percent,
        "unwrap": unwrap,
        "export_bounds": export_bounds,
    }


def solve_flatten(
    vertices,
    faces,
    input_unit="mm",
    method="LSCM",
    boundary_mode="outer_only",
):
    bounds = get_all_bounds(faces)
    if not bounds:
        raise ValueError(
            "Could not find open boundary loops to export. "
            "The selected mesh is likely a closed solid. "
            "Try selecting a single surface patch."
        )

    init_points_ids, init_points_pos, plan = init_unfold(vertices, faces, 0)
    unwrap = unfold(vertices, faces, init_points_ids, init_points_pos, method=method)
    export_bounds = _select_bounds_for_export(bounds, unwrap, mode=boundary_mode)
    strain_percent = compute_deformation(vertices, faces, unwrap, verbose=False)
    scale = get_unit_scale(input_unit)
    metrics = _compute_metrics(vertices, faces, unwrap, export_bounds, scale)
    return {
        "bounds_found": len(bounds),
        "bounds_exported": len(export_bounds),
        "method": method,
        "boundary_mode": boundary_mode,
        "metrics": metrics,
        "strain_percent_per_face": strain_percent,
        "unwrap": unwrap,
        "export_bounds": export_bounds,
    }


def main(
    path_input=None,
    path_output=None,
    path_png=None,
    vertice_init_id=0,
    show_plot=True,
    input_unit="mm",
    method="LSCM",
    boundary_mode="outer_only",
    seam_allowance_mm=0.0,
    align_to_x=False,
    label_text=None,
):
    if path_input is None or not os.path.isfile(path_input):
        root = tk.Tk()
        root.withdraw()
        path_input = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(__file__), "..", "data"),
            title="Please select a 3D surface file to unfold",
            filetypes=(("3D files", "*.stl *.STL *.obj *.3ds"), ("All files", "*.*")),
        )
        if not path_input:
            return None

    if path_png is None:
        stem = ".".join(os.path.basename(path_input).split(".")[:-1])
        path_png = os.path.join(os.path.dirname(path_input), f"{stem}.png")

    if path_output is None:
        stem = ".".join(os.path.basename(path_input).split(".")[:-1])
        path_output = os.path.join(os.path.dirname(path_input), f"{stem}.svg")

    vertices, faces = load(path_input, prefer_open_surface=True)
    return flatten_mesh(
        vertices=vertices,
        faces=faces,
        path_output=path_output,
        path_png=path_png,
        vertice_init_id=vertice_init_id,
        show_plot=show_plot,
        input_unit=input_unit,
        method=method,
        boundary_mode=boundary_mode,
        seam_allowance_mm=seam_allowance_mm,
        align_to_x=align_to_x,
        label_text=label_text,
    )
