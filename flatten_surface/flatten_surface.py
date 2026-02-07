import os
import tkinter as tk
from tkinter import filedialog

from .display import plot
from .igl_api import init_unfold, unfold, get_all_bounds
from .import_export import load, export_svg, export_dxf, get_unit_scale
from .score import compute_deformation


def main(path_input=None, path_output=None, path_png=None, vertice_init_id=0, show_plot=True, input_unit="mm", method="LSCM"):
    if path_input is None or not os.path.isfile(path_input):
        root = tk.Tk()
        root.withdraw()
        path_input = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(__file__), "..", "data"),
            title="Please select a 3D surface file to unfold",
            filetypes=(("3D files", "*.stl *.STL *.obj *.3ds"), ("All files", "*.*"))
        )
        if not path_input:
            return

    if path_png is None:
        path_png = os.path.join(os.path.dirname(path_input), ".".join(os.path.basename(path_input).split(".")[:-1]) + ".png")

    if path_output is None:
        path_output = os.path.join(os.path.dirname(path_input), ".".join(os.path.basename(path_input).split(".")[:-1]) + ".svg")

    vertices, faces = load(path_input)
    bounds = get_all_bounds(faces)
    init_points_ids, init_points_pos, plan = init_unfold(vertices, faces, vertice_init_id)
    unwrap = unfold(vertices, faces, init_points_ids, init_points_pos, method=method)
    deformation = compute_deformation(vertices, faces, unwrap)

    if show_plot:
        plot(vertices, faces, unwrap, init_points_pos, plan, init_points_ids, bounds, deformation, path_png)

    scale = get_unit_scale(input_unit)

    if path_output.lower().endswith('.dxf'):
        export_dxf(unwrap, bounds, path_output, scale=scale)
    else:
        export_svg(unwrap, bounds, path_output, scale=scale)
