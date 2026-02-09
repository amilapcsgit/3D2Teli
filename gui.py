import os
import sys
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flatten_surface.flatten_surface import flatten_mesh
    from flatten_surface.import_export import load_with_components, mesh_to_data
except ImportError:
    from flatten_surface.flatten_surface import flatten_mesh
    from flatten_surface.import_export import load_with_components, mesh_to_data


class FlattenApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("3D Surface Flattening Tool - Tent Manufacturing")
        self.geometry("1450x900")

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.unit_var = ctk.StringVar(value="mm")
        self.method_var = ctk.StringVar(value="ARAP")

        self.components = []
        self.selected_component_idx = 0

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self, width=360)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.label = ctk.CTkLabel(self.left_panel, text="FLATTEN TOOL", font=("Roboto", 24, "bold"))
        self.label.pack(pady=20)

        self.file_frame = ctk.CTkFrame(self.left_panel)
        self.file_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.file_frame, text="1. Load 3D Model:").pack(anchor="w", padx=5)
        self.input_btn = ctk.CTkButton(self.file_frame, text="Open STL/OBJ/3DS", command=self.browse_input)
        self.input_btn.pack(fill="x", padx=5, pady=5)

        self.comp_frame = ctk.CTkFrame(self.left_panel)
        self.comp_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.comp_frame, text="2. Select Surface:").pack(anchor="w", padx=5)
        self.comp_menu = ctk.CTkOptionMenu(self.comp_frame, values=["No model loaded"], command=self.on_component_select)
        self.comp_menu.pack(fill="x", padx=5, pady=5)

        self.comp_nav = ctk.CTkFrame(self.comp_frame, fg_color="transparent")
        self.comp_nav.pack(fill="x", padx=5, pady=(0, 6))
        self.prev_btn = ctk.CTkButton(self.comp_nav, text="Prev", width=80, command=self.select_prev_component)
        self.prev_btn.pack(side="left", padx=(0, 8))
        self.next_btn = ctk.CTkButton(self.comp_nav, text="Next", width=80, command=self.select_next_component)
        self.next_btn.pack(side="left")

        self.opt_frame = ctk.CTkFrame(self.left_panel)
        self.opt_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.opt_frame, text="3. Input Units:").pack(anchor="w", padx=5)
        self.unit_menu = ctk.CTkOptionMenu(self.opt_frame, values=["mm", "cm", "m", "inch"], variable=self.unit_var)
        self.unit_menu.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(self.opt_frame, text="4. Flattening Method:").pack(anchor="w", padx=5)
        self.method_menu = ctk.CTkOptionMenu(
            self.opt_frame,
            values=["LSCM (Fast)", "ARAP (Advanced - Best for Tent)"],
            command=self.on_method_change,
        )
        self.method_menu.pack(fill="x", padx=5, pady=5)
        self.method_menu.set("ARAP (Advanced - Best for Tent)")
        self.method_var.set("ARAP")

        self.out_frame = ctk.CTkFrame(self.left_panel)
        self.out_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.out_frame, text="5. Output:").pack(anchor="w", padx=5)
        self.out_btn = ctk.CTkButton(self.out_frame, text="Set Export Path", command=self.browse_output)
        self.out_btn.pack(fill="x", padx=5, pady=5)
        self.out_label = ctk.CTkLabel(self.out_frame, textvariable=self.output_path, font=("Roboto", 10), wraplength=320)
        self.out_label.pack(fill="x", padx=5)

        self.run_btn = ctk.CTkButton(
            self.left_panel,
            text="GENERATE PRODUCTION DXF",
            command=self.run_flattening,
            height=60,
            font=("Roboto", 18, "bold"),
            fg_color="green",
            hover_color="darkgreen",
        )
        self.run_btn.pack(pady=30, padx=10, fill="x")

        self.status_label = ctk.CTkLabel(self.left_panel, text="Ready", wraplength=320, justify="left")
        self.status_label.pack(pady=10)

        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.fig = plt.figure(figsize=(6, 6), facecolor="#2b2b2b")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#2b2b2b")
        self.ax.set_axis_off()

        self.toolbar_container = ctk.CTkFrame(self.right_panel)
        self.toolbar_container.pack(fill="x", padx=4, pady=(4, 0))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_container, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.viewer_help = ctk.CTkLabel(
            self.right_panel,
            text="Viewer: Rotate=left drag, Pan/Zoom/Home via toolbar",
            text_color="#bfc6d1",
        )
        self.viewer_help.pack(anchor="w", padx=8, pady=(2, 6))

    def on_method_change(self, val):
        self.method_var.set("ARAP" if "ARAP" in val else "LSCM")

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("3D files", "*.stl *.obj *.3ds"), ("All files", "*.*")])
        if filename:
            self.input_path.set(filename)
            base = os.path.splitext(filename)[0]
            self.output_path.set(base + ".dxf")
            self.load_model(filename)

    def _component_name(self, idx, mesh):
        return f"Surface {idx + 1} | Faces: {len(mesh.faces)} | Area: {mesh.area:.1f}"

    def load_model(self, filename):
        try:
            self.status_label.configure(text="Analyzing mesh and extracting selectable surfaces...", text_color="cyan")
            self.update()
            self.components = load_with_components(filename)

            if not self.components:
                raise ValueError("No selectable surfaces found")

            comp_names = [self._component_name(i, c) for i, c in enumerate(self.components)]
            self.comp_menu.configure(values=comp_names)
            self.comp_menu.set(comp_names[0])
            self.selected_component_idx = 0

            self.preview_component(0)
            self.status_label.configure(
                text=f"Loaded {len(self.components)} selectable surfaces. Use dropdown/Prev/Next and 3D viewer controls.",
                text_color="white",
            )
        except Exception as e:
            self.status_label.configure(text=f"Error loading: {str(e)}", text_color="red")

    def on_component_select(self, val):
        values = self.comp_menu.cget("values")
        if val not in values:
            return
        idx = values.index(val)
        self.selected_component_idx = idx
        self.preview_component(idx)

    def select_prev_component(self):
        if not self.components:
            return
        self.selected_component_idx = (self.selected_component_idx - 1) % len(self.components)
        self._sync_component_ui()

    def select_next_component(self):
        if not self.components:
            return
        self.selected_component_idx = (self.selected_component_idx + 1) % len(self.components)
        self._sync_component_ui()

    def _sync_component_ui(self):
        name = self._component_name(self.selected_component_idx, self.components[self.selected_component_idx])
        self.comp_menu.set(name)
        self.preview_component(self.selected_component_idx)

    def _preview_mesh_data(self, mesh, max_faces=30000):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) > max_faces:
            step = max(1, len(faces) // max_faces)
            faces = faces[::step]
        return vertices, faces

    def preview_component(self, idx):
        if not self.components:
            return

        mesh = self.components[idx]
        vertices, faces = self._preview_mesh_data(mesh)

        self.ax.clear()
        self.ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            cmap="viridis",
            edgecolor="grey",
            linewidth=0.1,
            alpha=0.85,
        )
        self.ax.set_title(f"Surface Preview ({len(mesh.faces)} faces)", color="white")

        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min(),
        ]).max() / 2.0
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        self.ax.set_axis_off()
        self.canvas.draw()

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("SVG files", "*.svg"), ("All files", "*.*")],
        )
        if filename:
            self.output_path.set(filename)

    def run_flattening(self):
        if not self.components:
            self.status_label.configure(text="Error: No model loaded", text_color="red")
            return

        out = self.output_path.get().strip()
        unit = self.unit_var.get()
        method = self.method_var.get()

        if not out:
            self.status_label.configure(text="Error: Set output path first", text_color="red")
            return

        try:
            surface_no = self.selected_component_idx + 1
            self.status_label.configure(text=f"Flattening Surface {surface_no} with {method}...", text_color="cyan")
            self.update()

            vertices, faces = mesh_to_data(self.components[self.selected_component_idx])
            result = flatten_mesh(
                vertices=vertices,
                faces=faces,
                path_output=out,
                show_plot=False,
                input_unit=unit,
                method=method,
            )

            metrics = result.get("metrics", {})
            area_err = metrics.get("area_error_pct", 0.0)
            perim_err = metrics.get("perimeter_error_pct", 0.0)

            quality_color = "green" if area_err <= 2.0 and perim_err <= 1.0 else "#f0ad4e"
            status = (
                f"Success: {os.path.basename(out)}\n"
                f"Area error: {area_err:.2f}% | Perimeter error: {perim_err:.2f}%"
            )
            self.status_label.configure(text=status, text_color=quality_color)
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")


if __name__ == "__main__":
    app = FlattenApp()
    app.mainloop()
