import os
import sys
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
from matplotlib.colors import Normalize

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flatten_surface.flatten_surface import flatten_mesh, solve_flatten
    from flatten_surface.import_export import load_with_components, mesh_to_data, component_quality, extract_source_label
except ImportError:
    from flatten_surface.flatten_surface import flatten_mesh, solve_flatten
    from flatten_surface.import_export import load_with_components, mesh_to_data, component_quality, extract_source_label


class FlattenApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tent-Maker Pro - Surface Flattening for PVC Production")
        self.geometry("1450x900")

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.unit_var = ctk.StringVar(value="mm")
        self.method_var = ctk.StringVar(value="ARAP")
        self.boundary_mode_var = ctk.StringVar(value="outer_only")
        self.face_pick_mode = ctk.BooleanVar(value=False)
        self.show_heatmap_var = ctk.BooleanVar(value=False)
        self.allow_dirty_export_var = ctk.BooleanVar(value=False)
        self.auto_relief_var = ctk.BooleanVar(value=True)
        self.relief_threshold_var = ctk.StringVar(value="3.0")
        self.seam_allowance_var = ctk.StringVar(value="12.0")
        self.align_to_x_enabled = False
        self.source_label = ""

        self.components = []
        self.selected_component_idx = 0
        self._pick_connection_id = None
        self._heatmap_cache = {}
        self._heatmap_colorbar = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self, width=360)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=0)
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.controls_frame = ctk.CTkScrollableFrame(self.left_panel)
        self.controls_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.actions_frame = ctk.CTkFrame(self.left_panel)
        self.actions_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(8, 0))

        self.label = ctk.CTkLabel(self.controls_frame, text="TENT-MAKER PRO", font=("Roboto", 24, "bold"))
        self.label.pack(pady=20)

        self.file_frame = ctk.CTkFrame(self.controls_frame)
        self.file_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.file_frame, text="1. Load 3D Model:").pack(anchor="w", padx=5)
        self.input_btn = ctk.CTkButton(self.file_frame, text="Open STL/OBJ/3DS", command=self.browse_input)
        self.input_btn.pack(fill="x", padx=5, pady=5)

        self.comp_frame = ctk.CTkFrame(self.controls_frame)
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
        self.face_pick_switch = ctk.CTkSwitch(
            self.comp_nav,
            text="Face Pick",
            variable=self.face_pick_mode,
            command=self.toggle_face_pick_mode,
            onvalue=True,
            offvalue=False,
            width=110,
        )
        self.face_pick_switch.pack(side="right")

        self.opt_frame = ctk.CTkFrame(self.controls_frame)
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

        ctk.CTkLabel(self.opt_frame, text="5. Boundary Export:").pack(anchor="w", padx=5)
        self.boundary_menu = ctk.CTkOptionMenu(
            self.opt_frame,
            values=["Outer Only (Recommended)", "Outer + Large Holes", "All Loops"],
            command=self.on_boundary_mode_change,
        )
        self.boundary_menu.pack(fill="x", padx=5, pady=5)
        self.boundary_menu.set("Outer Only (Recommended)")
        self.boundary_mode_var.set("outer_only")

        ctk.CTkLabel(self.opt_frame, text="6. Seam Allowance (mm):").pack(anchor="w", padx=5)
        self.seam_entry = ctk.CTkEntry(self.opt_frame, textvariable=self.seam_allowance_var)
        self.seam_entry.pack(fill="x", padx=5, pady=5)

        self.align_btn = ctk.CTkButton(self.opt_frame, text="Align to X-Axis: OFF", command=self.toggle_align_to_x)
        self.align_btn.pack(fill="x", padx=5, pady=5)

        self.heatmap_switch = ctk.CTkSwitch(
            self.opt_frame,
            text="Show Strain Heatmap",
            variable=self.show_heatmap_var,
            command=self.on_heatmap_toggle,
            onvalue=True,
            offvalue=False,
        )
        self.heatmap_switch.pack(anchor="w", padx=5, pady=(2, 6))
        self.allow_dirty_switch = ctk.CTkSwitch(
            self.opt_frame,
            text="Allow Export with Mesh Warnings",
            variable=self.allow_dirty_export_var,
            onvalue=True,
            offvalue=False,
        )
        self.allow_dirty_switch.pack(anchor="w", padx=5, pady=(0, 6))
        self.auto_relief_switch = ctk.CTkSwitch(
            self.opt_frame,
            text="Auto Relief-Cut Suggestion",
            variable=self.auto_relief_var,
            onvalue=True,
            offvalue=False,
        )
        self.auto_relief_switch.pack(anchor="w", padx=5, pady=(0, 4))
        ctk.CTkLabel(self.opt_frame, text="Relief Trigger Strain %:").pack(anchor="w", padx=5)
        self.relief_threshold_entry = ctk.CTkEntry(self.opt_frame, textvariable=self.relief_threshold_var)
        self.relief_threshold_entry.pack(fill="x", padx=5, pady=(0, 6))

        self.out_frame = ctk.CTkFrame(self.controls_frame)
        self.out_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.out_frame, text="7. Output:").pack(anchor="w", padx=5)
        self.out_btn = ctk.CTkButton(self.out_frame, text="Set Export Path", command=self.browse_output)
        self.out_btn.pack(fill="x", padx=5, pady=5)
        self.out_label = ctk.CTkLabel(self.out_frame, textvariable=self.output_path, font=("Roboto", 10), wraplength=320)
        self.out_label.pack(fill="x", padx=5)

        self.run_btn = ctk.CTkButton(
            self.actions_frame,
            text="GENERATE PRODUCTION DXF",
            command=self.run_flattening,
            height=60,
            font=("Roboto", 18, "bold"),
            fg_color="green",
            hover_color="darkgreen",
        )
        self.run_btn.pack(pady=(10, 10), padx=10, fill="x")

        self.status_label = ctk.CTkLabel(self.actions_frame, text="Ready", wraplength=320, justify="left")
        self.status_label.pack(pady=(0, 8), padx=10, anchor="w")

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
            text="Viewer: Rotate=left drag. Use toolbar for Pan/Zoom/Home. Enable Face Pick to click-select sheet.",
            text_color="#bfc6d1",
        )
        self.viewer_help.pack(anchor="w", padx=8, pady=(2, 6))
        self._pick_connection_id = self.canvas.mpl_connect("pick_event", self.on_pick)

    def on_method_change(self, val):
        self.method_var.set("ARAP" if "ARAP" in val else "LSCM")
        self._heatmap_cache.clear()
        if self.components:
            self.preview_component(self.selected_component_idx)

    def on_boundary_mode_change(self, val):
        mapping = {
            "Outer Only (Recommended)": "outer_only",
            "Outer + Large Holes": "outer_and_large_holes",
            "All Loops": "all_loops",
        }
        self.boundary_mode_var.set(mapping.get(val, "outer_only"))
        self._heatmap_cache.clear()
        if self.components:
            self.preview_component(self.selected_component_idx)

    def on_heatmap_toggle(self):
        if self.components:
            self.preview_component(self.selected_component_idx)

    def toggle_align_to_x(self):
        self.align_to_x_enabled = not self.align_to_x_enabled
        state = "ON" if self.align_to_x_enabled else "OFF"
        self.align_btn.configure(text=f"Align to X-Axis: {state}")

    def _get_seam_allowance_mm(self):
        raw = (self.seam_allowance_var.get() or "").strip()
        if raw == "":
            return 0.0
        val = float(raw)
        if val < 0:
            raise ValueError("Seam allowance must be >= 0")
        return val

    def _get_relief_threshold_pct(self):
        raw = (self.relief_threshold_var.get() or "").strip()
        if raw == "":
            return 3.0
        val = float(raw)
        if val <= 0:
            raise ValueError("Relief trigger strain must be > 0")
        return val

    def _ask_input_unit(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Input Units")
        dialog.geometry("360x170")
        dialog.transient(self)
        dialog.grab_set()

        result = {"unit": None}
        unit_var = ctk.StringVar(value=self.unit_var.get() or "mm")

        ctk.CTkLabel(
            dialog,
            text="Select units used by the loaded model file:",
            wraplength=330,
            justify="left",
        ).pack(padx=12, pady=(12, 8), anchor="w")

        unit_menu = ctk.CTkOptionMenu(dialog, values=["mm", "cm", "m", "inch"], variable=unit_var)
        unit_menu.pack(fill="x", padx=12, pady=4)

        row = ctk.CTkFrame(dialog, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(12, 10))

        def on_ok():
            result["unit"] = unit_var.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ctk.CTkButton(row, text="Cancel", command=on_cancel, width=90).pack(side="right", padx=(8, 0))
        ctk.CTkButton(row, text="Use Unit", command=on_ok, width=90).pack(side="right")

        self.wait_window(dialog)
        return result["unit"]

    def toggle_face_pick_mode(self):
        if not self.components:
            return
        self.preview_component(self.selected_component_idx)
        if self.face_pick_mode.get():
            self.status_label.configure(
                text="Face Pick enabled. Click a surface in the viewer to select it.",
                text_color="#8ac926",
            )
        else:
            self.status_label.configure(
                text="Face Pick disabled. Use dropdown/Prev/Next to select surfaces.",
                text_color="white",
            )

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("3D files", "*.stl *.obj *.3ds"), ("All files", "*.*")])
        if filename:
            picked_unit = self._ask_input_unit()
            if not picked_unit:
                return
            self.unit_var.set(picked_unit)
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
            self.source_label = extract_source_label(filename).get("label", os.path.splitext(os.path.basename(filename))[0])
            self._heatmap_cache.clear()

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
        indices = np.arange(len(faces), dtype=np.int64)
        if len(faces) > max_faces:
            step = max(1, len(faces) // max_faces)
            indices = indices[::step]
            faces = faces[::step]
        return vertices, faces, indices

    def _preview_mesh_data_all(self, mesh_count, mesh, max_total_faces=80000):
        per_mesh = max(1200, max_total_faces // max(mesh_count, 1))
        return self._preview_mesh_data(mesh, max_faces=per_mesh)

    def _setup_axes(self):
        self.ax.clear()
        self.ax.set_facecolor("#2b2b2b")
        self.ax.set_axis_off()

    def _set_equal_limits(self, all_vertices):
        vertices = np.asarray(all_vertices, dtype=np.float64)
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

    def _render_all_components_for_picking(self):
        self._clear_heatmap_colorbar()
        self._setup_axes()
        mesh_count = len(self.components)
        all_vertices = []

        for idx, mesh in enumerate(self.components):
            vertices, faces, _ = self._preview_mesh_data_all(mesh_count, mesh)
            if len(faces) == 0:
                continue
            all_vertices.append(vertices)

            base_color = cm.tab20(idx % 20)
            alpha = 0.85 if idx == self.selected_component_idx else 0.40
            edge_color = "#e8f0ff" if idx == self.selected_component_idx else "none"
            line_width = 0.08 if idx == self.selected_component_idx else 0.0

            collection = self.ax.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                color=base_color,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=alpha,
                shade=True,
            )
            collection.set_picker(True)
            setattr(collection, "_component_idx", idx)

        if all_vertices:
            merged = np.concatenate(all_vertices, axis=0)
            self._set_equal_limits(merged)

        active_mesh = self.components[self.selected_component_idx]
        self.ax.set_title(
            f"Face Pick Mode | Selected: Surface {self.selected_component_idx + 1} ({len(active_mesh.faces)} faces)",
            color="white",
        )
        self.canvas.draw()

    def _clear_heatmap_colorbar(self):
        if self._heatmap_colorbar is not None:
            self._heatmap_colorbar.remove()
            self._heatmap_colorbar = None

    def _get_component_strain(self, idx):
        key = (idx, self.method_var.get(), self.boundary_mode_var.get())
        if key in self._heatmap_cache:
            return self._heatmap_cache[key]
        vertices, faces = mesh_to_data(self.components[idx])
        solved = solve_flatten(
            vertices=vertices,
            faces=faces,
            input_unit=self.unit_var.get(),
            method=self.method_var.get(),
            boundary_mode=self.boundary_mode_var.get(),
        )
        strain = np.asarray(solved.get("strain_percent_per_face", []), dtype=np.float64)
        self._heatmap_cache[key] = strain
        return strain

    def preview_component(self, idx):
        if not self.components:
            return

        if self.face_pick_mode.get():
            self._render_all_components_for_picking()
            self._show_component_quality(self.components[idx])
            return

        mesh = self.components[idx]
        vertices, faces, face_indices = self._preview_mesh_data(mesh)

        self._setup_axes()
        collection = self.ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            cmap="viridis",
            edgecolor="grey",
            linewidth=0.1,
            alpha=0.85,
        )
        if self.show_heatmap_var.get() and self.method_var.get() == "ARAP":
            strain = self._get_component_strain(idx)
            if len(strain) > 0:
                strain_preview = strain[face_indices]
                cmap = cm.get_cmap("RdYlGn_r")
                norm = Normalize(vmin=0.0, vmax=3.0, clip=True)
                facecolors = cmap(norm(strain_preview))
                collection.set_facecolor(facecolors)
                collection.set_edgecolor("none")
                collection.set_linewidth(0.0)
                self._clear_heatmap_colorbar()
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                self._heatmap_colorbar = self.fig.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.02)
                self._heatmap_colorbar.set_label("Strain % (Green=0, Red>3)")
        else:
            self._clear_heatmap_colorbar()
        self.ax.set_title(f"Surface Preview ({len(mesh.faces)} faces)", color="white")

        self._set_equal_limits(vertices)
        self.canvas.draw()
        self._show_component_quality(mesh)

    def on_pick(self, event):
        if not self.face_pick_mode.get() or not self.components:
            return
        if getattr(self.toolbar, "mode", ""):
            return

        component_idx = getattr(event.artist, "_component_idx", None)
        if component_idx is None:
            return
        if component_idx < 0 or component_idx >= len(self.components):
            return

        self.selected_component_idx = component_idx
        self._sync_component_ui()
        self.status_label.configure(
            text=f"Face Pick: selected Surface {component_idx + 1}.",
            text_color="#8ac926",
        )

    def _show_component_quality(self, mesh):
        q = component_quality(mesh)
        if q["non_manifold_edges"] > 0 or q["degenerate_faces"] > 0:
            color = "#f0ad4e"
            flag = "Mesh warnings"
        else:
            color = "#8ac926"
            flag = "Mesh clean"
        details = (
            f"{flag}: faces={q['faces']}, open_edges={q['open_edges']}, "
            f"non_manifold={q['non_manifold_edges']}, degenerate={q['degenerate_faces']}"
        )
        self.status_label.configure(text=details, text_color=color)

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
        boundary_mode = self.boundary_mode_var.get()
        seam_allowance_mm = self._get_seam_allowance_mm()
        relief_threshold_pct = self._get_relief_threshold_pct()

        if not out:
            self.status_label.configure(text="Error: Set output path first", text_color="red")
            return

        try:
            surface_no = self.selected_component_idx + 1
            self.status_label.configure(text=f"Flattening Surface {surface_no} with {method}...", text_color="cyan")
            self.update()

            q = component_quality(self.components[self.selected_component_idx])
            has_hard_mesh_issue = q["non_manifold_edges"] > 0 or q["degenerate_faces"] > 0
            if has_hard_mesh_issue and not self.allow_dirty_export_var.get():
                self.status_label.configure(
                    text=(
                        "Export blocked: mesh has non-manifold or degenerate faces.\n"
                        "Enable 'Allow Export with Mesh Warnings' to override."
                    ),
                    text_color="red",
                )
                return

            vertices, faces = mesh_to_data(self.components[self.selected_component_idx])
            result = flatten_mesh(
                vertices=vertices,
                faces=faces,
                path_output=out,
                show_plot=False,
                input_unit=unit,
                method=method,
                boundary_mode=boundary_mode,
                seam_allowance_mm=seam_allowance_mm,
                align_to_x=self.align_to_x_enabled,
                label_text=self.source_label,
                auto_relief_cut=self.auto_relief_var.get(),
                relief_threshold_pct=relief_threshold_pct,
            )

            metrics = result.get("metrics", {})
            area_err = metrics.get("area_error_pct", 0.0)
            perim_err = metrics.get("perimeter_error_pct", 0.0)
            loops = result.get("bounds_exported", 0)
            relief_path = result.get("relief_path_2d")
            relief_note = "Yes" if relief_path is not None else "No"

            quality_color = "green" if area_err <= 2.0 and perim_err <= 1.0 else "#f0ad4e"
            warning_note = ""
            if area_err > 3.0 or perim_err > 3.0:
                warning_note = "\nProduction warning: high distortion. Consider different sheet/relief cuts."
            status = (
                f"Success: {os.path.basename(out)}\n"
                f"Area error: {area_err:.2f}% | Perimeter error: {perim_err:.2f}% | Loops: {loops}\n"
                f"Seam allowance: {seam_allowance_mm:.2f} mm | Align X: {'ON' if self.align_to_x_enabled else 'OFF'}\n"
                f"Relief cut suggested: {relief_note} (trigger>{relief_threshold_pct:.2f}%)"
                f"{warning_note}"
            )
            self.status_label.configure(text=status, text_color=quality_color)
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")


if __name__ == "__main__":
    app = FlattenApp()
    app.mainloop()
