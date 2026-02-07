import customtkinter as ctk
from tkinter import filedialog
import os
import sys
import numpy as np

# Matplotlib for 3D viewer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flatten_surface.flatten_surface import main as flatten_main
    from flatten_surface.import_export import load_with_components, mesh_to_data
except ImportError:
    from flatten_surface import flatten_surface as flatten_main
    from flatten_surface.import_export import load_with_components, mesh_to_data

class FlattenApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("3D Surface Flattening Tool - Tent Manufacturing")
        self.geometry("1400x900")

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.unit_var = ctk.StringVar(value="mm")
        self.method_var = ctk.StringVar(value="LSCM")

        self.components = []
        self.selected_component_idx = 0

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Panel ---
        self.left_panel = ctk.CTkFrame(self, width=350)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.label = ctk.CTkLabel(self.left_panel, text="FLATTEN TOOL", font=("Roboto", 24, "bold"))
        self.label.pack(pady=20)

        # File Selection
        self.file_frame = ctk.CTkFrame(self.left_panel)
        self.file_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.file_frame, text="1. Load 3D Model:").pack(anchor="w", padx=5)
        self.input_btn = ctk.CTkButton(self.file_frame, text="Open STL/OBJ/3DS", command=self.browse_input)
        self.input_btn.pack(fill="x", padx=5, pady=5)

        # Component Selection
        self.comp_frame = ctk.CTkFrame(self.left_panel)
        self.comp_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.comp_frame, text="2. Select Surface:").pack(anchor="w", padx=5)
        self.comp_menu = ctk.CTkOptionMenu(self.comp_frame, values=["No model loaded"], command=self.on_component_select)
        self.comp_menu.pack(fill="x", padx=5, pady=5)

        # Options
        self.opt_frame = ctk.CTkFrame(self.left_panel)
        self.opt_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.opt_frame, text="3. Input Units:").pack(anchor="w", padx=5)
        self.unit_menu = ctk.CTkOptionMenu(self.opt_frame, values=["mm", "cm", "m", "inch"], variable=self.unit_var)
        self.unit_menu.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(self.opt_frame, text="4. Flattening Method:").pack(anchor="w", padx=5)
        self.method_menu = ctk.CTkOptionMenu(self.opt_frame, values=["LSCM (Fast)", "ARAP (Advanced - Best for Tent)"],
                                           command=self.on_method_change)
        self.method_menu.pack(fill="x", padx=5, pady=5)
        self.method_var.set("LSCM")

        # Output Selection
        self.out_frame = ctk.CTkFrame(self.left_panel)
        self.out_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.out_frame, text="5. Output:").pack(anchor="w", padx=5)
        self.out_btn = ctk.CTkButton(self.out_frame, text="Set Export Path", command=self.browse_output)
        self.out_btn.pack(fill="x", padx=5, pady=5)
        self.out_label = ctk.CTkLabel(self.out_frame, textvariable=self.output_path, font=("Roboto", 10), wraplength=300)
        self.out_label.pack(fill="x", padx=5)

        # Run Button
        self.run_btn = ctk.CTkButton(self.left_panel, text="GENERATE PRODUCTION DXF", command=self.run_flattening,
                                    height=60, font=("Roboto", 18, "bold"), fg_color="green", hover_color="darkgreen")
        self.run_btn.pack(pady=30, padx=10, fill="x")

        self.status_label = ctk.CTkLabel(self.left_panel, text="Ready", wraplength=300)
        self.status_label.pack(pady=10)

        # --- Right Panel (3D Viewer) ---
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.fig = plt.figure(figsize=(5, 5), facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def on_method_change(self, val):
        if "ARAP" in val:
            self.method_var.set("ARAP")
        else:
            self.method_var.set("LSCM")

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("3D files", "*.stl *.obj *.3ds"), ("All files", "*.*")])
        if filename:
            self.input_path.set(filename)
            base = os.path.splitext(filename)[0]
            self.output_path.set(base + ".dxf")
            self.load_model(filename)

    def load_model(self, filename):
        try:
            self.status_label.configure(text="Splitting mesh into components...", text_color="cyan")
            self.update()
            self.components = load_with_components(filename)

            comp_names = [f"Surface {i+1} (Area: {c.area:.1f})" for i, c in enumerate(self.components)]
            self.comp_menu.configure(values=comp_names)
            self.comp_menu.set(comp_names[0])
            self.selected_component_idx = 0

            self.preview_component(0)
            self.status_label.configure(text=f"Loaded {len(self.components)} surfaces.", text_color="white")
        except Exception as e:
            self.status_label.configure(text=f"Error loading: {str(e)}", text_color="red")

    def on_component_select(self, val):
        idx = self.comp_menu.cget("values").index(val)
        self.selected_component_idx = idx
        self.preview_component(idx)

    def preview_component(self, idx):
        if not self.components: return
        mesh = self.components[idx]
        vertices = mesh.vertices
        faces = mesh.faces

        self.ax.clear()
        self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces, cmap='viridis', edgecolor='grey', linewidth=0.1, alpha=0.8)
        self.ax.set_title(f"Surface Preview ({len(faces)} faces)", color='white')

        # Set equal aspect ratio
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        self.ax.set_axis_off()

        self.canvas.draw()

    def browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".dxf",
                                               filetypes=[("DXF files", "*.dxf"), ("SVG files", "*.svg"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)

    def run_flattening(self):
        if not self.components:
            self.status_label.configure(text="Error: No model loaded", text_color="red")
            return

        out = self.output_path.get()
        unit = self.unit_var.get()
        method = self.method_var.get()

        if not out:
            self.status_label.configure(text="Error: Set output path first", text_color="red")
            return

        try:
            self.status_label.configure(text=f"Flattening surface {self.selected_component_idx+1}...", text_color="cyan")
            self.update()

            # Save selected component to a temporary file or pass data directly
            # For simplicity, we'll temporarily save it as STL
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                temp_stl = tmp.name

            self.components[self.selected_component_idx].export(temp_stl)

            flatten_main(path_input=temp_stl, path_output=out, show_plot=False, input_unit=unit, method=method)

            # Clean up
            if os.path.exists(temp_stl): os.remove(temp_stl)

            self.status_label.configure(text=f"Success! Exported to {os.path.basename(out)}", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")

if __name__ == "__main__":
    app = FlattenApp()
    app.mainloop()
