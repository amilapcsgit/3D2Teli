# Tent-Maker Pro (Qt + OpenGL)

Tent-Maker Pro is a CAD-oriented toolchain for tent-cloth workflows:
import 3D geometry, inspect/select surfaces, flatten to 2D, and export production DXF.

This branch (`opengl`) runs a hardware-accelerated viewport based on `pyqtgraph.opengl`.

## Current Progress (OpenGL Branch)

### Completed
- Qt ribbon workspace with Setup / Flatten / Production flow.
- OpenGL 3D viewport (`GLViewWidget`) with:
  - GPU rendering and shaded mesh display.
  - ViewCube face-click camera presets.
  - Floating HUD controls.
  - Split-screen 3D + 2D preview toggle.
  - Smart face selection (single + flood fill).
- STEP import path with fallback meshing through `gmsh`.
- Flattening workflow (LSCM / ARAP) + DXF export.
- Nesting pipeline integration.

### In Active Refinement
- Initial color fidelity for meshes that do not provide reliable embedded colors.
- Viewport lighting/material tuning for consistent "CAD-like" depth on first frame.

## Exact Technologies Used

### UI and Interaction
- `PySide6` (Qt widgets, signals/slots, threading, event handling)
- `qt-material` (styling support where applicable)

### 3D Viewport
- `pyqtgraph` (`pyqtgraph.opengl.GLViewWidget`, `GLMeshItem`, `GLGridItem`)
- `PyOpenGL` (lighting/material state, headlamp behavior)
- `numpy` (mesh/color arrays, ray math)

### Mesh IO and Geometry
- `trimesh` (mesh loading, normals, topology helpers, color visuals)
- `gmsh` (STEP `.stp/.step` tessellation fallback)
- `libigl` + internal flatten modules (surface flattening)

### 2D/Export/Nesting
- `ezdxf` (DXF read/write)
- `shapely` (2D geometry ops)
- `networkx` / `scipy` (graph + numeric helpers)
- `svgwrite` (supporting output tooling)

## Runtime Architecture (Qt Path)

1. `main.py` configures OpenGL surface format.
2. `qt_app/main_window.py` starts Qt app and creates the ribbon workspace.
3. `qt_app/ribbon_window.py`:
   - Runs model loading in a worker thread.
   - Emits payload to main UI thread.
   - Sends `vertices`, `faces`, preview faces, and available colors to viewport.
4. `qt_app/viewport.py`:
   - Builds mesh items.
   - Applies camera/selection/lighting logic.
   - Keeps ViewCube + HUD interactions in sync.

## Key Functions (Current Core)

### Model Loading / Handoff
- `qt_app/ribbon_window.py:74` `Worker._run_load_model`
- `qt_app/ribbon_window.py:801` `_apply_loaded_model`

### Viewport Rendering / Camera
- `qt_app/viewport.py:174` `initializeGL`
- `qt_app/viewport.py:193` `paintGL`
- `qt_app/viewport.py:681` `set_mesh`
- `qt_app/viewport.py:729` `update_model`
- `qt_app/viewport.py:777` `apply_view_preset`

### Viewport Selection
- `qt_app/viewport.py:234` `mousePressEvent`
- `qt_app/viewport.py:277` `mouseMoveEvent`
- `qt_app/viewport.py:1332` `smart_select`

### Flatten / Export
- `qt_app/ribbon_window.py:855` `run_flatten`
- `qt_app/ribbon_window.py:929` `export_dxf`

## 3D Viewport Controls (Current)

### Navigation
- `Alt + Middle Mouse Drag`: orbit
- `Middle Mouse Drag`: pan
- `Ctrl + Alt + Middle Mouse Drag`: smooth zoom drag
- `Mouse Wheel`: zoom step

### Selection
- `Left Click`: select (replace)
- `Ctrl + Left Click`: add to selection
- `Alt + Left Click`: remove from selection
- Hover with no buttons: preview highlight

### Shortcuts
- `Z`: frame selected region
- `Alt + W`: toggle 2D preview split

### Context Menu
- `Right Click`: Clear Selection / Invert Selection / Isolate / Flatten Selected

## Supported Import Formats
- STL (`.stl`)
- OBJ (`.obj`)
- STEP (`.stp`, `.step`)

## Run

Default launcher:

```bash
python main.py
```

Force Tkinter fallback:

```bash
set TENTMAKER_FORCE_TK=1
python main.py
```

Direct Tkinter app:

```bash
python gui.py
```

## Install

```bash
pip install -r requirements.txt
```

