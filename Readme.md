# Advanced Surface Flattener (Tent Manufacturing Edition)

This software is designed for tent cloth manufacturing, allowing users to flatten 3D surfaces into production-ready 2D DXF files with high precision.

## Features
- Dual UI support:
  - **Qt Workspace (default)** for professional CAD-style workflow.
  - **CustomTkinter fallback** for compatibility.
- 3D viewport upgrades (Qt):
  - Hardware-accelerated OpenGL setup (Core Profile 4.1 default format).
  - ViewCube with face-click navigation and explicit face pads.
  - Floating HUD controls: fit, pan, zoom-drag, orbit, split toggle.
  - Split-screen 3D/2D pattern preview with show/hide toggle.
- Multi-format import:
  - STL, OBJ
  - STEP (`.stp`, `.step`) with meshing fallback through `gmsh`.
- Surface selection:
  - Single pick and smart flood-fill (BFS) selection.
- Flattening methods:
  - **LSCM**: fast conformal mapping.
  - **ARAP**: better area preservation for production fabric workflows.
- Production workflow:
  - Seam allowance preview.
  - DXF export.
  - Nesting support.

## Requirements
- Python 3.10+
- Dependencies from `requirements.txt` (includes `PySide6`, `trimesh`, `libigl`, `gmsh`, etc.)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run
Default launcher (Qt, with fallback to Tkinter if Qt fails):
```bash
python main.py
```

Force Tkinter fallback:
```bash
set TENTMAKER_FORCE_TK=1
python main.py
```

Direct Tkinter launch:
```bash
python gui.py
```

## Qt Workflow Notes
1. Import model (`STL`, `OBJ`, `STP`, `STEP`).
2. Use ViewCube faces/pads to snap to top/front/right/etc.
3. Use HUD controls:
   - `⛶` fit view
   - `✋` pan drag
   - `⇵` zoom drag
   - `↻` orbit drag
   - `◫` show/hide 2D preview
4. Run flattening and export DXF.

## STEP Import Notes
- STEP import uses a direct load attempt first.
- If needed, it falls back to `gmsh` for surface meshing.
- If STEP import fails, verify:
  - `gmsh` is installed in the same Python environment.
  - The STEP file is valid and not empty/corrupt.

## Credits
This tool uses `libigl` and related geometry-processing libraries for flattening and production geometry operations.
