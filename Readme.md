# Advanced Surface Flattener (Tent Manufacturing Edition)

This software is designed for tent cloth manufacturing, allowing users to flatten 3D surfaces into production-ready 2D DXF files with high precision.

## Features
- **Windows Native UI**: Modern interface built with CustomTkinter.
- **Multi-format Support**: Import STL, OBJ, 3DS, and other 3D formats.
- **Interactive 3D Preview**: Rotate and inspect models before flattening.
- **Surface Selection**: Automatically split complex models into components and select the specific surface to flatten.
- **Precision Flattening**:
    - **LSCM**: Fast conformal mapping.
    - **ARAP (As-Rigid-As-Possible)**: Minimizes material stretch and preserves area (ideal for tent cloth).
- **Correct Scaling**: Export in mm with unit conversion (cm, m, inch supported).
- **Production Export**: High-quality DXF and SVG outputs.

## Requirements
- Python 3.10+
- libigl
- trimesh
- customtkinter
- ezdxf
- numpy, scipy, matplotlib, svgwrite

## Usage
Run the GUI:
```bash
python gui.py
```

1. **Load 3D Model**: Select your STL/OBJ file.
2. **Select Surface**: If the model has multiple parts, choose the one you want to flatten.
3. **Input Units**: Specify the units of the input file (e.g., if it was designed in cm).
4. **Method**: Choose "ARAP" for best area preservation.
5. **Export**: Set the output path and click "GENERATE PRODUCTION DXF".

## Credits
This tool uses the `libigl` library for advanced geometry processing.
