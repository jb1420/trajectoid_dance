# Trajectoids Curve-to-3D App

Python desktop app to draw a 2D trajectory, generate a 3D rolling body using the Trajectoids algorithm logic, and inspect the result in an interactive 360-degree 3D viewer.

## What It Includes

- 2D curve editor with:
  - freehand tool (optional smoothing-on-draw)
  - Bezier-style spline control-point tool
  - polyline tool
  - eraser (delete control points)
  - select/move control points
  - undo/redo
  - smooth / uniform resample
  - scale / rotate / translate
  - curve length and curvature plot
- 3D viewport with:
  - orbit/rotate, zoom, pan (mouse controls)
  - reset view
  - fit-to-screen
  - wireframe or shaded mode
  - simulation playback with final trajectory overlay
  - OpenGL/GPU rendering path (with Matplotlib fallback)
- Trajectoid generation:
  - adapted from `compute_trajectoid.py` rotation/orientation logic in the upstream Trajectoids repo
  - automatic mesh construction (no manual 3ds Max step)
  - manual binary STL export of the latest full-resolution generated mesh
  - failure messages with suggested fixes

## Project Files

- `app.py`: main Qt app
- `curve_editor.py`: 2D drawing/editing canvas
- `trajectoids_adapter.py`: Trajectoids algorithm adapter + mesh generation
- `requirements.txt`: Python dependencies

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

## Notes
- Algorithm adapted from: Sobolev, Y. I., Dong, R., Tlusty, T., Eckmann, J.-P., Granick, S., & Grzybowski, B. A. (2023). [*Solid-body trajectoids shaped to roll along desired pathways*](https://doi.org/10.1038/s41586-023-06306-y). *Nature*, 620(7973), 310–315.
- Reference implementation: [yaroslavsobolev/trajectoids](https://github.com/yaroslavsobolev/trajectoids)
- The app uses the core Trajectoids rolling-rotation path mapping logic to derive cut constraints from your trajectory.
- The embedded 3D viewport uses `pyqtgraph` + OpenGL for GPU-accelerated interaction.
- If OpenGL dependencies are unavailable, it falls back to a Matplotlib CPU viewer.
- After generating a mesh, click `Simulate` to run an 8-second rolling animation at a fixed 45-degree camera.
- Click `Export STL` after generation to save the full generated mesh as a binary `.stl` file.
- Simulation stops when one full trajectory has been displayed and leaves the final trail visible.
- If generation fails, use:
  - `Smooth`
  - `Resample`
  - endpoint closure (if closed mode is enabled)
  - lower `Grid` value for faster iteration
