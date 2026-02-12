from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Set, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QMenu, QToolButton, QVBoxLayout, QWidget


class ThreeDViewportWidget(QWidget):
    cameraChanged = Signal(float, float)
    facesSelected = Signal(list)
    modelDropped = Signal(str)
    flattenRequested = Signal()
    splitToggleRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color: #252525; border: none;")

        self.vertices: np.ndarray | None = None
        self.render_faces: np.ndarray | None = None
        self.pick_faces: np.ndarray | None = None
        self.mesh_name = ""
        self.selection_mode = "single"
        self.selected_faces: Set[int] = set()
        self._drag_start: Tuple[float, float] | None = None
        self._face_normals: np.ndarray | None = None
        self._face_adjacency: List[List[int]] | None = None
        self._pick_cache_token = None
        self._camera_position: Tuple[float, float, float] | None = None
        self._camera_up: Tuple[float, float, float] | None = None
        self.is_rotating = False
        self.is_panning = False
        self.is_zooming = False
        self._nav_last: Tuple[float, float] | None = None
        self.orbit_mode_enabled = True
        self.pan_mode_enabled = False
        self.zoom_mode_enabled = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.fig = Figure(figsize=(7, 5), facecolor="#252525")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color: #252525; border: none;")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#252525")
        self.ax.set_proj_type("ortho")
        self.ax.set_axis_off()
        layout.addWidget(self.canvas, 1)

        self.overlay = QLabel("Drop 3D Model Here\n(STL / OBJ / STEP)", self)
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setStyleSheet(
            "QLabel { color:#8fa0b7; font-size:24px; font-weight:600; background:transparent; }"
        )
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.overlay, 0, Qt.AlignmentFlag.AlignCenter)

        self.dim_label = QLabel("", self)
        self.dim_label.setStyleSheet(
            "QLabel { color:#d5deec; background-color:rgba(20,22,28,170); border:1px solid #3a4250; padding:4px 8px; }"
        )
        self.dim_label.move(14, 14)
        self.dim_label.resize(400, 28)
        self.dim_label.show()

        self._camera_debounce = QTimer(self)
        self._camera_debounce.setSingleShot(True)
        self._camera_debounce.setInterval(35)
        self._camera_debounce.timeout.connect(self._emit_camera_changed)

        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.canvas.mpl_connect("scroll_event", self._on_camera_event)
        self.canvas.mpl_connect("draw_event", self._on_camera_event)
        self._build_hud()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())
        self.dim_label.raise_()
        self._position_hud()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.is_rotating or self.is_panning or self.is_zooming:
            return
        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                p = urls[0].toLocalFile().lower()
                if p.endswith(".stl") or p.endswith(".obj") or p.endswith(".stp") or p.endswith(".step"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        self.modelDropped.emit(path)
        event.acceptProposedAction()

    def _build_hud(self) -> None:
        self.hud = QWidget(self)
        self.hud.setObjectName("ViewportHUD")
        self.hud.setStyleSheet(
            """
            QWidget#ViewportHUD {
                background-color: rgba(0, 0, 0, 150);
                border-radius: 8px;
            }
            QToolButton {
                min-width: 30px;
                min-height: 30px;
                max-width: 30px;
                max-height: 30px;
                color: #f2f2f2;
                background-color: #2a2a2a;
                border: 1px solid #565656;
                border-radius: 4px;
                font-size: 15px;
            }
            QToolButton:checked {
                border-color: #cfcfcf;
                background-color: #3c3c3c;
            }
            """
        )
        hud_layout = QVBoxLayout(self.hud)
        hud_layout.setContentsMargins(8, 8, 8, 8)
        hud_layout.setSpacing(6)

        self.hud_fit_btn = QToolButton(self.hud)
        self.hud_fit_btn.setText("\u26F6")
        self.hud_fit_btn.setToolTip("Zoom Fit")
        self.hud_fit_btn.clicked.connect(self.reset_camera)

        self.hud_pan_btn = QToolButton(self.hud)
        self.hud_pan_btn.setText("\u270B")
        self.hud_pan_btn.setToolTip("Pan Mode")
        self.hud_pan_btn.setCheckable(True)
        self.hud_pan_btn.toggled.connect(self._toggle_pan_mode)

        self.hud_zoom_btn = QToolButton(self.hud)
        self.hud_zoom_btn.setText("\u21F5")
        self.hud_zoom_btn.setToolTip("Zoom Drag Mode")
        self.hud_zoom_btn.setCheckable(True)
        self.hud_zoom_btn.toggled.connect(self._toggle_zoom_mode)

        self.hud_orbit_btn = QToolButton(self.hud)
        self.hud_orbit_btn.setText("\u21BB")
        self.hud_orbit_btn.setToolTip("Orbit Mode")
        self.hud_orbit_btn.setCheckable(True)
        self.hud_orbit_btn.setChecked(True)
        self.hud_orbit_btn.toggled.connect(self._toggle_orbit_mode)

        self.hud_split_btn = QToolButton(self.hud)
        self.hud_split_btn.setText("\u25EB")
        self.hud_split_btn.setToolTip("Toggle 2D Preview")
        self.hud_split_btn.clicked.connect(self.splitToggleRequested.emit)

        hud_layout.addWidget(self.hud_fit_btn)
        hud_layout.addWidget(self.hud_pan_btn)
        hud_layout.addWidget(self.hud_zoom_btn)
        hud_layout.addWidget(self.hud_orbit_btn)
        hud_layout.addWidget(self.hud_split_btn)
        self.hud.adjustSize()
        self.hud.show()
        self._position_hud()

    def _position_hud(self) -> None:
        margin = 16
        x = max(0, self.width() - self.hud.width() - margin)
        y = max(0, self.height() - self.hud.height() - margin)
        self.hud.move(x, y)
        self.hud.raise_()
        self.dim_label.raise_()

    def _toggle_pan_mode(self, enabled: bool) -> None:
        self.pan_mode_enabled = bool(enabled)
        if self.pan_mode_enabled and self.hud_orbit_btn.isChecked():
            self.hud_orbit_btn.blockSignals(True)
            self.hud_orbit_btn.setChecked(False)
            self.hud_orbit_btn.blockSignals(False)
            self.orbit_mode_enabled = False
        if self.pan_mode_enabled and self.hud_zoom_btn.isChecked():
            self.hud_zoom_btn.blockSignals(True)
            self.hud_zoom_btn.setChecked(False)
            self.hud_zoom_btn.blockSignals(False)
            self.zoom_mode_enabled = False
        self.is_panning = False
        self._nav_last = None

    def _toggle_orbit_mode(self, enabled: bool) -> None:
        self.orbit_mode_enabled = bool(enabled)
        if self.orbit_mode_enabled and self.hud_pan_btn.isChecked():
            self.hud_pan_btn.blockSignals(True)
            self.hud_pan_btn.setChecked(False)
            self.hud_pan_btn.blockSignals(False)
            self.pan_mode_enabled = False
        if self.orbit_mode_enabled and self.hud_zoom_btn.isChecked():
            self.hud_zoom_btn.blockSignals(True)
            self.hud_zoom_btn.setChecked(False)
            self.hud_zoom_btn.blockSignals(False)
            self.zoom_mode_enabled = False
        self.is_rotating = False
        self._nav_last = None

    def _toggle_zoom_mode(self, enabled: bool) -> None:
        self.zoom_mode_enabled = bool(enabled)
        if self.zoom_mode_enabled and self.hud_pan_btn.isChecked():
            self.hud_pan_btn.blockSignals(True)
            self.hud_pan_btn.setChecked(False)
            self.hud_pan_btn.blockSignals(False)
            self.pan_mode_enabled = False
        if self.zoom_mode_enabled and self.hud_orbit_btn.isChecked():
            self.hud_orbit_btn.blockSignals(True)
            self.hud_orbit_btn.setChecked(False)
            self.hud_orbit_btn.blockSignals(False)
            self.orbit_mode_enabled = False
        self.is_zooming = False
        self._nav_last = None

    def _pan_camera_by_delta(self, dx: float, dy: float) -> None:
        x0, x1 = self.ax.get_xlim3d()
        y0, y1 = self.ax.get_ylim3d()
        z0, z1 = self.ax.get_zlim3d()
        w = max(1.0, float(self.canvas.width()))
        h = max(1.0, float(self.canvas.height()))
        sx = -(dx / w) * (x1 - x0)
        sy = (dy / h) * (y1 - y0)
        self.ax.set_xlim3d(x0 + sx, x1 + sx)
        self.ax.set_ylim3d(y0 + sy, y1 + sy)
        self.ax.set_zlim3d(z0, z1)

    def _orbit_camera_by_delta(self, dx: float, dy: float) -> None:
        elev = float(self.ax.elev) - dy * 0.28
        azim = float(self.ax.azim) - dx * 0.32
        roll = float(getattr(self.ax, "roll", 0.0))
        self._set_camera_view(elev=elev, azim=azim, roll=roll, emit_signal=False)

    def _zoom_camera_by_delta(self, dy: float) -> None:
        factor = 1.0 + (float(dy) * 0.01)
        factor = max(0.2, min(5.0, factor))
        for getter, setter in (
            (self.ax.get_xlim3d, self.ax.set_xlim3d),
            (self.ax.get_ylim3d, self.ax.set_ylim3d),
            (self.ax.get_zlim3d, self.ax.set_zlim3d),
        ):
            lo, hi = getter()
            mid = (lo + hi) * 0.5
            half = max(1e-9, (hi - lo) * 0.5 * factor)
            setter(mid - half, mid + half)

    def clear_view(self) -> None:
        self.vertices = None
        self.render_faces = None
        self.pick_faces = None
        self.selected_faces.clear()
        self._face_normals = None
        self._face_adjacency = None
        self._pick_cache_token = None
        self._camera_position = None
        self._camera_up = None
        self.is_rotating = False
        self.is_panning = False
        self.is_zooming = False
        self._nav_last = None
        self.ax.clear()
        self.ax.set_facecolor("#252525")
        self.ax.set_proj_type("ortho")
        self.ax.set_axis_off()
        self.canvas.draw_idle()
        self.overlay.show()
        self.dim_label.setText("")

    def set_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        render_faces: np.ndarray,
        dims_mm: Tuple[float, float, float],
        pick_faces: np.ndarray | None = None,
    ) -> None:
        self.mesh_name = name
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.render_faces = np.asarray(render_faces, dtype=np.int64)
        self.pick_faces = np.asarray(pick_faces, dtype=np.int64) if pick_faces is not None else self.render_faces
        self.selected_faces.clear()
        self._face_normals = None
        self._face_adjacency = None
        self._pick_cache_token = (len(self.vertices), len(self.pick_faces))
        lx, ly, lz = dims_mm
        self.dim_label.setText(f"{name} | LxWxH: {lx:.1f} x {ly:.1f} x {lz:.1f} mm")
        self._redraw_mesh(reset_camera=True)

    def reset_camera(self) -> None:
        if self.vertices is None:
            return
        self._camera_position = None
        self._camera_up = None
        self._set_camera_view(elev=24.0, azim=-58.0, roll=0.0)

    def apply_view_preset(self, preset: str) -> None:
        if self.vertices is None:
            return
        bounds = self.vertices.max(axis=0) - self.vertices.min(axis=0)
        dist = max(float(np.max(bounds)), 1.0) * 1.8

        presets = {
            "FRONT": ((0.0, -dist, 0.0), (0.0, 0.0, 1.0), 0.0, -90.0, 0.0),
            "BACK": ((0.0, dist, 0.0), (0.0, 0.0, 1.0), 0.0, 90.0, 0.0),
            "LEFT": ((-dist, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, 180.0, 0.0),
            "RIGHT": ((dist, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, 0.0, 0.0),
            "TOP": ((0.0, 0.0, dist), (0.0, 1.0, 0.0), 90.0, -90.0, 0.0),
            "BOTTOM": ((0.0, 0.0, -dist), (0.0, -1.0, 0.0), -90.0, -90.0, 180.0),
        }
        if preset not in presets:
            return

        pos, up, elev, azim, roll = presets[preset]
        # Equivalent to camera.setParams: keep explicit position + up-vector state.
        self._camera_position = (float(pos[0]), float(pos[1]), float(pos[2]))
        self._camera_up = (float(up[0]), float(up[1]), float(up[2]))
        self._set_camera_view(elev=elev, azim=azim, roll=roll)

    def _set_camera_view(self, elev: float, azim: float, roll: float = 0.0, emit_signal: bool = True) -> None:
        try:
            self.ax.view_init(elev=elev, azim=azim, roll=roll)
        except TypeError:
            self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
        if emit_signal:
            self._emit_camera_changed()

    def set_selection_mode(self, mode: str) -> None:
        self.selection_mode = mode

    def clear_selection(self) -> None:
        if not self.selected_faces:
            return
        self.selected_faces.clear()
        self._redraw_mesh(reset_camera=False)
        self.facesSelected.emit([])

    def get_selected_faces(self) -> List[int]:
        return sorted(self.selected_faces)

    def _on_camera_event(self, _event) -> None:
        if self.vertices is None:
            return
        if self.is_rotating or self.is_panning or self.is_zooming:
            return
        self._camera_debounce.start()

    def _on_mouse_motion(self, event) -> None:
        if self.vertices is None or event.inaxes != self.ax:
            return
        if self.is_rotating and self._nav_last is not None:
            dx = float(event.x) - self._nav_last[0]
            dy = float(event.y) - self._nav_last[1]
            self._nav_last = (float(event.x), float(event.y))
            self._orbit_camera_by_delta(dx, dy)
            return
        if self.is_panning and self._nav_last is not None:
            dx = float(event.x) - self._nav_last[0]
            dy = float(event.y) - self._nav_last[1]
            self._nav_last = (float(event.x), float(event.y))
            self._pan_camera_by_delta(dx, dy)
            self.canvas.draw_idle()
            return
        if self.is_zooming and self._nav_last is not None:
            dy = float(event.y) - self._nav_last[1]
            self._nav_last = (float(event.x), float(event.y))
            self._zoom_camera_by_delta(dy)
            self.canvas.draw_idle()
            return
        # Avoid hover picking/raycast work while any navigation drag is active.
        if self.is_rotating or self.is_panning or self.is_zooming:
            return
        buttons = getattr(event, "buttons", None)
        if buttons not in (None, 0):
            return
        self._camera_debounce.start()

    def _on_mouse_press(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if self.orbit_mode_enabled and event.button == 3:
            self.is_rotating = True
            self.is_panning = False
            self._nav_last = (float(event.x), float(event.y))
            self._drag_start = None
            return
        if self.pan_mode_enabled and event.button in (1, 2):
            self.is_panning = True
            self.is_rotating = False
            self._nav_last = (float(event.x), float(event.y))
            self._drag_start = None
            return
        if self.zoom_mode_enabled and event.button in (1, 2):
            self.is_zooming = True
            self.is_rotating = False
            self.is_panning = False
            self._nav_last = (float(event.x), float(event.y))
            self._drag_start = None
            return
        self._drag_start = (float(event.x), float(event.y))

    def _on_mouse_release(self, event) -> None:
        if self.vertices is None or self.pick_faces is None or event.inaxes != self.ax:
            return
        if event.button == 3 and self.is_rotating:
            self.is_rotating = False
            self._nav_last = None
            self._emit_camera_changed()
            return
        if event.button in (1, 2) and self.is_panning:
            self.is_panning = False
            self._nav_last = None
            self._emit_camera_changed()
            return
        if event.button in (1, 2) and self.is_zooming:
            self.is_zooming = False
            self._nav_last = None
            self._emit_camera_changed()
            return
        self._on_camera_event(event)
        if event.button != 1:
            return
        if self.selection_mode == "off":
            return
        if self._drag_start is not None:
            dx = float(event.x) - self._drag_start[0]
            dy = float(event.y) - self._drag_start[1]
            if (dx * dx + dy * dy) > 25.0:
                return
        hit = self._raycast_face(float(event.x), float(event.y))
        if hit is None:
            return
        if self.selection_mode == "single":
            self.selected_faces = {int(hit)}
        elif self.selection_mode == "smart":
            self.selected_faces = self.smart_select(int(hit), 30.0)
        self._redraw_mesh(reset_camera=False)
        self.facesSelected.emit(self.get_selected_faces())

    def _emit_camera_changed(self) -> None:
        self.cameraChanged.emit(float(self.ax.elev), float(self.ax.azim))

    def _redraw_mesh(self, reset_camera: bool = False) -> None:
        if self.vertices is None or self.render_faces is None:
            return
        old_view = (float(self.ax.elev), float(self.ax.azim))
        old_roll = float(getattr(self.ax, "roll", 0.0))
        self.ax.clear()
        self.ax.set_facecolor("#252525")
        self.ax.set_proj_type("ortho")
        self.ax.set_axis_off()
        self.overlay.hide()

        self.ax.plot_trisurf(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.vertices[:, 2],
            triangles=self.render_faces,
            cmap="viridis",
            linewidth=0.05,
            edgecolor="none",
            antialiased=True,
            shade=True,
            alpha=0.95,
        )

        if self.selected_faces and self.pick_faces is not None:
            idx = np.asarray(sorted(self.selected_faces), dtype=np.int64)
            sel_tris = self.vertices[self.pick_faces[idx]]
            col = Poly3DCollection(
                sel_tris,
                facecolors=(1.0, 0.36, 0.0, 0.45),
                edgecolors=(1.0, 0.68, 0.3, 0.8),
                linewidths=0.15,
            )
            self.ax.add_collection3d(col)

        self._set_equal_axes(self.vertices)
        if reset_camera:
            self._camera_position = None
            self._camera_up = None
            self._set_camera_view(elev=24.0, azim=-58.0, roll=0.0)
        else:
            self._set_camera_view(elev=old_view[0], azim=old_view[1], roll=old_roll)

    def _set_equal_axes(self, vertices: np.ndarray) -> None:
        max_range = np.array(
            [
                vertices[:, 0].max() - vertices[:, 0].min(),
                vertices[:, 1].max() - vertices[:, 1].min(),
                vertices[:, 2].max() - vertices[:, 2].min(),
            ]
        ).max() / 2.0
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def _raycast_face(self, screen_x: float, screen_y: float) -> int | None:
        if self.vertices is None or self.pick_faces is None or len(self.pick_faces) == 0:
            return None

        x2, y2, z2 = proj3d.proj_transform(
            self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], self.ax.get_proj()
        )
        disp = self.ax.transData.transform(np.column_stack([x2, y2]))

        tri = disp[self.pick_faces]  # (F,3,2)
        bbox_min = tri.min(axis=1)
        bbox_max = tri.max(axis=1)
        candidates = np.where(
            (screen_x >= bbox_min[:, 0]) & (screen_x <= bbox_max[:, 0]) & (screen_y >= bbox_min[:, 1]) & (screen_y <= bbox_max[:, 1])
        )[0]
        if len(candidates) == 0:
            return None

        p = np.array([screen_x, screen_y], dtype=np.float64)
        t = tri[candidates]
        a = t[:, 0, :]
        b = t[:, 1, :]
        c = t[:, 2, :]
        v0 = b - a
        v1 = c - a
        v2 = p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        inside = (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)
        hits = candidates[inside]
        if len(hits) == 0:
            return None

        ztri = z2[self.pick_faces[hits]]
        depths = ztri.mean(axis=1)
        return int(hits[int(np.argmax(depths))])

    def _ensure_selection_topology(self) -> None:
        if self.vertices is None or self.pick_faces is None:
            return
        if self._face_normals is not None and self._face_adjacency is not None and self._pick_cache_token == (
            len(self.vertices),
            len(self.pick_faces),
        ):
            return

        faces = self.pick_faces
        tri = self.vertices[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lens = np.linalg.norm(normals, axis=1)
        lens = np.where(lens < 1e-12, 1e-12, lens)
        normals = normals / lens[:, None]

        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            edges = [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]
            for u, v in edges:
                e = (u, v) if u < v else (v, u)
                edge_to_faces.setdefault(e, []).append(fi)

        adjacency = [set() for _ in range(len(faces))]
        for flist in edge_to_faces.values():
            if len(flist) < 2:
                continue
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    a = flist[i]
                    b = flist[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        self._face_normals = normals
        self._face_adjacency = [sorted(list(n)) for n in adjacency]
        self._pick_cache_token = (len(self.vertices), len(self.pick_faces))

    def smart_select(self, seed_face: int, angle_limit_deg: float = 30.0) -> Set[int]:
        self._ensure_selection_topology()
        if self._face_normals is None or self._face_adjacency is None:
            return {seed_face}
        cos_thresh = math.cos(math.radians(angle_limit_deg))
        selected: Set[int] = set()
        visited: Set[int] = set()
        q = deque([int(seed_face)])
        visited.add(int(seed_face))
        while q:
            face = q.popleft()
            selected.add(face)
            n0 = self._face_normals[face]
            for nb in self._face_adjacency[face]:
                if nb in visited:
                    continue
                visited.add(nb)
                n1 = self._face_normals[nb]
                if float(np.dot(n0, n1)) >= cos_thresh:
                    q.append(nb)
        return selected

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if self.vertices is None or self.is_rotating or self.is_panning or self.is_zooming:
            return
        menu = QMenu(self)
        flatten_action = menu.addAction("Flatten Selected")
        chosen = menu.exec(event.globalPos())
        if chosen == flatten_action:
            self.flattenRequested.emit()

