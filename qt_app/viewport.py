from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Set, Tuple

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QVector3D, QVector4D
from PySide6.QtWidgets import QLabel, QMenu, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class ThreeDViewportWidget(gl.GLViewWidget):
    cameraChanged = Signal(float, float)
    facesSelected = Signal(list)
    modelDropped = Signal(str)
    flattenRequested = Signal()
    splitToggleRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ThreeDViewportWidget")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setBackgroundColor((43, 43, 43, 255))
        self.setStyleSheet("QOpenGLWidget#ThreeDViewportWidget { border: none; outline: none; background-color: #2b2b2b; }")

        self.vertices: np.ndarray | None = None
        self.vertices32: np.ndarray | None = None
        self.render_faces: np.ndarray | None = None
        self.render_faces32: np.ndarray | None = None
        self.pick_faces: np.ndarray | None = None
        self.mesh_name = ""
        self.selection_mode = "single"
        self.selected_faces: Set[int] = set()
        self._drag_start: Tuple[float, float] | None = None
        self._left_dragging = False
        self._face_normals: np.ndarray | None = None
        self._face_adjacency: List[List[int]] | None = None
        self._pick_cache_token = None
        self._camera_up: Tuple[float, float, float] | None = None
        self._nav_last: Tuple[float, float] | None = None
        self.is_rotating = False
        self.is_panning = False
        self.is_zooming = False
        self.orbit_mode_enabled = True
        self.pan_mode_enabled = False
        self.zoom_mode_enabled = False
        self._base_render_face_colors: np.ndarray | None = None
        self._pick_to_render: np.ndarray | None = None
        self._pick_tri_a: np.ndarray | None = None
        self._pick_e1: np.ndarray | None = None
        self._pick_e2: np.ndarray | None = None

        self.grid_item = gl.GLGridItem(color=(85, 85, 85, 110))
        self.grid_item.setSize(x=12000.0, y=12000.0, z=1.0)
        self.grid_item.setSpacing(10.0, 10.0, 1.0)
        self.addItem(self.grid_item)

        self.mesh_item = gl.GLMeshItem(drawFaces=True, drawEdges=False, smooth=False, shader="shaded")
        self.mesh_item.setGLOptions("opaque")
        self.addItem(self.mesh_item)

        self.selection_item = gl.GLMeshItem(drawFaces=True, drawEdges=False, smooth=False, shader="shaded", glOptions="translucent")
        self.selection_item.setVisible(False)
        self.addItem(self.selection_item)

        self.overlay = QLabel("Drop 3D Model Here\n(STL / OBJ / STEP)", self)
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setStyleSheet("QLabel { color:#8fa0b7; font-size:24px; font-weight:600; background:transparent; }")
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.dim_label = QLabel("", self)
        self.dim_label.setStyleSheet(
            "QLabel { color:#d5deec; background-color:rgba(20,22,28,170); border:1px solid #3a4250; padding:4px 8px; }"
        )
        self.dim_label.move(14, 14)
        self.dim_label.setMinimumSize(220, 28)
        self.dim_label.show()

        self._camera_debounce = QTimer(self)
        self._camera_debounce.setSingleShot(True)
        self._camera_debounce.setInterval(35)
        self._camera_debounce.timeout.connect(self._emit_camera_changed)

        self._build_hud()
        self.clear_view()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())
        self.dim_label.raise_()
        self._position_hud()
        self._update_grid_extent()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        self._nav_last = (float(pos.x()), float(pos.y()))
        self._drag_start = (float(pos.x()), float(pos.y()))
        self._left_dragging = False

        if event.button() == Qt.MouseButton.RightButton and self.orbit_mode_enabled:
            self.is_rotating = True
            self.is_panning = False
            self.is_zooming = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.is_rotating = False
            self.is_zooming = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.pan_mode_enabled:
            self.is_panning = True
            self.is_rotating = False
            self.is_zooming = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.zoom_mode_enabled:
            self.is_zooming = True
            self.is_rotating = False
            self.is_panning = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        cur = (float(event.position().x()), float(event.position().y()))
        if self._nav_last is None:
            self._nav_last = cur
        dx = cur[0] - self._nav_last[0]
        dy = cur[1] - self._nav_last[1]
        self._nav_last = cur

        if self.is_rotating:
            self.orbit(-dx, dy)
            self._on_camera_event()
            event.accept()
            return
        if self.is_panning:
            self.pan(dx, dy, 0.0, relative="view-upright")
            self._on_camera_event()
            event.accept()
            return
        if self.is_zooming:
            self._zoom_camera_by_delta(dy)
            self._on_camera_event()
            event.accept()
            return
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_start is not None:
            ddx = cur[0] - self._drag_start[0]
            ddy = cur[1] - self._drag_start[1]
            if (ddx * ddx + ddy * ddy) > 25.0:
                self._left_dragging = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton and self.is_rotating:
            self.is_rotating = False
            self._nav_last = None
            self._on_camera_event()
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton) and self.is_panning:
            self.is_panning = False
            self._nav_last = None
            self._on_camera_event()
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton) and self.is_zooming:
            self.is_zooming = False
            self._nav_last = None
            self._on_camera_event()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            try:
                if self.selection_mode == "off" or self.vertices is None or self.pick_faces is None or self._left_dragging:
                    event.accept()
                    return
                hit = self._raycast_face(float(event.position().x()), float(event.position().y()))
                if hit is None:
                    event.accept()
                    return
                self.selected_faces = {int(hit)} if self.selection_mode == "single" else self.smart_select(int(hit), 30.0)
                self._update_mesh_visuals()
                self.facesSelected.emit(self.get_selected_faces())
                event.accept()
                return
            finally:
                self._drag_start = None
                self._left_dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = int(event.angleDelta().x()) or int(event.angleDelta().y())
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.opts["fov"] = float(np.clip(self.opts["fov"] * (0.999 ** delta), 5.0, 120.0))
        else:
            self.opts["distance"] = float(np.clip(self.opts["distance"] * (0.999 ** delta), 1e-3, 1e12))
        self.update()
        self._on_camera_event()
        event.accept()

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
        if urls:
            self.modelDropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if self.vertices is None or self.is_rotating or self.is_panning or self.is_zooming:
            return
        menu = QMenu(self)
        flatten_action = menu.addAction("Flatten Selected")
        if menu.exec(event.globalPos()) == flatten_action:
            self.flattenRequested.emit()

    def _build_hud(self) -> None:
        self.hud = QWidget(self)
        self.hud.setObjectName("ViewportHUD")
        self.hud.setStyleSheet(
            """
            QWidget#ViewportHUD { background-color: rgba(0, 0, 0, 150); border-radius: 8px; }
            QToolButton { min-width:30px; min-height:30px; max-width:30px; max-height:30px; color:#f2f2f2; background-color:#2a2a2a; border:1px solid #565656; border-radius:4px; font-size:15px; }
            QToolButton:checked { border-color:#cfcfcf; background-color:#3c3c3c; }
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
        for w in (self.hud_fit_btn, self.hud_pan_btn, self.hud_zoom_btn, self.hud_orbit_btn, self.hud_split_btn):
            hud_layout.addWidget(w)
        self.hud.adjustSize()
        self.hud.show()
        self._position_hud()

    def _position_hud(self) -> None:
        margin = 16
        self.hud.move(max(0, self.width() - self.hud.width() - margin), max(0, self.height() - self.hud.height() - margin))
        self.hud.raise_()
        self.dim_label.raise_()

    def _toggle_pan_mode(self, enabled: bool) -> None:
        self.pan_mode_enabled = bool(enabled)
        if enabled:
            self.hud_orbit_btn.blockSignals(True)
            self.hud_orbit_btn.setChecked(False)
            self.hud_orbit_btn.blockSignals(False)
            self.hud_zoom_btn.blockSignals(True)
            self.hud_zoom_btn.setChecked(False)
            self.hud_zoom_btn.blockSignals(False)
            self.orbit_mode_enabled = False
            self.zoom_mode_enabled = False
        self.is_panning = False
        self._nav_last = None

    def _toggle_orbit_mode(self, enabled: bool) -> None:
        self.orbit_mode_enabled = bool(enabled)
        if enabled:
            self.hud_pan_btn.blockSignals(True)
            self.hud_pan_btn.setChecked(False)
            self.hud_pan_btn.blockSignals(False)
            self.hud_zoom_btn.blockSignals(True)
            self.hud_zoom_btn.setChecked(False)
            self.hud_zoom_btn.blockSignals(False)
            self.pan_mode_enabled = False
            self.zoom_mode_enabled = False
        self.is_rotating = False
        self._nav_last = None

    def _toggle_zoom_mode(self, enabled: bool) -> None:
        self.zoom_mode_enabled = bool(enabled)
        if enabled:
            self.hud_pan_btn.blockSignals(True)
            self.hud_pan_btn.setChecked(False)
            self.hud_pan_btn.blockSignals(False)
            self.hud_orbit_btn.blockSignals(True)
            self.hud_orbit_btn.setChecked(False)
            self.hud_orbit_btn.blockSignals(False)
            self.pan_mode_enabled = False
            self.orbit_mode_enabled = False
        self.is_zooming = False
        self._nav_last = None

    def _zoom_camera_by_delta(self, dy: float) -> None:
        self.opts["distance"] = float(np.clip(self.opts["distance"] * (1.0 + dy * 0.01), 1e-3, 1e12))
        self.update()

    def clear_view(self) -> None:
        self.vertices = None
        self.vertices32 = None
        self.render_faces = None
        self.render_faces32 = None
        self.pick_faces = None
        self.selected_faces.clear()
        self._face_normals = None
        self._face_adjacency = None
        self._pick_cache_token = None
        self._base_render_face_colors = None
        self._pick_to_render = None
        self._pick_tri_a = None
        self._pick_e1 = None
        self._pick_e2 = None
        self.mesh_item.setMeshData(vertexes=np.empty((0, 3), np.float32), faces=np.empty((0, 3), np.int32), shader="shaded", smooth=False)
        self.selection_item.setVisible(False)
        self.setCameraPosition(pos=QVector3D(0.0, 0.0, 0.0), distance=600.0, elevation=24.0, azimuth=-58.0)
        self.opts["fov"] = 45.0
        self._update_grid_extent()
        self.overlay.show()
        self.dim_label.setText("")
        self.update()

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
        self.vertices32 = np.ascontiguousarray(self.vertices.astype(np.float32, copy=False))
        self.render_faces = np.asarray(render_faces, dtype=np.int64)
        self.render_faces32 = np.ascontiguousarray(self.render_faces.astype(np.int32, copy=False))
        self.pick_faces = np.asarray(pick_faces, dtype=np.int64) if pick_faces is not None else self.render_faces
        self.selected_faces.clear()
        self._face_normals = None
        self._face_adjacency = None
        self._pick_cache_token = (len(self.vertices), len(self.pick_faces))
        self._base_render_face_colors = self._build_default_face_colors()
        self._build_pick_to_render_map()
        self._prepare_pick_raycast_cache()
        self._update_mesh_visuals()
        self._fit_camera_to_mesh()
        self._update_grid_extent()
        self.overlay.hide()
        lx, ly, lz = dims_mm
        self.dim_label.setText(f"{name} | LxWxH: {lx:.1f} x {ly:.1f} x {lz:.1f} mm")
        self._on_camera_event()

    def _fit_camera_to_mesh(self) -> None:
        if self.vertices is None:
            return
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        center = (mins + maxs) * 0.5
        span_xyz = maxs - mins
        span = float(max(np.max(span_xyz), np.linalg.norm(span_xyz), 1.0))
        self.setCameraPosition(pos=QVector3D(float(center[0]), float(center[1]), float(center[2])), distance=max(span * 1.8, 20.0), elevation=24.0, azimuth=-58.0)
        self.opts["fov"] = 40.0
        self.update()

    def reset_camera(self) -> None:
        self._fit_camera_to_mesh() if self.vertices is not None else self.clear_view()
        self._on_camera_event()

    def apply_view_preset(self, preset: str) -> None:
        if self.vertices is None:
            return
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        center = (mins + maxs) * 0.5
        dist = max(float(np.max(maxs - mins)) * 2.2, 20.0)
        mapping = {
            "FRONT": (0.0, -90.0, (0.0, 0.0, 1.0)),
            "BACK": (0.0, 90.0, (0.0, 0.0, 1.0)),
            "LEFT": (0.0, 180.0, (0.0, 0.0, 1.0)),
            "RIGHT": (0.0, 0.0, (0.0, 0.0, 1.0)),
            "TOP": (90.0, -90.0, (0.0, 1.0, 0.0)),
            "BOTTOM": (-90.0, -90.0, (0.0, -1.0, 0.0)),
        }
        if preset not in mapping:
            return
        elev, azim, up = mapping[preset]
        self.setCameraPosition(pos=QVector3D(float(center[0]), float(center[1]), float(center[2])), distance=dist, elevation=elev, azimuth=azim)
        self.opts["fov"] = 40.0
        self._camera_up = up
        self.update()
        self._on_camera_event()

    def set_selection_mode(self, mode: str) -> None:
        self.selection_mode = mode

    def clear_selection(self) -> None:
        if self.selected_faces:
            self.selected_faces.clear()
            self._update_mesh_visuals()
            self.facesSelected.emit([])

    def get_selected_faces(self) -> List[int]:
        return sorted(self.selected_faces)

    def _on_camera_event(self) -> None:
        self._camera_debounce.start()

    def _emit_camera_changed(self) -> None:
        center = self.opts["center"]
        cam = self.cameraPosition()
        vx = float(cam.x() - center.x())
        vy = float(cam.y() - center.y())
        vz = float(cam.z() - center.z())
        elev = float(math.degrees(math.atan2(vz, max(math.hypot(vx, vy), 1e-12))))
        azim = float(math.degrees(math.atan2(vy, vx)))
        self.cameraChanged.emit(elev, azim)

    def _update_grid_extent(self) -> None:
        if self.vertices is None:
            span = 3500.0
            cx = 0.0
            cy = 0.0
        else:
            mins = self.vertices.min(axis=0)
            maxs = self.vertices.max(axis=0)
            span = max(float(np.max(maxs - mins)) * 8.0, 800.0)
            cx = float((mins[0] + maxs[0]) * 0.5)
            cy = float((mins[1] + maxs[1]) * 0.5)
        aspect = max(1.0, float(self.width()) / max(1.0, float(self.height())))
        self.grid_item.resetTransform()
        self.grid_item.setSize(x=max(2000.0, min(200000.0, span * aspect * 2.0)), y=max(2000.0, min(200000.0, span * 2.0)), z=1.0)
        step = max(10.0, round((span / 80.0) / 10.0) * 10.0)
        self.grid_item.setSpacing(step, step, 1.0)
        self.grid_item.setColor((85, 85, 85, 110))
        self.grid_item.translate(cx, cy, 0.0)

    def _build_default_face_colors(self) -> np.ndarray:
        if self.vertices is None or self.render_faces is None or len(self.render_faces) == 0:
            return np.empty((0, 4), dtype=np.float32)
        c = self.vertices[self.render_faces].mean(axis=1)
        t = c[:, 2]
        t = (t - t.min()) / max(float(t.max() - t.min()), 1e-9)
        low = np.array([0.20, 0.24, 0.55], dtype=np.float32)
        high = np.array([0.98, 0.90, 0.12], dtype=np.float32)
        rgb = low[None, :] * (1.0 - t[:, None]) + high[None, :] * t[:, None]
        alpha = np.full((len(rgb), 1), 0.96, dtype=np.float32)
        return np.concatenate((rgb.astype(np.float32), alpha), axis=1)

    def _build_pick_to_render_map(self) -> None:
        if self.pick_faces is None or self.render_faces is None:
            self._pick_to_render = None
            return
        if len(self.pick_faces) == len(self.render_faces) and np.array_equal(self.pick_faces, self.render_faces):
            self._pick_to_render = np.arange(len(self.pick_faces), dtype=np.int64)
            return
        render_map = {tuple(sorted(map(int, f))): i for i, f in enumerate(self.render_faces)}
        self._pick_to_render = np.array([render_map.get(tuple(sorted(map(int, f))), -1) for f in self.pick_faces], dtype=np.int64)

    def _prepare_pick_raycast_cache(self) -> None:
        if self.vertices is None or self.pick_faces is None or len(self.pick_faces) == 0:
            self._pick_tri_a = None
            self._pick_e1 = None
            self._pick_e2 = None
            return
        tri = self.vertices[self.pick_faces]
        self._pick_tri_a = tri[:, 0, :]
        self._pick_e1 = tri[:, 1, :] - tri[:, 0, :]
        self._pick_e2 = tri[:, 2, :] - tri[:, 0, :]

    def _update_mesh_visuals(self) -> None:
        if self.vertices32 is None or self.render_faces32 is None:
            return
        colors = self._base_render_face_colors.copy() if self._base_render_face_colors is not None else self._build_default_face_colors()
        if self._pick_to_render is not None and self.selected_faces:
            idx = np.asarray(sorted(self.selected_faces), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(self._pick_to_render))]
            ridx = self._pick_to_render[idx]
            ridx = ridx[ridx >= 0]
            if len(ridx):
                colors[ridx] = np.array([1.0, 0.42, 0.05, 1.0], dtype=np.float32)
        self.mesh_item.setMeshData(vertexes=self.vertices32, faces=self.render_faces32, faceColors=colors, smooth=False, drawEdges=False, shader="shaded")
        if self.selected_faces and self.pick_faces is not None:
            idx = np.asarray(sorted(self.selected_faces), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(self.pick_faces))]
            if len(idx):
                sel_faces = np.ascontiguousarray(self.pick_faces[idx].astype(np.int32, copy=False))
                sel_colors = np.tile(np.array([1.0, 0.45, 0.0, 0.35], dtype=np.float32), (len(sel_faces), 1))
                self.selection_item.setMeshData(vertexes=self.vertices32, faces=sel_faces, faceColors=sel_colors, smooth=False, drawEdges=False, shader="shaded")
                self.selection_item.setVisible(True)
            else:
                self.selection_item.setVisible(False)
        else:
            self.selection_item.setVisible(False)
        self.update()

    def _screen_ray(self, sx: float, sy: float) -> Tuple[np.ndarray, np.ndarray] | None:
        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        proj = self.projectionMatrix((0, 0, w, h), self.getViewport())
        inv_mvp, ok = (proj * self.viewMatrix()).inverted()
        if not ok:
            return None
        x_ndc = (2.0 * sx / float(w)) - 1.0
        y_ndc = 1.0 - (2.0 * sy / float(h))
        near_h = inv_mvp.map(QVector4D(float(x_ndc), float(y_ndc), -1.0, 1.0))
        far_h = inv_mvp.map(QVector4D(float(x_ndc), float(y_ndc), 1.0, 1.0))
        if abs(float(near_h.w())) < 1e-12 or abs(float(far_h.w())) < 1e-12:
            return None
        near = np.array([near_h.x() / near_h.w(), near_h.y() / near_h.w(), near_h.z() / near_h.w()], dtype=np.float64)
        far = np.array([far_h.x() / far_h.w(), far_h.y() / far_h.w(), far_h.z() / far_h.w()], dtype=np.float64)
        d = far - near
        n = float(np.linalg.norm(d))
        return None if n < 1e-12 else (near, d / n)

    def _raycast_face(self, sx: float, sy: float) -> int | None:
        if self._pick_tri_a is None or self._pick_e1 is None or self._pick_e2 is None or self.pick_faces is None:
            return None
        ray = self._screen_ray(sx, sy)
        if ray is None:
            return None
        origin, direction = ray
        a = self._pick_tri_a
        e1 = self._pick_e1
        e2 = self._pick_e2
        eps = 1e-9
        pvec = np.cross(np.broadcast_to(direction, a.shape), e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        valid = np.abs(det) > eps
        if not np.any(valid):
            return None
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = origin - a
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)
        if not np.any(valid):
            return None
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,j->i", qvec, direction) * inv_det
        valid &= (v >= 0.0) & ((u + v) <= 1.0)
        if not np.any(valid):
            return None
        t = np.einsum("ij,ij->i", e2, qvec) * inv_det
        valid &= t > eps
        if not np.any(valid):
            return None
        cand = np.where(valid)[0]
        return int(cand[int(np.argmin(t[cand]))])

    def _ensure_selection_topology(self) -> None:
        if self.vertices is None or self.pick_faces is None:
            return
        if self._face_normals is not None and self._face_adjacency is not None and self._pick_cache_token == (len(self.vertices), len(self.pick_faces)):
            return
        tri = self.vertices[self.pick_faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lens = np.linalg.norm(normals, axis=1)
        lens = np.where(lens < 1e-12, 1e-12, lens)
        self._face_normals = normals / lens[:, None]
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi, (a, b, c) in enumerate(self.pick_faces):
            for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                e = (u, v) if u < v else (v, u)
                edge_to_faces.setdefault(e, []).append(fi)
        adj = [set() for _ in range(len(self.pick_faces))]
        for fs in edge_to_faces.values():
            if len(fs) < 2:
                continue
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    adj[fs[i]].add(fs[j])
                    adj[fs[j]].add(fs[i])
        self._face_adjacency = [sorted(list(a)) for a in adj]
        self._pick_cache_token = (len(self.vertices), len(self.pick_faces))

    def smart_select(self, seed_face: int, angle_limit_deg: float = 30.0) -> Set[int]:
        self._ensure_selection_topology()
        if self._face_normals is None or self._face_adjacency is None:
            return {seed_face}
        cos_thresh = math.cos(math.radians(angle_limit_deg))
        selected: Set[int] = set()
        visited: Set[int] = {int(seed_face)}
        q = deque([int(seed_face)])
        while q:
            face = q.popleft()
            selected.add(face)
            n0 = self._face_normals[face]
            for nb in self._face_adjacency[face]:
                if nb in visited:
                    continue
                visited.add(nb)
                if float(np.dot(n0, self._face_normals[nb])) >= cos_thresh:
                    q.append(nb)
        return selected
