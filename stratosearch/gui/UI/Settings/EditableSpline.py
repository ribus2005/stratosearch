from PySide6.QtGui import QPen, QColor, QBrush, QPainterPath
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsEllipseItem, QGraphicsItem
from PySide6.QtCore import Qt, QPointF


def clamp_point(p, rect):
    x = min(max(p.x(), rect.left()), rect.right())
    y = min(max(p.y(), rect.top()), rect.bottom())
    return QPointF(x, y)


class EditableSpline:
    def __init__(self, scene, points, color, handle_ratio, image_scene_rect):
        self.scene = scene
        self.points = points  # список QPointF
        self.handle_ratio = handle_ratio
        self.image_scene_rect = image_scene_rect
        self.bounding_rect_path = QPainterPath()
        self.bounding_rect_path.addRect(image_scene_rect)
        self.dragging = False

        # --- заливка ---
        self.fill_item = QGraphicsPathItem()
        fill_color = QColor(color)
        fill_color.setAlpha(80)
        self.fill_item.setBrush(QBrush(fill_color))
        self.fill_item.setPen(Qt.NoPen)
        self.fill_item.setZValue(2)
        scene.addItem(self.fill_item)

        # --- контур ---
        self.path_item = QGraphicsPathItem()
        self.path_item.setPen(QPen(QColor("red"), 2))
        self.path_item.setZValue(3)
        scene.addItem(self.path_item)

        # --- контрольные точки ---
        self.handles = []
        for i, p in enumerate(points):
            h = SplineHandle(p.x(), p.y(), i, self)
            scene.addItem(h)
            self.handles.append(h)

        self.update_path()

    def update_spline_from_handles(self):
        self.points = [h.pos() for h in self.handles]
        self.update_path()

    def update_path(self):
        if len(self.points) < 3:
            return

        smooth_path = QPainterPath()
        smooth_path.moveTo(self.points[0])
        n = len(self.points)

        for i in range(n):
            p0 = self.points[i]
            p1 = self.points[(i + 1) % n]
            p_prev = self.points[i - 1]
            p_next = self.points[(i + 2) % n]

            ctrl1 = QPointF(
                p0.x() + self.handle_ratio * (p1.x() - p_prev.x()),
                p0.y() + self.handle_ratio * (p1.y() - p_prev.y())
            )
            ctrl2 = QPointF(
                p1.x() - self.handle_ratio * (p_next.x() - p0.x()),
                p1.y() - self.handle_ratio * (p_next.y() - p0.y())
            )

            smooth_path.cubicTo(ctrl1, ctrl2, p1)

        smooth_path.closeSubpath()

        clipped_path = smooth_path.intersected(self.bounding_rect_path)

        self.path_item.setPath(clipped_path)  # красивая граница
        self.fill_item.setPath(clipped_path)  # заливка строго по маске

    def remove(self):
        for h in self.handles:
            self.scene.removeItem(h)
        self.scene.removeItem(self.path_item)
        self.scene.removeItem(self.fill_item)


class SplineHandle(QGraphicsEllipseItem):
    def __init__(self, x, y, index, editor):
        r = 4
        super().__init__(-r, -r, 2*r, 2*r)

        self.setPos(x, y)
        self.setBrush(QColor("yellow"))
        self.setPen(QPen(Qt.black, 3))
        self.setZValue(3)

        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges |
            QGraphicsItem.ItemIsSelectable
        )

        self.index = index        # номер точки в сплайне
        self.editor = editor      # ссылка на менеджер сплайна

    def mousePressEvent(self, event):
        # Пользователь начал тянуть точку
        self.editor.dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Пользователь отпустил точку
        self.editor.dragging = False
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            rect = self.editor.image_scene_rect
            new_pos = value
            return clamp_point(new_pos, rect)

        if change == QGraphicsItem.ItemPositionHasChanged:
            self.editor.update_spline_from_handles()

        return super().itemChange(change, value)