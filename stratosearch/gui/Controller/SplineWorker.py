import cv2
from PySide6.QtCore import QObject, Signal, Slot, QPointF

from stratosearch.calculate.calculate import extract_connected_region


class SplineWorker(QObject):
    finished = Signal(object, object)  # points, class_id
    error = Signal(str)

    def __init__(self, mask_array, ix, iy, epsilon=2.0):
        super().__init__()
        self.mask_array = mask_array
        self.ix = ix
        self.iy = iy
        self.epsilon = epsilon

    @Slot()
    def run(self):
        try:
            class_id = self.mask_array[self.iy, self.ix]

            # Получаем бинарную маску для связной области
            region_mask = extract_connected_region(self.mask_array, self.ix, self.iy)

            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return None

            # Выбираем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)

            # Аппроксимация
            approx = cv2.approxPolyDP(largest_contour, self.epsilon, closed=True)

            points = [QPointF(p[0][0], p[0][1]) for p in approx]  # перевод в QPointF
            if len(points) == 0:
                points = None

            self.finished.emit(points, class_id)
        except Exception as e:
            self.error.emit(str(e))
