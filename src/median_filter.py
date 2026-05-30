from collections import deque
import numpy as np

class MedianFilter3D:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, point_3d):
        """
        Add a new 3D point (X, Y, Z) and return the smoothed median point.
        """
        if point_3d is None:
            return None
        self.buffer.append(point_3d)
        if len(self.buffer) < 2:
            return point_3d
        arr = np.array(self.buffer)
        return np.median(arr, axis=0)  # Median along each axis

    def reset(self):
        """
        Clear the buffer history.
        """
        self.buffer.clear()
