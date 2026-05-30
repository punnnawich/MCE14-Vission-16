import numpy as np

class ReleaseDetector:
    def __init__(self, vel_threshold=0.5):
        """
        vel_threshold: Minimum velocity in meters/second to register a release.
        """
        self.prev_pos = None
        self.prev_time = None
        self.released = False
        self.vel_threshold = vel_threshold

    def update(self, pos_3d, timestamp):
        """
        pos_3d: np.array([x, y, z]) representing the 3D coordinates in meters.
        timestamp: float in seconds.
        """
        if self.released:
            return True

        if self.prev_pos is None:
            self.prev_pos = pos_3d
            self.prev_time = timestamp
            return False

        dt = timestamp - self.prev_time
        if dt <= 0:
            return self.released

        # Calculate 3D velocity norm
        velocity = np.linalg.norm(pos_3d - self.prev_pos) / dt

        self.prev_pos = pos_3d
        self.prev_time = timestamp

        if velocity > self.vel_threshold:
            self.released = True

        return self.released

    def reset(self):
        """
        Reset release detection status.
        """
        self.released = False
        self.prev_pos = None
        self.prev_time = None
