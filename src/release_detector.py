import numpy as np

class ReleaseDetector:
    def __init__(self, vel_threshold=1.5, displacement_threshold=0.15):
        """
        vel_threshold: Minimum velocity in meters/second to register a release.
        displacement_threshold: Minimum distance in meters from starting position to trigger a release.
        """
        self.prev_pos = None
        self.prev_time = None
        self.start_pos = None
        self.released = False
        self.vel_threshold = vel_threshold
        self.displacement_threshold = displacement_threshold

    def update(self, pos_3d, timestamp):
        """
        pos_3d: np.array([x, y, z]) representing the 3D coordinates in meters.
        timestamp: float in seconds.
        """
        if self.released:
            return True

        if self.start_pos is None:
            self.start_pos = pos_3d.copy()

        if self.prev_pos is None:
            self.prev_pos = pos_3d.copy()
            self.prev_time = timestamp
            return False

        dt = timestamp - self.prev_time
        if dt <= 0:
            return self.released

        # Calculate 3D velocity norm
        velocity = np.linalg.norm(pos_3d - self.prev_pos) / dt

        # Calculate 3D displacement from initial starting position
        displacement = np.linalg.norm(pos_3d - self.start_pos)

        self.prev_pos = pos_3d.copy()
        self.prev_time = timestamp

        # Dual-thresholding: must exceed both velocity and displacement limit
        # This prevents depth noise spikes on a stationary ball from triggering a false release
        if velocity > self.vel_threshold and displacement > self.displacement_threshold:
            self.released = True

        return self.released

    def reset(self):
        """
        Reset release detection status.
        """
        self.released = False
        self.prev_pos = None
        self.prev_time = None
        self.start_pos = None
