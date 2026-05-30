import unittest
import numpy as np
import cv2
import sys
import os

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from median_filter import MedianFilter3D
from release_detector import ReleaseDetector
from projectile_predictor import ProjectilePredictor
from ball_detector import BallDetector

class TestMedianFilter(unittest.TestCase):
    def test_filter_smoothing(self):
        filt = MedianFilter3D(window_size=5)
        # Feed steady points
        for _ in range(3):
            filt.update(np.array([1.0, 2.0, 3.0]))
        
        # Feed outlier
        filt.update(np.array([10.0, 20.0, 30.0]))
        
        # Get next point
        res = filt.update(np.array([1.1, 2.1, 3.1]))
        
        # The outlier should be suppressed by median filter
        self.assertLess(res[0], 2.0)
        self.assertLess(res[1], 3.0)
        self.assertLess(res[2], 4.0)

class TestReleaseDetector(unittest.TestCase):
    def test_no_release_on_slow_motion(self):
        detector = ReleaseDetector(vel_threshold=1.0)
        # Slow movements
        self.assertFalse(detector.update(np.array([0.0, 0.0, 1.0]), 0.0))
        self.assertFalse(detector.update(np.array([0.01, 0.01, 1.0]), 0.1))
        self.assertFalse(detector.update(np.array([0.02, 0.02, 1.0]), 0.2))

    def test_release_on_fast_motion(self):
        detector = ReleaseDetector(vel_threshold=1.0)
        detector.update(np.array([0.0, 0.0, 1.0]), 0.0)
        # Sudden fast movement (2 meters in 0.1 seconds -> 20 m/s)
        self.assertTrue(detector.update(np.array([0.0, 0.0, 3.0]), 0.1))

class TestProjectilePredictor(unittest.TestCase):
    def test_prediction_accuracy(self):
        config = {
            "predictor": {
                "min_points": 4,
                "z_floor": 0.0,
                "drag_correction": 1.0 # Use 1.0 to check analytical gravity fit
            }
        }
        predictor = ProjectilePredictor(config)
        
        # Generate simulated parabolic trajectory points
        # Z(t) = -0.5 * g * t^2 + Vz0 * t + Z0
        # Let g = 9.81, Vz0 = 4.0 m/s, Z0 = 2.0 m
        # X(t) = Vx0 * t + X0 (Vx0 = 2.0 m/s, X0 = 0.5 m)
        # Y(t) = Vy0 * t + Y0 (Vy0 = -1.0 m/s, Y0 = 0.2 m)
        # Analytical landing: Z(t) = -4.905*t^2 + 4*t + 2 = 0
        # Roots of -4.905*t^2 + 4*t + 2 = 0:
        # t = (-4 - sqrt(16 - 4*(-4.905)*2)) / (2 * -4.905)
        # t = (-4 - sqrt(16 + 39.24)) / -9.81 = (-4 - 7.432) / -9.81 = 1.165 seconds
        # Expected X_land = 2.0 * 1.165 + 0.5 = 2.83 meters
        # Expected Y_land = -1.0 * 1.165 + 0.2 = -0.965 meters

        g = 9.81
        vx0, vy0, vz0 = 2.0, -1.0, 4.0
        x0, y0, z0 = 0.5, 0.2, 2.0
        
        t_points = [0.0, 0.1, 0.2, 0.3, 0.4]
        for t in t_points:
            x = vx0 * t + x0
            y = vy0 * t + y0
            z = -0.5 * g * t**2 + vz0 * t + z0
            predictor.add_point(np.array([x, y, z]), t)
            
        pred = predictor.predict_landing()
        self.assertIsNotNone(pred)
        
        # Validate prediction output
        self.assertAlmostEqual(pred["x"], 2.83, delta=0.05)
        self.assertAlmostEqual(pred["y"], -0.965, delta=0.05)
        self.assertAlmostEqual(pred["z"], 0.0, delta=0.01)

class TestBallDetector(unittest.TestCase):
    def test_ball_detection_mock(self):
        config = {
            "hsv": {
                "lower_red_1": [0, 100, 80],
                "upper_red_1": [10, 255, 255],
                "lower_red_2": [170, 100, 80],
                "upper_red_2": [180, 255, 255]
            },
            "blob": {
                "min_area": 50,
                "max_area": 50000,
                "min_circularity": 0.7
            }
        }
        detector = BallDetector(config)
        
        # Create a mock 400x400 black image with a bright red circle at center (200, 200)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Red color in BGR is (0, 0, 255)
        cv2.circle(img, (200, 200), 20, (0, 0, 255), -1)
        
        mask = detector.detect_red_ball(img)
        res = detector.find_ball_centroid(mask)
        
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["cx"], 200, delta=2)
        self.assertAlmostEqual(res["cy"], 200, delta=2)

if __name__ == "__main__":
    unittest.main()
