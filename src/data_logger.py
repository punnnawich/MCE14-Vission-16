import os
import csv
from datetime import datetime

class DataLogger:
    def __init__(self, output_dir="logs"):
        """
        Initialize the logger. Creates the output directory if it doesn't exist.
        Each run generates a new timestamped CSV file.
        """
        # Get absolute path relative to this script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(os.path.dirname(script_dir), output_dir)
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"[Logger] Error creating log directory {self.output_dir}: {e}")
            self.output_dir = script_dir  # Fallback to src directory

        # Generate filename with date and time
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.output_dir, f"trajectory_log_{timestamp_str}.csv")
        self.file = None
        self.writer = None
        self.is_active = False

        # Define CSV header fields
        self.headers = [
            "timestamp_seconds",
            "is_calibrated",
            "raw_world_x",
            "raw_world_y",
            "raw_world_z",
            "filtered_world_x",
            "filtered_world_y",
            "filtered_world_z",
            "is_released",
            "elapsed_since_release_seconds",
            "predicted_landing_x",
            "predicted_landing_y",
            "predicted_landing_z",
            "unclamped_landing_x",
            "unclamped_landing_y",
            "is_clamped",
            "time_to_impact_seconds",
            "vertical_velocity_vz"
        ]

    def start(self):
        """
        Open the log file and write headers.
        """
        try:
            self.file = open(self.filename, mode="w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
            self.writer.writeheader()
            self.is_active = True
            print(f"[Logger] Started logging to: {self.filename}")
        except Exception as e:
            print(f"[Logger] Failed to start logging: {e}")
            self.is_active = False

    def log(self, timestamp, is_calibrated, raw_pos, filt_pos, is_released, prediction):
        """
        Logs a single frame's data.
        
        raw_pos: np.array([x, y, z]) or None
        filt_pos: np.array([x, y, z]) or None
        prediction: dict with keys "x", "y", "z", "t_land_from_now", "vz" or None
        """
        if not self.is_active or self.writer is None:
            return

        row = {
            "timestamp_seconds": f"{timestamp:.6f}",
            "is_calibrated": int(is_calibrated),
            "raw_world_x": f"{raw_pos[0]:.4f}" if raw_pos is not None else "",
            "raw_world_y": f"{raw_pos[1]:.4f}" if raw_pos is not None else "",
            "raw_world_z": f"{raw_pos[2]:.4f}" if raw_pos is not None else "",
            "filtered_world_x": f"{filt_pos[0]:.4f}" if filt_pos is not None else "",
            "filtered_world_y": f"{filt_pos[1]:.4f}" if filt_pos is not None else "",
            "filtered_world_z": f"{filt_pos[2]:.4f}" if filt_pos is not None else "",
            "is_released": int(is_released),
            "elapsed_since_release_seconds": f"{prediction['elapsed_since_release']:.4f}" if (prediction is not None and 'elapsed_since_release' in prediction) else "",
            "predicted_landing_x": f"{prediction['x']:.4f}" if prediction is not None else "",
            "predicted_landing_y": f"{prediction['y']:.4f}" if prediction is not None else "",
            "predicted_landing_z": f"{prediction['z']:.4f}" if prediction is not None else "",
            "unclamped_landing_x": f"{prediction['unclamped_x']:.4f}" if (prediction is not None and 'unclamped_x' in prediction) else "",
            "unclamped_landing_y": f"{prediction['unclamped_y']:.4f}" if (prediction is not None and 'unclamped_y' in prediction) else "",
            "is_clamped": int(prediction['is_clamped']) if (prediction is not None and 'is_clamped' in prediction) else "",
            "time_to_impact_seconds": f"{prediction['t_land_from_now']:.4f}" if prediction is not None else "",
            "vertical_velocity_vz": f"{prediction['vz']:.4f}" if (prediction is not None and 'vz' in prediction) else ""
        }

        try:
            self.writer.writerow(row)
        except Exception as e:
            print(f"[Logger] Error writing row to CSV: {e}")

    def stop(self):
        """
        Flush and close the log file.
        """
        if self.file is not None:
            try:
                self.file.flush()
                self.file.close()
                print(f"[Logger] Log saved successfully. ({self.filename})")
            except Exception as e:
                print(f"[Logger] Error closing log file: {e}")
            finally:
                self.file = None
                self.writer = None
                self.is_active = False
