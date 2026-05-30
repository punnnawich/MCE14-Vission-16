import socket
import json
import threading
import time

class RobotComms:
    def __init__(self, config):
        """
        Initialize UDP socket for communicating with ESP32.
        """
        comm_cfg = config.get("communication", {})
        self.ip = comm_cfg.get("esp32_ip", "192.168.137.100")
        self.port = comm_cfg.get("esp32_port", 5005)
        self.heartbeat_interval = comm_cfg.get("heartbeat_interval_ms", 100) / 1000.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.heartbeat_thread = None
        self.last_error = None

    def start(self):
        """
        Start the background heartbeat thread.
        """
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        """
        Periodically transmits the heartbeat payload to the robot.
        """
        payload = json.dumps({"heartbeat": 1})
        while self.running:
            try:
                self.sock.sendto(payload.encode(), (self.ip, self.port))
                self.last_error = None
            except Exception as e:
                self.last_error = str(e)
            time.sleep(self.heartbeat_interval)

    def send_target(self, x_cm, y_cm):
        """
        Sends the calculated landing coordinates (in cm) to the ESP32.
        """
        payload = json.dumps({
            "x": round(x_cm, 1),
            "y": round(y_cm, 1)
        })
        try:
            self.sock.sendto(payload.encode(), (self.ip, self.port))
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def stop(self):
        """
        Stops the heartbeat loop and releases the socket.
        """
        self.running = False
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=0.5)
        try:
            self.sock.close()
        except Exception:
            pass
