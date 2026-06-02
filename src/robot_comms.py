import socket
import struct
import threading
import time

# ---------------------------------------------------------------------------
# UDP Binary Protocol (ตาม robot_flow.md)
# ---------------------------------------------------------------------------
# PC → ESP32 : [seq: uint32][x: float32][y: float32][extra: uint32]  = 16 bytes
#   extra = 0 → BALL_POS  (พิกัดเป้าหมาย)
#   extra = 1 → ROBOT_POS (พิกัดหุ่นปัจจุบัน ตอบ REQUEST_POS)
#
# ESP32 → PC : "REQUEST_POS" (11 bytes ASCII) → ESP32 ขอพิกัดหุ่น
#              "READY"       ( 5 bytes ASCII) → ESP32 พร้อมรับ Target ใหม่
#              "OKAY"        ( 4 bytes ASCII) → (legacy) เหมือน READY
# ---------------------------------------------------------------------------

_STRUCT_FORMAT = "<IffI"   # little-endian: uint32, float, float, uint32
_PACKET_SIZE   = struct.calcsize(_STRUCT_FORMAT)  # = 16 bytes


class RobotComms:
    def __init__(self, config):
        """
        Initialize UDP socket for communicating with ESP32.
        Binary protocol: 16-byte fixed-size packets.
        """
        comm_cfg = config.get("communication", {})
        self.ip   = comm_cfg.get("esp32_ip",   "192.168.137.123")
        self.port = comm_cfg.get("esp32_port",  12345)

        # One socket for both send and receive
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.running          = False
        self._recv_thread     = None
        self.last_error       = None
        self.sequence_number  = 0

        # Flags set by _recv_loop — read by vision_pipeline
        self.robot_ready          = True   # True = ESP32 พร้อมรับ Target ใหม่ (set by READY/OKAY)
        self.pending_request_pos  = False  # True = ESP32 ขอพิกัดหุ่น
        self.last_target_time     = 0.0    # Last time a BALL_POS was sent (for timeout recovery)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Bind the socket and start the receive listener thread."""
        try:
            self.sock.bind(("", self.port))
        except OSError as e:
            print(f"[RobotComms] WARNING: Could not bind to port {self.port}: {e}")
            print("[RobotComms]   Receive (REQUEST_POS / OKAY) will be unavailable.")

        # Optimize socket buffers (1MB) to reduce packet drop under CPU load
        from performance import optimize_socket
        optimize_socket(self.sock)

        self.sock.settimeout(0.15)   # non-blocking receive with short timeout
        self.running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="robot-recv", daemon=True
        )
        self._recv_thread.start()
        print(f"[RobotComms] Started — target {self.ip}:{self.port} | packet={_PACKET_SIZE} bytes")

    def stop(self):
        """Stop the receive loop and close the socket."""
        self.running = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)
        try:
            self.sock.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Receive loop (runs in background thread)
    # ------------------------------------------------------------------

    def _recv_loop(self):
        """
        Listens for ASCII messages from ESP32:
          - "REQUEST_POS" → sets pending_request_pos = True
          - "OKAY"        → sets robot_ready = True
        """
        while self.running:
            try:
                data, _ = self.sock.recvfrom(64)
                msg = data.decode("utf-8", errors="ignore").strip()
                if msg == "REQUEST_POS":
                    self.pending_request_pos = True
                    print("[RobotComms] ← REQUEST_POS received")
                elif msg == "OKAY":
                    self.robot_ready = True
                    print("[RobotComms] ← OKAY received (robot ready)")
                elif msg == "READY":
                    self.robot_ready = True
                    print("[RobotComms] ← READY received (ESP32 booted)")
                else:
                    print(f"[RobotComms] ← Unknown msg: {repr(msg)}")
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    self.last_error = str(e)

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def _send_packet(self, x_cm: float, y_cm: float, extra: int) -> bool:
        """
        Pack and send a 16-byte binary UDP packet.
        Format: [seq:uint32][x:float32][y:float32][extra:uint32]
        """
        self.sequence_number += 1
        payload = struct.pack(_STRUCT_FORMAT,
                              self.sequence_number,
                              float(x_cm),
                              float(y_cm),
                              int(extra))
        try:
            self.sock.sendto(payload, (self.ip, self.port))
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def send_target(self, x_cm: float, y_cm: float) -> bool:
        """
        Send ball landing coordinates (cm) as BALL_POS packet (extra = 0).
        Call only when robot_ready == True.
        Sends 3x redundantly to survive WiFi packet loss.
        ESP32 deduplicates via sequence number.
        """
        ok = False
        for i in range(3):
            ok = self._send_packet(x_cm, y_cm, extra=0)
            if not ok:
                break
            if i < 2:
                time.sleep(0.001)  # 1ms gap between redundant sends
        if ok:
            self.robot_ready = False   # รอ OKAY จาก ESP32 ก่อนส่งครั้งถัดไป
            self.last_target_time = time.time()  # Track for timeout recovery
            print(f"[RobotComms] → BALL_POS  seq={self.sequence_number:4d}  "
                  f"x={x_cm:+7.1f} cm  y={y_cm:+7.1f} cm  (3x sent)")
        return ok

    def send_robot_pos(self, rx_cm: float, ry_cm: float) -> bool:
        """
        Send current robot position (cm) as ROBOT_POS packet (extra = 1).
        Called in response to REQUEST_POS from ESP32.
        """
        ok = self._send_packet(rx_cm, ry_cm, extra=1)
        if ok:
            self.pending_request_pos = False
            print(f"[RobotComms] → ROBOT_POS seq={self.sequence_number:4d}  "
                  f"rx={rx_cm:+7.1f} cm  ry={ry_cm:+7.1f} cm")
        return ok
