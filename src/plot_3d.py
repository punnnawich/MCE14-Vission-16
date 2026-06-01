"""
MCE14 Vission 16 - Real-time 3D Ball Trajectory Plot
=====================================================
สคริปต์สำหรับแสดงผลตำแหน่ง 3 มิติของลูกบอลแบบ Real-time
รับข้อมูลผ่าน UDP จาก vision_pipeline.py (localhost:5006)

วิธีใช้:
  1. รัน vision_pipeline.py ก่อน
  2. รัน plot_3d.py ในอีก Terminal หนึ่ง

รูปแบบข้อมูล UDP (CSV):
  X_cm, Y_cm, Z_cm, pred_X_cm, pred_Y_cm, pred_Z_cm
  (pred = None ถ้ายังไม่มีการทำนาย)
"""

import socket
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# ==============================
# การตั้งค่า
# ==============================
UDP_IP = "127.0.0.1"
UDP_PORT = 5006          # ต้องตรงกับ plot_udp_port ใน config.yaml
MAX_TRAIL_POINTS = 30    # ความยาว trail (จำนวนจุด)
FLOOR_Z = 0.0            # ระนาบพื้น (cm)

# ==============================
# สร้าง UDP Socket
# ==============================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# ==============================
# ตั้งค่า Matplotlib 3D
# ==============================
plt.ion()
fig = plt.figure(figsize=(10, 7))
fig.patch.set_facecolor('#1a1a2e')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#16213e')

# กำหนดขอบเขตเริ่มต้น
ax.set_xlim([-150, 150])   # X = ซ้าย-ขวา (Width)
ax.set_ylim([-50, 350])    # Y = แนวลึก (Depth)
ax.set_zlim([-10, 250])    # Z = ความสูง (Height)
ax.set_xlabel('X (Width - cm)', color='white', fontsize=10)
ax.set_ylabel('Y (Depth - cm)', color='white', fontsize=10)
ax.set_zlabel('Z (Height - cm)', color='white', fontsize=10)
ax.set_title('MCE14-Vission-16 — Real-time 3D Ball Tracking', color='white', fontsize=13, pad=15)
ax.view_init(elev=20, azim=-120)

# สีของ tick labels
ax.tick_params(axis='x', colors='#aaaaaa')
ax.tick_params(axis='y', colors='#aaaaaa')
ax.tick_params(axis='z', colors='#aaaaaa')

# วาดระนาบพื้น (z=0) แบบ grid
floor_x = np.linspace(-150, 150, 5)
floor_y = np.linspace(-50, 350, 5)
floor_X, floor_Y = np.meshgrid(floor_x, floor_y)
floor_Z = np.zeros_like(floor_X)
ax.plot_surface(floor_X, floor_Y, floor_Z, alpha=0.08, color='cyan')

# เตรียมอ็อบเจ็กต์กราฟ
scatter_history = ax.plot([], [], [], marker='o', linestyle='None',
                          c='#888888', alpha=0.4, markersize=4)[0]
scatter_latest = ax.plot([], [], [], marker='o', linestyle='None',
                         c='#ff4444', markersize=10, zorder=5)[0]
plot_line = ax.plot([], [], [], c='#4488ff', alpha=0.6, linewidth=1.5)[0]
scatter_landing = ax.plot([], [], [], marker='X', linestyle='None',
                          c='#00ff88', markersize=14, markeredgewidth=2,
                          label='จุดตกที่ทำนาย')[0]
plot_curve = ax.plot([], [], [], c='#ffaa00', linestyle='--',
                     linewidth=2, alpha=0.8, label='วิถีพยากรณ์')[0]

# Text annotation สำหรับแสดงพิกัดล่าสุด
info_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                      color='white', fontsize=9, verticalalignment='top',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a2a', alpha=0.8))

ax.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e',
          edgecolor='#444444', labelcolor='white')

# ==============================
# ตัวแปรเก็บข้อมูล
# ==============================
xs = deque(maxlen=MAX_TRAIL_POINTS)
ys = deque(maxlen=MAX_TRAIL_POINTS)
zs = deque(maxlen=MAX_TRAIL_POINTS)
frame_count = 0

print("=" * 55)
print("  MCE14-Vission-16 - 3D Ball Trajectory Plotter")
print("=" * 55)
print(f"  Waiting for UDP data at {UDP_IP}:{UDP_PORT} ...")
print(f"  Make sure vision_pipeline.py is running")
print(f"  Press Ctrl+C or close graph window to exit")
print("=" * 55)

try:
    while True:
        data = None
        # อ่านข้อมูลล่าสุดจาก buffer
        while True:
            try:
                packet, addr = sock.recvfrom(1024)
                data = packet
            except BlockingIOError:
                break
            except Exception:
                pass

        if data:
            decoded = data.decode('utf-8')
            parts = decoded.split(',')
            if len(parts) >= 3:
                try:
                    raw_x = float(parts[0])
                    raw_y = float(parts[1])
                    raw_z = float(parts[2])

                    xs.append(raw_x)
                    ys.append(raw_y)
                    zs.append(raw_z)
                    frame_count += 1

                    pred_x, pred_y, pred_z = None, None, None

                    # Parse prediction data
                    if len(parts) >= 6 and parts[3] != 'None':
                        pred_x = float(parts[3])
                        pred_y = float(parts[4])
                        pred_z = float(parts[5])

                        # แสดงจุดตก
                        scatter_landing.set_data([pred_x], [pred_y])
                        scatter_landing.set_3d_properties([pred_z])

                        # คำนวณเส้นโค้งพยากรณ์ (Parabolic interpolation)
                        # จากตำแหน่งปัจจุบันไปยังจุดตก
                        n_curve = 30
                        t_arr = np.linspace(0, 1, n_curve)
                        # เส้นตรง X,Y + พาราโบลา Z
                        curve_x = raw_x + (pred_x - raw_x) * t_arr
                        curve_y = raw_y + (pred_y - raw_y) * t_arr
                        # Z: parabolic arc from current height to floor
                        peak_extra = max(raw_z * 0.1, 5)  # เพิ่มส่วนโค้งเล็กน้อย
                        curve_z = raw_z * (1 - t_arr) + pred_z * t_arr + \
                                  peak_extra * 4 * t_arr * (1 - t_arr)
                        # Ensure end is at floor
                        curve_z[-1] = pred_z

                        plot_curve.set_data(curve_x, curve_y)
                        plot_curve.set_3d_properties(curve_z)
                    else:
                        scatter_landing.set_data([], [])
                        scatter_landing.set_3d_properties([])
                        plot_curve.set_data([], [])
                        plot_curve.set_3d_properties([])

                    # อัปเดต trajectory trail
                    xs_list = list(xs)
                    ys_list = list(ys)
                    zs_list = list(zs)

                    if len(xs_list) > 1:
                        scatter_history.set_data(xs_list[:-1], ys_list[:-1])
                        scatter_history.set_3d_properties(zs_list[:-1])
                    else:
                        scatter_history.set_data([], [])
                        scatter_history.set_3d_properties([])

                    # จุดล่าสุด
                    scatter_latest.set_data([xs_list[-1]], [ys_list[-1]])
                    scatter_latest.set_3d_properties([zs_list[-1]])

                    # เส้นเชื่อม
                    plot_line.set_data(xs_list, ys_list)
                    plot_line.set_3d_properties(zs_list)

                    # Dynamic axis scaling
                    all_x = xs_list + ([pred_x] if pred_x is not None else [])
                    all_y = ys_list + ([pred_y] if pred_y is not None else [])

                    max_x = max(max(abs(v) for v in all_x) * 1.3, 150) if all_x else 150
                    max_y = max(max(abs(v) for v in all_y) * 1.3, 350) if all_y else 350
                    max_z = max(max(zs_list) * 1.3, 250) if zs_list else 250

                    ax.set_xlim([-max_x, max_x])
                    ax.set_ylim([-50, max_y])
                    ax.set_zlim([-10, max_z])

                    # อัปเดต info text
                    info = f"Pos: ({raw_x:.1f}, {raw_y:.1f}, {raw_z:.1f}) cm\n"
                    info += f"Pts: {frame_count}"
                    if pred_x is not None:
                        info += f"\nLanding: ({pred_x:.1f}, {pred_y:.1f}) cm"
                    info_text.set_text(info)

                except ValueError:
                    pass

        # Redraw
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Check if graph window is closed
        if not plt.fignum_exists(fig.number):
            print("\nGraph window closed. Exiting...")
            break

except KeyboardInterrupt:
    print("\nTerminated by user (Ctrl+C)")
finally:
    sock.close()
    plt.ioff()
    plt.close('all')
    print("Socket and plot closed successfully.")
