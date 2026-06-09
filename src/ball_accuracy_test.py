"""
MCE14 Ball Localization Accuracy Test
======================================
วัดความแม่นยำของการตรวจจับตำแหน่ง 3D ลูกบอลโดยเทียบกับตำแหน่งจริง

วิธีใช้:
  1. รัน vision_pipeline.py และกด 'z' เพื่อ SET ZERO ก่อน
  2. ปิด plot_3d.py (port conflict)
  3. รัน: python ball_accuracy_test.py
  4. วางลูกบอลที่ตำแหน่งที่รู้จริง (วัดด้วยตลับเมตรจากจุด origin)
  5. กด Enter เพื่อ capture (~1 วินาที)
  6. พิมพ์ ground truth X Y Z (cm) คั่นด้วย space
  7. ทำซ้ำ ≥5 ตำแหน่ง — กด Ctrl+C เพื่อดูสถิติ

ระบบพิกัด (เหมือนกับ vision_pipeline):
  X = ซ้าย/ขวา (cm)  Y = ลึก (cm)  Z = สูง (cm)
  origin = จุดที่กด SET ZERO
"""

import socket
import time
import numpy as np
import csv
import os
import msvcrt
from datetime import datetime

UDP_IP       = "127.0.0.1"
UDP_PORT     = 5006
CAPTURE_N    = 60     # จำนวน frames ต่อ 1 การวัด (~2s ที่ 30fps)
NOISE_WARN   = 1.5    # cm — เตือนถ้า std สูงกว่านี้ (ลูกขยับ)


def drain_latest(sock):
    """Drain UDP queue, return latest (x, y, z) or None."""
    latest = None
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            parts = data.decode().split(',')
            if len(parts) >= 3 and parts[0] != 'None':
                latest = (float(parts[0]), float(parts[1]), float(parts[2]))
        except BlockingIOError:
            break
        except Exception:
            break
    return latest


def show_live_and_wait_enter(sock):
    """แสดงตำแหน่งแบบ real-time จนกว่าผู้ใช้กด Enter"""
    print("  [กด Enter เพื่อเริ่ม capture]", flush=True)
    while True:
        pos = drain_latest(sock)
        if pos:
            print(f"\r  Live: X={pos[0]:+7.1f}  Y={pos[1]:+7.1f}  Z={pos[2]:+7.1f} cm   [Enter]   ",
                  end='', flush=True)
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b'\r', b'\n'):
                print()
                return
        time.sleep(1.0 / 30)


def capture(sock, n=CAPTURE_N):
    """เก็บ n frames แล้วคืนค่า mean และ std (cm)."""
    samples = []
    print(f"  Capturing {n} frames ", end='', flush=True)
    while len(samples) < n:
        try:
            data, _ = sock.recvfrom(1024)
            parts = data.decode().split(',')
            if len(parts) >= 3 and parts[0] != 'None':
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                samples.append((x, y, z))
                if len(samples) % 15 == 0:
                    print('.', end='', flush=True)
        except BlockingIOError:
            time.sleep(0.005)
        except Exception:
            pass
    print(' done')
    arr = np.array(samples)
    return arr.mean(axis=0), arr.std(axis=0)


def ask_ground_truth():
    """รับ ground truth X Y Z (cm) จากผู้ใช้"""
    while True:
        try:
            raw = input("  Ground truth X Y Z (cm): ").strip()
            vals = list(map(float, raw.split()))
            if len(vals) == 3:
                return np.array(vals)
            print("  ⚠ กรุณาใส่ 3 ค่า เช่น: 0 50 25")
        except ValueError:
            print("  ⚠ ค่าไม่ถูกต้อง")


def print_stats(results):
    if not results:
        return
    errors = np.array([(r['err_x'], r['err_y'], r['err_z']) for r in results])
    euclid = np.array([r['err_3d'] for r in results])

    print()
    print("=" * 62)
    print("  ACCURACY STATISTICS")
    print("=" * 62)
    print(f"  Measurements : {len(results)}")
    print()
    print(f"  {'Axis':<10} {'Bias (mean)':>12} {'Noise (std)':>12} {'RMSE':>10} {'Max |err|':>12}")
    print(f"  {'-'*58}")
    labels = ['X (lateral)', 'Y (depth)', 'Z (height)']
    for i, lbl in enumerate(labels):
        e = errors[:, i]
        bias = e.mean()
        noise = e.std()
        rmse  = float(np.sqrt((e ** 2).mean()))
        mx    = float(np.max(np.abs(e)))
        print(f"  {lbl:<10} {bias:>+11.1f}cm {noise:>11.1f}cm {rmse:>9.1f}cm {mx:>11.1f}cm")
    print()
    print(f"  3D Euclidean  mean={euclid.mean():.1f}cm   std={euclid.std():.1f}cm   max={euclid.max():.1f}cm")
    print("=" * 62)

    # ประเมินผล
    rmse_3d = float(np.sqrt((euclid ** 2).mean()))
    print()
    if rmse_3d < 2.0:
        grade = "EXCELLENT  ✓✓"
    elif rmse_3d < 5.0:
        grade = "GOOD       ✓"
    elif rmse_3d < 10.0:
        grade = "ACCEPTABLE ~"
    else:
        grade = "POOR       ✗"
    print(f"  Overall RMSE: {rmse_3d:.1f} cm  →  {grade}")
    print()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    log_dir  = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            f'accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    print()
    print("=" * 62)
    print("  MCE14-Vission-16  —  Ball Localization Accuracy Test")
    print("=" * 62)
    print(f"  UDP : {UDP_IP}:{UDP_PORT}")
    print(f"  Log : {log_path}")
    print()
    print("  ต้องการก่อนเริ่ม:")
    print("   • vision_pipeline.py รันอยู่และกด 'z' SET ZERO แล้ว")
    print("   • plot_3d.py ปิดอยู่")
    print()
    print("  แนะนำตำแหน่งทดสอบ (≥5 จุด):")
    print("   • หน้ากล้อง: Y=50,100,150 cm  ที่ X=0, Z=25 cm")
    print("   • ด้านข้าง:  X=±20 cm  ที่ Y=100 cm, Z=25 cm")
    print("   • ระดับสูง:  Z=0,25,50 cm  ที่ X=0, Y=100 cm")
    print()
    print("  Ctrl+C = จบและแสดงสถิติ")
    print("=" * 62)

    results = []

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n',
            'gt_x_cm', 'gt_y_cm', 'gt_z_cm',
            'meas_x_cm', 'meas_y_cm', 'meas_z_cm',
            'err_x_cm', 'err_y_cm', 'err_z_cm', 'err_3d_cm',
            'noise_x_cm', 'noise_y_cm', 'noise_z_cm',
        ])

        try:
            n = 1
            while True:
                print(f"\n── ตำแหน่งที่ {n} {'─'*40}")
                show_live_and_wait_enter(sock)

                mean, std = capture(sock)
                print(f"  Measured : X={mean[0]:+7.1f}  Y={mean[1]:+7.1f}  Z={mean[2]:+7.1f} cm")
                print(f"  Noise std: X={std[0]:5.2f}   Y={std[1]:5.2f}   Z={std[2]:5.2f} cm")

                if std.max() > NOISE_WARN:
                    ans = input(f"  ⚠ Noise สูง ({std.max():.1f} cm) ลูกบอลอาจขยับ\n"
                                f"  Capture ใหม่? [y = ใช่ / Enter = ข้ามไป]: ").strip().lower()
                    if ans == 'y':
                        continue

                gt = ask_ground_truth()
                err    = mean - gt
                err_3d = float(np.linalg.norm(err))

                print(f"  Error    : X={err[0]:+7.1f}  Y={err[1]:+7.1f}  Z={err[2]:+7.1f} cm  |3D|={err_3d:.1f} cm")

                results.append({
                    'err_x': err[0], 'err_y': err[1], 'err_z': err[2],
                    'err_3d': err_3d,
                })

                writer.writerow([
                    n,
                    f'{gt[0]:.1f}', f'{gt[1]:.1f}', f'{gt[2]:.1f}',
                    f'{mean[0]:.2f}', f'{mean[1]:.2f}', f'{mean[2]:.2f}',
                    f'{err[0]:.2f}', f'{err[1]:.2f}', f'{err[2]:.2f}',
                    f'{err_3d:.2f}',
                    f'{std[0]:.2f}', f'{std[1]:.2f}', f'{std[2]:.2f}',
                ])
                f.flush()
                n += 1

        except KeyboardInterrupt:
            pass

    print_stats(results)
    if results:
        print(f"  ผลทั้งหมดบันทึกที่: {log_path}")
    sock.close()


if __name__ == '__main__':
    main()
