import sys
import os
import time
import yaml
import socket
import struct

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from robot_comms import RobotComms

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def main():
    print("=" * 65)
    print(" 🚀  MCE14 — Interactive ESP32 Communication Test Tool")
    print("=" * 65)

    # 1. Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        return

    comm_cfg = config.get("communication", {})
    esp_ip = comm_cfg.get("esp32_ip", "192.168.137.123")
    esp_port = comm_cfg.get("esp32_port", 12345)

    print(f"📍 Current Configured ESP32 IP : {esp_ip}")
    print(f"📍 Current Configured Port     : {esp_port}")
    print("-" * 65)

    # Ask if IP needs to be updated
    new_ip = input(f"Enter ESP32 IP from Serial Monitor (Press Enter to keep '{esp_ip}'): ").strip()
    if new_ip:
        esp_ip = new_ip
        config["communication"]["esp32_ip"] = esp_ip
        try:
            save_config(config)
            print(f"💾 Updated config.yaml with new ESP32 IP: {esp_ip}")
        except Exception as e:
            print(f"⚠️ Could not save config.yaml: {e}")

    # Initialize communication
    comms = RobotComms(config)
    comms.start()

    print("\n[System] Listener thread started. Ready to send/receive data.")
    print("[System] Menu:")
    print("  [1] Send BALL_POS (Target Coordinate)   -> (extra = 0)")
    print("  [2] Send ROBOT_POS (Robot Location)      -> (extra = 1)")
    print("  [3] Start Auto-Responder (Fully simulate PC-side logic)")
    print("  [4] 🎯 Manual Send Mode (พิมพ์ x,y ส่งต่อเนื่อง)")
    print("  [5] Send CAMERA_READY (Camera Ready)    -> (extra = 2)")
    print("  [q] Quit")
    print("-" * 65)

    try:
        while True:
            choice = input("\nSelect Option [1/2/3/4/5/q]: ").strip().lower()
            if choice == "q":
                break
            
            elif choice == "5":
                comms.send_camera_ready()
            
            elif choice == "1":
                try:
                    x = float(input("Enter Target X (cm): ").strip())
                    y = float(input("Enter Target Y (cm): ").strip())
                    # Temporarily force robot_ready to True to allow sending in manual mode
                    comms.robot_ready = True
                    comms.send_target(x, y)
                except ValueError:
                    print("❌ Invalid input! X and Y must be numbers.")
            
            elif choice == "2":
                try:
                    rx = float(input("Enter Robot RX (cm): ").strip())
                    ry = float(input("Enter Robot RY (cm): ").strip())
                    comms.send_robot_pos(rx, ry)
                except ValueError:
                    print("❌ Invalid input! RX and RY must be numbers.")
            
            elif choice == "3":
                print("\n🤖 Auto-Responder Active! Press Ctrl+C to exit Auto-Responder mode.")
                print("Listening for REQUEST_POS and OKAY from ESP32...")
                print("When REQUEST_POS is received, PC will automatically send (RX=2.0cm, RY=-1.5cm) back.")
                print("-" * 65)
                try:
                    while True:
                        if comms.pending_request_pos:
                            print("\n[Auto] ⚡ ESP32 requested position (REQUEST_POS)!")
                            time.sleep(0.5) # Wait 500ms
                            # Simulate sending small error of 2.0 cm, -1.5 cm (dist is ~2.5 cm <= 3.0 cm)
                            rx, ry = 2.0, -1.5
                            print(f"[Auto] ➔ Sending ROBOT_POS: RX={rx}cm, RY={ry}cm (Error: {float((rx**2+ry**2)**0.5):.2f}cm)")
                            comms.send_robot_pos(rx, ry)
                        
                        if comms.robot_ready:
                            # If ready, we can send a mock ball target if they want
                            pass
                        
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n[Auto] Exited Auto-Responder mode.")

            elif choice == "4":
                print("\n" + "=" * 65)
                print(" 🎯  Manual Send Mode — พิมพ์ x,y ส่ง BALL_POS ต่อเนื่อง")
                print("=" * 65)
                print(" วิธีใช้:")
                print("   พิมพ์ค่า:  x,y    เช่น  10,20   หรือ  -5.5,12.3")
                print("   ส่งซ้ำ:    Enter  (ส่งค่าเดิมอีกครั้ง)")
                print("   ออก:       q")
                print("-" * 65)

                comms.robot_ready = True  # Force ready for manual testing
                last_x, last_y = 0.0, 0.0
                send_count = 0

                try:
                    while True:
                        prompt = f"[#{send_count}] x,y (last: {last_x:.1f},{last_y:.1f}) > "
                        raw = input(prompt).strip()
                        
                        if raw.lower() == "q":
                            break
                        
                        if raw == "":
                            # Enter = ส่งค่าเดิมซ้ำ
                            x, y = last_x, last_y
                        else:
                            try:
                                parts = raw.replace(" ", ",").split(",")
                                if len(parts) != 2:
                                    print("   ❌ ใส่ 2 ค่า คั่นด้วย , เช่น 10,20")
                                    continue
                                x = float(parts[0])
                                y = float(parts[1])
                            except ValueError:
                                print("   ❌ ค่าไม่ถูกต้อง ใส่ตัวเลข เช่น 10.5,-3.2")
                                continue
                        
                        # Force ready and send
                        comms.robot_ready = True
                        comms.send_target(x, y)
                        send_count += 1
                        last_x, last_y = x, y
                        print(f"   ✅ Sent BALL_POS → X={x:.1f}cm Y={y:.1f}cm (seq={comms.sequence_number})")
                        
                except KeyboardInterrupt:
                    pass
                print(f"\n[Manual] Exited. Total sent: {send_count} packets")

            else:
                print("⚠️ Invalid choice. Select 1, 2, 3, 4, or q.")
    finally:
        print("\n[System] Stopping communication...")
        comms.stop()
        print("Done.")

if __name__ == "__main__":
    main()

