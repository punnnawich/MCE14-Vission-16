# MCE14 Vission 16
### ระบบ 3D Vision สำหรับหุ่นยนต์รับบอล — OAK-D Lite + ESP32 Omni Robot

ระบบประมวลผล 3D Vision เวอร์ชัน 16 ใช้กล้อง OAK-D Lite ตรวจจับลูกบอลแดง คำนวณจุดตกด้วย Projectile Curve Fitting แล้วส่งพิกัดเป้าหมายผ่าน UDP ไปยังหุ่นยนต์ Omni-Wheel 3 ล้อ (ESP32)

---

## สถาปัตยกรรมระบบ

```
┌─────────────────────────────────────────────────────────┐
│                    PC (Python)                          │
│                                                         │
│  [OAK-D Lite] ─── RGB + Depth (30 FPS)                 │
│       │                                                 │
│       ▼                                                 │
│  HSV Color Segmentation ── Binary mask สีแดง            │
│       │                                                 │
│       ▼                                                 │
│  Blob Detection ── centroid (cx, cy)                    │
│       │                                                 │
│       ▼                                                 │
│  Depth Lookup + 3D Projection ── (X, Y, Z) เมตร        │
│       │                                                 │
│       ▼                                                 │
│  Median Filter ── ลด depth noise                        │
│       │                                                 │
│       ▼                                                 │
│  Release Detection ── ตรวจจับการปล่อยลูก                │
│       │           (vel > 1.5 m/s + disp > 15cm)         │
│       ▼                                                 │
│  Projectile Predictor ── Curve Fitting หาจุดตก          │
│       │              (Parabolic Z + Linear X,Y)         │
│       ▼                                                 │
│  UDP Binary Packet ── 16 bytes [seq|x|y|extra]          │
│                                                         │
└────────────────────┬────────────────────────────────────┘
                     │ WiFi UDP (port 12345)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  ESP32 (Arduino)                        │
│                                                         │
│  Core 0: UDP Task ── รับ/ส่ง packet (1 kHz)             │
│       │                                                 │
│  Core 1: Control Loop (100 Hz)                          │
│       ├── Velocity PID ── 3 มอเตอร์ (100 Hz)           │
│       ├── Position PID + Trapezoidal Profile (20 Hz)    │
│       ├── IMU Yaw Correction (20 Hz)                    │
│       └── State Machine:                                │
│            IDLE → MOVE → WAIT → CHECK_POS → BACK        │
│                                  → BACK_CHECK            │
│                                  → BACK_CORRECT → IDLE   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## โปรโตคอล UDP (Binary 16 bytes)

### PC → ESP32
| Field | Type | Size | Description |
|-------|------|------|-------------|
| `seq` | uint32 | 4B | Sequence number |
| `x` | float32 | 4B | พิกัด X (cm) |
| `y` | float32 | 4B | พิกัด Y (cm) |
| `extra` | uint32 | 4B | 0 = BALL_POS, 1 = ROBOT_POS |

### ESP32 → PC
| Message | Description |
|---------|-------------|
| `REQUEST_POS` | ขอตำแหน่งหุ่นจากกล้อง |
| `OKAY` | พร้อมรับคำสั่งใหม่ |

---

## โครงสร้างโฟลเดอร์

```
MCE14-Vission-16/
├── src/
│   ├── vision_pipeline.py       # Main Loop — orchestrator ทั้งระบบ
│   ├── ball_detector.py         # ตรวจจับลูกบอล (HSV + Contour)
│   ├── median_filter.py         # Median Filter 3D
│   ├── release_detector.py      # ตรวจจับการปล่อยลูก (Velocity + Displacement)
│   ├── projectile_predictor.py  # ทำนายจุดตก (Parabolic Curve Fitting)
│   ├── robot_tracker.py         # ติดตามหุ่นยนต์ (HSV Gold Marker)
│   ├── robot_comms.py           # UDP Binary Communication (16-byte packets)
│   ├── debug_visualizer.py      # GUI Overlay แสดงพิกัด + เส้นทาง
│   ├── data_logger.py           # บันทึก trajectory ลง CSV
│   ├── latency_profiler.py      # วัดเวลาแต่ละ stage ของ pipeline
│   ├── calibrate_camera.py      # Calibrate กล้อง (Checkerboard)
│   ├── plot_3d.py               # 3D Real-time Plot (UDP receiver)
│   ├── test_esp_comms.py        # Interactive test tool สำหรับ UDP
│   └── config.yaml              # ศูนย์รวมการตั้งค่าทั้งหมด
├── Robot/
│   └── ROBOT_CONTROLahhh.ino    # ESP32 Firmware (FreeRTOS + PID + State Machine)
├── logs/                        # Trajectory logs (CSV)
├── tools/                       # Utility scripts
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```
> ต้องใช้ **DepthAI v3+** (`pip install depthai --upgrade`)

### 2. เชื่อมต่อ WiFi
PC และ ESP32 ต้องอยู่ WiFi วงเดียวกัน (SSID: `MCE14` / Password: `12345678`)

### 3. Flash ESP32
เปิด `Robot/ROBOT_CONTROLahhh.ino` ใน Arduino IDE แล้ว Upload

### 4. รันระบบ Vision
```bash
cd src
python vision_pipeline.py
```

### 5. Calibrate (SET ZERO)
1. วางลูกบอลที่จุดศูนย์กลางหุ่นยนต์
2. กด `z` ในหน้าต่าง RGB
3. รอ 10 วินาที (warmup) จนขึ้น `✅ Transmission ENABLED`
4. เริ่มโยนลูกบอลได้

### คีย์ลัด
| คีย์ | หน้าที่ |
|------|--------|
| `z` | SET ZERO — ตั้งจุดเริ่มต้น |
| `q` | ออกจากโปรแกรม |

---

## เงื่อนไขการส่ง BALL_POS

ระบบจะส่งจุดตกไปยัง ESP32 เมื่อผ่านเงื่อนไข **ทั้ง 4 ข้อ**:

1. ✅ `is_calibrated` — กด SET ZERO แล้ว
2. ✅ `warmup_ok` — ผ่านไป 10 วินาทีหลัง SET ZERO
3. ✅ `elapsed_since_release ≤ 1.0s` — prediction ภายในหน้าต่างเวลา
4. ✅ `robot_ready` — ESP32 ส่ง OKAY มาแล้ว (พร้อมรับคำสั่ง)

---

## State Machine (ESP32)

```
IDLE ──(BALL_POS)──→ MOVE ──(เสร็จ)──→ WAIT (2s)
  ▲                                       │
  │                                       ▼
  │                                  CHECK_POS
  │                                  (REQUEST_POS)
  │                                       │
  │                                       ▼
  │                                    BACK
  │                                       │
  │                                       ▼
  │                                  BACK_CHECK ◄──┐
  │                                    │    │      │
  │                        (err ≤ 3cm) │    │ (err > 3cm)
  │                                    │    └──→ BACK_CORRECT
  └────────(OKAY)──────────────────────┘         (max 3 ครั้ง)
```

| State | LED | หน้าที่ |
|-------|-----|--------|
| `S_IDLE` | 🟡 เหลือง | รอ BALL_POS |
| `S_MOVE` | 🟢 เขียว | วิ่งไปเป้าหมาย (Trapezoidal Profile) |
| `S_WAIT` | 🟡 กะพริบ | รอ 2 วินาที |
| `S_CHECK_POS` | 🔴 กะพริบ | ส่ง REQUEST_POS → รอพิกัดจากกล้อง |
| `S_BACK` | 🔴 | วิ่งกลับ Home |
| `S_BACK_CHECK` | 🔴+🟡 | ตรวจตำแหน่ง (ถ้าเกิน 3cm → แก้ไข) |
| `S_BACK_CORRECT` | 🔴+🟢 | วิ่งแก้ตำแหน่ง (สูงสุด 3 ครั้ง) |

---

## Config Reference (`config.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `camera` | `fps` | 30 | Frame rate |
| `camera` | `resolution_w/h` | 640×360 | ความละเอียด |
| `camera` | `preset_mode` | FAST_ACCURACY | Stereo preset |
| `communication` | `esp32_ip` | 10.252.108.237 | IP ของ ESP32 |
| `communication` | `esp32_port` | 12345 | UDP port |
| `predictor` | `min_points` | 3 | จุดขั้นต่ำสำหรับ curve fit |
| `predictor` | `z_catch` | 0.25 | ความสูงรับลูก (เมตร) |
| `predictor` | `workspace_radius_m` | 0.5 | รัศมี workspace (เมตร) |
| `release` | `vel_threshold` | 1.5 | ความเร็วขั้นต่ำ (m/s) |
| `release` | `max_transmission_delay_s` | 1.0 | หน้าต่างเวลาส่ง |
| `extrinsics` | `T[2]` | ~0.86 | ความสูงกล้อง (เมตร) — auto-calibrate ได้ |

---

## ESP32 Firmware Features

- **Dual-Core FreeRTOS**: Core 0 = UDP, Core 1 = Control Loop
- **Cascaded PID**: Position (20 Hz) → Velocity (100 Hz)
- **Trapezoidal Motion Profile**: Smooth acceleration/deceleration
- **IMU Yaw Correction**: MPU6050 gyroscope heading lock
- **Auto-IP Learning**: เรียนรู้ IP ของ PC จาก packet แรก
- **Task Watchdog**: 5 วินาที timeout ป้องกัน firmware ค้าง
- **Position Correction Loop**: ตรวจและแก้ตำแหน่งอัตโนมัติ (สูงสุด 3 ครั้ง)

---

## Tools

### Test Communication
```bash
cd src
python test_esp_comms.py
```
ทดสอบส่ง BALL_POS / ROBOT_POS ไปยัง ESP32 แบบ interactive

### 3D Real-time Plot
```bash
python plot_3d.py
```
แสดง trajectory 3D แบบ real-time (รับข้อมูลผ่าน UDP port 5006)

### Camera Calibration
```bash
python calibrate_camera.py
```
Calibrate intrinsics ด้วย checkerboard pattern
