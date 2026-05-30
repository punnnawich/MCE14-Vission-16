# MCE14 Vission 16
### ระบบตรวจจับลูกบอลและคำนวณจุดตก 3 มิติ (HSV + Curve Fitting) สำหรับหุ่นยนต์รับบอล

ระบบประมวลผล 3D Vision เวอร์ชัน 16 พัฒนาขึ้นตามข้อกำหนดใหม่ โดยเปลี่ยนผ่านจากระบบโมเดล Deep Learning (YOLOv8) และ Kalman Filter มาเป็นระบบประมวลผลภาพแบบดั้งเดิม (Computer Vision) ที่ทำงานได้รวดเร็ว ลดความต้องการของทรัพยากรระบบ และรองรับการติดตามหุ่นยนต์ไปในตัว

## สถาปัตยกรรมระบบ (Pipeline Overview)

```
[OAK-D Lite]
    │
    ▼
Image Capture          ── RGB frame + Depth map (synchronized)
    │
    ▼
HSV Color Segmentation ── Binary mask ของพื้นที่สีแดง
    │
    ▼
Blob Detection         ── หา centroid (cx, cy) ของลูกบอล
    │
    ▼
Depth Lookup + 3D Proj ── แปลง pixel → (X, Y, Z) เมตร
    │
    ▼
Median Filter          ── ลด noise ของพิกัด 3D
    │
    ▼
Release Detection      ── ตรวจสอบการปล่อยลูกบอลออกจากมือ
    │
    ▼ (เริ่มเก็บ trajectory เมื่อ release = True)
Trajectory Predictor   ── ฟิตสมการ Projectile Curve Fitting หาจุดตก
    │
    ├── Robot Tracking ── ติดตามพิกัดหุ่นยนต์ (ArUco / Color Marker)
    │
    ▼
Coordinate Transform   ── แปลงพิกัดกล้อง → พิกัดหุ่นยนต์
    │
    ▼
UDP Transmission       ── ส่งพิกัดเป้าหมาย (JSON) + Heartbeat ไปที่ ESP32
```

## โครงสร้างโฟลเดอร์

```
MCE14-Vission-16/
├── src/
│   ├── ball_detector.py         # ตัวตรวจจับลูกบอล (HSV + Contour Filtering)
│   ├── median_filter.py         # ตัวกรองสัญญาณรบกวน 3 มิติ (Median Filter)
│   ├── release_detector.py      # ตัวตรวจจับการโยน/ปล่อยลูกบอล (Velocity + Separation)
│   ├── projectile_predictor.py  # ตัวทำนายจุดตก (Curve Fitting: Linear X/Y, Parabolic Z)
│   ├── robot_tracker.py         # ตัวติดตามพิกัดหุ่นยนต์ด้วย ArUco
│   ├── robot_comms.py           # ตัวจัดการส่ง UDP JSON และ Heartbeat
│   ├── debug_visualizer.py      # ตัววาด GUI Overlay แสดงพิกัดและเส้นทาง
│   ├── latency_profiler.py      # ตัวจับเวลาการทำงานของแต่ละส่วนประมวลผล
│   ├── vision_pipeline.py       # ไฟล์รันหลัก (Main Loop Orchestrator)
│   └── config.yaml              # ศูนย์รวมการตั้งค่าทั้งหมด
├── tools/
│   └── calibration.py           # สคริปต์ Calibrate หาความสูงกล้องและเมทริกซ์หมุนพิกัด
├── firmware/
│   └── esp32_robot.ino          # ซอร์สโค้ดสำหรับบอร์ดหุ่นยนต์ ESP32 (UDP JSON parsing)
├── README.md
└── requirements.txt
```

## Quick Start

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. กำหนดค่าต่างๆ ใน `src/config.yaml`
ปรับแต่งค่าเครือข่าย ค่าพิกัด และความสูงกล้อง ให้สอดคล้องกับสภาพสนามจริง

### 3. รันโปรแกรมหลัก
```bash
cd src
python vision_pipeline.py
```

### 4. การ Calibrate พิกัดกล้อง
วางแผ่นตารางหมากรุก (Checkerboard) หรือแผ่น ArUco ที่จุดอ้างอิงพิกัดหุ่นยนต์แล้วรัน:
```bash
python tools/calibration.py
```
นำค่าความสูง (`z_floor`) และเมทริกซ์แปลงพิกัด (`R`, `T`) ที่ได้ไปอัปเดตใน `src/config.yaml`
