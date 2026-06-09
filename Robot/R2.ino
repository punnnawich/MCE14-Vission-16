// ════════════════════════════════════════════════════════════
//   robot_main.cpp — Omni Robot Firmware (CAMERA_READY Sync v5.1)
//
//   State Flow:
//     BOOT → S_WAIT_CAMERA → S_CHECK_HOME → [S_CORRECT_HOME →] S_IDLE
//          → S_MOVE → S_WAIT → S_BACK → S_WAIT_POS → [S_CORRECT_MOVE →] S_IDLE
//
//   Protocol (Vision PC → ESP32):
//     Binary 16 bytes: [seq:uint32][x:float32][y:float32][extra:uint32]
//     extra = 0  → BALL_POS  (หน่วย cm)
//     extra = 1  → ROBOT_POS (ตอบ REQUEST_POS)
//     extra = 2  → CAMERA_READY (ปลดล็อกสเตตแรกเริ่ม / รีเซ็ต Sequence)
// ════════════════════════════════════════════════════════════

#include <Arduino.h>
#include <math.h>
#include <ESP32Encoder.h>
#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WiFiUdp.h>

ESP32Encoder wheel1, wheel2, wheel3;
MPU6050 mpu;

// ════════════════════════════════════
// CONFIG — WiFi
// ════════════════════════════════════
const char* WIFI_SSID = "MCE14";
const char* WIFI_PASS = "12345678";
const uint16_t UDP_PORT  = 12345;

IPAddress PC_IP(192, 168, 137, 1);
bool       has_learned_pc_ip = false;
unsigned long reconnect_timer = 0;

// ── Packet struct ──────────────────────
#pragma pack(push, 1)
struct VisionPacket {
    uint32_t seq;
    float    x;
    float    y;
    uint32_t extra;   // 0 = BALL_POS, 1 = ROBOT_POS, 2 = CAMERA_READY
};
#pragma pack(pop)
static_assert(sizeof(VisionPacket) == 16, "Packet size must be 16 bytes");

// ── LED ──────────────────────────────
#define LED_RED    4
#define LED_GREEN  2
#define LED_YELLOW 15

// ── Motor Pins ────────────────────────
#define Upper1  16
#define Lower1  17
#define Upper2  18
#define Lower2  19
#define Upper3  23
#define Lower3  25

// ── Encoder Pins ─────────────────────
#define ENC1_A  26
#define ENC1_B  27
#define ENC2_A  13
#define ENC2_B  14
#define ENC3_A  32
#define ENC3_B  33

// ── Robot params ──────────────────────
#define PPR       17
#define N_GEAR    19
#define WHEEL_R   0.041f
#define ROBOT_L   0.2715f
#define DT        0.01f

const float PULSE_TO_RAD = (2.0f * PI) / (PPR * N_GEAR);

#define CORRECT_THRESHOLD_CM  3.0f
#define POS_TOLERANCE_RAD     0.35f

// ── IMU ──────────────────────────────
float yaw         = 0;
float gyroBiasZ   = 0;
float targetYaw   = 0;
unsigned long imuPrevTime = 0;

void calibrateGyro(int samples = 1000) {
    long sum = 0;
    int16_t ax, ay, az, gx, gy, gz;
    Serial.print("Calibrating gyro");
    for (int i = 0; i < samples; i++) {
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        sum += gz;
        if (i % 100 == 0) Serial.print(".");
        delay(2);
    }
    gyroBiasZ = sum / (float)samples;
   // Serial.printf("\nGyro bias Z: %.2f\n", gyroBiasZ);
}

void updateYaw() {
    unsigned long now = micros();
    float dt = (now - imuPrevTime) / 1e6f;
    imuPrevTime = now;
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    float gz_dps = (gz - gyroBiasZ) / 65.5f;
    if (fabsf(gz_dps) < 0.3f) gz_dps = 0;
    yaw += gz_dps * (PI / 180.0f) * dt;
    if (yaw >  PI) yaw -= 2.0f * PI;
    if (yaw < -PI) yaw += 2.0f * PI;
}

// ── Encoder / shared vars ─────────────
volatile long pos1, pos2, pos3;
volatile long last_pos1, last_pos2, last_pos3;
volatile bool isr_flag = false;
volatile bool running  = false;

float actual_pos1 = 0, actual_vel1 = 0;
float actual_pos2 = 0, actual_vel2 = 0;
float actual_pos3 = 0, actual_vel3 = 0;
float t_elapsed   = 0;
float theta_corr  = 0;
float motion_accel = 3.0f;

float target_vel1 = 0, target_vel2 = 0, target_vel3 = 0;
float duty1_out = 0, duty2_out = 0, duty3_out = 0;
float target_pos1 = 0, target_pos2 = 0, target_pos3 = 0;

// ── Trapezoidal profile ───────────────
float prof_a = 0, prof_vmax = 0, prof_dist = 0, tacce = 0, sd = 0;
float cos_angle = 1, sin_angle = 0;

void profile_init(float tx, float ty, float acc) {
    prof_dist = sqrtf(tx*tx + ty*ty);
    float ang = atan2f(ty, tx);
    cos_angle = cosf(ang);
    sin_angle = sinf(ang);
    prof_a    = acc;
    prof_vmax = sqrtf(acc * prof_dist);
    tacce     = prof_vmax / acc;
    sd        = 0.5f * acc * tacce * tacce;
}

float profile_vel(float t) {
    if      (t <= tacce)         return prof_a * t;
    else if (t <= 2.0f * tacce)  return prof_vmax - prof_a * (t - tacce);
    else                         return 0.0f;
}

float profile_pos(float t) {
    if (t <= tacce) {
        return 0.5f * prof_a * t * t;
    } else if (t <= 2.0f * tacce) {
        float dt2 = t - tacce;
        return sd + prof_vmax*dt2 - 0.5f*prof_a*dt2*dt2;
    } else {
        return prof_dist;
    }
}

// ── PID ──────────────────────────────
struct PID {
    float Kp, Ki, integral, limit;
    float update(float err) {
        integral = constrain(integral + err * DT, -limit, limit);
        return constrain(Kp * err + Ki * integral, -limit, limit);
    }
    void reset() { integral = 0; }
};

PID pid_pos1 = { 5.6415f, 0.0f, 0.0f, 40.0f };
PID pid_pos2 = { 5.6415f, 0.0f, 0.0f, 40.0f };
PID pid_pos3 = { 5.6415f, 0.0f, 0.0f, 40.0f };
PID pid_vel1 = { 0.8f, 10.0f, 0.0f, 12.0f };
PID pid_vel2 = { 0.8f, 10.0f, 0.0f, 12.0f };
PID pid_vel3 = { 0.8f, 10.0f, 0.0f, 12.0f };
PID pid_yaw  = { 15.0f, 0.0f, 0.0f, 6.04f };

// ── Motor ─────────────────────────────
void motor_init() {
    ledcSetup(0, 5000, 8); ledcAttachPin(Upper1, 0);
    ledcSetup(1, 5000, 8); ledcAttachPin(Lower1, 1);
    ledcSetup(2, 5000, 8); ledcAttachPin(Upper2, 2);
    ledcSetup(3, 5000, 8); ledcAttachPin(Lower2, 3);
    ledcSetup(4, 5000, 8); ledcAttachPin(Upper3, 4);
    ledcSetup(5, 5000, 8); ledcAttachPin(Lower3, 5);
}

void set_motor(int rpwm, int lpwm, float duty) {
    duty = constrain(duty, -12.0f, 12.0f);
    uint8_t pwm = (uint8_t)((fabsf(duty) / 12.0f) * 255);
    if (duty > 0.05f) {
        ledcWrite(rpwm, pwm); ledcWrite(lpwm, 0);
    } else if (duty < -0.05f) {
        ledcWrite(rpwm, 0);   ledcWrite(lpwm, pwm);
    } else {
        ledcWrite(rpwm, 0);   ledcWrite(lpwm, 0);
    }
}

void stop_all() { for (int i = 0; i < 6; i++) ledcWrite(i, 0); }

// ── LED helpers ───────────────────────
void led_init() {
    pinMode(LED_RED,    OUTPUT);
    pinMode(LED_GREEN,  OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    digitalWrite(LED_RED,    LOW);
    digitalWrite(LED_GREEN,  LOW);
    digitalWrite(LED_YELLOW, LOW);
}

void setLED(bool r, bool g, bool y) {
    digitalWrite(LED_RED,    r ? HIGH : LOW);
    digitalWrite(LED_GREEN,  g ? HIGH : LOW);
    digitalWrite(LED_YELLOW, y ? HIGH : LOW);
}

// ════════════════════════════════════
// WiFi + UDP
// ════════════════════════════════════
WiFiUDP udp;

void comms_init() {
    Serial.println("\n[Comms] Initializing...");
    WiFi.disconnect(true);
    delay(200);

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
        if (millis() - t0 > 15000) { ESP.restart(); }
    }
    WiFi.setSleep(false);
    //Serial.printf("\n[Comms] IP: %s\n", WiFi.localIP().toString().c_str());
    udp.begin(UDP_PORT);
}

void comms_send(const char* msg) {
    if (WiFi.status() != WL_CONNECTED) return;
    udp.beginPacket(PC_IP, UDP_PORT);
    udp.write((const uint8_t*)msg, strlen(msg));
    udp.endPacket();
   // Serial.printf("[TX] \"%s\"\n", msg);
}

void comms_send_broadcast(const char* msg) {
    if (WiFi.status() != WL_CONNECTED) return;
    IPAddress bc = WiFi.localIP();
    bc[3] = 255;
    udp.beginPacket(bc, UDP_PORT);
    udp.write((const uint8_t*)msg, strlen(msg));
    udp.endPacket();
    //Serial.printf("[TX-BC] \"%s\"\n", msg);
}

void send_request_pos_once() {
    if (!has_learned_pc_ip)
        comms_send_broadcast("REQUEST_POS");
    else
        comms_send("REQUEST_POS");
}

void send_ready_once() {
    if (!has_learned_pc_ip)
        comms_send_broadcast("READY");
    else
        comms_send("READY");
}

// ── Shared inter-task variables ────────
SemaphoreHandle_t xMutex;

volatile float cmd_x      = 0, cmd_y    = 0;
volatile float robot_x    = 0, robot_y  = 0;
volatile bool  new_ball      = false;
volatile bool  new_robot_pos = false;
volatile bool  camera_ready  = false; // สัญญาณปล่อยตัวจาก Vision PC

volatile uint32_t lastSeqNum  = 0;
volatile bool     firstPacket = true;

float active_target_x = 0.0f, active_target_y = 0.0f;

bool comms_receive() {
    int size = udp.parsePacket();
    if (size <= 0) return false;

    if (size < (int)sizeof(VisionPacket)) {
        udp.flush(); // ล้างบัฟเฟอร์ขยะทิ้งทันที คอมไม่ค้าง
        return false;
    }

    IPAddress remote_ip = udp.remoteIP();
    if (!has_learned_pc_ip || PC_IP != remote_ip) {
        PC_IP = remote_ip;
        has_learned_pc_ip = true;
       // Serial.printf("[Comms] PC IP learned: %s\n", PC_IP.toString().c_str());
    }

    VisionPacket pkt;
    udp.read((uint8_t*)&pkt, sizeof(pkt));

    // ── ดักจับ CAMERA_READY (extra == 2) ก่อนเช็ก Sequence เผื่อคอมรีสตาร์ท ──
    if (pkt.extra == 2) {
        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
            camera_ready = true;
            firstPacket  = true;  // ปลดล็อกให้ยอมรับค่า seq เริ่มต้นใหม่ได้
            lastSeqNum   = 0;
            xSemaphoreGive(xMutex);
        }
       // Serial.println("[RX] >>> CAMERA_READY Signal Received <<<");
        return true;
    }

    if (pkt.extra == 0) { // Check sequence number ONLY for BALL_POS
        if (!firstPacket && pkt.seq <= lastSeqNum) return false;
        lastSeqNum  = pkt.seq;
        firstPacket = false;
    }

    if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
        if (pkt.extra == 0) {
            cmd_x    = pkt.x;
            cmd_y    = pkt.y;
            new_ball = true;
          //  Serial.printf("[RX] BALL_POS  x=%.1f y=%.1f cm\n", pkt.x, pkt.y);
        } else if (pkt.extra == 1) {
            robot_x       = pkt.x;
            robot_y       = pkt.y;
            new_robot_pos = true;
          //  Serial.printf("[RX] ROBOT_POS rx=%.1f ry=%.1f cm\n", pkt.x, pkt.y);
        }
        xSemaphoreGive(xMutex);
    }
    return true;
}

// ── UDP Task (Core 0) ─────────────────
void udpTask(void* pvParameters) {
    for (;;) {
        if (WiFi.status() != WL_CONNECTED) {
            if (millis() - reconnect_timer >= 2000) {
                reconnect_timer = millis();
                WiFi.begin(WIFI_SSID, WIFI_PASS);
            }
            vTaskDelay(10); continue;
        }
        comms_receive();
        vTaskDelay(1);
    }
}

// ── Timer ISR ─────────────────────────
hw_timer_t* timer    = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR onTimer() {
    portENTER_CRITICAL_ISR(&timerMux);
    pos1 = wheel1.getCount();
    pos2 = wheel2.getCount();
    pos3 = wheel3.getCount();

    actual_pos1 = pos1 * PULSE_TO_RAD;
    actual_pos2 = pos2 * PULSE_TO_RAD;
    actual_pos3 = pos3 * PULSE_TO_RAD;

    actual_vel1 = (pos1 - last_pos1) * PULSE_TO_RAD / DT;
    actual_vel2 = (pos2 - last_pos2) * PULSE_TO_RAD / DT;
    actual_vel3 = (pos3 - last_pos3) * PULSE_TO_RAD / DT;

    last_pos1 = pos1; last_pos2 = pos2; last_pos3 = pos3;
    
    if (running) {
        t_elapsed += DT;
    }
    isr_flag = true;
    portEXIT_CRITICAL_ISR(&timerMux);
}

// ── Utility ───────────────────────────
void reset_motion() {
    portENTER_CRITICAL(&timerMux);
    t_elapsed   = 0;
    target_vel1 = target_vel2 = target_vel3 = 0;
    duty1_out   = duty2_out   = duty3_out   = 0;
    pos1 = pos2 = pos3 = 0;
    last_pos1 = last_pos2 = last_pos3 = 0;
    actual_pos1 = actual_pos2 = actual_pos3 = 0;
    actual_vel1 = actual_vel2 = actual_vel3 = 0;
    target_pos1 = target_pos2 = target_pos3 = 0;
    yaw = 0; targetYaw = 0;
    wheel1.clearCount(); wheel2.clearCount(); wheel3.clearCount();
    portEXIT_CRITICAL(&timerMux);

    pid_pos1.reset(); pid_pos2.reset(); pid_pos3.reset();
    pid_vel1.reset(); pid_vel2.reset(); pid_vel3.reset();
    pid_yaw.reset();
    lastSeqNum  = 0;
    firstPacket = true;
}

void reset_all() {
    reset_motion();
    yaw = 0; targetYaw = 0;
}

uint32_t ledBlinkTime = 0;
bool     ledBlinkOn   = false;

void updateBlinkLED(int pin, uint32_t intervalMs) {
    if (millis() - ledBlinkTime >= intervalMs) {
        ledBlinkTime = millis();
        ledBlinkOn   = !ledBlinkOn;
        digitalWrite(pin, ledBlinkOn ? HIGH : LOW);
    }
}

bool is_position_reached() {
    return fabsf(target_pos1 - actual_pos1) < POS_TOLERANCE_RAD &&
           fabsf(target_pos2 - actual_pos2) < POS_TOLERANCE_RAD &&
           fabsf(target_pos3 - actual_pos3) < POS_TOLERANCE_RAD;
}

// ════════════════════════════════════
// State Machine
// ════════════════════════════════════
enum State {
    S_WAIT_CAMERA,   // เริ่มต้น: รอกล้องบูตเสร็จ
    S_IDLE,
    S_MOVE,
    S_WAIT,
    S_BACK,
    S_CHECK_HOME,
    S_CORRECT_HOME,
    S_WAIT_POS,
    S_CORRECT_MOVE
};

State    state      = S_WAIT_CAMERA;
uint32_t t_wait     = 0;
uint32_t t_req_sent = 0;

// ── ฟังก์ชัน transition เข้าแต่ละ State ──
void enter_wait_camera() {
    stop_all();
    target_vel1 = target_vel2 = target_vel3 = 0;
    portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);
    if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
        camera_ready = false;
        xSemaphoreGive(xMutex);
    }
    state = S_WAIT_CAMERA;
    setLED(true, false, true); // เปิด แดง + เหลือง รอเริ่มต้นคอยสัญญาณ
   // Serial.println("[SM] -> S_WAIT_CAMERA | Waiting for Vision PC warm up...");
}

void enter_check_home() {
    stop_all();
    target_vel1 = target_vel2 = target_vel3 = 0;
    portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);
    if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
        new_robot_pos = false;
        xSemaphoreGive(xMutex);
    }
    firstPacket = true;        
    send_request_pos_once();   
    t_req_sent = millis();
    state = S_CHECK_HOME;
    setLED(true, false, false);
   // Serial.println("[SM] -> S_CHECK_HOME");
}

void enter_wait_pos() {
    stop_all();
    target_vel1 = target_vel2 = target_vel3 = 0;
    portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);
    if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
        new_robot_pos = false;
        xSemaphoreGive(xMutex);
    }
    firstPacket = true;        
    send_request_pos_once();   
    t_req_sent = millis();
    state = S_WAIT_POS;
    setLED(true, false, false);
   // Serial.println("[SM] -> S_WAIT_POS");
}

void enter_idle() {
    reset_all();
    firstPacket = true;        
    state = S_IDLE;
    setLED(false, false, true);
    send_ready_once();         
    //Serial.println("[SM] -> S_IDLE | READY sent");
}

// ════════════════════════════════════
// Setup
// ════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("════ BOOT ════");

    led_init();
    setLED(true, true, true);

    wheel1.attachFullQuad(ENC1_A, ENC1_B);
    wheel2.attachFullQuad(ENC2_A, ENC2_B);
    wheel3.attachFullQuad(ENC3_A, ENC3_B);
    wheel1.clearCount(); wheel2.clearCount(); wheel3.clearCount();
    Serial.println("✓ Encoder");

    motor_init(); stop_all();
    Serial.println("✓ Motor");

    Wire.begin(21, 22);
    delay(200);
    Wire.setClock(400000);
    mpu.initialize();
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    calibrateGyro(1000);
    targetYaw   = 0;
    imuPrevTime = micros();
    Serial.println("✓ IMU");

    comms_init();
    Serial.println("✓ WiFi");

    profile_init(0.0f, 0.0f, 3.0f);

    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 10000, true);
    timerAlarmEnable(timer);
    Serial.println("✓ Timer 100 Hz");

    xMutex = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(udpTask, "UDP_Task", 8192, NULL, 1, NULL, 0);
    Serial.println("✓ UDP Task");

    // บูตเข้าสู่สเตตรอกล้องเซ็ตอัพ
    enter_wait_camera();
    Serial.println("════ BOOT DONE ════");
}

// ════════════════════════════════════
// Loop
// ════════════════════════════════════
void loop() {
    if (!isr_flag) return;
    isr_flag = false;

    static uint8_t cnt = 0;
    cnt++;

    // ── 1. Velocity PID (10 ms) ──────
    bool is_running;
    portENTER_CRITICAL(&timerMux); is_running = running; portEXIT_CRITICAL(&timerMux);

    duty1_out = pid_vel1.update(target_vel1 - actual_vel1) * is_running;
    duty2_out = pid_vel2.update(target_vel2 - actual_vel2) * is_running;
    duty3_out = pid_vel3.update(target_vel3 - actual_vel3) * is_running;
    set_motor(0, 1, duty1_out);
    set_motor(2, 3, duty2_out);
    set_motor(4, 5, duty3_out);

    // ── 2. Profile + IK + IMU (50 ms) ─
    if (cnt % 5 == 0) {
        float local_t;
        portENTER_CRITICAL(&timerMux); local_t = t_elapsed; portEXIT_CRITICAL(&timerMux);

        float vel = profile_vel(local_t);
        float pos = profile_pos(local_t);
        float vx = vel * cos_angle, vy = vel * sin_angle;
        float px = pos * cos_angle, py = pos * sin_angle;

        updateYaw();
        float yaw_err = targetYaw - yaw;
        theta_corr    = pid_yaw.update(yaw_err);

        float w1 = ( 0.000f*vx - 1.000f*vy + theta_corr*ROBOT_L) / WHEEL_R;
        float w2 = ( 0.866f*vx + 0.500f*vy + theta_corr*ROBOT_L) / WHEEL_R;
        float w3 = (-0.866f*vx + 0.500f*vy + theta_corr*ROBOT_L) / WHEEL_R;

        target_pos1 = ( 0.000f*px - 1.000f*py) / WHEEL_R;
        target_pos2 = ( 0.866f*px + 0.500f*py) / WHEEL_R;
        target_pos3 = (-0.866f*px + 0.500f*py) / WHEEL_R;

        target_vel1 = w1 + pid_pos1.update(target_pos1 - actual_pos1);
        target_vel2 = w2 + pid_pos2.update(target_pos2 - actual_pos2);
        target_vel3 = w3 + pid_pos3.update(target_pos3 - actual_pos3);
    }

    // ── 3. Serial Debug (100 ms) ──────
    if (cnt % 10 == 0) {
        cnt = 0;
       // Serial.printf("%.3f,%.3f,%.3f,%.2f,%.3f,%.2f\n",
          //  actual_pos1, actual_pos2, actual_pos3,
           // yaw*(180.0f/PI), theta_corr, targetYaw*(180.0f/PI));
    }

    // ── 4. State Machine ──────────────
    float current_t;
    portENTER_CRITICAL(&timerMux); current_t = t_elapsed; portEXIT_CRITICAL(&timerMux);

    switch (state) {
        // ═════════════════════════════
        // S_WAIT_CAMERA (เวอร์ชันส่งสัญญานกู้ชีพขอบอกพิกัด)
        // ═════════════════════════════
        case S_WAIT_CAMERA: {  
            stop_all();
            target_vel1 = target_vel2 = target_vel3 = 0;
            
            // 1. หลอดไฟกะพริบเตือนสถานะปกติ
            if (millis() - ledBlinkTime >= 200) {
                ledBlinkTime = millis();
                ledBlinkOn   = !ledBlinkOn;
                digitalWrite(LED_RED,    ledBlinkOn ? HIGH : LOW);
                digitalWrite(LED_YELLOW, ledBlinkOn ? HIGH : LOW);
                digitalWrite(LED_GREEN,  ledBlinkOn ? HIGH : LOW);
            }

            // ✨ 2. ทะลวงจุดบอด: ยิง UDP ขอบอกพิกัดจาก PC ทุกๆ 500ms 
            static uint32_t lastRequestTime = 0;
            if (millis() - lastRequestTime >= 500) { 
                lastRequestTime = millis();
              //  Serial.println("[SM] 🛰️ Robot Rebooted! Requesting position from PC...");
                
                // 🚀 เรียกใช้ฟังก์ชันเดิมที่คุณเขียนไว้แล้วได้เลย มันจะ Broadcast หา PC อัตโนมัติถ้าเพิ่งเปิดเครื่อง
                send_request_pos_once();
            }

            // 3. เช็กว่า PC ตอบกลับมาหรือยังผ่าน Mutex
            bool camera_is_active = false;
            if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                if (camera_ready || new_ball || new_robot_pos) {
                    camera_is_active = true; 
                }
                xSemaphoreGive(xMutex);
            }

            // 4. จังหวะทะลวง: พอ PC ได้ยินเสียงสะกิด แล้วส่งพิกัดสวนกลับมา... โดดเข้าสเตตแรกทันที!
            if (camera_is_active) {
                digitalWrite(LED_RED,    LOW);
                digitalWrite(LED_YELLOW, LOW);
                digitalWrite(LED_GREEN,  LOW);
                
               // Serial.println("[SM] 🎉 Got Response from PC! Plunging to CHECK_HOME");
                enter_check_home(); 
            }
            break;
        }

        // ═════════════════════════════
        // S_CHECK_HOME
        // เช็กสเตตระยะห่าง Home และทำการ Retry ทุกๆ 2.0 วินาทีอย่างแม่นยำ (คอมไม่ค้าง)
        // ═════════════════════════════
        case S_CHECK_HOME:
            updateBlinkLED(LED_RED, 500);

            // แก้ไข: โครงสร้างป้องกันลูปสแปมแพ็กเก็ต ทำงานห่างกันครั้งละ 2 วินาทีชัดเจนเมื่อคอมเงียบ
            if (millis() - t_req_sent >= 5000) {
                t_req_sent = millis(); // บังคับอัปเดตเวลารอตั้งแต่ต้นบล็อก
                if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
                    bool still_waiting = !new_robot_pos;
                    xSemaphoreGive(xMutex);
                    if (still_waiting) {
                        send_request_pos_once();
                      //  Serial.println("[SM] CHECK_HOME: retry REQUEST_POS");
                    }
                }
            }

            if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
                if (new_robot_pos) {
                    new_robot_pos = false;
                    float rx = robot_x / 100.0f;
                    float ry = robot_y / 100.0f;
                    xSemaphoreGive(xMutex);

                    digitalWrite(LED_RED, LOW);
                    float err_cm = sqrtf(robot_x*robot_x + robot_y*robot_y);
                   // Serial.printf("[SM] CHECK_HOME: err=%.2f cm\n", err_cm);

                    if (err_cm > CORRECT_THRESHOLD_CM) {
                        reset_motion();
                        profile_init(-rx, -ry, motion_accel);
                        portENTER_CRITICAL(&timerMux); running = true; portEXIT_CRITICAL(&timerMux);
                        state = S_CORRECT_HOME;
                        setLED(false, true, false);
                      //  Serial.printf("[SM] CHECK_HOME->CORRECT_HOME | err=%.2f cm\n", err_cm);
                    } else {
                        enter_idle();  
                    }
                } else {
                    xSemaphoreGive(xMutex);
                }
            }
            break;

        case S_CORRECT_HOME:
            updateBlinkLED(LED_GREEN, 200);
            if ((current_t >= 2.0f * tacce) && (is_position_reached() || (current_t >= 2.0f * tacce + 1.0f))) {
                digitalWrite(LED_GREEN, LOW);
                stop_all();
                portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);
                enter_idle();  
            }
            break;

        case S_IDLE:
            stop_all();
            target_vel1 = target_vel2 = target_vel3 = 0;

            if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
                if (new_ball) {
                    new_ball = false;
                    active_target_x = cmd_x / 100.0f;
                    active_target_y = cmd_y / 100.0f;
                    xSemaphoreGive(xMutex);

                    reset_motion();
                    profile_init(active_target_x, active_target_y, motion_accel);
                    portENTER_CRITICAL(&timerMux); running = true; portEXIT_CRITICAL(&timerMux);

                    state = S_MOVE;
                    setLED(false, true, false);
                   // Serial.printf("[SM] IDLE->MOVE (%.3f, %.3f)m\n", active_target_x, active_target_y);
                } else {
                    xSemaphoreGive(xMutex);
                }
            }
            break;

        case S_MOVE:
            if ((current_t >= 2.0f * tacce) && (is_position_reached() || (current_t >= 2.0f * tacce + 1.0f))) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
                portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);
                t_wait = millis();
                state  = S_WAIT;
                setLED(false, false, true);
               // Serial.println("[SM] MOVE->WAIT");
            }
            break;

        case S_WAIT:
            stop_all();
            target_vel1 = target_vel2 = target_vel3 = 0;
            updateBlinkLED(LED_YELLOW, 200);

            if (millis() - t_wait >= 5000) {
                digitalWrite(LED_YELLOW, LOW);
                reset_motion();
                profile_init(-active_target_x, -active_target_y, motion_accel);
                portENTER_CRITICAL(&timerMux); running = true; portEXIT_CRITICAL(&timerMux);
                state = S_BACK;
                setLED(true, false, false);
              //  Serial.println("[SM] WAIT->BACK");
            }
            break;

        case S_BACK:
            if ((current_t >= 2.0f * tacce) && (is_position_reached() || (current_t >= 2.0f * tacce + 1.0f))) {
                enter_wait_pos();  
            }
            break;

        case S_WAIT_POS:
            updateBlinkLED(LED_RED, 500);

            // แก้ไข: คุมจังหวะให้ส่งขอข้อมูลแก้ตำแหน่งกลับ ทุกๆ 2 วินาทีอย่างเสถียร
            if (millis() - t_req_sent >= 5000) {
                t_req_sent = millis(); // บังคับล็อกช่วงเวลารอไว้ก่อน
                if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
                    bool still_waiting = !new_robot_pos;
                    xSemaphoreGive(xMutex);
                    if (still_waiting) {
                        send_request_pos_once();
                      //  Serial.println("[SM] WAIT_POS: retry REQUEST_POS");
                    }
                }
            }

            if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
                if (new_robot_pos) {
                    new_robot_pos = false;
                    float rx = robot_x / 100.0f;
                    float ry = robot_y / 100.0f;
                    xSemaphoreGive(xMutex);

                    digitalWrite(LED_RED, LOW);
                    float err_cm = sqrtf(robot_x*robot_x + robot_y*robot_y);
                   // Serial.printf("[SM] WAIT_POS: err=%.2f cm\n", err_cm);

                    if (err_cm > CORRECT_THRESHOLD_CM) {
                        reset_motion();
                        profile_init(-rx, -ry, motion_accel);
                        portENTER_CRITICAL(&timerMux); running = true; portEXIT_CRITICAL(&timerMux);
                        state = S_CORRECT_MOVE;
                        setLED(false, true, false);
                       // Serial.printf("[SM] WAIT_POS->CORRECT_MOVE | err=%.2f cm\n", err_cm);
                    } else {
                        enter_idle();  
                    }
                } else {
                    xSemaphoreGive(xMutex);
                }
            }
            break;

        case S_CORRECT_MOVE:
            updateBlinkLED(LED_GREEN, 200);
            if ((current_t >= 2.0f * tacce) && (is_position_reached() || (current_t >= 2.0f * tacce + 1.0f))) {
                digitalWrite(LED_GREEN, LOW);
                stop_all();
                portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);
                enter_wait_pos();  
            }
            break;
    }
}