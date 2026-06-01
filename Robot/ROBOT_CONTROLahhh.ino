
// ════════════════════════════════════════════════════════════
//  robot_main.cpp — Omni Robot Firmware
//  State: IDLE → MOVE → WAIT → CHECK_POS → BACK → IDLE
//
//  Protocol (Vision PC → ESP32):
//    Binary 16 bytes: [seq:uint32][x:float32][y:float32][extra:uint32]
//    extra = 0  → BALL_POS  (หน่วย cm)
//    extra = 1  → ROBOT_POS (ตอบ REQUEST_POS)
//
//  Protocol (ESP32 → Vision PC):
//    "REQUEST_POS"  11 bytes ASCII
//    "OKAY"          4 bytes ASCII
// ════════════════════════════════════════════════════════════

#include <Arduino.h>
#include <math.h>
#include <esp_task_wdt.h>
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
// 💡 ต่อ Hotspot มือถือหรือเน็ตวงอื่น → ใส่ // หน้า #define USE_STATIC_IP
//    ระบบจะสลับเป็น DHCP + Auto-IP Learning อัตโนมัติ
//#define USE_STATIC_IP    // ← Comment เพื่อใช้ DHCP (WiFi มือถือ/Router)

const char*    WIFI_SSID = "MCE14";
const char*    WIFI_PASS = "12345678";
const uint16_t UDP_PORT  = 12345;

IPAddress STATIC_IP (192, 168, 137, 123);
IPAddress GATEWAY   (192, 168, 137,   1);
IPAddress SUBNET    (255, 255, 255,   0);

// IP ของ PC — อัปเดตอัตโนมัติจาก packet แรกที่รับ (Auto-IP Learning)
IPAddress PC_IP(192, 168, 137, 1);
bool      has_learned_pc_ip = false;
unsigned long reconnect_timer = 0;

// ── Packet struct ──────────────────────
// ตรงกับ struct.pack("<IffI") ใน robot_comms.py
#pragma pack(push, 1)
struct VisionPacket {
    uint32_t seq;
    float    x;       // หน่วย cm
    float    y;       // หน่วย cm
    uint32_t extra;   // 0 = BALL_POS, 1 = ROBOT_POS
};
#pragma pack(pop)
static_assert(sizeof(VisionPacket) == 16, "Packet size must be 16 bytes");

// ── LED ──────────────────────────────
//   เหลือง (solid)   S_IDLE
//   เขียว            S_MOVE
//   เหลืองกะพริบ     S_WAIT
//   แดงกะพริบ        S_CHECK_POS
//   แดง (solid)      S_BACK
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
#define PPR      17
#define N_GEAR   19
#define WHEEL_R  0.041f
#define ROBOT_L  0.2715f
#define DT       0.01f
#define CORRECTION_THRESHOLD_M  0.03f  // 3 cm — ตำแหน่งเกินนี้จะ correct อัตโนมัติ
#define MAX_CORRECTIONS  3             // จำนวนครั้งสูงสุดที่ correction loop จะวน
#define WDT_TIMEOUT_S    5             // Task Watchdog timeout (วินาที)

const float PULSE_TO_RAD = (2.0f * PI) / (PPR * N_GEAR);
const float VEL_SCALE    = PULSE_TO_RAD / DT;   // precomputed: pulse→rad/s

// ── IK Precomputed Constants (OPT-5) ──
static const float INV_R      = 1.0f / WHEEL_R;
static const float L_OVER_R   = ROBOT_L / WHEEL_R;
static const float SIN60_R    = 0.86602540f * INV_R;   // sin(60°)/R
static const float COS60_R    = 0.50000000f * INV_R;   // cos(60°)/R
static const float ONE_OVER_R = 1.0f * INV_R;

// ── IMU ──────────────────────────────
float yaw        = 0;
float gyroBiasZ  = 0;
float targetYaw  = 0;
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
    Serial.printf("\nGyro bias Z: %.2f\n", gyroBiasZ);
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

volatile float a = 3.0f;

float target_vel1 = 0, target_vel2 = 0, target_vel3 = 0;
float duty1_out = 0, duty2_out = 0, duty3_out = 0;

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
    if      (t <= tacce)          return prof_a * t;
    else if (t <= 2.0f * tacce)   return prof_vmax - prof_a * (t - tacce);
    else                          return 0.0f;
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
// WiFi + UDP (comms layer)
// ════════════════════════════════════
WiFiUDP udp;

void comms_init() {
    Serial.println("\n[Comms] Initializing...");
    Serial.printf("[Comms] Target SSID: %s\n", WIFI_SSID);

    WiFi.disconnect(true);
    delay(200);

#ifdef USE_STATIC_IP
    Serial.println("[Comms] Mode: STATIC IP");
    if (!WiFi.config(STATIC_IP, GATEWAY, SUBNET)) {
        Serial.println("[Comms] Static IP config failed!");
    }
#else
    Serial.println("[Comms] Mode: DHCP");
#endif

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
        if (millis() - t0 > 15000) {
            Serial.println("\n[Comms] Timeout! Restarting...");
            ESP.restart();
        }
    }

    WiFi.setSleep(false);   // ปิด WiFi Sleep → ลด latency / แก้หลุดๆ ติดๆ
    Serial.println("\n[Comms] WiFi Sleep Disabled (Low-Latency Mode)");
    Serial.printf("[Comms] ESP32 IP : %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[Comms] Gateway  : %s\n", WiFi.gatewayIP().toString().c_str());
    Serial.printf("[Comms] Subnet   : %s\n", WiFi.subnetMask().toString().c_str());

    udp.begin(UDP_PORT);
    Serial.printf("[Comms] UDP listening on port %d\n", UDP_PORT);
}

void comms_send(const char* msg) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[Comms] Send failed: WiFi not connected!");
        return;
    }
    udp.beginPacket(PC_IP, UDP_PORT);
    udp.write((const uint8_t*)msg, strlen(msg));
    udp.endPacket();
    Serial.printf("[Comms] Sent to PC (%s): \"%s\"\n", PC_IP.toString().c_str(), msg);
}

void comms_send_okay()        { comms_send("OKAY"); }
void comms_send_request_pos() { comms_send("REQUEST_POS"); }

// ── Shared inter-task variables ────────
SemaphoreHandle_t xMutex;

volatile float cmd_x      = 0, cmd_y    = 0;
volatile float robot_x    = 0, robot_y  = 0;
volatile bool  new_ball      = false;
volatile bool  new_robot_pos = false;

volatile bool  send_okay_flag    = false;
volatile bool  send_req_pos_flag = false;

uint32_t lastSeqNum  = 0;
bool     firstPacket = true;

float active_target_x = 0.0f, active_target_y = 0.0f;

// ── รับ packet → อัปเดต shared vars ──
bool comms_receive() {
    int size = udp.parsePacket();
    if (size <= 0) return false;

    if (size < (int)sizeof(VisionPacket)) {
        Serial.printf("[Comms] Bad packet size: %d bytes\n", size);
        uint8_t dummy[128]; udp.read(dummy, sizeof(dummy));
        return false;
    }

    // Auto-IP Learning
    IPAddress remote_ip = udp.remoteIP();
    if (!has_learned_pc_ip || PC_IP != remote_ip) {
        PC_IP = remote_ip;
        has_learned_pc_ip = true;
        Serial.printf("[Comms] PC IP learned: %s\n", PC_IP.toString().c_str());
    }

    VisionPacket pkt;
    udp.read((uint8_t*)&pkt, sizeof(pkt));

    if (!firstPacket && pkt.seq <= lastSeqNum) {
        if (lastSeqNum - pkt.seq > 100) {
            // PC restart detected — seq dropped significantly, reset tracker
            Serial.printf("[Comms] PC restart detected (seq %u->%u) — accepting\n",
                          lastSeqNum, pkt.seq);
        } else {
            Serial.printf("[Comms] Ignored old seq=%u\n", pkt.seq);
            return false;
        }
    }
    lastSeqNum  = pkt.seq;
    firstPacket = false;

    if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
        if (pkt.extra == 0) {
            cmd_x    = pkt.x;
            cmd_y    = pkt.y;
            new_ball = true;
            Serial.printf("[Comms] BALL_POS  seq=%u  x=%.2f cm  y=%.2f cm\n",
                          pkt.seq, pkt.x, pkt.y);
        } else if (pkt.extra == 1) {
            robot_x       = pkt.x;
            robot_y       = pkt.y;
            new_robot_pos = true;
            Serial.printf("[Comms] ROBOT_POS seq=%u  rx=%.2f cm  ry=%.2f cm\n",
                          pkt.seq, pkt.x, pkt.y);
        } else {
            Serial.printf("[Comms] Unknown extra=%u\n", pkt.extra);
        }
        xSemaphoreGive(xMutex);
    }
    return true;
}

// ── UDP Task (Core 0) ─────────────────
void udpTask(void* pvParameters) {
    Serial.println("[UDP Task] started");
    for (;;) {

        // Auto Reconnect
        if (WiFi.status() != WL_CONNECTED) {
            if (millis() - reconnect_timer >= 2000) {
                reconnect_timer = millis();
                Serial.println("[WiFi] Lost! Reconnecting...");
                WiFi.begin(WIFI_SSID, WIFI_PASS);
            }
            vTaskDelay(10); continue;
        }

        // ส่งออก
        if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
            if (send_okay_flag)    { comms_send_okay();        send_okay_flag    = false; }
            if (send_req_pos_flag) { comms_send_request_pos(); send_req_pos_flag = false; }
            xSemaphoreGive(xMutex);
        }

        // รับเข้า
        comms_receive();

        vTaskDelay(1);
    }
}

// ── Timer ISR ─────────────────────────
hw_timer_t*  timer    = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// OPT-1: ISR ทำแค่อ่าน encoder (integer) แล้วตั้ง flag
// float calculation ทั้งหมดย้ายไป updateEncoderCalc() ใน loop()
void IRAM_ATTR onTimer() {
    portENTER_CRITICAL_ISR(&timerMux);
    pos1 = wheel1.getCount();
    pos2 = wheel2.getCount();
    pos3 = wheel3.getCount();
    isr_flag = true;
    portEXIT_CRITICAL_ISR(&timerMux);
}

// OPT-1 + OPT-2: คำนวณ position/velocity/t_elapsed ใน loop context
// เรียกทุก 10ms หลัง isr_flag triggered
void updateEncoderCalc() {
    portENTER_CRITICAL(&timerMux);
    long p1 = pos1, p2 = pos2, p3 = pos3;
    portEXIT_CRITICAL(&timerMux);

    actual_pos1 = p1 * PULSE_TO_RAD;
    actual_pos2 = p2 * PULSE_TO_RAD;
    actual_pos3 = p3 * PULSE_TO_RAD;

    actual_vel1 = (p1 - last_pos1) * VEL_SCALE;
    actual_vel2 = (p2 - last_pos2) * VEL_SCALE;
    actual_vel3 = (p3 - last_pos3) * VEL_SCALE;

    last_pos1 = p1; last_pos2 = p2; last_pos3 = p3;

    // OPT-2: t_elapsed อัปเดตใน loop → ไม่มี race condition กับ ISR
    if (running) t_elapsed += DT;
}

// ── Utility ───────────────────────────
// OPT-3: ครอบ ISR-shared vars ด้วย critical section
void reset_all() {
    portENTER_CRITICAL(&timerMux);
    t_elapsed = 0;
    pos1 = pos2 = pos3 = 0;
    last_pos1 = last_pos2 = last_pos3 = 0;
    portEXIT_CRITICAL(&timerMux);

    actual_pos1 = actual_pos2 = actual_pos3 = 0;
    actual_vel1 = actual_vel2 = actual_vel3 = 0;
    target_vel1 = target_vel2 = target_vel3 = 0;
    duty1_out   = duty2_out   = duty3_out   = 0;
    wheel1.clearCount(); wheel2.clearCount(); wheel3.clearCount();
    pid_pos1.reset(); pid_pos2.reset(); pid_pos3.reset();
    pid_vel1.reset(); pid_vel2.reset(); pid_vel3.reset();
    pid_yaw.reset();
    yaw = 0; targetYaw = 0;
}

// ── LED blink helper ──────────────────
uint32_t ledBlinkTime = 0;
bool     ledBlinkOn   = false;

void updateBlinkLED(int pin, uint32_t intervalMs) {
    if (millis() - ledBlinkTime >= intervalMs) {
        ledBlinkTime = millis();
        ledBlinkOn   = !ledBlinkOn;
        digitalWrite(pin, ledBlinkOn ? HIGH : LOW);
    }
}

// ════════════════════════════════════
// State Machine
// ════════════════════════════════════
enum State { S_IDLE, S_MOVE, S_WAIT, S_CHECK_POS, S_BACK, S_BACK_CHECK, S_BACK_CORRECT };
State    state      = S_IDLE;
State    prev_state = (State)-1;  // OPT-4: ตรวจ state entry
uint32_t t_wait     = 0;
uint32_t t_req_sent = 0;
uint8_t  correction_count = 0;    // OPT-6: นับจำนวน correction loops

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
    Serial.println("✓ IMU");
    calibrateGyro(1000);
    targetYaw   = 0;
    imuPrevTime = micros();
    Serial.println("✓ Yaw locked");

    comms_init();

    profile_init(0.0f, 0.0f, 3.0f);

    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 10000, true);
    timerAlarmEnable(timer);
    Serial.println("✓ Timer 100 Hz");

    xMutex = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(udpTask, "UDP_Task", 8192, NULL, 1, NULL, 0);

    // OPT-6: Task Watchdog — ป้องกัน loop ค้าง
    esp_task_wdt_init(WDT_TIMEOUT_S, true);
    esp_task_wdt_add(NULL);
    Serial.printf("✓ Watchdog %ds\n", WDT_TIMEOUT_S);

    setLED(false, false, true);
    Serial.println("════ READY — รอ BALL_POS จาก UDP ════");
}

// ════════════════════════════════════
// Loop
// ════════════════════════════════════
void loop() {
    // OPT-6: Watchdog reset ต้องอยู่ก่อน isr_flag gate
    esp_task_wdt_reset();

    if (!isr_flag) return;
    isr_flag = false;

    static uint8_t cnt = 0;
    cnt++;

    // OPT-1 + OPT-2: คำนวณ encoder position/velocity + t_elapsed
    updateEncoderCalc();

    // ── 1. Velocity PID (10 ms) ──────
    duty1_out = pid_vel1.update(target_vel1 - actual_vel1) * running;
    duty2_out = pid_vel2.update(target_vel2 - actual_vel2) * running;
    duty3_out = pid_vel3.update(target_vel3 - actual_vel3) * running;
    set_motor(0, 1, duty1_out);
    set_motor(2, 3, duty2_out);
    set_motor(4, 5, duty3_out);

    // ── 2. Profile + IK + IMU (50 ms) ─
    if (cnt % 5 == 0) {
        float vel = profile_vel(t_elapsed);
        float pos = profile_pos(t_elapsed);
        float vx = vel * cos_angle,  vy = vel * sin_angle;
        float px = pos * cos_angle,  py = pos * sin_angle;

        updateYaw();
        float yaw_err = targetYaw - yaw;
        theta_corr    = pid_yaw.update(yaw_err);

        // OPT-5: IK ใช้ precomputed constants (multiply แทน divide)
        float yaw_term = theta_corr * L_OVER_R;
        float w1 =              - ONE_OVER_R * vy + yaw_term;
        float w2 =  SIN60_R * vx + COS60_R  * vy + yaw_term;
        float w3 = -SIN60_R * vx + COS60_R  * vy + yaw_term;
        float p1 =              - ONE_OVER_R * py;
        float p2 =  SIN60_R * px + COS60_R  * py;
        float p3 = -SIN60_R * px + COS60_R  * py;

        target_vel1 = w1 + pid_pos1.update(p1 - actual_pos1);
        target_vel2 = w2 + pid_pos2.update(p2 - actual_pos2);
        target_vel3 = w3 + pid_pos3.update(p3 - actual_pos3);
    }

    // ── 3. Serial Debug (100 ms) ──────
    if (cnt % 10 == 0) {
        cnt = 0;
        Serial.printf("%.3f,%.3f,%.3f,%.2f,%.3f,%.2f\n",
            actual_pos1, actual_pos2, actual_pos3,
            yaw * (180.0f / PI), theta_corr,
            targetYaw * (180.0f / PI));
    }

    // OPT-4: ตรวจจับ state entry (เรียก stop_all / init ครั้งเดียว)
    bool state_entered = (state != prev_state);
    prev_state = state;

    // ── 4. State Machine ──────────────
    switch (state) {

        // ─────────────────────────────
        // S_IDLE — รอ BALL_POS
        // LED: เหลือง
        // ─────────────────────────────
        case S_IDLE:
            if (state_entered) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
                correction_count = 0;  // OPT-6: reset correction counter
            }

            if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(2)) == pdTRUE) {
                if (new_ball) {
                    new_ball = false;
                    active_target_x = cmd_x / 100.0f;
                    active_target_y = cmd_y / 100.0f;
                    xSemaphoreGive(xMutex);

                    reset_all();
                    profile_init(active_target_x, active_target_y, a);

                    portENTER_CRITICAL(&timerMux);
                    running = true;
                    portEXIT_CRITICAL(&timerMux);

                    state = S_MOVE;
                    setLED(false, true, false);
                    Serial.printf("[SM] IDLE->MOVE | (%.3f, %.3f)m  dur=%.2fs\n",
                                  active_target_x, active_target_y, 2.0f*tacce);
                } else {
                    xSemaphoreGive(xMutex);
                }
            }
            break;

        // ─────────────────────────────
        // S_MOVE — วิ่งไปเป้าหมาย
        // LED: เขียว
        // ─────────────────────────────
        case S_MOVE:
            if (t_elapsed >= 2.0f * tacce) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
                portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);

                t_wait = millis();
                state  = S_WAIT;
                setLED(false, false, true);
                Serial.println("[SM] MOVE->WAIT 2s");
            }
            break;

        // ─────────────────────────────
        // S_WAIT — รอ 2 วินาที
        // LED: เหลืองกะพริบ
        // ─────────────────────────────
        case S_WAIT:
            if (state_entered) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
            }
            updateBlinkLED(LED_YELLOW, 200);

            if (millis() - t_wait >= 2000) {
                digitalWrite(LED_YELLOW, LOW);

                if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
                    new_robot_pos     = false;
                    send_req_pos_flag = true;
                    xSemaphoreGive(xMutex);
                }
                t_req_sent = millis();

                state = S_CHECK_POS;
                setLED(true, false, false);
                Serial.println("[SM] WAIT->CHECK_POS | REQUEST_POS sent");
            }
            break;

        // ─────────────────────────────
        // S_CHECK_POS — รอ ROBOT_POS จากกล้อง
        // Retry ทุก 1 วินาที
        // LED: แดงกะพริบ
        // ─────────────────────────────
        case S_CHECK_POS:
            if (state_entered) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
            }
            updateBlinkLED(LED_RED, 150);

            if (millis() - t_req_sent >= 1000) {
                if (xSemaphoreTake(xMutex, 0) == pdTRUE) {
                    if (!new_robot_pos) {
                        send_req_pos_flag = true;
                        Serial.println("[SM] Retry REQUEST_POS");
                    }
                    xSemaphoreGive(xMutex);
                }
                t_req_sent = millis();
            }

            if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(2)) == pdTRUE) {
                if (new_robot_pos) {
                    new_robot_pos = false;
                    float rx = robot_x / 100.0f;
                    float ry = robot_y / 100.0f;
                    xSemaphoreGive(xMutex);

                    digitalWrite(LED_RED, LOW);
                    Serial.printf("[SM] ROBOT_POS: (%.3f, %.3f) m\n", rx, ry);

                    reset_all();
                    profile_init(-rx, -ry, a);

                    portENTER_CRITICAL(&timerMux);
                    t_elapsed = 0.0f; running = true;
                    portEXIT_CRITICAL(&timerMux);

                    state = S_BACK;
                    setLED(true, false, false);
                    Serial.printf("[SM] CHECK_POS->BACK | dist=%.3fm  dur=%.2fs\n",
                                  sqrtf(rx*rx + ry*ry), 2.0f*tacce);
                } else {
                    xSemaphoreGive(xMutex);
                }
            }
            break;

        // ─────────────────────────────
        // S_BACK — วิ่งกลับ Home
        // LED: แดง
        // ─────────────────────────────
        case S_BACK:
            if (t_elapsed >= 2.0f * tacce) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
                portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);

                // ส่ง REQUEST_POS เพื่อตรวจว่ากลับถึง Home จริงหรือไม่
                if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
                    new_robot_pos     = false;
                    send_req_pos_flag = true;
                    xSemaphoreGive(xMutex);
                }
                t_req_sent = millis();

                state = S_BACK_CHECK;
                setLED(true, false, true);
                Serial.println("[SM] BACK->BACK_CHECK | REQUEST_POS sent");
            }
            break;

        // ─────────────────────────────
        // S_BACK_CHECK — ตรวจตำแหน่งหลังกลับ Home
        // ถ้าคลาดเคลื่อน > 3cm → แก้ไขอัตโนมัติ
        // LED: แดง+เหลืองกะพริบ
        // ─────────────────────────────
        case S_BACK_CHECK:
            if (state_entered) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
            }
            updateBlinkLED(LED_RED, 150);

            // Retry REQUEST_POS ทุก 1 วินาที
            if (millis() - t_req_sent >= 1000) {
                if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(2)) == pdTRUE) {
                    if (!new_robot_pos) {
                        send_req_pos_flag = true;
                        Serial.println("[SM] Retry REQUEST_POS (back check)");
                    }
                    xSemaphoreGive(xMutex);
                }
                t_req_sent = millis();
            }

            if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(2)) == pdTRUE) {
                if (new_robot_pos) {
                    new_robot_pos = false;
                    float rx = robot_x / 100.0f;
                    float ry = robot_y / 100.0f;
                    xSemaphoreGive(xMutex);

                    float err = sqrtf(rx * rx + ry * ry);
                    digitalWrite(LED_RED, LOW);
                    Serial.printf("[SM] BACK_CHECK pos: (%.3f, %.3f)m  err=%.3fm\n", rx, ry, err);

                    if (err > CORRECTION_THRESHOLD_M) {
                        // OPT-6: จำกัดจำนวน correction loops
                        correction_count++;
                        if (correction_count > MAX_CORRECTIONS) {
                            Serial.printf("[SM] Max corrections (%d) reached! Forcing OKAY (err=%.3fm)\n",
                                          MAX_CORRECTIONS, err);
                            if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
                                send_okay_flag = true;
                                xSemaphoreGive(xMutex);
                            }
                            reset_all();
                            correction_count = 0;
                            state = S_IDLE;
                            setLED(false, false, true);
                        } else {
                            // ตำแหน่งคลาดเคลื่อนเกิน → วิ่งแก้ไข
                            reset_all();
                            profile_init(-rx, -ry, a);
                            portENTER_CRITICAL(&timerMux);
                            running = true;
                            portEXIT_CRITICAL(&timerMux);

                            state = S_BACK_CORRECT;
                            setLED(true, true, false);
                            Serial.printf("[SM] BACK_CHECK->BACK_CORRECT #%d | err=%.3fm\n",
                                          correction_count, err);
                        }
                    } else {
                        // ตำแหน่งเป๊ะ → ส่ง OKAY กลับ PC
                        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
                            send_okay_flag = true;
                            xSemaphoreGive(xMutex);
                        }
                        reset_all();
                        state = S_IDLE;
                        setLED(false, false, true);
                        Serial.printf("[SM] BACK_CHECK->IDLE | OKAY sent (err=%.3fm OK)\n", err);
                    }
                } else {
                    xSemaphoreGive(xMutex);
                }
            }
            break;

        // ─────────────────────────────
        // S_BACK_CORRECT — วิ่งแก้ไขตำแหน่งกลับ Home
        // LED: แดง+เขียว
        // ─────────────────────────────
        case S_BACK_CORRECT:
            if (t_elapsed >= 2.0f * tacce) {
                stop_all();
                target_vel1 = target_vel2 = target_vel3 = 0;
                portENTER_CRITICAL(&timerMux); running = false; portEXIT_CRITICAL(&timerMux);

                // แก้ไขเสร็จ → ตรวจสอบตำแหน่งอีกครั้ง
                if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
                    new_robot_pos     = false;
                    send_req_pos_flag = true;
                    xSemaphoreGive(xMutex);
                }
                t_req_sent = millis();

                state = S_BACK_CHECK;
                setLED(true, false, true);
                Serial.println("[SM] BACK_CORRECT->BACK_CHECK | REQUEST_POS sent");
            }
            break;
    }
}
