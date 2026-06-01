/*
  MCE14 Vission 16 — ESP32 Robot Control Firmware

  This firmware connects to the host computer's hotspot, listens for incoming
  UDP JSON packets on port 5005, parses target (x, y) coordinates and
  heartbeats, and handles safety timeouts (falls back if no packet is received
  for 500ms).

  Dependencies:
  - ArduinoJson library (Install via Arduino Library Manager)
*/

#include <ArduinoJson.h>
#include <WiFi.h>
#include <WiFiUdp.h>


// --- WiFi Configurations ---
const char *ssid = "MCE14_Vision_Hotspot"; // Match your computer's Hotspot name
const char *password =
    "mce14password"; // Match your computer's Hotspot password

// --- Network Port ---
const unsigned int localPort = 5005;

// --- UDP Buffer & Socket ---
WiFiUDP udp;
char packetBuffer[255];

// --- Safety & Timing Variables ---
unsigned long lastPacketTime = 0;
const unsigned long timeoutInterval = 500; // 500 ms safety timeout
bool isTimedOut = false;

// --- Robot Target State ---
float targetX = 0.0;
float targetY = 0.0;
bool hasTarget = false;

void setup() {
  Serial.begin(115200);
  delay(10);

  // Connect to WiFi network
  Serial.println();
  Serial.print("Connecting to Hotspot: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Start UDP Listener
  Serial.print("Starting UDP listener on port ");
  Serial.println(localPort);
  udp.begin(localPort);

  lastPacketTime = millis();
}

void loop() {
  // 1. Read and parse incoming UDP packets (Non-blocking)
  int packetSize = udp.parsePacket();
  if (packetSize) {
    // Read the packet into the buffer
    int len = udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = 0; // Null-terminate string
    }

    // Parse JSON
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, packetBuffer);

    if (!error) {
      lastPacketTime = millis(); // Reset timeout timer on any valid packet
      isTimedOut = false;

      // Check if it is a heartbeat packet
      if (doc.containsKey("heartbeat")) {
        // Serial.println("Heartbeat received.");
      }
      // Check if it is a landing coordinate target
      else if (doc.containsKey("x") && doc.containsKey("y")) {
        targetX = doc["x"];
        targetY = doc["y"];
        hasTarget = true;

        Serial.printf("[Vision Target] Move to -> X: %.1f cm, Y: %.1f cm\n",
                      targetX, targetY);

        // --- TODO: Insert your robot motion control code here ---
        // e.g. command_motors(targetX, targetY);
      }
    } else {
      Serial.print("JSON Deserialization failed: ");
      Serial.println(error.c_str());
    }
  }

  // 2. Safety Timeout Check
  if (millis() - lastPacketTime > timeoutInterval) {
    if (!isTimedOut) {
      isTimedOut = true;
      Serial.println("\n[WARNING] Connection lost! Vision system timeout (no "
                     "packets for >500ms).");
      Serial.println("Executing safety fallback: Stopping motors...");

      // --- TODO: Insert safety action here ---
      // stop_motors();
      hasTarget = false;
    }
  }

  // 3. Robot Control Loop (Runs continuously)
  if (hasTarget && !isTimedOut) {
    // Perform standard drive / positioning loop to reach targetX, targetY
  }
}
