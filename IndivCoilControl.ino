// ============================
//  DUAL-YOKE LEVITATION CONTROL
//  + ROLL STABILIZATION (IMU)
// ============================

#include <Adafruit_MPU6050.h>   // same library you installed earlier
#include <Adafruit_Sensor.h>
#include <Wire.h>

// ----------------------------
// IMU (Gyro/Accel) for roll control
// ----------------------------
Adafruit_MPU6050 mpu;            // IMU device (MPU6050)

// Complementary filter state (combines gyro+accel into a stable angle)
// roll_deg:  rotation around X axis; flat ≈ 0°, left ≈ −90°, right ≈ +90°
float roll_deg = 0.0f;           // X axis rotation (degrees)
float pitch_deg = 0.0f;          // Y axis rotation (degrees) — computed but unused here
bool  first_sample = true;       // seed filter from accel on first sample
const float ALPHA = 0.98f;       // 0.98 = mostly gyro (short-term), 0.02 = accel (long-term)

// Inductive Sensor Mapping Struct
typedef struct IndSensorMap {
  double A;
  double K;
  double B;
  double C;
  double v;
} IndSensorMap;

IndSensorMap ind0Map = {-8.976076325826309, 913.5463710698101, 0.29767471011439534, 5.6686184386250025, 0.3627635461289861};
IndSensorMap ind1Map = {-4.831976283950702, 885.9877001844566, 0.2793284618109283, 3.8852507844119217, 0.2389935455347361};
IndSensorMap ind2Map = {-9.824360913609562, 871.4744633266955, 0.2909366235093304, 4.3307594408159495, 0.2822807132259202};

// Adjust sign so left tilt is negative and right tilt is positive.
// Flip to +1.0f if your physical mounting is reversed.
const float ROLL_SIGN = -1.0f;

// ----------------------------
// Roll PID (anti-rotation)
// ----------------------------
// These gains generate a differential torque command (±PWM) that we add to one coil
// and subtract from the other to cancel rotation (like twisting the U-yokes opposite).
float Kp_roll = 4.0f;            // proportional gain (start 3–6)
float Ki_roll = 0.4f;            // integral gain (start 0.2–0.6)
float Kd_roll = 0.0f;            // derivative gain (start 0; add if needed for damping)
float roll_ref = 0.0f;           // target roll (degrees) — we want level
float roll_e_int = 0.0f;         // integral accumulator for roll
float roll_e_prev = 0.0f;        // previous roll error for derivative term
const float ROLL_INT_CAP = 300.0f;   // anti-windup clamp for integral
const int   TORQUE_CAP   = 80;       // max |differential PWM| used to twist the pair (tune)

// ----------------------------
// PIN MAPPING
// ----------------------------
const int ind0Pin  = A0;
const int ind1Pin  = A1;
#define pwmZero 6
#define dirZero 7
#define pwmOne 9
#define dirOne 8
#define pwmTwo 3 // this is a random placeholder value

const int range2Pin = A4;
const int rangePin  = A3;

// ----------------------------
// variables
// ----------------------------
int dist_raw, oor, t0, tcurr, tprior, telapsed, pwm, pwm2, dist2_raw, oor2;
float dist, ecurr, eprior, derror, ecum, ff, dist2, ecurr2, eprior2, derror2, ecum2, ff2;

const int CAP = 200;                             // absolute PWM clamp (per channel)

// ----------------------------
// CONTROLLER CONSTANTS
// ----------------------------
float kp, ki, kd, K;
float MAX_INTEGRAL_TERM = 1e4;

const float ref  = 21.0;   // left yoke distance setpoint
const float ref2 = 22.0;   // right yoke distance setpoint

const float K_current  = 1;    // (kept for compatibility)
const float ki_current = 0;    // (kept for compatibility)

// FOR MC 1 (left yoke): separate gains when falling vs attracting
const float K_f  = 40;     // falling (error > 0) — push down / reduce lift
const float ki_f = 0.01;
const float kd_f = 7;

const float K_a  = 20;     // attracting (error < 0) — pull up / increase lift
const float ki_a = ki_f;
const float kd_a = 20;

// FOR MC 2 (right yoke): mirror of MC1
const float K2_f  = K_f;
const float ki2_f = ki_f;
const float kd2_f = kd_f;

const float K2_a  = K_a;
const float ki2_a = ki_a;
const float kd2_a = kd_a;

const int sampling_rate   = 1000;                 // 1 kHz main control cadence
const int dt_micros       = 1000000 / sampling_rate;
const uint32_t SAMPLE_PERIOD_MS = 1;              // ~1 kHz guard in loop()

#define LEV_ON
int ON = 0;                                       // serial toggle; "0" to off, anything else = on

// ----------------------------
// helpers
// ----------------------------
static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static inline float clamp90(float x) { return clampf(x, -90.0f, 90.0f); }

// prototypes (so we can keep your function layout)
void send_pwm1(int val);
void send_pwm2(int val);
float ind2mm(IndSensorMap ind, unsigned int raw);
int levitate(float e, float de, float ecum_local, int oor);
int levitate2(float e, float de, float ecum_local, int oor);

// ----------------------------
// setup()
// ----------------------------
void setup() {
  // put your setup code here, to run once:
  Serial.begin(57600);

  pinMode(ind0Pin,  INPUT);
  pinMode(rangePin, INPUT);
  pinMode(ind1Pin, INPUT);
  pinMode(range2Pin, INPUT);

  // positive pwm is A; negative is B
  // ATTRACT IS B  // REPEL IS A
  // (Your driver polarity is preserved; only the torque split is added later.)

  // Motor control pins were not explicitly configured before — add OUTPUT modes
  pinMode(ind0Pin,  OUTPUT);
  pinMode(pwmZero,  OUTPUT);
  pinMode(ind1Pin, OUTPUT);
  pinMode(pwmOne, OUTPUT);

  // --- IMU init (new) ---
  Wire.begin();
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050");
    while (1) { delay(10); }
  }
  // Use moderate ranges and bandwidth to reduce noise in the tilt estimate
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  t0 = micros();
  tprior = t0;

  ecum  = 0;    // left integral (height)
  ecum2 = 0;    // right integral (height)

  // zero outputs at boot
  send_pwm1(0);
  send_pwm2(0);
}

// ----------------------------
// loop()
// ----------------------------
void loop() {
  // Serial command: send "0" to turn levitation OFF, anything else to turn ON
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');  // Read the full input
    inputString.trim();                                 // strip whitespace
    if (inputString == "0") { ON = 0; } else { ON = 1; }
  }

  telapsed = micros() - tprior;

  if (telapsed >= dt_micros) {
    // ----------------------------
    // Read height sensors (yours)
    // ----------------------------
    dist_raw = analogRead(ind0Pin);
    dist     = ind2mm(ind0Map, dist_raw)
    Serial.print(dist);
    Serial.print(", ");

    dist2_raw = analogRead(ind1Pin);
    // dist2     = float_map(dist2_raw, 189, 950, 16, 26);
    Serial.print(dist2);
    Serial.print(", ");

    // naive out-of-range flags (kept from your code)
    oor  = (dist_raw  <= 195);
    oor2 = (dist2_raw <= 195);

    // ----------------------------
    // Height loop errors (yours)
    // ----------------------------
    ecurr   = ref  - dist;     // error > 0: too low (need to fall less / lift more depending on design)
    derror  = ecurr - eprior;

    ecurr2  = ref2 - dist2;
    derror2 = ecurr2 - eprior2;

    // integrate (your ecum variables existed but were not updated; add standard integrator)
    float dt_h = telapsed / 1e6f;
    ecum  += ecurr  * dt_h;
    ecum2 += ecurr2 * dt_h;
    ecum  = clampf(ecum,  -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
    ecum2 = clampf(ecum2, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);

    // ----------------------------
    // Base levitation outputs (yours)
    // ----------------------------
    int pwm1_base = 0;
    int pwm2_base = 0;

    if (ON) {
      pwm1_base = levitate(ecurr,  derror,  ecum,  oor);   // left yoke base command
      pwm2_base = levitate2(ecurr2, derror2, ecum2, oor2); // right yoke base command
      Serial.print(pwm1_base);
      Serial.print(", ");
      Serial.print(pwm2_base);
      Serial.print(", ");
    } else {
      send_pwm1(0);
      send_pwm2(0);
      Serial.print(0);
      Serial.print(", ");
      Serial.print(0);
      Serial.print(", ");
    }

    // ----------------------------
    // IMU update (new)
    // ----------------------------
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    if (first_sample) {
      // Seed angles from accelerometer so flat starts near 0°
      first_sample = false;
      roll_deg  = atan2f(a.acceleration.y, a.acceleration.z) * 180.0f / PI;
      pitch_deg = atan2f(-a.acceleration.x,
                         sqrtf(a.acceleration.y * a.acceleration.y +
                               a.acceleration.z * a.acceleration.z)) * 180.0f / PI;
    } else {
      // Complementary filter step: gyro integrate + accel correct
      float gx_deg_s = g.gyro.x * 180.0f / PI;  // gyro is rad/s → deg/s
      float gy_deg_s = g.gyro.y * 180.0f / PI;

      float roll_gyro  = roll_deg  + gx_deg_s * dt_h;
      float pitch_gyro = pitch_deg + gy_deg_s * dt_h;

      float roll_acc  = atan2f(a.acceleration.y, a.acceleration.z) * 180.0f / PI;
      float pitch_acc = atan2f(-a.acceleration.x,
                               sqrtf(a.acceleration.y * a.acceleration.y +
                                     a.acceleration.z * a.acceleration.z)) * 180.0f / PI;

      roll_deg  = ALPHA * roll_gyro  + (1.0f - ALPHA) * roll_acc;
      pitch_deg = ALPHA * pitch_gyro + (1.0f - ALPHA) * pitch_acc;
    }

    // Map/sign and clamp roll to [-90°, +90°] as requested
    float roll_mapped = clamp90(ROLL_SIGN * roll_deg);

    // ----------------------------
    // Roll PID → differential torque (new)
    // ----------------------------
    // The idea: create a "twist" command that adds to left PWM and subtracts from right PWM.
    // This generates opposite magnetic forces that resist rotation without changing total lift much.
    float roll_e = roll_ref - roll_mapped;               // we want 0° roll (level)
    roll_e_int  += roll_e * dt_h;                        // integrate error
    roll_e_int   = clampf(roll_e_int, -ROLL_INT_CAP, ROLL_INT_CAP);
    float roll_e_der = (roll_e - roll_e_prev) / dt_h;    // derivative
    roll_e_prev = roll_e;

    float torque_cmd_f = Kp_roll * roll_e + Ki_roll * roll_e_int + Kd_roll * roll_e_der;
    int   torque_cmd   = (int)clampf(torque_cmd_f, -TORQUE_CAP, TORQUE_CAP);

    // Apply differential torque around the vertical axis:
    // left gets +torque, right gets -torque (or vice-versa depending on ROLL_SIGN)
    int out1 = (int)clampf((float)pwm1_base + torque_cmd, -CAP, CAP);
    int out2 = (int)clampf((float)pwm2_base - torque_cmd, -CAP, CAP);

    if (ON) {
      // Override earlier base sends with torque-compensated outputs
      send_pwm1(out1);
      send_pwm2(out2);
    }

    // ----------------------------
    // Telemetry (kept + added roll)
    // ----------------------------
    Serial.print(ON);
    Serial.print(", ");
    Serial.print(roll_mapped, 2);   // roll (deg) for logging/plotting
    Serial.println();

    // ----------------------------
    // advance time/error history (yours)
    // ----------------------------
    tprior = micros();
    eprior = ecurr;
    eprior2 = ecurr2;
  }
}

// ----------------------------
// levitate() — left yoke (yours)
// ----------------------------
// Selects gains based on sign of error (fall vs attract) and computes a base PWM.
// We keep your polarity & saturation and send immediately to channel 1.
int levitate(float e, float de, float ecum_local, int oor) {
  if (oor) {
    pwm = 0;
  } else {
    if (e < 0) { // dist > ref => ATTRACT branch
      kd = kd_a; ki = ki_a; K = K_a;
    } else {     // dist <= ref => FALL branch
      kd = kd_f; ki = ki_f; K = K_f;
    }
    pwm = constrain(K * (e + ki * ecum_local + kd * de), -CAP, CAP);
  }
  send_pwm1(pwm);
  return (int)pwm;
}

// ----------------------------
// levitate2() — right yoke (yours)
// ----------------------------
int levitate2(float e, float de, float ecum_local, int oor) {
  if (oor) {
    pwm2 = 0;
  } else {
    if (e < 0) { // dist > ref => ATTRACT branch
      kd = kd2_a; ki = ki2_a; K = K2_a;
    } else {     // dist <= ref => FALL branch
      kd = kd2_f; ki = ki2_f; K = K2_f;
    }
    pwm2 = constrain(K * (e + ki * ecum_local + kd * de), -CAP, CAP);
  }
  send_pwm2(pwm2);
  return (int)pwm2;
}

// ----------------------------
// send_pwm1() — left motor driver (yours)
// ----------------------------
// "Positive" values drive one polarity (REPEL=A), "negative" the opposite (ATTRACT=B).
void send_pwm1(int val) {
  if (val > 0) { digitalWrite(dirZero, LOW); }
  else         { digitalWrite(dirZero, HIGH); }
  analogWrite(pwmZero, abs(val));
}

// ----------------------------
// send_pwm2() — right motor driver (yours)
// ----------------------------
void send_pwm2(int val) {
  if (val > 0) { digitalWrite(dirOne, LOW); }
  else         { digitalWrite(dirOne, HIGH); }
  analogWrite(pwmTwo, abs(val));
}

// ----------------------------
// float_map() — ADC→distance mapping (yours)
// ----------------------------
// Uses John and Adi's sensor calibration to return the millimeter reading from an analog sensor value.
float ind2mm(IndSensorMap ind, unsigned int raw) {
  return ind.C - (1.0 / ind.B) * log(pow((ind.K - ind.A) / ((float)raw - ind.A), ind.v) - 1.0);
}
