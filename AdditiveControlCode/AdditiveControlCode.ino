#include <Arduino.h>
#include "IndSensorMap.hpp"
#include "Controller.hpp"
#include "ADC.hpp"
#include "FastPWM.hpp"

// ── PID Gains (Kp, Ki, Kd) ──────────────────────────────────
// Height loop: controls average gap → additive PWM on all coils
PIDGains heightGains = { 100.0f, 0.0f, 8.0f };

// Roll loop: corrects left/right tilt → differential L/R
PIDGains rollGains   = { 0.6f, 0.0f, -0.1f };

// Pitch loop: corrects front/back tilt → differential F/B
PIDGains pitchGains  = { 50.0f, 0.0f, 1.9f };

// ── Reference ────────────────────────────────────────────────
float avgRef = 12.36f;  // Target gap height (mm) — 9.4 kg equilibrium

// ── Feedforward ──────────────────────────────────────────────
bool useFeedforward = true;  // Set false to disable feedforward LUT

// ── Sampling ─────────────────────────────────────────────────
#define SAMPLING_RATE 200  // Hz (controller tick rate)

// ── EMA filter alpha (all sensors) ───────────────────────────
#define ALPHA_VAL 1.0f

// ═══════════════════════════════════════════════════════════════
// ABOVE THIS LINE IS TUNING VALUES ONLY, BELOW IS ACTUAL CODE.
// ═══════════════════════════════════════════════════════════════

unsigned long tprior;
unsigned int tDiffMicros;

FullController controller(indL, indR, indF, indB,
                          heightGains, rollGains, pitchGains,
                          avgRef, useFeedforward);

const int dt_micros = 1000000 / SAMPLING_RATE;

int ON = 0;

void setup() {
  Serial.begin(2000000);
  setupADC();
  setupFastPWM();

  indL.alpha = ALPHA_VAL;
  indR.alpha = ALPHA_VAL;
  indF.alpha = ALPHA_VAL;
  indB.alpha = ALPHA_VAL;

  tprior = micros();

  pinMode(dirFL, OUTPUT);
  pinMode(pwmFL, OUTPUT);
  pinMode(dirBL, OUTPUT);
  pinMode(pwmBL, OUTPUT);
  pinMode(dirFR, OUTPUT);
  pinMode(pwmFR, OUTPUT);
  pinMode(dirBR, OUTPUT);
  pinMode(pwmBR, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    // REF,avgRef — update target gap height
    if (cmd.startsWith("REF,")) {
      float newRef = cmd.substring(4).toFloat();
      avgRef = newRef;
      controller.updateReference(avgRef);
      Serial.print("Updated Ref: ");
      Serial.println(avgRef);
    }
    // PID,loop,kp,ki,kd — update gains (loop: 0=height, 1=roll, 2=pitch)
    else if (cmd.startsWith("PID,")) {
      int c1 = cmd.indexOf(',');
      int c2 = cmd.indexOf(',', c1 + 1);
      int c3 = cmd.indexOf(',', c2 + 1);
      int c4 = cmd.indexOf(',', c3 + 1);
      
      if (c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0) {
        int loop = cmd.substring(c1 + 1, c2).toInt();
        float kp  = cmd.substring(c2 + 1, c3).toFloat();
        float ki  = cmd.substring(c3 + 1, c4).toFloat();
        float kd  = cmd.substring(c4 + 1).toFloat();
        
        PIDGains g = { kp, ki, kd };
        
        switch (loop) {
          case 0:
            heightGains = g;
            controller.updateHeightPID(g);
            Serial.println("Updated Height PID");
            break;
          case 1:
            rollGains = g;
            controller.updateRollPID(g);
            Serial.println("Updated Roll PID");
            break;
          case 2:
            pitchGains = g;
            controller.updatePitchPID(g);
            Serial.println("Updated Pitch PID");
            break;
          default:
            Serial.println("Invalid loop (0=height, 1=roll, 2=pitch)");
            break;
        }
      }
    }
    // FF,0 or FF,1 — toggle feedforward
    else if (cmd.startsWith("FF,")) {
      bool en = (cmd.charAt(3) != '0');
      useFeedforward = en;
      controller.setFeedforward(en);
      Serial.print("Feedforward: ");
      Serial.println(en ? "ON" : "OFF");
    }
    else {
      // Original on/off command (any char except '0' turns on)
      controller.outputOn = (cmd.charAt(0) != '0');
    }
  }
  
  tDiffMicros = micros() - tprior;

  if (tDiffMicros >= dt_micros) {
    controller.update();
    controller.report();
    controller.sendOutputs();
    
    tprior = micros();
  }
}