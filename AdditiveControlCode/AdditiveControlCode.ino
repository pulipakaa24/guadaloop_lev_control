#include <Arduino.h>
#include "IndSensorMap.hpp"
#include "Controller.hpp"
#include "ADC.hpp"
#include "FastPWM.hpp"

// K, Ki, Kd Constants
Constants repelling = {250, 0, 1000};
Constants attracting = {250, 0, 1000};

Constants RollLeftUp = {0, 0, 100};
Constants RollLeftDown = {0, 0, 100};

Constants RollFrontUp = {0, 0, 500};
Constants RollFrontDown = {0, 0, 500};

// Reference values for average dist, 
float avgRef = 11.0; // TBD: what is our equilibrium height with this testrig?
float LRDiffRef = -2.0; // TBD: what is our left-right balance equilibrium? Positive -> left is above right
float FBDiffRef = 0.0; // TBD: what is front-back balance equilibrium? Positive -> front above back.

// Might be useful for things like jitter or lag.
#define sampling_rate 1000 // Hz

// EMA filter alpha value (all sensors use same alpha)
#define alphaVal 0.3f

// ABOVE THIS LINE IS TUNING VALUES ONLY, BELOW IS ACTUAL CODE.

unsigned long tprior;
unsigned int tDiffMicros;

FullConsts fullConsts = {
  {repelling, attracting},
  {RollLeftDown, RollLeftUp},
  {RollFrontDown, RollFrontUp}
};

FullController controller(indL, indR, indF, indB, fullConsts, avgRef, LRDiffRef, FBDiffRef);

const int dt_micros = 1e6/sampling_rate;

#define LEV_ON

int ON = 0;

void setup() {
  Serial.begin(2000000);
  setupADC();
  setupFastPWM();

  indL.alpha = alphaVal;
  indR.alpha = alphaVal;
  indF.alpha = alphaVal;
  indB.alpha = alphaVal;

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
    
    // Check if it's a reference update command (format: REF,avgRef,lrDiffRef,fbDiffRef)
    if (cmd.startsWith("REF,")) {
      int firstComma = cmd.indexOf(',');
      int secondComma = cmd.indexOf(',', firstComma + 1);
      int thirdComma = cmd.indexOf(',', secondComma + 1);
      
      if (firstComma > 0 && secondComma > 0 && thirdComma > 0) {
        float newAvgRef = cmd.substring(firstComma + 1, secondComma).toFloat();
        float newLRDiffRef = cmd.substring(secondComma + 1, thirdComma).toFloat();
        float newFBDiffRef = cmd.substring(thirdComma + 1).toFloat();
        
        avgRef = newAvgRef;
        LRDiffRef = newLRDiffRef;
        FBDiffRef = newFBDiffRef;
        
        controller.updateReferences(avgRef, LRDiffRef, FBDiffRef);
        Serial.print("Updated References: Avg=");
        Serial.print(avgRef);
        Serial.print(", LR=");
        Serial.print(LRDiffRef);
        Serial.print(", FB=");
        Serial.println(FBDiffRef);
      }
    }
    // Check if it's a PID tuning command (format: PID,mode,kp,ki,kd)
    else if (cmd.startsWith("PID,")) {
      int firstComma = cmd.indexOf(',');
      int secondComma = cmd.indexOf(',', firstComma + 1);
      int thirdComma = cmd.indexOf(',', secondComma + 1);
      int fourthComma = cmd.indexOf(',', thirdComma + 1);
      
      if (firstComma > 0 && secondComma > 0 && thirdComma > 0 && fourthComma > 0) {
        int mode = cmd.substring(firstComma + 1, secondComma).toInt();
        float kp = cmd.substring(secondComma + 1, thirdComma).toFloat();
        float ki = cmd.substring(thirdComma + 1, fourthComma).toFloat();
        float kd = cmd.substring(fourthComma + 1).toFloat();
        
        Constants newConst = {kp, ki, kd};
        
        // Mode mapping:
        // 0: Repelling
        // 1: Attracting
        // 2: RollLeftDown
        // 3: RollLeftUp
        // 4: RollFrontDown
        // 5: RollFrontUp
        
        switch(mode) {
          case 0: // Repelling
            repelling = newConst;
            controller.updateAvgPID(repelling, attracting);
            Serial.println("Updated Repelling PID");
            break;
          case 1: // Attracting
            attracting = newConst;
            controller.updateAvgPID(repelling, attracting);
            Serial.println("Updated Attracting PID");
            break;
          case 2: // RollLeftDown
            RollLeftDown = newConst;
            controller.updateLRPID(RollLeftDown, RollLeftUp);
            Serial.println("Updated RollLeftDown PID");
            break;
          case 3: // RollLeftUp
            RollLeftUp = newConst;
            controller.updateLRPID(RollLeftDown, RollLeftUp);
            Serial.println("Updated RollLeftUp PID");
            break;
          case 4: // RollFrontDown
            RollFrontDown = newConst;
            controller.updateFBPID(RollFrontDown, RollFrontUp);
            Serial.println("Updated RollFrontDown PID");
            break;
          case 5: // RollFrontUp
            RollFrontUp = newConst;
            controller.updateFBPID(RollFrontDown, RollFrontUp);
            Serial.println("Updated RollFrontUp PID");
            break;
          default:
            Serial.println("Invalid mode");
            break;
        }
      }
    } else {
      // Original control on/off command
      controller.outputOn = (cmd.charAt(0) != '0');
    }
  }
  
  tDiffMicros = micros() - tprior;

  if (tDiffMicros >= dt_micros){
    controller.update();
    controller.report();
    controller.sendOutputs(); 
    // this and the previous line can be switched if you want the PWMs to display 0 when controller off.
    
    tprior = micros(); // maybe we have to move this line to before the update commands?
    // since the floating point arithmetic may take a while...
  }

  //Serial.println(telapsed);
}