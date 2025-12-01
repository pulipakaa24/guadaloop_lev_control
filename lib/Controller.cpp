#include "Controller.hpp"
#include <Arduino.h>

// CONTROLLER CONSTANTS
float MAX_INTEGRAL_TERM = 1e4;

void FullController::update() {
  Left.readMM();
  Right.readMM();
  Front.readMM();
  Back.readMM(); // read and update dists/oor for all sensors.

  oor = Left.oor || Right.oor || Front.oor || Back.oor;

  avgControl();
  LRControl(); // run pwm functions.
  FBControl();

  FLPWM = constrain(avgPWM + LDiffPWM + FDiffPWM, -CAP, CAP);
  BLPWM = constrain(avgPWM + LDiffPWM + BDiffPWM, -CAP, CAP);
  FRPWM = constrain(avgPWM + RDiffPWM + FDiffPWM, -CAP, CAP);
  BRPWM = constrain(avgPWM + RDiffPWM + BDiffPWM, -CAP, CAP);

  // FLPWM = avgPWM;
  // BLPWM = avgPWM;
  // FRPWM = avgPWM;
  // BRPWM = avgPWM;
}

void FullController::zeroPWMs() {
  FLPWM = 0;
  BLPWM = 0;
  FRPWM = 0;
  BRPWM = 0;
}

void FullController::sendOutputs() {
  if (!outputOn) {
    zeroPWMs();
  }

  // The following assumes 0 direction drives repulsion and 1 direction drives
  // attraction. Using direct register writes to maintain fast PWM mode set by
  // setupFastPWM()
  digitalWrite(dirFL, FLPWM < 0);
  OCR2A = abs(FLPWM); // Pin 11 -> Timer 2A
  digitalWrite(dirBL, BLPWM < 0);
  OCR1A = abs(BLPWM); // Pin 9 -> Timer 1A
  digitalWrite(dirFR, FRPWM < 0);
  OCR2B = abs(FRPWM); // Pin 3 -> Timer 2B
  digitalWrite(dirBR, BRPWM < 0);
  OCR1B = abs(BRPWM); // Pin 10 -> Timer 1B
}

void FullController::avgControl() {
  avg = (Left.mmVal + Right.mmVal + Front.mmVal + Back.mmVal) * 0.25f;
  float eCurr = AvgRef - avg;

  avgError.eDiff = eCurr - avgError.e;
  if (!oor) {
    avgError.eInt += eCurr;
    avgError.eInt =
        constrain(avgError.eInt, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
  }
  avgError.e = eCurr;

  avgPWM = pwmFunc(avgConsts, avgError);
}

void FullController::LRControl() {
  float diff = Right.mmVal - Left.mmVal; // how far above the right is the left?
  float eCurr = diff - LRDiffRef; // how different is that from the reference?
                                  // positive -> Left repels, Right attracts.
  K_MAP rConsts = {
      LConsts.attracting,
      LConsts.repelling}; // apply attracting to repelling and vice versa.

  LRDiffErr.eDiff = eCurr - LRDiffErr.e;

  if (!oor) {
    LRDiffErr.eInt += eCurr;
    LRDiffErr.eInt =
        constrain(LRDiffErr.eInt, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
  }

  LRDiffErr.e = eCurr;

  LDiffPWM = pwmFunc(LConsts, LRDiffErr);
  RDiffPWM = -pwmFunc(rConsts, LRDiffErr);
}

void FullController::FBControl() {
  float diff = Back.mmVal - Front.mmVal; // how far above the back is the front?
  float eCurr = diff - FBDiffRef; // how different is that from ref? pos.->Front
                                  // must repel, Back must attract
  K_MAP bConsts = {FConsts.attracting, FConsts.repelling};

  FBDiffErr.eDiff = eCurr - FBDiffErr.e;

  if (!oor) {
    FBDiffErr.eInt += eCurr;
    FBDiffErr.eInt =
        constrain(FBDiffErr.eInt, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
  }

  FBDiffErr.e = eCurr;

  FDiffPWM = pwmFunc(FConsts, FBDiffErr);
  BDiffPWM = -pwmFunc(bConsts, FBDiffErr);
}

int16_t FullController::pwmFunc(K_MAP consts, Errors errs) {
  if (oor)
    return 0;
  Constants constants = (errs.e < 0) ? consts.attracting : consts.repelling;
  return (int)(constants.kp * errs.e + constants.ki * errs.eInt +
                            constants.kd * errs.eDiff);
}

void FullController::report() {
  // CSV Format: Left,Right,Front,Back,Avg,FLPWM,BLPWM,FRPWM,BRPWM,ControlOn
  Serial.print(Left.mmVal);
  Serial.print(",");
  Serial.print(Right.mmVal);
  Serial.print(",");
  Serial.print(Front.mmVal);
  Serial.print(",");
  Serial.print(Back.mmVal);
  Serial.print(",");
  Serial.print(avg);
  Serial.print(",");

  Serial.print(FLPWM);
  Serial.print(",");
  Serial.print(BLPWM);
  Serial.print(",");
  Serial.print(FRPWM);
  Serial.print(",");
  Serial.print(BRPWM);
  Serial.print(",");

  Serial.println(outputOn);
}

void FullController::updateAvgPID(Constants repel, Constants attract) {
  avgConsts.repelling = repel;
  avgConsts.attracting = attract;
}

void FullController::updateLRPID(Constants down, Constants up) {
  LConsts.repelling = down;
  LConsts.attracting = up;
}

void FullController::updateFBPID(Constants down, Constants up) {
  FConsts.repelling = down;
  FConsts.attracting = up;
}

void FullController::updateReferences(float avgReference, float lrDiffReference, float fbDiffReference) {
  AvgRef = avgReference;
  LRDiffRef = lrDiffReference;
  FBDiffRef = fbDiffReference;
}