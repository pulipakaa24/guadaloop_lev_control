#include "Controller.hpp"
#include <Arduino.h>

// CONTROLLER CONSTANTS
float MAX_INTEGRAL_TERM = 1e4;

void FullController::update(float tDiff) {
  this->tDiff = tDiff; // store time step for use in differential and integral portions

  Left.readMM();
  Right.readMM();
  Front.readMM();
  Back.readMM(); // read and update dists/oor for all sensors.

  oor = Left.oor || Right.oor || Front.oor || Back.oor;

  avgControl();
  LRControl(); // run pwm functions.
  FBControl();

  int16_t flTarget = avgPWM + LDiffPWM + FDiffPWM;
  int16_t blTarget = avgPWM + LDiffPWM + BDiffPWM;
  int16_t frTarget = avgPWM + RDiffPWM + FDiffPWM;
  int16_t brTarget = avgPWM + RDiffPWM + BDiffPWM;

  FLPWM = slewLimit(flTarget, FLPrev);
  BLPWM = slewLimit(blTarget, BLPrev);
  FRPWM = slewLimit(frTarget, FRPrev);
  BRPWM = slewLimit(brTarget, BRPrev);

  FLPrev = FLPWM;
  BLPrev = BLPWM;
  FRPrev = FRPWM;
  BRPrev = BRPWM;
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

  // The following assumes 0 direction drives repulsion and 1 direction drives attraction.
  digitalWrite(dirFL, FLPWM < 0);
  analogWrite(pwmFL, abs(FLPWM));
  digitalWrite(dirBL, BLPWM < 0);
  analogWrite(pwmBL, abs(BLPWM));
  digitalWrite(dirFR, FRPWM < 0);
  analogWrite(pwmFR, abs(FRPWM));
  digitalWrite(dirBR, BRPWM < 0);
  analogWrite(pwmBR, abs(BRPWM));
}

void FullController::avgControl() {
  float avg = (Left.mmVal + Right.mmVal + Front.mmVal + Back.mmVal) * 0.25f;
  float eCurr = AvgRef - avg;

  avgError.eDiff = (tDiff == 0.0) ? 0:(eCurr - avgError.e) / tDiff; // rise over run
  avgError.eInt += eCurr * tDiff;
  avgError.eInt = constrain(avgError.eInt, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
  avgError.e = eCurr;

  avgPWM = pwmFunc(avgConsts, avgError);
}

void FullController::LRControl() {
  float diff = Right.mmVal - Left.mmVal; // how far above the right is the left?
  float eCurr = diff - LRDiffRef; // how different is that from the reference? positive -> Left repels, Right attracts.
  K_MAP rConsts = {LConsts.attracting, LConsts.repelling}; // apply attracting to repelling and vice versa.

  LRDiffErr.eDiff = (tDiff == 0.0) ? 0:(eCurr - LRDiffErr.e) / tDiff;
  LRDiffErr.eInt += eCurr * tDiff;
  LRDiffErr.eInt = constrain(LRDiffErr.eInt, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
  LRDiffErr.e = eCurr;

  LDiffPWM = pwmFunc(LConsts, LRDiffErr);
  RDiffPWM = -pwmFunc(rConsts, LRDiffErr);
}

void FullController::FBControl() {
  float diff = Back.mmVal - Front.mmVal; // how far above the back is the front?
  float eCurr = diff - FBDiffRef; // how different is that from ref? pos.->Front must repel, Back must attract
  K_MAP bConsts = {FConsts.attracting, FConsts.repelling};

  FBDiffErr.eDiff = (tDiff == 0.0) ? 0:(eCurr - FBDiffErr.e) / tDiff;
  FBDiffErr.eInt += eCurr * tDiff;
  FBDiffErr.eInt = constrain(FBDiffErr.eInt, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
  FBDiffErr.e = eCurr;

  FDiffPWM = pwmFunc(FConsts, FBDiffErr);
  BDiffPWM = -pwmFunc(bConsts, FBDiffErr);
}

int16_t FullController::pwmFunc(K_MAP consts, Errors errs) {
  if (oor) return 0;
  Constants constants = (errs.e < 0) ? consts.attracting : consts.repelling;
  return (int)constrain(constants.K*(errs.e + constants.ki*errs.eInt + constants.kd*errs.eDiff), -(float)CAP,(float)CAP);
}

int16_t FullController::slewLimit(int16_t target, int16_t prev) {
  int16_t maxChange = (int16_t)(slewRateLimit * tDiff);
  int16_t delta = target - prev;
  if (abs(delta) <= maxChange) return target;
  return prev + (delta > 0 ? maxChange : -maxChange);
}

void FullController::report() {
  Serial.print("SENSORS - Left: ");
  Serial.print(Left.mmVal);
  Serial.print("mm, Right: ");
  Serial.print(Right.mmVal);
  Serial.print("mm, Front: ");
  Serial.print(Front.mmVal);
  Serial.print("mm, Back: ");
  Serial.print(Back.mmVal);
  Serial.print("mm,\n");

  Serial.print("PWMS - FL_PWM: ");
  Serial.print(FLPWM);
  Serial.print(", BL_PWM: ");
  Serial.print(BLPWM);
  Serial.print("FR_PWM: ");
  Serial.print(FRPWM);
  Serial.print("BR_PWM: ");
  Serial.print(BRPWM);
  Serial.print("\n");

  Serial.print("CONTROL ON - ");
  Serial.print(outputOn);
  Serial.print("\n");
}