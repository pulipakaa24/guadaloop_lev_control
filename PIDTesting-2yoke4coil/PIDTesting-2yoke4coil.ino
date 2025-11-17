#include <Arduino.h>
#include "IndSensorMap.hpp"
#include "Controller.hpp"

// K, Ki, Kd Constants
Constants repelling = {40, 0.01, 7};
Constants attracting = {20, 0.01, 20};

Constants RollLeftUp = {30, 0.01, 15};
Constants RollLeftDown = {30, 0.01, 15};

Constants RollFrontUp = {30, 0.01, 15};
Constants RollFrontDown = {30, 0.01, 15};

// Reference values for average dist, 
float avgRef = 21.0; // TBD: what is our equilibrium height with this testrig?
float LRDiffRef = 0.0; // TBD: what is our left-right balance equilibrium? Positive -> left is above right
float FBDiffRef = 0.0; // TBD: what is front-back balance equilibrium? Positive -> front above back.

float slewRateLimit = 100.0; // max PWM change per control cycle (determined by 1 second / sampling rate)
// this was implemented by Claude and we can see if it helps.
// Set it at or above 255 to make it have no effect.

// Might be useful for things like jitter or lag.
#define sampling_rate 1000 // Hz

// ABOVE THIS LINE IS TUNING VALUES ONLY, BELOW IS ACTUAL CODE.

unsigned long tprior;
unsigned int tDiffMicros;

FullConsts fullConsts = {
  {repelling, attracting},
  {RollLeftDown, RollLeftUp},
  {RollFrontDown, RollFrontUp}
};

FullController controller(indL, indR, indF, indB, fullConsts, avgRef, LRDiffRef, FBDiffRef, slewRateLimit);

const int dt_micros = 1e6/sampling_rate;

#define LEV_ON

int ON = 0;

void setup() {
  Serial.begin(57600);

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
    // this might need to be changed if we have trouble getting serial to read. 
    char c = Serial.read();
    while(Serial.available()) Serial.read(); // flush remaining

    controller.outputOn = (c != '0');
  }
  
  tDiffMicros = micros() - tprior;

  if (tDiffMicros >= dt_micros){
    controller.update((float)tDiffMicros / (float)1e6);
    controller.report();
    controller.sendOutputs(); 
    // this and the previous line can be switched if you want the PWMs to display 0 when controller off.
    
    tprior = micros(); // maybe we have to move this line to before the update commands?
    // since the floating point arithmetic may take a while...
  }

  //Serial.println(telapsed);
}