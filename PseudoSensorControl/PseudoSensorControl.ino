#include <Arduino.h>
#include "IndSensorMap.hpp"
#include "PseudoSensorControl.hpp"
#include "ADC.hpp"

float refs[4] = {12.9,12.3,12.6,12};

Constants repelling = {250, 0, 20000};
Constants attracting = {250, 0, 20000};

K_MAP consts = {repelling, attracting};

#define slewRateLimit 100 // max PWM change per control cycle (determined by 1 second / sampling rate)
// this was implemented by Claude and we can see if it helps.
// Set it at or above 255 to make it have no effect.

// Might be useful for things like jitter or lag.
#define sampling_rate 1000 // Hz

// EMA filter alpha value (all sensors use same alpha)
#define alphaVal 0.3f


// ABOVE THIS LINE IS TUNING VALUES ONLY, BELOW IS ACTUAL CODE.

unsigned long tprior;
unsigned int tDiffMicros;

PseudoSensorController controller(indL, indR, indF, indB, consts, refs, slewRateLimit);

const int dt_micros = 1e6/sampling_rate;

#define LEV_ON

int ON = 0;

void setup() {
  Serial.begin(115200);

  setupADC();

  indL.alpha = alphaVal;
  indR.alpha = alphaVal;
  indF.alpha = alphaVal;
  indB.alpha = alphaVal;

  tprior = micros();
  for (PinPair& mc : pinMap) {
    pinMode(mc.dir, OUTPUT);
    pinMode(mc.pwm, OUTPUT);
  }
}

void loop() {
  if (Serial.available() > 0) {
    // this might need to be changed if we have trouble getting serial to read. 
    char c = Serial.read();
    while(Serial.available()) Serial.read(); // flush remaining

    controller.outputOn = (c != '0' && c != 'r'); // planning to add r command to set refernce or smth
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
}