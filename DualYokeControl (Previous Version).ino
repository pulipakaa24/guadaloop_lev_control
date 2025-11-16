#include <Arduino.h>
#include "IndSensorMap.hpp"

// PIN MAPPING

#define dirFR 2
#define pwmFR 3
#define dirBR 4
#define pwmBR 5
#define pwmFL 6
#define dirFL 7
#define dirBL 8
#define pwmBL 9

// variables

int dist_raw, tprior, telapsed, pwm, pwm2, dist2_raw;
bool oor, oor2;
float dist,ecurr, eprior, derror, ecum, ff,dist2,ecurr2, eprior2, derror2, ecum2, ff2;

#define CAP 200


// CONTROLLER CONSTANTS
float MAX_INTEGRAL_TERM = 1e4;

//// FOR MC 1
//const float K_f = 50; // gain for when we want to fall (ref > dist or error > 0)
//const float ki_f = 0.01;
//const float kd_f = 8; // 25
//
//const float K_a = 20;
//const float ki_a = ki_f;
//const float kd_a = 10; //30;

typedef struct Constants {
  float K;
  float ki;
  float kd;
} Constants;

typedef struct K_MAP {
  Constants falling;
  Constants attracting;
} K_MAP;

typedef struct Collective {
  K_MAP constants;
  float e;
  float eDiff;
  float eInt;
  float ref;
} Collective;

Collective collLeft = {{{40, 0.01, 7}, {20, 0.01, 20}}, 0, 0, 0, 21.0};
Collective collRight = {{{40, 0.01, 7}, {20, 0.01, 20}}, 0, 0, 0, 22.0};

int levCollective(Collective collective, bool oor){
  if (oor){
    pwm = 0;
  }
  else{
    Constants pidConsts;

    // this means that dist > ref so we gotta attract to track now vv
    if (collective.e < 0) pidConsts = collective.constants.attracting;
    // this is falling vv
    else pidConsts = collective.constants.falling;

    pwm = constrain(pidConsts.K*(collective.e + pidConsts.ki*collective.eInt + pidConsts.kd*collective.eDiff), -CAP,CAP);
  }
  return (int)pwm;
}

#define sampling_rate 1000
const int dt_micros = 1e6/sampling_rate;

#define LEV_ON

int ON = 0;

void setup() {
  // put your setup code here, to run once:

  Serial.begin(57600);

  tprior = micros();
  ecum = 0;
  ecum2 = 0;

  // positive pwm is A
  // negative is B

  // ATTRACT IS B  // REPEL IS A

  //when error is negative, I want to attract.
  send_pwmFL(0);
  send_pwmFR(0);

}

void loop() {
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');  // Read the full input
    inputString.trim();  // Remove leading/trailing whitespace, including \n and \r

    // Conditional pipeline
    if (inputString == "0") {
        ON=0;
    } 
    else {
        ON=1;
    }
  }
  
  telapsed = micros() - tprior;

  if (telapsed >= dt_micros){
    // put your main code here, to run repeatedly:
    indL.read();
    indR.read();
    indF.read();
    indB.read();
    dist_raw = analogRead(indL);
    if (dist_raw > 870) oor = true;
    dist = indToMM(ind0Map, dist_raw); // 189->950, 16->26
    Serial.print(dist);
    Serial.print(", ");

    dist2_raw = analogRead(indR);
    if (dist2_raw > 870) oor2 = true;
    dist2 = indToMM(ind1Map, dist2_raw);
    Serial.print(dist2);
    Serial.print(", ");

    ecurr = ref - dist;
    derror = ecurr - eprior;

    ecurr2 = ref2 - dist2;
    derror2 = ecurr2 - eprior2;
    
    ecum += ecurr * (telapsed / 1e6);
    ecum = constrain(ecum, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);
    ecum2 += ecurr2 * (telapsed / 1e6);
    ecum2 = constrain(ecum2, -MAX_INTEGRAL_TERM, MAX_INTEGRAL_TERM);

    
    if (ON) {
      int collective1 = 
      send_pwmFL(pwm);
      send_pwmFR(pwm2);
      Serial.print(pwm);
      Serial.print(", ");
      Serial.print(pwm2);
      Serial.print(", ");
    }
    else {
      send_pwmFL(0);
      send_pwmFR(0);
      Serial.print(0);
      Serial.print(", ");
      Serial.print(0);
      Serial.print(", ");
    }

    Serial.print(ON);
    
    tprior = micros();
    eprior = ecurr;
    eprior2 = ecurr2;
//    //Serial.print(ecurr); Serial.print(","); Serial.print(oor); Serial.print(","); Serial.print(derror); Serial.print(","); Serial.print(pwm); Serial.print(";  "); Serial.print(ecurr2); Serial.print(","); Serial.print(oor2); Serial.print(","); Serial.print(derror2); Serial.print(","); Serial.print(pwm2);
//    Serial.print(ecurr); Serial.print(","); Serial.print(ecurr2); Serial.print(","); Serial.print(ecum); Serial.print(",");Serial.print(ecum2); Serial.print(",");
//    
    Serial.println();
  }

  //Serial.println(telapsed);
}

void send_pwmFL(int val){
  if (val > 0) {
    digitalWrite(dirFL, LOW);
  }
  else{
    digitalWrite(dirFL,HIGH);
  }
  analogWrite(pwmFL,abs(val));
}

void send_pwmFR(int val){
  if (val > 0) {
    digitalWrite(dirFR, LOW);
  }
  else{
    digitalWrite(dirFR,HIGH);
  }

  analogWrite(pwmFR,abs(val));

}
