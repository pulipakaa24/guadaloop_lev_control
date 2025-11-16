#include <Arduino.h>
#include "IndSensorMap.hpp"

// PIN MAPPING

#define indL A0
#define indR A1

#define dirFR 2
#define pwmFR 3
#define dirBR 4
#define pwmBR 5
#define pwmFL 6
#define dirFL 7
#define dirBL 8
#define pwmBL 9

// variables

int dist_raw, tprior, telapsed, pwm, pwm2, oor, oor2, dist2_raw;
float dist,ecurr, eprior, derror, ecum, ff,dist2,ecurr2, eprior2, derror2, ecum2, ff2;

#define CAP 200


// CONTROLLER CONSTANTS

float ki, kd, K;
float MAX_INTEGRAL_TERM = 1e4;

const float ref = 21.0; //14.9;
const float ref2 = 22.0; //14.9;

const float refL = (ref + ref2) / 2;
const float refR = refL;

const float K_current = 1;
const float ki_current = 0;

//// FOR MC 1
//const float K_f = 50; // gain for when we want to fall (ref > dist or error > 0)
//const float ki_f = 0.01;
//const float kd_f = 8; // 25
//
//const float K_a = 20;
//const float ki_a = ki_f;
//const float kd_a = 10; //30;

// FOR MC 1
const float K_f = 40; // gain for when we want to fall (ref > dist or error > 0)
const float ki_f = 0.01;
const float kd_f = 7; // 25

const float K_a = 20;
const float ki_a = ki_f;
const float kd_a = 20; //30;

// FOR MC 2

const float K2_f = K_f;// gain for when we want to fall (ref > dist or error > 0)
const float ki2_f = ki_f;
const float kd2_f = kd_f;

const float K2_a = K_a;
const float ki2_a = ki_a;
const float kd2_a = kd_a;

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
    dist_raw = analogRead(indL);
    if (dist_raw > 870) oor = true;
    dist = ind2mm(ind0Map, dist_raw); // 189->950, 16->26
    Serial.print(dist);
    Serial.print(", ");

    dist2_raw = analogRead(indR);
    if (dist2_raw > 870) oor2 = true;
    dist2 = ind2mm(ind1Map, dist2_raw);
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
      int collective1 = levitate(ecurr, derror, ecum, oor);
      int collective2 = levitate2(ecurr2, derror2, ecum2, oor2);
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

int levitate(float e, float de, float ecum, int oor){
  if (oor){
    pwm = 0;
  }
  else{
    if (e < 0) { // this means that dist > ref so we gotta attract to track now
      kd = kd_a;
      ki = ki_a;
      K = K_a;
    }
    else{
      kd = kd_f;
      ki = ki_f;
      K = K_f;
    }

    pwm = constrain(K*(e + ki*ecum + kd*de), -CAP,CAP);

  }
  return (int)pwm;
}

int levitate2(float e, float de, float ecunm, int oor){
  if (oor){
    pwm2 = 0;
  }
  else{
    if (e < 0) { // this means that dist > ref so we gotta attract to track now
      kd = kd2_a;
      ki = ki2_a;
      K = K2_a;
    }
    else{
      kd = kd2_f;
      ki = ki2_f;
      K = K2_f;
    }

    pwm2 = constrain(K*(e + ki*ecum + kd*de), -CAP,CAP);

  }
  return (int)pwm2;

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
