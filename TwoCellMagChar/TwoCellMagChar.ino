#include "HX711.h"
#include "CalibConsts.hpp"

HX711 c0, c1;

//  adjust pins if needed.
#define c0Data 2
#define c0Clock 3
#define c1Data 4
#define c1Clock 5

#define DIR0 7
#define PWM0 6
#define DIR1 8
#define PWM1 9

void setup()
{
  pinMode(DIR0, OUTPUT);
  pinMode(PWM0, OUTPUT);
  pinMode(DIR1, OUTPUT);
  pinMode(PWM1, OUTPUT);

  Serial.begin(115200);
  Serial.println();
  Serial.println(__FILE__);
  Serial.print("HX711_LIB_VERSION: ");
  Serial.println(HX711_LIB_VERSION);
  Serial.println();

  c0.begin(c0Data, c0Clock);
  c0.set_offset(OFFSET0);
  c0.set_scale(SCALE0);

  c1.begin(c1Data, c1Clock);
  c1.set_offset(OFFSET1);
  c1.set_scale(SCALE1);
}

void loop() {
  while (Serial.available()) Serial.read();
  Serial.println("Enter any key to begin reading");
  while (!Serial.available());
  Serial.println("Reading Load Cells (Mean over 10 Readings)");
  float c0Avgs[11] = {0};
  float c1Avgs[11] = {0};
  uint8_t ind = 0;

  for (int16_t i = -250; i < 255; i+=50) {
    Serial.print("Running PWM: ");
    Serial.println(i);
    digitalWrite(DIR0, i >= 0);
    digitalWrite(DIR1, i >= 0);
    analogWrite(PWM0, abs(i));
    analogWrite(PWM1, abs(i));
    delay(200);
    for (uint8_t j = 0; j < 10; j++) c0Avgs[ind] += c0.get_units();
    c0Avgs[ind] /= 10;
    for (uint8_t j = 0; j < 10; j++) c1Avgs[ind] += c1.get_units();
    c1Avgs[ind] /= 10;
    ind++;
  }

  digitalWrite(DIR0, 0);
  digitalWrite(DIR1, 0);
  digitalWrite(PWM0, 0);
  digitalWrite(PWM1, 0);

  Serial.println("Average Values:");
  for (uint8_t i = 0; i < 11; i++) {
    Serial.print("PWM ");
    Serial.print((int) -250+i*50);
    Serial.print(" - Cell 0: ");
    Serial.print(c0Avgs[i]);
    Serial.print("g, Cell 1: ");
    Serial.print(c1Avgs[i]);
    Serial.println("g");
  }  
}