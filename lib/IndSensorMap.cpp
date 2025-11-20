#include "IndSensorMap.hpp"
#include <Arduino.h>
#include <math.h>

// Sensor calibration data
IndSensorMap ind0Map = {-8.976076325826309, 913.5463710698101, 0.29767471011439534, 5.6686184386250025, 0.3627635461289861};
IndSensorMap ind1Map = {-4.831976283950702, 885.9877001844566, 0.2793284618109283, 3.8852507844119217, 0.2389935455347361};
IndSensorMap ind2Map = {-9.824360913609562, 871.4744633266955, 0.2909366235093304, 4.3307594408159495, 0.2822807132259202};
IndSensorMap ind3Map = {-13.891292062248292, 990.6819962477331, 0.16376045588859353, -0.074904004740735, 0.17727132893449118};

// IndSensor class implementation
IndSensor::IndSensor(IndSensorMap calibration, uint8_t analogPin)
  : consts(calibration), pin(analogPin), oor(false) {}

// Convert raw analog reading to millimeters using sensor calibration
float IndSensor::toMM(unsigned int raw) {
  return consts.C - (1.0 / consts.B) * log(pow((consts.K - consts.A) / ((float)raw - consts.A), consts.v) - 1.0);
}

// Read sensor directly from pin and convert to millimeters
float IndSensor::readMM() {
  unsigned int raw = analogRead(pin);
  oor = (raw == 0 || raw > 870);  // Update out-of-range flag
  mmVal = toMM(raw);
  return mmVal;
}

// Predefined sensor instances
IndSensor indL(ind1Map, A0);
IndSensor indR(ind0Map, A1);
IndSensor indF(ind3Map, A5);
IndSensor indB(ind2Map, A4);
