#include "IndSensorMap.hpp"
#include <Arduino.h>
#include <math.h>
#include "ADC.hpp"

// Sensor calibration data with pre-computed constants
IndSensorMap ind0Map = {
  -8.976076325826309, 913.5463710698101, 0.29767471011439534, 5.6686184386250025, 0.3627635461289861,
  1.0f/0.29767471011439534f, 913.5463710698101f - (-8.976076325826309f)
};
IndSensorMap ind1Map = {
  -4.831976283950702, 885.9877001844566, 0.2793284618109283, 3.8852507844119217, 0.2389935455347361,
  1.0f/0.2793284618109283f, 885.9877001844566f - (-4.831976283950702f)
};
IndSensorMap ind2Map = {
  -9.824360913609562, 871.4744633266955, 0.2909366235093304, 4.3307594408159495, 0.2822807132259202,
  1.0f/0.2909366235093304f, 871.4744633266955f - (-9.824360913609562f)
};
IndSensorMap ind3Map = {
  -13.8907146886418, 990.6824637304771, 0.16376005385006073, -0.07513804021312243, 0.1772655198934789,
  1.0f/0.16376005385006073f, 990.6824637304771f - (-13.8907146886418f)
};

// IndSensor class implementation
IndSensor::IndSensor(IndSensorMap calibration, uint8_t analogPin, float emaAlpha)
  : consts(calibration), pin(analogPin), alpha(emaAlpha), oor(false), filteredRaw(0) {
  // Initialize filtered value with first reading
  filteredRaw = analogRead(pin);
}

// Convert raw analog reading to millimeters using sensor calibration
float IndSensor::toMM(uint16_t raw) {
  // Optimized version using pre-computed constants and faster math functions
  float raw_minus_A = (float)raw - consts.A;
  float ratio = consts.K_minus_A / raw_minus_A;
  float powered = powf(ratio, consts.v);
  float inside_log = powered - 1.0f;
  return consts.C - consts.invB * logf(inside_log);
}

// Read sensor directly from pin and convert to millimeters
float IndSensor::readMM() {
  uint8_t index = pin - A0;
  index = (index > 3) ? index - 2 : index;
  uint16_t raw;

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    raw = constrain(adc_results[index], 0, 900);
  }
  
  // Exponential moving average filter
  filteredRaw = alpha * raw + (1.0f - alpha) * filteredRaw;
  
  analog = (uint16_t)filteredRaw;
  oor = (analog < 10 || analog > 870);
  mmVal = toMM(analog);
  return mmVal;
}

// Predefined sensor instances
IndSensor indF(ind1Map, A0);
IndSensor indB(ind0Map, A1);
IndSensor indR(ind3Map, A5);
IndSensor indL(ind2Map, A4);