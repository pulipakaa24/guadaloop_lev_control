#ifndef IND_SENSOR_MAP_HPP
#define IND_SENSOR_MAP_HPP

#include <stdint.h>

// Inductive Sensor Mapping Struct
typedef struct IndSensorMap {
  float A;
  float K;
  float B;
  float C;
  float v;
} IndSensorMap;

class IndSensor {
  public:
    bool oor;
    float mmVal;

    // Constructor
    IndSensor(IndSensorMap calibration, uint8_t analogPin);
    // Read sensor directly from pin and convert to millimeters
    float readMM();

  private:
    IndSensorMap consts;
    uint8_t pin;

    // helper function to convert analog reading to millimeters
    float toMM(unsigned int raw);
};

// sensor instances
extern IndSensor indL;
extern IndSensor indR;
extern IndSensor indF;
extern IndSensor indB;

#endif // IND_SENSOR_MAP_HPP
