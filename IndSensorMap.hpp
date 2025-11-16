#ifndef IND_SENSOR_MAP_HPP
#define IND_SENSOR_MAP_HPP

// Inductive Sensor Mapping Struct
typedef struct IndSensorMap {
  double A;
  double K;
  double B;
  double C;
  double v;
} IndSensorMap;

class IndSensor {
public:
  bool oor;
  int mmVal;

  // Constructor
  IndSensor(IndSensorMap calibration, uint8_t analogPin);
  // Read sensor directly from pin and convert to millimeters
  float read();

private:
  IndSensorMap consts;
  uint8_t pin;

  // Read sensor and convert to millimeters
  float toMM(unsigned int raw);
};

// sensor instances
extern IndSensor indL;
extern IndSensor indR;
extern IndSensor indF;
extern IndSensor indB;

#endif // IND_SENSOR_MAP_HPP
