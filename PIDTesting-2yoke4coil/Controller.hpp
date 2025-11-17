#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <stdint.h>
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

#define CAP 200

typedef struct Constants {
  float K;
  float ki;
  float kd;
} Constants;

typedef struct K_MAP {
  Constants repelling;
  Constants attracting;
} K_MAP;

typedef struct FullConsts {
  K_MAP avg;
  K_MAP lColl; // repelling is applied to attracting and vice versa for the Right and Back collectives.
  K_MAP fColl;
} FullConsts;

typedef struct Errors {
  float e;
  float eDiff;
  float eInt;
} Errors;

class FullController {
  public:
    bool oor;
    bool outputOn;

    FullController(IndSensor& l, IndSensor& r, IndSensor& f, IndSensor& b,
      FullConsts fullConsts, float avgRef, float lrDiffRef, float fbDiffRef, float slewRate)
      : Left(l), Right(r), Front(f), Back(b), AvgRef(avgRef), LRDiffRef(lrDiffRef),
      FBDiffRef(fbDiffRef), avgConsts(fullConsts.avg), LConsts(fullConsts.lColl),
      FConsts(fullConsts.fColl), avgError({0,0,0}), LRDiffErr({0,0,0}), 
      FBDiffErr({0,0,0}), oor(false), outputOn(false),
      FLPrev(0), BLPrev(0), FRPrev(0), BRPrev(0), slewRateLimit(slewRate) {}

    void update(float tDiff);
    void zeroPWMs();
    void sendOutputs();
    void report();

  private:
    void avgControl();
    void LRControl();
    void FBControl();
    int16_t pwmFunc(K_MAP consts, Errors errs);
    int16_t slewLimit(int16_t target, int16_t prev);

    IndSensor& Front;
    IndSensor& Back;
    IndSensor& Right;
    IndSensor& Left;

    K_MAP avgConsts;
    K_MAP LConsts;
    K_MAP FConsts;

    Errors avgError;
    Errors LRDiffErr;
    Errors FBDiffErr;

    float AvgRef;
    float LRDiffRef;
    float FBDiffRef;
    float slewRateLimit;

    int16_t avgPWM;
    int16_t LDiffPWM;
    int16_t RDiffPWM;
    int16_t FDiffPWM;
    int16_t BDiffPWM;
    // Initially, I was going to make the Right and Back just the negatives,
    // but having separate control functions running for each of these outputs might prove useful.

    int16_t FLPWM;
    int16_t BLPWM;
    int16_t FRPWM;
    int16_t BRPWM;

    int16_t FLPrev;
    int16_t BLPrev;
    int16_t FRPrev;
    int16_t BRPrev;

    float tDiff;
};
#endif // CONTROLLER_HPP