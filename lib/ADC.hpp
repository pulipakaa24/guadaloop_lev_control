#ifndef ADC_H
#define ADC_H

#include <Arduino.h>
#include <util/atomic.h>

extern volatile uint16_t adc_results[];

void setupADC();

#endif