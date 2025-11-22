#include "ADC.hpp"

#define NUM_PINS 4

const uint8_t adc_pins[] = {0, 1, 4, 5}; // A0, A1, A4, A5

volatile uint16_t adc_results[NUM_PINS];
volatile uint8_t current_channel_index = 0;

void setupADC() {
  // 1. Set Reference to AVCC (5V)
  // REFS0 = 1, REFS1 = 0
  ADMUX = (1 << REFS0);

  // 2. Set ADC Prescaler to 128 (16MHz / 128 = 125KHz)
  // Good balance of speed and accuracy. 
  // Bits: ADPS2, ADPS1, ADPS0
  ADCSRA |= (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);

  // 3. Enable ADC and Enable ADC Interrupt
  ADCSRA |= (1 << ADEN) | (1 << ADIE);

  // 4. Set initial channel to the first pin in our list
  ADMUX = (ADMUX & 0xF0) | (adc_pins[0] & 0x0F);

  // 5. Start the first conversion!
  ADCSRA |= (1 << ADSC); 
}

ISR(ADC_vect) {
  // 1. Read the result (must read ADCL first, then ADCH, or just use ADC word)
  adc_results[current_channel_index] = ADC;

  // 2. Increment to next channel
  current_channel_index++;
  if (current_channel_index >= NUM_PINS) {
    current_channel_index = 0;
  }

  // 3. Switch MUX to next channel
  // Clear bottom 4 bits of ADMUX, then OR in the new pin number
  ADMUX = (ADMUX & 0xF0) | (adc_pins[current_channel_index] & 0x0F);

  // 4. Start next conversion
  ADCSRA |= (1 << ADSC);
}