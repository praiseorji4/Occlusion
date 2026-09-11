/*********************************************************************
 *  encoder_driver.ino  -  Quadrature encoder handling (ESP32)
 *
 *  Uses a 16-entry lookup table to decode all four Gray-code
 *  transitions per channel per edge, giving full 4× resolution.
 *  This is more robust than the state-sum if/else approach under
 *  high interrupt rates because it handles missed edges gracefully
 *  and eliminates branching inside the ISR.
 *
 *  Both channels (A and B) trigger the same ISR on CHANGE so every
 *  edge is captured — this is required for 4× quadrature decoding.
 *
 *  Only the FRONT wheels are instrumented.  Rear wheels are
 *  mechanically slaved to the front and need no encoder.
 *********************************************************************/

#include "encoder_driver.h"
#include "commands.h"   // LEFT / RIGHT

// ── 4-bit Gray-code lookup table ──────────────────────────────────
// Index = (prev_A << 3) | (prev_B << 2) | (curr_A << 1) | curr_B
// Value = direction: +1 (CW / forward), -1 (CCW / reverse), 0 (invalid / no move)
static const int8_t QUAD_LUT[16] = {
   0,  1, -1,  0,   // prev=00
  -1,  0,  0,  1,   // prev=01
   1,  0,  0, -1,   // prev=10
   0, -1,  1,  0    // prev=11
};

// ── Encoder accumulators (ISR-written, main reads with noInterrupts) ─
volatile long encoderPositionM1 = 0;
volatile long encoderPositionM2 = 0;

// ── Per-encoder state (2-bit history kept in low nibble) ──────────
static volatile uint8_t encStateM1 = 0;
static volatile uint8_t encStateM2 = 0;

// ── ISR — Left (M1) ──────────────────────────────────────────────
void IRAM_ATTR handleEncoderM1()
{
    uint8_t a = (uint8_t)digitalRead(ENCODER_M1_A);
    uint8_t b = (uint8_t)digitalRead(ENCODER_M1_B);
    // Shift previous state into high bits, insert current state in low bits
    encStateM1 = ((encStateM1 << 2) | (a << 1) | b) & 0x0F;
    encoderPositionM1 += QUAD_LUT[encStateM1];
}

// ── ISR — Right (M2) ─────────────────────────────────────────────
void IRAM_ATTR handleEncoderM2()
{
    uint8_t a = (uint8_t)digitalRead(ENCODER_M2_A);
    uint8_t b = (uint8_t)digitalRead(ENCODER_M2_B);
    encStateM2 = ((encStateM2 << 2) | (a << 1) | b) & 0x0F;
    encoderPositionM2 += QUAD_LUT[encStateM2];
}

// ── Public functions ──────────────────────────────────────────────

void initEncoders()
{
    pinMode(ENCODER_M1_A, INPUT_PULLUP);
    pinMode(ENCODER_M1_B, INPUT_PULLUP);
    pinMode(ENCODER_M2_A, INPUT_PULLUP);
    pinMode(ENCODER_M2_B, INPUT_PULLUP);

    // Seed the state registers with the current pin levels so the
    // first edge produces a valid LUT index instead of index 0.
    encStateM1 = ((uint8_t)digitalRead(ENCODER_M1_A) << 1)
               |  (uint8_t)digitalRead(ENCODER_M1_B);
    encStateM2 = ((uint8_t)digitalRead(ENCODER_M2_A) << 1)
               |  (uint8_t)digitalRead(ENCODER_M2_B);

    // Both channels trigger the same ISR — required for 4x decoding
    attachInterrupt(digitalPinToInterrupt(ENCODER_M1_A), handleEncoderM1, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENCODER_M1_B), handleEncoderM1, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENCODER_M2_A), handleEncoderM2, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENCODER_M2_B), handleEncoderM2, CHANGE);
}

void resetEncoders()
{
    noInterrupts();
    encoderPositionM1 = 0;
    encoderPositionM2 = 0;
    interrupts();
}

void resetEncoder(int i)
{
    noInterrupts();
    if      (i == LEFT)  encoderPositionM1 = 0;
    else if (i == RIGHT) encoderPositionM2 = 0;
    interrupts();
}

long readEncoder(int i)
{
    if      (i == LEFT)  return encoderPositionM1;
    else if (i == RIGHT) return encoderPositionM2;
    return 0;
}
