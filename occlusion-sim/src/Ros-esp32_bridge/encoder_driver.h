/*********************************************************************
 *  encoder_driver.h  -  Quadrature encoder pin definitions
 *
 *  Robot: Horizon (4WD, front-wheel encoders only)
 *  MCU  : ESP32 NodeMCU
 *  Type : Optical quadrature, both channels interrupt-driven
 *
 *  Only the FRONT wheels carry encoders.  The rear wheels are
 *  mechanically coupled to their front counterpart (same axle /
 *  belt), so no additional instrumentation is needed.
 *
 *  Wiring:
 *    Left  front encoder  A -> GPIO 18   B -> GPIO 19
 *    Right front encoder  A -> GPIO 23   B -> GPIO 22
 *********************************************************************/

#ifndef ENCODER_DRIVER_H
#define ENCODER_DRIVER_H

// ── Pin map ────────────────────────────────────────────────────────
#define ENCODER_M1_A  18   // Left  encoder channel A
#define ENCODER_M1_B  19   // Left  encoder channel B
#define ENCODER_M2_A  23   // Right encoder channel A
#define ENCODER_M2_B  22   // Right encoder channel B

// ── Public API ────────────────────────────────────────────────────
void initEncoders();
long readEncoder(int i);
void resetEncoder(int i);
void resetEncoders();

#endif
