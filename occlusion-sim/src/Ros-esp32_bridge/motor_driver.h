/*********************************************************************
 *  motor_driver.h  -  BTS7960 pin map and function declarations
 *
 *  Hardware: 2x BTS7960 43A H-Bridge
 *  MCU     : ESP32 NodeMCU
 *
 *  Motor 1 = LEFT  (front-left + rear-left mechanically coupled)
 *  Motor 2 = RIGHT (front-right + rear-right mechanically coupled)
 *
 *  The BTS7960 uses separate PWM lines for each direction:
 *    RPWM = forward (positive speed)
 *    LPWM = reverse (negative speed)
 *  Both EN pins are tied HIGH at startup and left there.
 *********************************************************************/

#ifndef MOTOR_DRIVER_H
#define MOTOR_DRIVER_H

// ── Motor 1 (Left) ────────────────────────────────────────────────
#define RPWM1  14
#define LPWM1  12
#define R_EN1  27
#define L_EN1  26

// ── Motor 2 (Right) ───────────────────────────────────────────────
#define RPWM2  33
#define LPWM2  32
#define R_EN2  4
#define L_EN2  17

// ── Public API ────────────────────────────────────────────────────
void initMotorController();
void setMotorSpeed(int i, int spd);
void setMotorSpeeds(int leftSpeed, int rightSpeed);
void setMotorBrake(int i, int brk);
void setMotorBrakes(int leftBrake, int rightBrake);

#endif
