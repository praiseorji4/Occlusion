/*********************************************************************
 *  DualBTS7960MotorShieldESP32.cpp  -  BTS7960 43A H-Bridge for ESP32
 *********************************************************************/
#include "DualBTS7960MotorShieldESP32.h"

// ── Constructors ──────────────────────────────────────────────────

DualBTS7960MotorShieldESP32::DualBTS7960MotorShieldESP32()
    : _RPWM1(14), _LPWM1(12), _R_EN1(27), _L_EN1(26),
      _RPWM2(32), _LPWM2(33), _R_EN2(4),  _L_EN2(17)
{}

DualBTS7960MotorShieldESP32::DualBTS7960MotorShieldESP32(
    unsigned char RPWM1, unsigned char LPWM1,
    // unsigned char R_EN1, unsigned char L_EN1,
    unsigned char RPWM2, unsigned char LPWM2)
    // unsigned char R_EN2, unsigned char L_EN2)
    // : _RPWM1(RPWM1), _LPWM1(LPWM1), _R_EN1(R_EN1), _L_EN1(L_EN1),
    //   _RPWM2(RPWM2), _LPWM2(LPWM2), _R_EN2(R_EN2), _L_EN2(L_EN2)
    : _RPWM1(RPWM1), _LPWM1(LPWM1),
      _RPWM2(RPWM2), _LPWM2(LPWM2)
{}

// ── init() ────────────────────────────────────────────────────────

void DualBTS7960MotorShieldESP32::init()
{
    // Enable pins — BTS7960 EN is active-HIGH
    // pinMode(_R_EN1, OUTPUT); digitalWrite(_R_EN1, HIGH);
    // pinMode(_L_EN1, OUTPUT); digitalWrite(_L_EN1, HIGH);
    // pinMode(_R_EN2, OUTPUT); digitalWrite(_R_EN2, HIGH);
    // pinMode(_L_EN2, OUTPUT); digitalWrite(_L_EN2, HIGH);

    // PWM: 20 kHz, 8-bit — above audible range, well within BTS7960 spec
    const int freq = 20000;
    const int res  = 8;
    ledcAttachChannel(_RPWM1, freq, res, 0);
    ledcAttachChannel(_LPWM1, freq, res, 1);
    ledcAttachChannel(_RPWM2, freq, res, 2);
    ledcAttachChannel(_LPWM2, freq, res, 3);

    ledcWrite(_RPWM1, 0); ledcWrite(_LPWM1, 0);
    ledcWrite(_RPWM2, 0); ledcWrite(_LPWM2, 0);
}

// ── Speed control ─────────────────────────────────────────────────

void DualBTS7960MotorShieldESP32::setM1Speed(int speed)
{
    speed = constrain(speed, -255, 255);
    if (speed > 0) {
        ledcWrite(_RPWM1, speed);  ledcWrite(_LPWM1, 0);
    } else if (speed < 0) {
        ledcWrite(_RPWM1, 0);     ledcWrite(_LPWM1, -speed);
    } else {
        ledcWrite(_RPWM1, 0);     ledcWrite(_LPWM1, 0);
    }
}

void DualBTS7960MotorShieldESP32::setM2Speed(int speed)
{
    speed = constrain(speed, -255, 255);
    if (speed > 0) {
        ledcWrite(_RPWM2, speed);  ledcWrite(_LPWM2, 0);
    } else if (speed < 0) {
        ledcWrite(_RPWM2, 0);     ledcWrite(_LPWM2, -speed);
    } else {
        ledcWrite(_RPWM2, 0);     ledcWrite(_LPWM2, 0);
    }
}

void DualBTS7960MotorShieldESP32::setSpeeds(int m1Speed, int m2Speed)
{
    setM1Speed(m1Speed);
    setM2Speed(m2Speed);
}

// ── Braking (both PWM lines high → regenerative brake) ───────────

void DualBTS7960MotorShieldESP32::setM1Brake(int brake)
{
    brake = constrain(abs(brake), 0, 255);
    ledcWrite(_RPWM1, brake);
    ledcWrite(_LPWM1, brake);
}

void DualBTS7960MotorShieldESP32::setM2Brake(int brake)
{
    brake = constrain(abs(brake), 0, 255);
    ledcWrite(_RPWM2, brake);
    ledcWrite(_LPWM2, brake);
}

void DualBTS7960MotorShieldESP32::setBrakes(int m1Brake, int m2Brake)
{
    setM1Brake(m1Brake);
    setM2Brake(m2Brake);
}
