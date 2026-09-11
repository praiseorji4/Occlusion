/*********************************************************************
 *  DualBTS7960MotorShieldESP32.h  -  BTS7960 43A H-Bridge for ESP32
 *********************************************************************/
#pragma once
#include <Arduino.h>

class DualBTS7960MotorShieldESP32
{
public:
    DualBTS7960MotorShieldESP32();
    DualBTS7960MotorShieldESP32(unsigned char RPWM1, unsigned char LPWM1,
                                // unsigned char R_EN1, unsigned char L_EN1,
                                unsigned char RPWM2, unsigned char LPWM2);
                                // unsigned char R_EN2, unsigned char L_EN2);

    void init();
    void setM1Speed(int speed);   // -255 .. 255
    void setM2Speed(int speed);
    void setSpeeds(int m1Speed, int m2Speed);
    void setM1Brake(int brake);   //   0 .. 255
    void setM2Brake(int brake);
    void setBrakes(int m1Brake, int m2Brake);

private:
    unsigned char _RPWM1, _LPWM1, _R_EN1, _L_EN1;
    unsigned char _RPWM2, _LPWM2, _R_EN2, _L_EN2;
};
