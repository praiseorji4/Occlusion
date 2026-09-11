/*********************************************************************
 *  motor_driver.ino  -  BTS7960 motor driver implementation
 *********************************************************************/

#include "DualBTS7960MotorShieldESP32.h"
#include "motor_driver.h"
#include "commands.h"   // LEFT / RIGHT

DualBTS7960MotorShieldESP32 drive(RPWM1, LPWM1,
                                   RPWM2, LPWM2);

void initMotorController()
{
    drive.init();
}

void setMotorSpeed(int i, int spd)
{
    if      (i == LEFT)  drive.setM1Speed(spd);
    else if (i == RIGHT) drive.setM2Speed(spd);
}

void setMotorSpeeds(int leftSpeed, int rightSpeed)
{
    drive.setM1Speed(leftSpeed);
    drive.setM2Speed(rightSpeed);
}

void setMotorBrake(int i, int brk)
{
    if      (i == LEFT)  drive.setM1Brake(brk);
    else if (i == RIGHT) drive.setM2Brake(brk);
}

void setMotorBrakes(int leftBrake, int rightBrake)
{
    drive.setM1Brake(leftBrake);
    drive.setM2Brake(rightBrake);
}
