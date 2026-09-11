/*********************************************************************
 *  commands.h  -  Serial command definitions
 *
 *  ── ROS SERIAL CHANNEL  (Serial0 / USB, 115200 8N1) ─────────────
 *  Host → ESP32 :  <cmd> [<arg1>] [<arg2>]\r
 *  ESP32 → Host :  response (see below), always terminated \r\n
 *
 *  cmd  args          ESP32 response
 *  ---  ----          --------------
 *  'b'  -             <baudrate>
 *  'e'  -             <leftEnc> <rightEnc>
 *  'r'  -             OK
 *  'm'  tpf_L tpf_R   OK          (ticks-per-frame, signed ints)
 *  'o'  pwm_L pwm_R   OK          (raw PWM, -255..255)
 *  'u'  Kp:Kd:Ki:Min  OK          (both sides, legacy format)
 *  'l'  Kp:Ki:Kd:Min  OK          (left side only)
 *  'f'  Kp:Ki:Kd:Min  OK          (right side only)
 *  'q'  -             D <14 space-separated fields>\r\n
 *
 *  ── DIAGNOSTIC STREAM  (Serial2 GPIO-16, 115200 TX-only, 30 Hz) ─
 *  Emitted automatically each PID tick (moving or stopped).
 *  Also emitted on demand via 'q' on the ROS serial channel.
 *
 *  Wire format (single line, \r\n terminated):
 *
 *   D lEnc rEnc lTgtRpm rTgtRpm lRpm rRpm
 *     lErr rErr lInt rInt lOut rOut linVel angVel
 *
 *  Field  Name        Unit     PlotJuggler path
 *  -----  ----------  ----     ----------------
 *  0      D           -        (header, skip)
 *  1      lEnc        ticks    /diag/enc/left
 *  2      rEnc        ticks    /diag/enc/right
 *  3      lTgtRpm     RPM      /diag/rpm/left_target
 *  4      rTgtRpm     RPM      /diag/rpm/right_target
 *  5      lRpm        RPM      /diag/rpm/left_actual
 *  6      rRpm        RPM      /diag/rpm/right_actual
 *  7      lErr        RPM      /diag/pid/left_error
 *  8      rErr        RPM      /diag/pid/right_error
 *  9      lInt        -        /diag/pid/left_integral
 *  10     rInt        -        /diag/pid/right_integral
 *  11     lOut        PWM      /diag/pid/left_output
 *  12     rOut        PWM      /diag/pid/right_output
 *  13     linVel      m/s      /diag/vel/linear
 *  14     angVel      rad/s    /diag/vel/angular
 *
 *  PlotJuggler setup:
 *    Streamer -> "Serial IO"  ->  /dev/ttyUSB1  115200
 *    Separator: space,  Skip lines not starting with 'D'
 *********************************************************************/

#ifndef COMMANDS_H
#define COMMANDS_H

#define LEFT  1
#define RIGHT 2

#define GET_BAUDRATE      'b'
#define READ_ENCODERS     'e'
#define MOTOR_SPEEDS      'm'
#define MOTOR_RAW_PWM     'o'
#define RESET_ENCODERS    'r'
#define UPDATE_PID        'u'
#define UPDATE_LEFT_PID   'l'
#define UPDATE_RIGHT_PID  'f'
#define READ_DIAGNOSTICS  'q'

#endif
