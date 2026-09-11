/*********************************************************************
 *  Ros-esp32_bridge.ino  -  Serial bridge: ESP32 <-> ROS 2 (Jazzy)
 *
 *  Hardware
 *  --------
 *  MCU         : ESP32 NodeMCU
 *  Motor driver: 2x BTS7960 43A H-Bridge
 *  Drive config: 4WD — front wheels only carry encoders;
 *                rear wheels are mechanically slaved to their
 *                respective front counterpart (same axle / belt).
 *
 *  Serial channels
 *  ---------------
 *  Serial  (USB, GPIO1/3)  115200  ROS command/response channel
 *                                  Handled by arduino_comms.hpp on Pi
 *  Serial2 (TX GPIO16)     115200  Diagnostic stream (TX-only)
 *                                  Connect to /dev/ttyUSB1 on Pi
 *                                  Feed directly into PlotJuggler
 *
 *  Command protocol — Serial channel
 *  -----------------------------------
 *  See commands.h for full table.  Key commands:
 *    'e'         -> "<lEnc> <rEnc>\r\n"
 *    'm' L R     -> "OK\r\n"   (set target ticks-per-frame)
 *    'q'         -> diagnostic line (same format as Serial2 stream)
 *
 *  Diagnostic stream — Serial2 (30 Hz)
 *  -------------------------------------
 *  See commands.h and diff_controller.h for full field list.
 *  PlotJuggler: Streamer -> Serial IO -> /dev/ttyUSB1 -> 115200
 *               Token separator: space,  filter lines starting with 'D'
 *
 *  Calibration checklist (do in order)
 *  ------------------------------------
 *  1. Spin left wheel 10 turns by hand.
 *     ros2 topic pub ... 'e' and watch /diag/enc/left in PlotJuggler.
 *     Expected delta: ~39560 (10 x 3956).
 *     Adjust ENC_CPR_LEFT in diff_controller.h if wrong.
 *
 *  2. Find MIN_PWM: send 'o 60 60' and decrease until wheels stall.
 *     Set leftMinPwm / rightMinPwm just above that.
 *
 *  3. Tune KP: command 0.2 m/s forward.  Watch lTgtRpm vs lRpm in
 *     PlotJuggler.  Raise KP until tracking is tight without oscillation.
 *
 *  4. Add KD (try 0.05) if oscillation appears.
 *
 *  5. Add KI (try 0.5) only if steady-state RPM error persists.
 *
 *  6. Drive 1 m straight -> check /diff_drive_controller/odom x ≈ 1.0
 *     Adjust WHEEL_RADIUS_M in diff_controller.h AND ubot_ros2_control.xacro.
 *
 *  7. Spin 360° in place -> check yaw returns to ≈ 0.0
 *     Adjust WHEEL_SEPARATION_M in diff_controller.h AND ubot_controllers.yaml.
 *********************************************************************/

#include <Arduino.h>
#include "DualBTS7960MotorShieldESP32.h"
#include "commands.h"
#include "motor_driver.h"
#include "encoder_driver.h"
#include "diff_controller.h"

// ── Serial config ─────────────────────────────────────────────────
#define BAUDRATE  115200

// ── Timing ────────────────────────────────────────────────────────
const int    PID_INTERVAL      = 1000 / PID_RATE;   // ms per PID tick
unsigned long nextPID          = PID_INTERVAL;

#define AUTO_STOP_INTERVAL     10000                  // ms
long lastMotorCommand          = AUTO_STOP_INTERVAL;

// ── Command parser state ──────────────────────────────────────────
static int  arg   = 0;
static int  indx  = 0;
static char chr;
static char cmd;
static char argv1[16];
static char argv2[16];
static long arg1;
static long arg2;

// ─────────────────────────────────────────────────────────────────
void resetCommand()
{
    cmd = '\0';
    memset(argv1, 0, sizeof(argv1));
    memset(argv2, 0, sizeof(argv2));
    arg1 = arg2 = 0;
    arg  = indx = 0;
}

// ─────────────────────────────────────────────────────────────────
int runCommand()
{
    char  *p   = argv1;
    char  *str;
    float  pid_args[4];
    int    idx;

    arg1 = atoi(argv1);
    arg2 = atoi(argv2);

    switch (cmd) {

        // ── 'b'  get baudrate ────────────────────────────────────
        case GET_BAUDRATE:
            Serial.println(BAUDRATE);
            break;

        // ── 'e'  read encoder counts ─────────────────────────────
        case READ_ENCODERS: {
            noInterrupts();
            long le = readEncoder(LEFT);
            long re = readEncoder(RIGHT);
            interrupts();
            Serial.printf("%ld %ld\r\n", le, re);
            Serial.flush();
            break;
        }

        // ── 'r'  reset encoders + PID ────────────────────────────
        case RESET_ENCODERS:
            resetEncoders();
            resetPID();
            Serial.println("OK");
            break;

        // ── 'm'  set motor speeds (ticks-per-frame) ──────────────
        case MOTOR_SPEEDS:
            lastMotorCommand = millis();
            if (arg1 == 0 && arg2 == 0) {
                setMotorBrakes(200, 200);
                resetPID();
                moving = 0;
            } else {
                moving = 1;
                cmdStartTime   = millis();
                cmdTimerActive = true;
                leftPID.integral    = 0.0f;  rightPID.integral    = 0.0f;
                leftPID.last_error  = 0.0f;  rightPID.last_error  = 0.0f;
            }
            leftPID.TargetTicksPerFrame  = arg1;
            rightPID.TargetTicksPerFrame = arg2;
            Serial.println("OK");
            break;

        // ── 'o'  raw PWM (bypasses PID) ──────────────────────────
        case MOTOR_RAW_PWM:
            lastMotorCommand = millis();
            resetPID();
            moving = 0;
            setMotorSpeeds(arg1, arg2);
            Serial.println("OK");
            Serial.flush();
            break;

        // ── 'u'  update both PID gains (legacy: Kp:Kd:Ki:Min) ───
        case UPDATE_PID: {
            idx = 0;
            while ((str = strtok_r(p, ":", &p)) != nullptr && idx < 4)
                pid_args[idx++] = atof(str);
            if (idx < 4) { Serial.println("Error: need Kp:Kd:Ki:MinPwm"); break; }
            leftKp  = rightKp  = pid_args[0];
            leftKd  = rightKd  = pid_args[1];
            leftKi  = rightKi  = pid_args[2];
            leftMinPwm = rightMinPwm = (int)pid_args[3];
            Serial.println("OK");
            break;
        }

        // ── 'l'  update left PID gains (Kp:Ki:Kd:Min) ───────────
        case UPDATE_LEFT_PID: {
            idx = 0;
            while ((str = strtok_r(p, ":", &p)) != nullptr && idx < 4)
                pid_args[idx++] = atof(str);
            if (idx < 4) { Serial.println("Error: need Kp:Ki:Kd:MinPwm"); break; }
            leftKp = pid_args[0]; leftKi = pid_args[1];
            leftKd = pid_args[2]; leftMinPwm = (int)pid_args[3];
            Serial.println("OK");
            break;
        }

        // ── 'f'  update right PID gains (Kp:Ki:Kd:Min) ──────────
        case UPDATE_RIGHT_PID: {
            idx = 0;
            while ((str = strtok_r(p, ":", &p)) != nullptr && idx < 4)
                pid_args[idx++] = atof(str);
            if (idx < 4) { Serial.println("Error: need Kp:Ki:Kd:MinPwm"); break; }
            rightKp = pid_args[0]; rightKi = pid_args[1];
            rightKd = pid_args[2]; rightMinPwm = (int)pid_args[3];
            Serial.println("OK");
            break;
        }

        // ── 'q'  on-demand full diagnostic snapshot ───────────────
        // Emits the same 14-field "D ..." line as the Serial2 stream
        // so arduino_comms.hpp can poll it without a second serial port.
        case READ_DIAGNOSTICS: {
            noInterrupts();
            leftPID.Encoder  = readEncoder(LEFT);
            rightPID.Encoder = readEncoder(RIGHT);
            interrupts();

            float leftTargetRpm  = (leftPID.TargetTicksPerFrame  / (float)ENC_CPR_LEFT)
                                   * 60.0f * PID_RATE;
            float rightTargetRpm = (rightPID.TargetTicksPerFrame / (float)ENC_CPR_RIGHT)
                                   * 60.0f * PID_RATE;

            float vL     = (leftPID.rpm_filtered  / 60.0f) * WHEEL_CIRC_M;
            float vR     = (rightPID.rpm_filtered / 60.0f) * WHEEL_CIRC_M;
            float linVel = (vL + vR) * 0.5f;
            float angVel = (vR - vL) / WHEEL_SEPARATION_M;

            // Write to ROS serial channel (arduino_comms reads this)
            Serial.printf(
                "D %ld %ld %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %ld %ld %.3f %.3f\r\n",
                leftPID.Encoder, rightPID.Encoder,
                leftTargetRpm, rightTargetRpm,
                leftPID.rpm_filtered, rightPID.rpm_filtered,
                leftPID.last_error, rightPID.last_error,
                leftPID.integral, rightPID.integral,
                leftPID.output, rightPID.output,
                linVel, angVel
            );
            Serial.flush();
            break;
        }

        default:
            Serial.println("Invalid Command");
            break;
    }
    return 0;
}

// ═════════════════════════════════════════════════════════════════
// SETUP
// ═════════════════════════════════════════════════════════════════
void setup()
{
    // ROS bridge channel
    Serial.begin(BAUDRATE);

    // Diagnostic-only stream (TX on GPIO16, no RX needed)
    Serial2.begin(115200, SERIAL_8N1, -1, 16);

    initEncoders();
    initMotorController();
    resetPID();
}

// ═════════════════════════════════════════════════════════════════
// LOOP
// ═════════════════════════════════════════════════════════════════
void loop()
{
    // ── Command parser ────────────────────────────────────────────
    while (Serial.available() > 0) {
        chr = Serial.read();

        if (chr == 13) {                   // CR terminates command
            if (arg == 1) argv1[indx] = '\0';
            else if (arg == 2) argv2[indx] = '\0';
            runCommand();
            resetCommand();
        }
        else if (chr == ' ') {
            if (arg == 0) {
                arg = 1;
            } else if (arg == 1) {
                argv1[indx] = '\0';
                arg  = 2;
                indx = 0;
            }
            // ignore additional spaces
        }
        else {
            if      (arg == 0) { cmd = chr; }
            else if (arg == 1) { argv1[indx++] = chr; }
            else if (arg == 2) { argv2[indx++] = chr; }
        }
    }

    // ── PID tick ──────────────────────────────────────────────────
    if (millis() > nextPID) {
        updatePID();
        nextPID += PID_INTERVAL;
    }

    // ── Auto-stop ─────────────────────────────────────────────────
    if ((millis() - lastMotorCommand) > AUTO_STOP_INTERVAL) {
        setMotorBrakes(200, 200);
        resetPID();
        moving = 0;
    }
}
