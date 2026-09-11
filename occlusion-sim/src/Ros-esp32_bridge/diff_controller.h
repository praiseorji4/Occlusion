/*********************************************************************
 *  diff_controller.h  -  RPM-based PID for differential drive
 *********************************************************************/

#ifndef DIFF_CONTROLLER_H
#define DIFF_CONTROLLER_H

// ── Robot geometry ────────────────────────────────────────────────
#define WHEEL_RADIUS_M      0.033f
#define WHEEL_SEPARATION_M  0.264204f
#define WHEEL_CIRC_M        (2.0f * 3.14159265f * WHEEL_RADIUS_M)

// ── Encoder resolution ────────────────────────────────────────────
#define ENC_CPR_LEFT   3956
#define ENC_CPR_RIGHT  3956

// ── Control rate ──────────────────────────────────────────────────
#define PID_RATE    30
#define PID_DT_S    (1.0f / (float)PID_RATE)

// ── 10-second run timer ───────────────────────────────────────────
#define CMD_TIMEOUT_MS  10000UL
static unsigned long cmdStartTime   = 0;
static bool          cmdTimerActive = false;

// ── Per-wheel PID state ───────────────────────────────────────────
typedef struct {
    double TargetTicksPerFrame;
    long   Encoder;
    long   PrevEnc;
    float  rpm_filtered;
    float  rpm_prev;
    float  integral;
    float  last_error;
    float  last_derivative;
    long   output;
    long   last_ticks;       // ticks/frame from last PID call (for debug)
} SetPointInfo;

SetPointInfo leftPID, rightPID;
int moving = 0;

// ── Per-wheel gains ───────────────────────────────────────────────
// Starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)
float leftKp  = 5.0f;  float leftKi  = 9.0f;  float leftKd  = 0.0f;
float rightKp = 5.0f;  float rightKi = 9.0f;  float rightKd = 0.0f;

int leftMinPwm  = 50;
int rightMinPwm = 50;

// ═════════════════════════════════════════════════════════════════
// getRpm()  -  Jimeno low-pass filtered RPM from encoder delta
// ═════════════════════════════════════════════════════════════════
float getRpm(SetPointInfo *p, int cpr)
{
    long delta    = p->Encoder - p->PrevEnc;
    p->last_ticks = delta;                        // store for debug print
    p->PrevEnc    = p->Encoder;

    float raw_rpm = ((float)delta / (float)cpr) * (60.0f / PID_DT_S);

    p->rpm_filtered = 0.854f * p->rpm_filtered
                    + 0.0728f * raw_rpm
                    + 0.0728f * p->rpm_prev;
    p->rpm_prev = raw_rpm;
    return p->rpm_filtered;
}

// ═════════════════════════════════════════════════════════════════
// doPID()  -  RPM PID with anti-windup
// ═════════════════════════════════════════════════════════════════
void doPID(SetPointInfo *p, float target_rpm,
           float kp, float ki, float kd, int min_pwm)
{
    float error    = target_rpm - p->rpm_filtered;

    p->integral   += error * PID_DT_S;
    p->integral    = constrain(p->integral, -50.0f, 50.0f);

    float derivative   = (error - p->last_error) / PID_DT_S;
    p->last_error      = error;
    p->last_derivative = derivative;

    float signal = (kp * error) + (ki * p->integral) + (kd * derivative);

    p->output = (long)constrain(signal, -255.0f, 255.0f);
}

// ═════════════════════════════════════════════════════════════════
// resetPID()  -  clear state on both wheels
// ═════════════════════════════════════════════════════════════════
void resetPID()
{
    leftPID.TargetTicksPerFrame  = 0;
    leftPID.Encoder              = readEncoder(LEFT);
    leftPID.PrevEnc              = leftPID.Encoder;
    leftPID.rpm_filtered         = 0.0f;
    leftPID.rpm_prev             = 0.0f;
    leftPID.integral             = 0.0f;
    leftPID.last_error           = 0.0f;
    leftPID.last_derivative      = 0.0f;
    leftPID.output               = 0;
    leftPID.last_ticks           = 0;

    rightPID.TargetTicksPerFrame = 0;
    rightPID.Encoder             = readEncoder(RIGHT);
    rightPID.PrevEnc             = rightPID.Encoder;
    rightPID.rpm_filtered        = 0.0f;
    rightPID.rpm_prev            = 0.0f;
    rightPID.integral            = 0.0f;
    rightPID.last_error          = 0.0f;
    rightPID.last_derivative     = 0.0f;
    rightPID.output              = 0;
    rightPID.last_ticks          = 0;

    cmdTimerActive = false;
}

// ═════════════════════════════════════════════════════════════════
// emitDiagnostics()  -  14-field diagnostic line to Serial2
// ═════════════════════════════════════════════════════════════════
void emitDiagnostics(float leftTargetRpm, float rightTargetRpm)
{
    float vL     = (leftPID.rpm_filtered  / 60.0f) * WHEEL_CIRC_M;
    float vR     = (rightPID.rpm_filtered / 60.0f) * WHEEL_CIRC_M;
    float linVel = (vL + vR) * 0.5f;
    float angVel = (vR - vL) / WHEEL_SEPARATION_M;

    Serial2.printf(
        "D %ld %ld %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %ld %ld %.3f %.3f\r\n",
        leftPID.Encoder,       rightPID.Encoder,
        leftTargetRpm,         rightTargetRpm,
        leftPID.rpm_filtered,  rightPID.rpm_filtered,
        leftPID.last_error,    rightPID.last_error,
        leftPID.integral,      rightPID.integral,
        leftPID.output,        rightPID.output,
        linVel, angVel
    );
}

// ═════════════════════════════════════════════════════════════════
// updatePID()  -  called at PID_RATE Hz from main loop
// ═════════════════════════════════════════════════════════════════
void updatePID()
{
    noInterrupts();
    leftPID.Encoder  = readEncoder(LEFT);
    rightPID.Encoder = readEncoder(RIGHT);
    interrupts();

    float leftTargetRpm  = (leftPID.TargetTicksPerFrame  / (float)ENC_CPR_LEFT)
                           * 60.0f * PID_RATE;
    float rightTargetRpm = (rightPID.TargetTicksPerFrame / (float)ENC_CPR_RIGHT)
                           * 60.0f * PID_RATE;

    // getRpm() updates last_ticks and PrevEnc internally
    getRpm(&leftPID,  ENC_CPR_LEFT);
    getRpm(&rightPID, ENC_CPR_RIGHT);

    // ── 10-second auto-stop ───────────────────────────────────────
    if (cmdTimerActive && (millis() - cmdStartTime >= CMD_TIMEOUT_MS)) {
        Serial2.println("--- 10s timeout: stopping motors ---");
        setMotorBrakes(200, 200);
        resetPID();
        moving = 0;
        emitDiagnostics(0.0f, 0.0f);
        return;
    }

    if (!moving) {
        if (leftPID.rpm_filtered != 0.0f || rightPID.rpm_filtered != 0.0f)
            resetPID();
        emitDiagnostics(0.0f, 0.0f);
        return;
    }

    if (leftTargetRpm == 0.0f && rightTargetRpm == 0.0f) {
        setMotorBrakes(200, 200);
        resetPID();
        moving = 0;
        emitDiagnostics(0.0f, 0.0f);
        return;
    }

    // ── Run PID ───────────────────────────────────────────────────
    doPID(&leftPID,  leftTargetRpm,  leftKp,  leftKi,  leftKd,  leftMinPwm);
    doPID(&rightPID, rightTargetRpm, rightKp, rightKi, rightKd, rightMinPwm);

    // ── Apply feedforward dead-zone offset ────────────────────────
    long l_cmd = leftPID.output;
    long r_cmd = rightPID.output;
    if (l_cmd != 0) { int d = (l_cmd > 0) ? 1 : -1; l_cmd = constrain(d * leftMinPwm  + l_cmd, -255L, 255L); }
    if (r_cmd != 0) { int d = (r_cmd > 0) ? 1 : -1; r_cmd = constrain(d * rightMinPwm + r_cmd, -255L, 255L); }
    setMotorSpeeds(l_cmd, r_cmd);

    // ── Debug print (~2x per second) ─────────────────────────────
    static uint8_t dbg_tick = 0;
    if (++dbg_tick >= 15) {
        dbg_tick = 0;
        Serial2.printf("TGT L:%.1f R:%.1f | ACT L:%.1f R:%.1f | TICKS L:%ld R:%ld | PWM L:%ld R:%ld\n",
            leftTargetRpm,         rightTargetRpm,
            leftPID.rpm_filtered,  rightPID.rpm_filtered,
            leftPID.last_ticks,    rightPID.last_ticks,
            l_cmd, r_cmd);
    }

    emitDiagnostics(leftTargetRpm, rightTargetRpm);
}

#endif  // DIFF_CONTROLLER_H