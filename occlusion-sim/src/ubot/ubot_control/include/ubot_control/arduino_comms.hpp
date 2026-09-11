/*********************************************************************
 *  arduino_comms.hpp  -  Serial bridge: Pi <-> ESP32
 *
 *  Transport: libserial (direct UART, no micro-ROS)
 *
 *  Commands sent to ESP32  (terminated with \r):
 *    'e'              -> "<lEnc> <rEnc>\r\n"
 *    'm' <L> <R>      -> "OK\r\n"    (ticks-per-frame, signed int)
 *    'u' Kp:Kd:Ki:Min -> "OK\r\n"    (both sides)
 *    'l' Kp:Ki:Kd:Min -> "OK\r\n"    (left side)
 *    'f' Kp:Ki:Kd:Min -> "OK\r\n"    (right side)
 *    'q'              -> "D <14 fields>\r\n"  (full diagnostic snapshot)
 *
 *  Diagnostic fields returned by 'q' (space-separated):
 *    D lEnc rEnc lTgtRpm rTgtRpm lRpm rRpm
 *      lErr rErr lInt rInt lOut rOut linVel angVel
 *********************************************************************/

#ifndef UBOT_CONTROL_ARDUINO_COMMS_HPP_
#define UBOT_CONTROL_ARDUINO_COMMS_HPP_

#include <string>
#include <sstream>
#include <cstdlib>
#include <iostream>

#include <libserial/SerialPort.h>

namespace ubot_control
{

// ── Diagnostic snapshot from the 'q' command ─────────────────────
struct DiagSnapshot
{
  long   l_enc     = 0;      // raw tick count, left
  long   r_enc     = 0;      // raw tick count, right
  float  l_tgt_rpm = 0.0f;   // PID target RPM, left
  float  r_tgt_rpm = 0.0f;
  float  l_rpm     = 0.0f;   // filtered actual RPM, left
  float  r_rpm     = 0.0f;
  float  l_err     = 0.0f;   // PID error (target - actual), left
  float  r_err     = 0.0f;
  float  l_int     = 0.0f;   // integral accumulator, left
  float  r_int     = 0.0f;
  long   l_out     = 0;      // final PWM output, left
  long   r_out     = 0;
  float  lin_vel   = 0.0f;   // computed body linear  velocity (m/s)
  float  ang_vel   = 0.0f;   // computed body angular velocity (rad/s)
  bool   valid     = false;  // false if parse failed
};

// ── Baud rate helper ─────────────────────────────────────────────
inline LibSerial::BaudRate convert_baud_rate(int baud_rate)
{
  switch (baud_rate)
  {
    case 1200:   return LibSerial::BaudRate::BAUD_1200;
    case 1800:   return LibSerial::BaudRate::BAUD_1800;
    case 2400:   return LibSerial::BaudRate::BAUD_2400;
    case 4800:   return LibSerial::BaudRate::BAUD_4800;
    case 9600:   return LibSerial::BaudRate::BAUD_9600;
    case 19200:  return LibSerial::BaudRate::BAUD_19200;
    case 38400:  return LibSerial::BaudRate::BAUD_38400;
    case 57600:  return LibSerial::BaudRate::BAUD_57600;
    case 115200: return LibSerial::BaudRate::BAUD_115200;
    case 230400: return LibSerial::BaudRate::BAUD_230400;
    default:
      std::cerr << "[ArduinoComms] Unsupported baud rate " << baud_rate
                << ", defaulting to 115200" << std::endl;
      return LibSerial::BaudRate::BAUD_115200;
  }
}

// ═════════════════════════════════════════════════════════════════
class ArduinoComms
{
public:
  ArduinoComms() = default;

  // ── Connection lifecycle ──────────────────────────────────────

  void connect(const std::string & serial_device, int baud_rate, int timeout_ms)
  {
    timeout_ms_ = timeout_ms;
    serial_conn_.Open(serial_device);
    serial_conn_.SetBaudRate(convert_baud_rate(baud_rate));
  }

  void disconnect()
  {
    if (serial_conn_.IsOpen())
      serial_conn_.Close();
  }

  bool connected() const
  {
    return serial_conn_.IsOpen();
  }

  // ── Low-level send / receive ──────────────────────────────────

  std::string send_msg(const std::string & msg, bool print = false)
  {
    serial_conn_.FlushIOBuffers();
    serial_conn_.Write(msg);

    std::string response;
    try
    {
      serial_conn_.ReadLine(response, '\n', timeout_ms_);
    }
    catch (const LibSerial::ReadTimeout &)
    {
      std::cerr << "[ArduinoComms] Read timeout on: " << msg << std::endl;
    }

    // Strip trailing \r if present
    if (!response.empty() && response.back() == '\r')
        response.pop_back();
        
    if (print)
      std::cout << "TX: " << msg << "  RX: " << response << std::endl;

    return response;
  }

  // ── Encoder poll  ('e') ───────────────────────────────────────
  // Response: "<lEnc> <rEnc>\r\n"

  void read_encoder_values(int & left, int & right)
  {
    const std::string response = send_msg("e\r");
    if (response.empty()) return;

    try
    {
      const auto pos = response.find(' ');
      if (pos == std::string::npos) return;
      const int new_left  = std::stoi(response.substr(0, pos));
      const int new_right = std::stoi(response.substr(pos + 1));
      left  = new_left;
      right = new_right;
    }
    catch (...)
    {
      std::cerr << "[ArduinoComms] Encoder parse error, skipping frame." << std::endl;
    }
  }

  // ── Motor speed command  ('m') ────────────────────────────────
  // left / right are ticks-per-frame (signed int)

  void set_motor_values(int left, int right)
  {
    std::ostringstream ss;
    ss << "m " << left << " " << right << "\r";
    send_msg(ss.str());
  }

  // ── PID update commands ───────────────────────────────────────
  // 'u' sets both sides identically (legacy: Kp:Kd:Ki:MinPwm)
  // 'l' sets left only            (Kp:Ki:Kd:MinPwm)
  // 'f' sets right only           (Kp:Ki:Kd:MinPwm)

  void set_pid_values(float kp, float kd, float ki, int min_pwm)
  {
    std::ostringstream ss;
    ss << "u " << kp << ":" << kd << ":" << ki << ":" << min_pwm << "\r";
    send_msg(ss.str());
  }

  void set_pid_values_left(float kp, float ki, float kd, int min_pwm)
  {
    std::ostringstream ss;
    ss << "l " << kp << ":" << ki << ":" << kd << ":" << min_pwm << "\r";
    send_msg(ss.str());
  }

  void set_pid_values_right(float kp, float ki, float kd, int min_pwm)
  {
    std::ostringstream ss;
    ss << "f " << kp << ":" << ki << ":" << kd << ":" << min_pwm << "\r";
    send_msg(ss.str());
  }

  // ── Full diagnostic snapshot  ('q') ──────────────────────────
  // Response: "D lEnc rEnc lTgtRpm rTgtRpm lRpm rRpm
  //             lErr rErr lInt rInt lOut rOut linVel angVel\r\n"
  //
  // Call this from read() when you want PlotJuggler-quality data
  // on ROS topics instead of (or alongside) the Serial2 stream.

  DiagSnapshot read_diagnostics()
  {
    DiagSnapshot snap;
    const std::string response = send_msg("q\r");
    if (response.size() < 4 || response[0] != 'D')
      return snap;   // valid stays false

    // Skip the leading "D " token then parse 14 floats/ints
    std::istringstream ss(response.substr(2));

    long  lEnc, rEnc, lOut, rOut;
    float lTgt, rTgt, lRpm, rRpm, lErr, rErr, lInt, rInt, linV, angV;

    // All fields space-delimited; use >> for robustness
    if (!(ss >> lEnc >> rEnc
             >> lTgt >> rTgt
             >> lRpm >> rRpm
             >> lErr >> rErr
             >> lInt >> rInt
             >> lOut >> rOut
             >> linV >> angV))
    {
      std::cerr << "[ArduinoComms] Diagnostics parse error: " << response << std::endl;
      return snap;
    }

    snap.l_enc     = lEnc;  snap.r_enc     = rEnc;
    snap.l_tgt_rpm = lTgt;  snap.r_tgt_rpm = rTgt;
    snap.l_rpm     = lRpm;  snap.r_rpm     = rRpm;
    snap.l_err     = lErr;  snap.r_err     = rErr;
    snap.l_int     = lInt;  snap.r_int     = rInt;
    snap.l_out     = lOut;  snap.r_out     = rOut;
    snap.lin_vel   = linV;  snap.ang_vel   = angV;
    snap.valid     = true;
    return snap;
  }

private:
  LibSerial::SerialPort serial_conn_;
  int timeout_ms_{0};
};

}  // namespace ubot_control

#endif  // UBOT_CONTROL_ARDUINO_COMMS_HPP_
