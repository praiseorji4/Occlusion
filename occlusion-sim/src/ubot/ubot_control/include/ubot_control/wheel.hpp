/*********************************************************************
 *  wheel.hpp  -  Per-wheel kinematics state
 *
 *  Stores encoder ticks, computes cumulative position in radians,
 *  and holds the velocity command from diff_drive_controller.
 *
 *  Only TWO Wheel objects are instantiated (left + right front).
 *  Rear wheels mirror front state via the hardware interface's
 *  export_state_interfaces() — no separate Wheel object needed.
 *********************************************************************/

#ifndef UBOT_HARDWARE_WHEEL_HPP
#define UBOT_HARDWARE_WHEEL_HPP

#include <string>
#include <cmath>

namespace ubot_control
{

class Wheel
{
public:
  std::string name        = "";
  int         enc         = 0;    // current encoder tick count (from ESP32)
  int         last_enc    = 0;    // previous tick count (for delta)
  double      cmd         = 0.0;  // velocity command (rad/s) from controller
  double      pos         = 0.0;  // accumulated position (rad)
  double      vel         = 0.0;  // computed velocity (rad/s)
  double      rads_per_count = 0.0;

  void setup(const std::string & wheel_name, int counts_per_rev)
  {
    name           = wheel_name;
    rads_per_count = (2.0 * M_PI) / static_cast<double>(counts_per_rev);
  }

  // Accumulate position from the delta since last call.
  // Call this once per read() cycle AFTER updating enc.
  void update_position()
  {
    const int diff = enc - last_enc;
    pos     += static_cast<double>(diff) * rads_per_count;
    last_enc = enc;
  }
};

}  // namespace ubot_control

#endif  // UBOT_HARDWARE_WHEEL_HPP
