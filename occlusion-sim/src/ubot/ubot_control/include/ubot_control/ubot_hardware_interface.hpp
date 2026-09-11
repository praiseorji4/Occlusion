/*********************************************************************
 *  ubot_hardware_interface.hpp  -  ros2_control hardware interface
 *
 *  Transport: direct serial via arduino_comms.hpp (libserial)
 *
 *  Drive config: 4WD — only front wheels carry encoders.
 *  Rear wheels are mechanically coupled to front and their state
 *  is mirrored in export_state_interfaces() — no separate encoder.
 *
 *  Joints exposed to ros2_control:
 *    Commands (velocity):  front_left, front_right
 *    States  (pos + vel):  front_left, front_right,
 *                           rear_left,  rear_right  (mirrors of front)
 *
 *  Optional diagnostics:
 *    If diag_publish_rate > 0 (set in URDF params), read() polls
 *    the ESP32 'q' command and publishes a Float32MultiArray on
 *    /ubot/diagnostics for PlotJuggler.
 *********************************************************************/

#ifndef UBOT_HARDWARE_INTERFACE_HPP_
#define UBOT_HARDWARE_INTERFACE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "std_msgs/msg/float32_multi_array.hpp"

#include "ubot_control/arduino_comms.hpp"
#include "ubot_control/wheel.hpp"

namespace ubot_control
{

class UbotHardware : public hardware_interface::SystemInterface
{
  // ── URDF / xacro parameter block ────────────────────────────
  struct Config
  {
    // Joint names (must match URDF)
    std::string left_wheel_name  = "";
    std::string right_wheel_name = "";

    // Encoder
    int    enc_counts_per_rev_left  = 3956;
    int    enc_counts_per_rev_right = 3956;

    // Geometry (must match diff_controller.h on the ESP32)
    double wheel_separation = 0.264204;
    double wheel_radius     = 0.033;

    // Serial
    std::string serial_device = "/dev/ttyUSB0";
    int    baud_rate          = 115200;
    int    serial_timeout_ms  = 1000;

    // Control
    double loop_rate          = 30.0;

    // Diagnostics  (0 = disabled, >0 = publish every N read() cycles)
    int    diag_publish_rate  = 0;
  };

public:
  RCLCPP_SHARED_PTR_DEFINITIONS(UbotHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  std::vector<hardware_interface::StateInterface>   export_state_interfaces()   override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  Config       cfg_;
  Wheel        wheel_l_;
  Wheel        wheel_r_;
  ArduinoComms comms_;

  // Diagnostics publisher (optional — enabled when diag_publish_rate > 0)
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr diag_pub_;
  int diag_cycle_count_ = 0;   // incremented each read(), publish when reaches rate
};

}  // namespace ubot_control

#endif  // UBOT_HARDWARE_INTERFACE_HPP_
