// Copyright 2024 ubot Development Team

#include "ubot_control/ubot_hardware_interface.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "hardware_interface/lexical_casts.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ubot_control
{

// ═════════════════════════════════════════════════════════════════
// on_init  -  parse URDF params, validate joint configuration
// ═════════════════════════════════════════════════════════════════
hardware_interface::CallbackReturn UbotHardware::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (hardware_interface::SystemInterface::on_init(params) !=
      hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // ── Joint names ───────────────────────────────────────────────
  cfg_.left_wheel_name  = info_.hardware_parameters.at("left_wheel_name");
  cfg_.right_wheel_name = info_.hardware_parameters.at("right_wheel_name");

  // ── Encoder counts ───────────────────────────────────────────
  cfg_.enc_counts_per_rev_left  =
    std::stoi(info_.hardware_parameters.at("enc_counts_per_rev_left"));
  cfg_.enc_counts_per_rev_right =
    std::stoi(info_.hardware_parameters.at("enc_counts_per_rev_right"));

  // ── Geometry ─────────────────────────────────────────────────
  cfg_.wheel_separation =
    hardware_interface::stod(info_.hardware_parameters.at("wheel_separation"));
  cfg_.wheel_radius =
    hardware_interface::stod(info_.hardware_parameters.at("wheel_radius"));

  // ── Serial ────────────────────────────────────────────────────
  cfg_.serial_device      = info_.hardware_parameters.at("serial_device");
  cfg_.baud_rate          = std::stoi(info_.hardware_parameters.at("baud_rate"));
  cfg_.serial_timeout_ms  = std::stoi(info_.hardware_parameters.at("serial_timeout_ms"));

  // ── Control / diagnostics ─────────────────────────────────────
  cfg_.loop_rate         =
    hardware_interface::stod(info_.hardware_parameters.at("loop_rate"));
  cfg_.diag_publish_rate =
    info_.hardware_parameters.count("diag_publish_rate")
      ? std::stoi(info_.hardware_parameters.at("diag_publish_rate"))
      : 0;

  // ── Wheel objects ─────────────────────────────────────────────
  wheel_l_.setup(cfg_.left_wheel_name,  cfg_.enc_counts_per_rev_left);
  wheel_r_.setup(cfg_.right_wheel_name, cfg_.enc_counts_per_rev_right);

  // ── Validate joint count: must be 4 (2 front + 2 rear) ───────
  if (info_.joints.size() != 4)
  {
    RCLCPP_FATAL(get_logger(),
      "[UbotHardware] Expected 4 joints (2 front + 2 rear mirrors), found %zu",
      info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // ── Validate interface specs per joint ───────────────────────
  for (const auto & joint : info_.joints)
  {
    // Command interfaces: 0 or 1 allowed, must be velocity if present
    if (!joint.command_interfaces.empty())
    {
      if (joint.command_interfaces.size() != 1 ||
          joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
      {
        RCLCPP_FATAL(get_logger(),
          "[UbotHardware] Joint '%s': expected 0 or 1 velocity command interface.",
          joint.name.c_str());
        return hardware_interface::CallbackReturn::ERROR;
      }
    }

    // State interfaces: exactly 2 (position + velocity)
    if (joint.state_interfaces.size() != 2)
    {
      RCLCPP_FATAL(get_logger(),
        "[UbotHardware] Joint '%s' has %zu state interfaces, expected 2.",
        joint.name.c_str(), joint.state_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

// ═════════════════════════════════════════════════════════════════
// export_state_interfaces
//   Front wheels: backed by real Wheel objects
//   Rear  wheels: point to the same doubles as front (mirror)
// ═════════════════════════════════════════════════════════════════
std::vector<hardware_interface::StateInterface>
UbotHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Front left
  state_interfaces.emplace_back(
    "front_left_wheel_joint", hardware_interface::HW_IF_POSITION, &wheel_l_.pos);
  state_interfaces.emplace_back(
    "front_left_wheel_joint", hardware_interface::HW_IF_VELOCITY, &wheel_l_.vel);

  // Front right
  state_interfaces.emplace_back(
    "front_right_wheel_joint", hardware_interface::HW_IF_POSITION, &wheel_r_.pos);
  state_interfaces.emplace_back(
    "front_right_wheel_joint", hardware_interface::HW_IF_VELOCITY, &wheel_r_.vel);

  // Rear left — mirrors front left (same motor / belt drive)
  state_interfaces.emplace_back(
    "rear_left_wheel_joint", hardware_interface::HW_IF_POSITION, &wheel_l_.pos);
  state_interfaces.emplace_back(
    "rear_left_wheel_joint", hardware_interface::HW_IF_VELOCITY, &wheel_l_.vel);

  // Rear right — mirrors front right
  state_interfaces.emplace_back(
    "rear_right_wheel_joint", hardware_interface::HW_IF_POSITION, &wheel_r_.pos);
  state_interfaces.emplace_back(
    "rear_right_wheel_joint", hardware_interface::HW_IF_VELOCITY, &wheel_r_.vel);

  return state_interfaces;
}

// ═════════════════════════════════════════════════════════════════
// export_command_interfaces
//   Only front wheels accept velocity commands.
//   Rear wheels have no command interface — they follow physically.
// ═════════════════════════════════════════════════════════════════
std::vector<hardware_interface::CommandInterface>
UbotHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  command_interfaces.emplace_back(
    "front_left_wheel_joint", hardware_interface::HW_IF_VELOCITY, &wheel_l_.cmd);
  command_interfaces.emplace_back(
    "front_right_wheel_joint", hardware_interface::HW_IF_VELOCITY, &wheel_r_.cmd);

  return command_interfaces;
}

// ═════════════════════════════════════════════════════════════════
// on_configure  -  open serial port + optional diagnostics pub
// ═════════════════════════════════════════════════════════════════
hardware_interface::CallbackReturn UbotHardware::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[UbotHardware] Configuring...");

  // Open serial connection to ESP32
  comms_.connect(cfg_.serial_device, cfg_.baud_rate, cfg_.serial_timeout_ms);
  if (!comms_.connected())
  {
    RCLCPP_FATAL(get_logger(),
      "[UbotHardware] Failed to open serial port '%s'",
      cfg_.serial_device.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }
  RCLCPP_INFO(get_logger(),
    "[UbotHardware] Serial open: %s @ %d baud",
    cfg_.serial_device.c_str(), cfg_.baud_rate);

  // Optional diagnostics publisher
  if (cfg_.diag_publish_rate > 0)
  {
    diag_pub_ = get_node()->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/ubot/diagnostics", rclcpp::QoS(10));
    RCLCPP_INFO(get_logger(),
      "[UbotHardware] Diagnostics publisher enabled (every %d read cycles)",
      cfg_.diag_publish_rate);
  }

  RCLCPP_INFO(get_logger(), "[UbotHardware] Configuration complete.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

// ═════════════════════════════════════════════════════════════════
// on_cleanup  -  close serial port
// ═════════════════════════════════════════════════════════════════
hardware_interface::CallbackReturn UbotHardware::on_cleanup(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[UbotHardware] Cleaning up...");
  comms_.disconnect();
  diag_pub_ = nullptr;
  RCLCPP_INFO(get_logger(), "[UbotHardware] Cleaned up.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

// ═════════════════════════════════════════════════════════════════
// on_activate  -  zero wheel state, seed encoder baseline
// ═════════════════════════════════════════════════════════════════
hardware_interface::CallbackReturn UbotHardware::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[UbotHardware] Activating...");

  // Zero kinematic state
  wheel_l_.enc  = 0; wheel_r_.enc  = 0;
  wheel_l_.pos  = 0.0; wheel_r_.pos  = 0.0;
  wheel_l_.vel  = 0.0; wheel_r_.vel  = 0.0;
  wheel_l_.cmd  = 0.0; wheel_r_.cmd  = 0.0;

  // Read current tick count from ESP32 and use it as the baseline so
  // the first read() delta is zero rather than a huge jump.
  comms_.read_encoder_values(wheel_l_.enc, wheel_r_.enc);
  wheel_l_.last_enc = wheel_l_.enc;
  wheel_r_.last_enc = wheel_r_.enc;

  // Stop robot on activation
  comms_.set_motor_values(0, 0);

  RCLCPP_INFO(get_logger(), "[UbotHardware] Activated (L_enc=%d, R_enc=%d).",
    wheel_l_.enc, wheel_r_.enc);
  return hardware_interface::CallbackReturn::SUCCESS;
}

// ═════════════════════════════════════════════════════════════════
// on_deactivate  -  stop robot
// ═════════════════════════════════════════════════════════════════
hardware_interface::CallbackReturn UbotHardware::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[UbotHardware] Deactivating...");
  comms_.set_motor_values(0, 0);
  RCLCPP_INFO(get_logger(), "[UbotHardware] Deactivated.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

// ═════════════════════════════════════════════════════════════════
// read  -  poll encoders, compute wheel kinematics
//           optionally poll diagnostics and publish
// ═════════════════════════════════════════════════════════════════
hardware_interface::return_type UbotHardware::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  // ── Encoder update ────────────────────────────────────────────
  comms_.read_encoder_values(wheel_l_.enc, wheel_r_.enc);

  const double dt = period.seconds();

  const double prev_pos_l = wheel_l_.pos;
  wheel_l_.update_position();
  wheel_l_.vel = (dt > 0.0) ? (wheel_l_.pos - prev_pos_l) / dt : 0.0;

  const double prev_pos_r = wheel_r_.pos;
  wheel_r_.update_position();
  wheel_r_.vel = (dt > 0.0) ? (wheel_r_.pos - prev_pos_r) / dt : 0.0;

  // ── Optional diagnostics poll ─────────────────────────────────
  // Polls the ESP32 'q' command and publishes a Float32MultiArray
  // that PlotJuggler can subscribe to as /ubot/diagnostics.
  // Field order matches DiagSnapshot / diff_controller.h comments:
  //   [0]  l_enc      [1]  r_enc
  //   [2]  l_tgt_rpm  [3]  r_tgt_rpm
  //   [4]  l_rpm      [5]  r_rpm
  //   [6]  l_err      [7]  r_err
  //   [8]  l_int      [9]  r_int
  //   [10] l_out      [11] r_out
  //   [12] lin_vel    [13] ang_vel
  if (diag_pub_ && cfg_.diag_publish_rate > 0)
  {
    ++diag_cycle_count_;
    if (diag_cycle_count_ >= cfg_.diag_publish_rate)
    {
      diag_cycle_count_ = 0;
      const DiagSnapshot snap = comms_.read_diagnostics();
      if (snap.valid)
      {
        std_msgs::msg::Float32MultiArray msg;
        msg.data = {
          static_cast<float>(snap.l_enc),
          static_cast<float>(snap.r_enc),
          snap.l_tgt_rpm,  snap.r_tgt_rpm,
          snap.l_rpm,      snap.r_rpm,
          snap.l_err,      snap.r_err,
          snap.l_int,      snap.r_int,
          static_cast<float>(snap.l_out),
          static_cast<float>(snap.r_out),
          snap.lin_vel,    snap.ang_vel
        };
        diag_pub_->publish(msg);
      }
    }
  }

  return hardware_interface::return_type::OK;
}

// ═════════════════════════════════════════════════════════════════
// write  -  convert rad/s commands to ticks-per-frame, send 'm'
// ═════════════════════════════════════════════════════════════════
hardware_interface::return_type UbotHardware::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // wheel.cmd is in rad/s (from diff_drive_controller)
  // Convert: rad/s -> rev/s -> rev/frame -> ticks/frame
  const double rps_l = wheel_l_.cmd / (2.0 * M_PI);
  const double rps_r = wheel_r_.cmd / (2.0 * M_PI);

  const int tpf_l =
    static_cast<int>(rps_l * cfg_.enc_counts_per_rev_left  / cfg_.loop_rate);
  const int tpf_r =
    static_cast<int>(rps_r * cfg_.enc_counts_per_rev_right / cfg_.loop_rate);

  comms_.set_motor_values(tpf_l, tpf_r);

  return hardware_interface::return_type::OK;
}

}  // namespace ubot_control

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  ubot_control::UbotHardware, hardware_interface::SystemInterface)
