"""Robot side, LiDAR-less: wheels, IMU, and the OAK-D's RGB stream. No LiDAR.

    ros2 launch ubot_mono_nav real_robot_mono.launch.py

A copy of ubot_bringup/launch/real_robot.launch.py with `ldlidar_node` removed
and `oak_rgb_node` added. The original is untouched, so the LiDAR stack still
runs for comparison -- unplug nothing, just launch the other file.

Everything heavy (the depth network, nav2, rviz) runs on the laptop; see
mono_nav_laptop.launch.py. This side only streams JPEG frames and drives wheels.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_description = get_package_share_directory('ubot_description')
    pkg_bringup = get_package_share_directory('ubot_bringup')
    pkg_mono = get_package_share_directory('ubot_mono_nav')

    # camera_pitch: CORRECTION on top of the URDF's CAD tilt, positive = UP.
    # The CamCase already tilts the OAK-D 4.09 deg DOWN in the URDF and that was
    # confirmed against the robot (lens centre 0.152 m off the floor), so the
    # correction is 0. Only set this if the mount is physically bent.
    # (Was 0.059 when the URDF still said 0 pitch; leaving it there now would
    # cancel most of the real tilt.)
    # which puts every camera obstacle at the wrong height, so it is passed in
    # here from what calibrate_pose.py measured.
    robot_description_config = ParameterValue(
        Command(['xacro ',
                 os.path.join(pkg_description, 'urdf', 'body', 'ubot_robot.urdf.xacro'),
                 ' use_gazebo:=false',
                 ' camera_pitch:=', LaunchConfiguration('camera_pitch')]),
        value_type=str)

    node_robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config, 'use_sim_time': False}])

    controller_manager = Node(
        package='controller_manager', executable='ros2_control_node',
        parameters=[{'robot_description': robot_description_config},
                    os.path.join(pkg_bringup, 'config', 'ubot_controllers.yaml'),
                    {'use_sim_time': False}],
        output='screen')

    joint_state_broadcaster_spawner = TimerAction(
        period=5.0,
        actions=[Node(package='controller_manager', executable='spawner',
                      arguments=['joint_state_broadcaster'],
                      parameters=[{'use_sim_time': False}])])

    diff_drive_controller_spawner = TimerAction(
        period=6.0,
        actions=[Node(package='controller_manager', executable='spawner',
                      arguments=['diff_drive_controller'],
                      parameters=[{'use_sim_time': False}])])

    twist_stamper = Node(
        package='twist_stamper', executable='twist_stamper',
        parameters=[{'use_sim_time': False, 'frame_id': 'base_footprint'}],
        remappings=[('/cmd_vel_in', '/cmd_vel'),
                    ('/cmd_vel_out', '/diff_drive_controller/cmd_vel')])

    oak_rgb = Node(
        package='ubot_mono_nav', executable='oak_rgb_node',
        name='oak_rgb_node', output='screen',
        parameters=[os.path.join(pkg_mono, 'config', 'mono_perception.yaml')],
        remappings=[('~/image_raw', '/camera/rgb/image_raw'),
                    ('~/image_raw/compressed', '/camera/rgb/image_raw/compressed'),
                    ('~/camera_info', '/camera/rgb/camera_info')])

    return LaunchDescription([
        DeclareLaunchArgument('camera_pitch', default_value='0.0',
                              description='radians, positive = up: a CORRECTION on top of '
                                          'the 4.09 deg down tilt the URDF already has. '
                                          '0 unless the mount is physically bent'),
        node_robot_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        diff_drive_controller_spawner,
        twist_stamper,
        oak_rgb,
    ])
