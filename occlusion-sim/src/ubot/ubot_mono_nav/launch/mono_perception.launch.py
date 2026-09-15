"""RGB -> depth -> /scan_mono, plus the command watchdog.

The perception half of the LiDAR-less stack, shared by the simulation and the
real robot. Runs wherever the depth network runs: the LAPTOP, not the Pi.

    ros2 launch ubot_mono_nav mono_perception.launch.py \
        image_topic:=/camera/rgb/image_raw camera_info_topic:=/camera/rgb/camera_info

Command chain, once nav2 is also running:
    controller -> velocity_smoother -> collision_monitor -> /cmd_vel_raw
               -> scan_watchdog -> /cmd_vel -> twist_stamper -> wheels
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ubot_mono_nav')
    default_params = os.path.join(pkg, 'config', 'mono_perception.yaml')

    params_file = LaunchConfiguration('params_file')
    image_topic = LaunchConfiguration('image_topic')
    info_topic = LaunchConfiguration('camera_info_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    compressed = LaunchConfiguration('compressed')
    depth_scale_json = LaunchConfiguration('depth_scale_json')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('image_topic', default_value='/camera/rgb/image_raw',
                              description='raw RGB the depth model consumes'),
        DeclareLaunchArgument('camera_info_topic',
                              default_value='/camera/rgb/camera_info'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('compressed', default_value='false',
                              description='true when the robot streams JPEG; a '
                                          'republish node then decodes it here'),
        # Overrides the value in params_file. The calibration belongs to a
        # camera and a mount, so it is given per run rather than committed.
        DeclareLaunchArgument('depth_scale_json', default_value='',
                              description='depth_scale.json from '
                                          'occlusion/eval/depth_affine.py; empty '
                                          'means uncalibrated metres'),
    ]

    # The Pi sends JPEG to keep the link under ~0.5 MB/s. Decoding happens on the
    # laptop, next to the model, so the network never carries raw frames.
    republish = Node(
        package='image_transport', executable='republish',
        name='rgb_republish', output='screen',
        condition=IfCondition(compressed),
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', PythonExpression(["'", image_topic, "' + '/compressed'"])),
            ('out', image_topic),
        ],
    )

    mono_depth = Node(
        package='ubot_mono_nav', executable='mono_depth_node',
        name='mono_depth_node', output='screen',
        parameters=[params_file,
                    {'use_sim_time': use_sim_time,
                     'depth_scale_json': depth_scale_json}],
        remappings=[
            ('~/image', image_topic),
            ('~/camera_info', info_topic),
            ('~/depth', '/mono/depth'),
            ('~/depth/camera_info', '/mono/depth/camera_info'),
        ],
    )

    depth_to_scan = Node(
        package='ubot_mono_nav', executable='depth_to_scan',
        name='depth_to_scan', output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('~/depth', '/mono/depth'),
            ('~/depth/camera_info', '/mono/depth/camera_info'),
            ('~/scan', '/scan_mono'),
        ],
    )

    watchdog = Node(
        package='ubot_mono_nav', executable='scan_watchdog',
        name='scan_watchdog', output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('~/cmd_vel_in', '/cmd_vel_raw'),
            ('~/cmd_vel_out', '/cmd_vel'),
            ('~/scan', '/scan_mono'),
        ],
    )

    return LaunchDescription(args + [republish, mono_depth, depth_to_scan, watchdog])
