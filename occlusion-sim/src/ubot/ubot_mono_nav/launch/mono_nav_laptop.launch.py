"""Laptop side of the LiDAR-less robot: depth network, scan, nav2, rviz.

    ros2 launch ubot_mono_nav mono_nav_laptop.launch.py \
        depth_scale_json:=/path/to/depth_scale.json

Pair with `real_robot_mono.launch.py` on the Pi, which streams JPEG RGB.

No slam_toolbox and no map: every frame is `odom`, so goals are relative to
wherever the robot started. Sending a goal in `map` would simply never resolve.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_mono = get_package_share_directory('ubot_mono_nav')
    pkg_description = get_package_share_directory('ubot_description')
    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    depth_scale_json = LaunchConfiguration('depth_scale_json')

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_mono, 'launch', 'mono_perception.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'compressed': LaunchConfiguration('compressed'),
            'depth_scale_json': depth_scale_json,
        }.items())

    # The calibration is passed in rather than baked into the config, because it
    # belongs to a camera and a mount, not to the code. Re-run depth_affine.py
    # after the mount moves.
    # The default nav2 trees call Spin and BackUp, which this stack does not load
    # (both move the robot through space the camera cannot see). Point bt_navigator
    # at the trimmed trees shipped in this package, or the first recovery hangs on an
    # action server that never appears.
    bt_dir = os.path.join(pkg_mono, 'behavior_trees')
    params = RewrittenYaml(
        source_file=os.path.join(pkg_mono, 'config', 'nav2_params_mono.yaml'),
        param_rewrites={
            'default_nav_to_pose_bt_xml':
                os.path.join(bt_dir, 'navigate_to_pose_mono.xml'),
            'default_nav_through_poses_bt_xml':
                os.path.join(bt_dir, 'navigate_through_poses_mono.xml'),
        }, convert_types=True)

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
        launch_arguments={'use_sim_time': 'false', 'params_file': params}.items())

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': False}])

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/rgb/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/rgb/camera_info'),
        DeclareLaunchArgument('compressed', default_value='true',
                              description='the Pi streams JPEG by default'),
        DeclareLaunchArgument('depth_scale_json', default_value='',
                              description='from occlusion/eval/depth_affine.py; '
                                          'without it the metres are uncalibrated'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(pkg_description, 'rviz', 'slam.rviz')),
        perception,
        nav2,
        rviz_node,
    ])
