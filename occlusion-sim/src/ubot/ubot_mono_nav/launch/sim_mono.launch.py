"""Gazebo + the ubot + nav2 driven by the CAMERA ONLY. No SLAM, no map.

    ros2 launch ubot_mono_nav sim_mono.launch.py

This mirrors ubot_bringup/launch/sim.launch.py, with three differences:

  * nav2 uses config/nav2_params_mono.yaml - obstacles from /scan_mono, and
    every frame in `odom` rather than `map`.
  * slam_toolbox is NOT launched. Nothing publishes map->odom, which is exactly
    the point: the robot navigates relative to where it started.
  * the perception chain (mono_perception.launch.py) turns /camera/rgb/image_raw
    into /scan_mono.

The Gazebo LiDAR is still bridged on /scan, unused by navigation. That is
deliberate: it is the reference to grade the camera scan against, and the A/B
comparison needs both from the same run.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  TextSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_description = get_package_share_directory('ubot_description')
    pkg_bringup = get_package_share_directory('ubot_bringup')
    pkg_mono = get_package_share_directory('ubot_mono_nav')
    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    robot_description_config = ParameterValue(
        Command(['xacro ',
                 os.path.join(pkg_description, 'urdf', 'body', 'ubot_robot.urdf.xacro'),
                 ' use_gazebo:=true']),
        value_type=str)

    node_robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config, 'use_sim_time': True}])

    # Same world handling as ubot_bringup/launch/sim.launch.py: 'world' names both the
    # .sdf in ubot_bringup/worlds and the <world name> inside it, because the spawner
    # addresses the world by name. Defaults to the indoor apartment, spawned in the
    # right-hand hall facing down the corridor.
    world = LaunchConfiguration('world')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    world_file = PathJoinSubstitution([
        pkg_bringup, 'worlds', [world, TextSubstitution(text='.sdf')],
    ])

    # model://apartment lives in ubot_bringup/models; Gazebo needs the CONTAINING dir.
    gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.join(pkg_bringup, 'models'))

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('ros_gz_sim'), 'launch',
                          'gz_sim.launch.py')]),
        launch_arguments={'gz_args': [TextSubstitution(text='-r -v 4 '), world_file]}.items())

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        remappings=[
            ('/camera/image', '/camera/rgb/image_raw'),
            ('/camera/depth_image', '/camera/depth/image_raw'),
            ('/camera/camera_info', '/camera/rgb/camera_info'),
            ('/imu', '/imu/data'),
        ],
        output='screen', parameters=[{'use_sim_time': True}])

    spawn_entity = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description', '-name', 'ubot',
                   '-world', world,
                   '-x', spawn_x, '-y', spawn_y, '-z', spawn_z, '-Y', spawn_yaw])

    load_joint_state_broadcaster = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--param-file',
                   os.path.join(pkg_bringup, 'config', 'sim_ubot_controllers.yaml')],
        parameters=[{'use_sim_time': True}])

    load_diff_drive_controller = Node(
        package='controller_manager', executable='spawner',
        arguments=['diff_drive_controller', '--param-file',
                   os.path.join(pkg_bringup, 'config', 'sim_ubot_controllers.yaml')],
        parameters=[{'use_sim_time': True}])

    node_twist_stamper = Node(
        package='twist_stamper', executable='twist_stamper',
        parameters=[{'use_sim_time': True, 'frame_id': 'base_footprint'}],
        remappings=[('/cmd_vel_in', '/cmd_vel'),
                    ('/cmd_vel_out', '/diff_drive_controller/cmd_vel')])

    ekf_node = Node(
        package='robot_localization', executable='ekf_node', name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(pkg_bringup, 'config', 'sim_ekf.yaml')])

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_mono, 'launch', 'mono_perception.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'image_topic': '/camera/rgb/image_raw',
            'camera_info_topic': '/camera/rgb/camera_info',
            'compressed': 'false',
        }.items())

    # Same params file as the real robot, with the sim's odometry topic swapped
    # in. The sim runs an EKF and publishes /odometry/filtered; the robot does
    # not, and uses the controller's own odometry.
    # The default nav2 trees call Spin and BackUp, which this stack does not load
    # (both move the robot through space the camera cannot see). Point bt_navigator
    # at the trimmed trees shipped in this package, or the first recovery hangs on an
    # action server that never appears.
    bt_dir = os.path.join(pkg_mono, 'behavior_trees')
    mono_params = RewrittenYaml(
        source_file=os.path.join(pkg_mono, 'config', 'nav2_params_mono.yaml'),
        param_rewrites={
            'odom_topic': '/odometry/filtered',
            'default_nav_to_pose_bt_xml':
                os.path.join(bt_dir, 'navigate_to_pose_mono.xml'),
            'default_nav_through_poses_bt_xml':
                os.path.join(bt_dir, 'navigate_through_poses_mono.xml'),
        },
        convert_types=True)

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
        launch_arguments={'use_sim_time': 'true', 'params_file': mono_params}.items())

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='apartment',
                              description='world .sdf in ubot_bringup/worlds AND its <world name>'),
        DeclareLaunchArgument('spawn_x', default_value='5.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.1'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.5708'),
        gz_resource_path,
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(pkg_description, 'rviz', 'slam.rviz'),
            description='add /scan_mono and the costmaps to see the camera working'),
        node_robot_state_publisher,
        gazebo,
        bridge,
        spawn_entity,
        node_twist_stamper,
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[load_joint_state_broadcaster, load_diff_drive_controller])),
        ekf_node,
        perception,
        nav2,
        rviz_node,
    ])
