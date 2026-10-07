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
from launch.conditions import IfCondition
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression, TextSubstitution)
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
        launch_arguments={'gz_args': [
            TextSubstitution(text='-r -v 4 '),
            # Same two switches as ubot_bringup/launch/sim.launch.py, and for the
            # same reason: a campaign runs this with no display.
            #   headless           -> '-s', the server only, no GUI process.
            #   headless_rendering -> ogre2 renders offscreen through EGL.
            # Without the second, the server still tries to open a GLX display
            # for the camera sensor, fails, and SEGFAULTS - the camera topics
            # exist and never publish. Without either, a container gets a GUI
            # Gazebo and an rviz that cannot start.
            PythonExpression(["'-s ' if '", LaunchConfiguration('headless'),
                              "'.lower() in ('true','1') else ''"]),
            PythonExpression(["'--headless-rendering ' if '",
                              LaunchConfiguration('headless_rendering'),
                              "'.lower() in ('true','1') else ''"]),
            world_file]}.items())

    # The <world name> INSIDE the sdf, which is NOT always the file name:
    # parking_garage_occlusion.sdf declares <world name='parking_garage'> and
    # sonoma_occlusion.sdf declares 'sensors'. Both the spawner and every scoped
    # gz topic address the world by this name, so passing the file name works
    # only for apartment.sdf and fails silently elsewhere - the spawner waits for
    # a world that does not exist. sim.launch.py already separates the two.
    world_name = LaunchConfiguration('world_name')

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

            # GROUND-TRUTH POSE from the model's PosePublisher, not from
            # SceneBroadcaster's dynamic_pose/info - that one converts to
            # transforms with an empty child_frame_id. See the longer note in
            # ubot_bringup/launch/sim.launch.py. Remapped to /gt/tf.
            '/model/ubot/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        remappings=[
            ('/camera/image', '/camera/rgb/image_raw'),
            ('/camera/depth_image', '/camera/depth/image_raw'),
            ('/camera/camera_info', '/camera/rgb/camera_info'),
            ('/imu', '/imu/data'),
            ('/model/ubot/pose', '/gt/tf'),
        ],
        output='screen', parameters=[{'use_sim_time': True}])

    spawn_entity = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description', '-name', 'ubot',
                   '-world', world_name,
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
            'depth_scale_json': LaunchConfiguration('depth_scale_json'),
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
            # USE_SIM_TIME MUST BE REWRITTEN HERE. nav2_params_mono.yaml says
            # False on every node, which is right for the REAL robot - this file
            # serves both. In sim, leaving it False gives nav2 nodes a WALL
            # clock while every sensor topic carries SIM stamps, and the
            # collision monitor then computes the scan's age as the difference
            # between the two epochs:
            #   [mono_scan]: Latest source and current collision monitor node
            #   timestamps differ on 1789585390.29 seconds. Ignoring the source.
            #   Robot to stop due to invalid source.
            # It stops the robot permanently - 102 s of silence on /cmd_vel and
            # 2.7 cm of travel - while every other part of the stack looks
            # healthy. The LiDAR arm never hit this because its params file is
            # sim-only and already says True.
            'use_sim_time': 'true',
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

    # use_rviz:=false is REQUIRED for a campaign: rviz is a GUI for a human
    # watching one run, it competes with Gazebo and nav2 for the same cores, and
    # on a headless host it dies with SIGABRT partway through an episode. It was
    # unconditional here while sim.launch.py already had the switch, so the
    # camera arm could not be run headless at all.
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='apartment',
                              description='world FILE in ubot_bringup/worlds, without .sdf'),
        # Separate from the file name on purpose: the spawner and every scoped gz
        # topic (ground truth, contacts) address the world by the <world name>
        # INSIDE the sdf, and the two differ - parking_garage_occlusion.sdf
        # declares 'parking_garage', sonoma_occlusion.sdf declares 'sensors'.
        # Passing the file name spawns into a world that does not exist, and the
        # failure is a silent wait. sim.launch.py makes the same distinction.
        DeclareLaunchArgument('world_name', default_value='apartment',
                              description="<world name> inside that sdf "
                                          "(apartment, parking_garage, sensors)"),
        # THE DEPTH CALIBRATION IS NOT OPTIONAL IN SIM.
        # Uncalibrated, the model reads about a*z + b with a = 1.31 and
        # b = +1.04 m, so a surface 1 m away is reported at 2.3 m. Every
        # point depth_to_scan projects then lands at the wrong HEIGHT,
        # falls outside its 0.05-0.60 m band, and /scan_mono comes out with
        # no finite range at all - readiness refuses the episode with
        # "0 scans with a finite range" while the chain runs and looks
        # healthy. Regenerate with ubot_eval/scripts/calibrate_sim_depth.py
        # (or docker/run.sh calibrate) if the camera pose or model changes.
        DeclareLaunchArgument(
            'depth_scale_json',
            default_value=os.path.join(pkg_mono, 'config', 'depth_scale_sim.json'),
            description='depth calibration; "" means uncalibrated, which cannot navigate'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='server only, no Gazebo GUI window'),
        DeclareLaunchArgument('headless_rendering', default_value='false',
                              description='render via EGL; required where there is no display'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='false for campaigns and headless hosts'),
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
