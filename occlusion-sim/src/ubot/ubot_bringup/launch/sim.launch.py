import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression, TextSubstitution)
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_description = get_package_share_directory('ubot_description')
    
    # 1. Process URDF with use_gazebo:=true
    robot_description_config = ParameterValue(
        Command([
            'xacro ',
            os.path.join(pkg_description, 'urdf', 'body', 'ubot_robot.urdf.xacro'),
            ' use_gazebo:=true',
            ' sensor_profile:=', LaunchConfiguration('sensor_profile')
        ]),
        value_type=str
    )
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config, 
            'use_sim_time': True
        }]
    )
    
    # 2. Gazebo world.
    #    'world' picks the .sdf file in ubot_bringup/worlds (without the extension);
    #    'world_name' is the <world name='...'> declared INSIDE that file, which the
    #    spawner needs because it addresses the world by name, not by filename.
    #    Indoor apartment world instead of the raceway:
    #      ros2 launch ubot_bringup sim.launch.py world:=apartment world_name:=apartment \
    #          spawn_x:=5.0 spawn_y:=0.0 spawn_z:=0.1 spawn_yaw:=1.5708
    world = LaunchConfiguration('world')
    world_name = LaunchConfiguration('world_name')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    declare_args = [
        DeclareLaunchArgument('world', default_value='sonoma_occlusion',
                              description="World FILE in ubot_bringup/worlds, without .sdf"),
        # The spawner addresses the world by the <world name='...'> INSIDE the sdf, which is
        # not always the filename: sonoma_occlusion.sdf declares <world name='sensors'>.
        DeclareLaunchArgument('world_name', default_value='sensors',
                              description="<world name> inside that sdf (spawner addresses it by name)"),
        # Default spawn: on the Sonoma pit lane beside the occlusion scene (cars/pedestrians
        # sit around x 245..275, y -132..-115, ground z ~3.3).
        #
        # THE SPAWN POSE BELONGS TO THE WORLD. Overriding `world` WITHOUT also
        # overriding these four puts the robot at Sonoma's coordinates inside
        # whatever world you asked for. In `apartment` that is ~300 m outside
        # the building, in empty space, where the LiDAR honestly returns .inf
        # for every beam and the camera sees nothing. There is no error: the
        # robot spawns, falls 3.4 m to the floor and sits there, and the only
        # symptom is that readiness refuses on the scan check. Cost a long
        # debugging session once already. For the apartment use:
        #   spawn_x:=5.0 spawn_y:=0.0 spawn_z:=0.1 spawn_yaw:=1.5708
        # ubot_eval passes these per course; see ubot_eval/config/courses.yaml.
        DeclareLaunchArgument('spawn_x', default_value='272.0'),
        DeclareLaunchArgument('spawn_y', default_value='-136.0'),
        DeclareLaunchArgument('spawn_z', default_value='3.45'),
        DeclareLaunchArgument('spawn_yaw', default_value='2.40'),
        # The raceway world is expensive. Turning these off leaves just Gazebo + the robot,
        # which is usually what you want for occlusion capture, and makes startup reliable:
        #   ros2 launch ubot_bringup sim.launch.py use_slam:=false use_nav2:=false use_rviz:=false
        DeclareLaunchArgument('use_slam', default_value='true'),
        # Empty means "pick from use_slam" (see the note by nav2_params_path).
        # Set it only to override both, e.g. for a tuning sweep.
        DeclareLaunchArgument('params_file', default_value='',
                              description='nav2 params override; empty = choose by use_slam'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        # headless:=true runs the Gazebo server only (no GUI window). Sensors still work.
        # Much lighter, and the only way to run on a machine without a display.
        DeclareLaunchArgument('headless', default_value='false'),
        # Separate from `headless` on purpose: headless means "no GUI window",
        # this means "render offscreen through EGL". A machine with a display
        # (WSLg, a desktop) wants headless:=true on its own; a container with no
        # display needs BOTH, or Gazebo segfaults when it tries to open one.
        DeclareLaunchArgument('headless_rendering', default_value='false',
                              description='render via EGL; required where there is no display at all'),
        # start_gazebo:=false attaches to a Gazebo that is ALREADY running, e.g. one you
        # started by hand with:  gz sim -r sonoma_occlusion.sdf
        # Leaving this true while a server is already up starts a SECOND server: both
        # advertise the same world name, the robot is spawned into one of them while you
        # watch the other, and it looks like the spawn silently failed.
        DeclareLaunchArgument('start_gazebo', default_value='true'),
        # sensor_profile:=fast halves the camera resolution and lowers sensor rates.
        # Rendering the depth camera + lidar is what limits simulation speed: in the full
        # raceway world 'full' runs near RTF 0.2 without GPU passthrough (/scan ~2 Hz).
        DeclareLaunchArgument('sensor_profile', default_value='full'),
    ]

    #    Gazebo finds model://apartment through GZ_SIM_RESOURCE_PATH, which must point at
    #    the directory CONTAINING the model folder (…/share/ubot_bringup/models).
    gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('ubot_bringup'), 'models'))

    #    Short-name aliases for Fuel models whose materials reference e.g. model://suv/...
    #    Created by scripts/fix_fuel_textures.sh; harmless if the directory does not exist.
    gz_fuel_aliases = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(os.path.expanduser('~'), '.gz', 'fuel_aliases'))

    world_file = PathJoinSubstitution([
        get_package_share_directory('ubot_bringup'), 'worlds',
        [world, TextSubstitution(text='.sdf')],
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': [
            TextSubstitution(text='-r -v 4 '),
            PythonExpression(["'-s ' if '", LaunchConfiguration('headless'), "'.lower() in ('true','1') else ''"]),
            # --headless-rendering makes ogre2 render offscreen through EGL
            # instead of opening a GLX display. `-s` alone only means "no GUI
            # process": the SERVER still renders the camera and LiDAR, and on a
            # machine with no display at all it logs
            #   Unable to open display / unable to find OpenGL 3+ Rendering
            #   Subsystem
            # and then SEGFAULTS (exit 139), taking the sensors with it. WSL has
            # a display via WSLg so it never needed this; a container does.
            PythonExpression(["'--headless-rendering ' if '",
                              LaunchConfiguration('headless_rendering'),
                              "'.lower() in ('true','1') else ''"]),
            world_file]}.items(),
        condition=IfCondition(LaunchConfiguration('start_gazebo'))
    )


    # 3. ROS-Gazebo Bridge (Clock, Cmd_vel, Odom, TF, Lidar, Camera)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            
            # FIXED RGB-D BRIDGE MAPPING
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',

            # IMU BRIDGE
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',

            # CONTACT BRIDGE - the collision metric for the navigation trials.
            #
            # The Contact system IGNORES the <topic> set on the sensor and always
            # publishes on the fully scoped name below, which carries the world
            # name - hence the substitution. Bridging a short name instead gives
            # a ROS topic that exists, is subscribed, and never receives a
            # message: a silent zero-collision result. Remapped to
            # /bumper/contacts for everything downstream.
            PathJoinSubstitution([
                TextSubstitution(text='/world'), world_name,
                TextSubstitution(
                    text='model/ubot/link/base_footprint/sensor/'
                         'base_contact_sensor/contact'
                         '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'),
            ]),

            # GROUND-TRUTH POSE - the reference every path metric is measured
            # against, and the only thing that can catch odometry lying. With no
            # SLAM, nav2 believes odom, so when odom drifts nav2 reports
            # SUCCEEDED from the wrong place (`false_success` in
            # ubot_eval/schema.py).
            #
            # This is the PosePublisher on the model (see ubot_gazebo.urdf.xacro),
            # NOT SceneBroadcaster's /world/<w>/dynamic_pose/info. That free
            # stream carries entity names but no per-pose header data, and the
            # bridge reads frame ids from the header data - so it converts to
            # transforms with an EMPTY child_frame_id, which cannot be told apart
            # from a moving pedestrian's. The model topic needs no world-name
            # substitution either, since it is scoped by model, not by world.
            '/model/ubot/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        remappings=[
            (PathJoinSubstitution([
                TextSubstitution(text='/world'), world_name,
                TextSubstitution(text='model/ubot/link/base_footprint/sensor/'
                                      'base_contact_sensor/contact')]),
             '/bumper/contacts'),
            ('/model/ubot/pose', '/gt/tf'),
            ('/camera/image', '/camera/rgb/image_raw'),
            ('/camera/depth_image', '/camera/depth/image_raw'),
            ('/camera/camera_info', '/camera/rgb/camera_info'),
            ('/imu', '/imu/data')
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # 4. Spawn Robot Entity
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'ubot',
            '-world', world_name,
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', spawn_yaw,
        ],
    )
    
    # 5. Controller Spawners
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--param-file",
            os.path.join(get_package_share_directory('ubot_bringup'),
                        'config', 'sim_ubot_controllers.yaml'),
            # A heavy world (Sonoma) needs ~15 s before gz_ros2_control has the robot
            # description and starts controller_manager services. The spawner's default
            # 10 s timeout expires first and it dies with "Failed to acquire lock".
            "--controller-manager-timeout", "180",
            "--switch-timeout", "60",
            # gz_ros2_control answers switch_controller from the simulation loop, which in a
            # heavy world can stall well past the 10 s default and kill the spawner.
            "--service-call-timeout", "60",
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    load_diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--param-file",
            os.path.join(get_package_share_directory('ubot_bringup'),
                        'config', 'sim_ubot_controllers.yaml'),
            "--controller-manager-timeout", "180",
            "--switch-timeout", "60",
            # gz_ros2_control answers switch_controller from the simulation loop, which in a
            # heavy world can stall well past the 10 s default and kill the spawner.
            "--service-call-timeout", "60",
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    # 6. Twist Stamper (for teleop compatibility)
    node_twist_stamper = Node(
        package='twist_stamper',
        executable='twist_stamper',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'base_footprint'
        }],
        remappings=[
            ('/cmd_vel_in', '/cmd_vel'),
            ('/cmd_vel_out', '/diff_drive_controller/cmd_vel'),
        ]
    )

    ubot_bringup_dir = get_package_share_directory('ubot_bringup')
    ubot_description_dir = get_package_share_directory('ubot_description')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    # Construction of the variable
    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    # Path to SLAM parameters
    slam_params_file = os.path.join(ubot_bringup_dir, 'config', 'mapper_params_online_async.yaml')
    
    # Path to RViz configuration
    rviz_config_file = os.path.join(ubot_description_dir, 'rviz', 'slam.rviz')

    # 7. Include SLAM Toolbox
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': slam_params_file,
            'use_sim_time': 'true'
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_slam'))
    )

    # REMOVED: a static identity map->odom published when use_slam:=false.
    #
    # It was added because nav2's docking_server failed its CONFIGURE transition
    # with SLAM off, and lifecycle_manager aborts the whole bringup when any one
    # node fails. The supporting experiment - the same params file with
    # use_slam:=true, where bringup succeeded - was CONFOUNDED: that run also
    # killed a leftover opennav_docking from a previous stack, and the real
    # message was
    #   Unable to start transition 1 from current state active
    # i.e. the docking server was a zombie that was ALREADY active, not a node
    # missing a frame. campaign.py::kill_survivors now removes those between
    # episodes, and bringup succeeds with no map frame at all.
    #
    # Nothing here plans in `map`: nav2_params_lidar_odom.yaml sets global_frame
    # to odom, and position truth comes from Gazebo via /gt/tf. A transform that
    # asserts map == odom while claiming to be scaffolding is a localisation
    # claim nobody measured, so it does not stay for comfort.

    # 8. RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    # WHICH nav2 PARAMS: the frame has to follow use_slam, because only SLAM
    # publishes map->odom.
    #
    #   use_slam:=true   -> sim_nav2_params.yaml       (map-framed, static_layer)
    #   use_slam:=false  -> nav2_params_lidar_odom.yaml (odom-framed, rolling)
    #
    # Getting this wrong is SILENT: with use_slam:=false and the map-framed file,
    # nothing publishes map->odom, the global costmap falls back to a default
    # 5x5 m grid at the origin, the robot is logged "out of bounds of the
    # costmap", and nav2 aborts after ~15 recoveries with plenty of clearance.
    # That was an actual run, not a guess, so the choice is made here rather
    # than left to whoever writes the command line.
    _cfg_dir = os.path.join(get_package_share_directory('ubot_bringup'), 'config')
    nav2_params_path = PythonExpression([
        "'", LaunchConfiguration('params_file'), "' or ("
        "'", os.path.join(_cfg_dir, 'sim_nav2_params.yaml'), "' "
        "if '", LaunchConfiguration('use_slam'), "'.lower() in ('true', '1') "
        "else '", os.path.join(_cfg_dir, 'nav2_params_lidar_odom.yaml'), "')"
    ])

    ekf_config_path = os.path.join(
        get_package_share_directory('ubot_bringup'),
        'config', 'sim_ekf.yaml'
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path]
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params_path
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_nav2'))
    )

    return LaunchDescription(declare_args + [
        gz_resource_path,
        gz_fuel_aliases,
        node_robot_state_publisher,
        gazebo,
        bridge,
        spawn_entity,
        node_twist_stamper,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster, load_diff_drive_controller],
            )
        ),
        ekf_node,
        rviz_node,
        slam_toolbox,
        nav2
    ])

# import os
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription, RegisterEventHandler
# from launch.event_handlers import OnProcessExit
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import Command
# from launch_ros.actions import Node

# def generate_launch_description():
#     pkg_description = get_package_share_directory('ubot_description')
    
#     # 1. Process URDF with use_gazebo:=true
#     robot_description_config = Command([
#         'xacro ', 
#         os.path.join(pkg_description, 'urdf', 'body', 'ubot_robot.urdf.xacro'), 
#         ' use_gazebo:=true'
#     ])
    
#     node_robot_state_publisher = Node(
#         package='robot_state_publisher',
#         executable='robot_state_publisher',
#         output='screen',
#         parameters=[{
#             'robot_description': robot_description_config, 
#             'use_sim_time': True
#         }]
#     )
    
#     # 2. Gazebo
#     gazebo = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource([os.path.join(
#             get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
#         launch_arguments={'gz_args': '-r -v 4 sensors.sdf'}.items(),
#     )
    
#     # 3. ROS-Gazebo Bridge (Clock, Cmd_vel, Odom, TF, Lidar, Camera)
#     bridge = Node(
#         package='ros_gz_bridge',
#         executable='parameter_bridge',
#         arguments=[
#             '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
#             '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
#             '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
#             '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
#             '/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            
#             # EXACT RGB-D BRIDGE MAPPING FOR HARMONIC
#             '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
#             '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
#             '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
#             '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
#         ],
#         output='screen',
#         parameters=[{'use_sim_time': True}]
#     )
    
#     # 4. Spawn Robot Entity
#     spawn_entity = Node(
#         package='ros_gz_sim',
#         executable='create',
#         output='screen',
#         arguments=[
#             '-topic', 'robot_description', 
#             '-name', 'ubot', 
#             '-z', '0.1'
#         ],
#     )
    
#     # 5. Controller Spawners
#     load_joint_state_broadcaster = Node(
#         package="controller_manager",
#         executable="spawner",
#         arguments=[
#             "joint_state_broadcaster", 
#             "--param-file", 
#             os.path.join(get_package_share_directory('ubot_bringup'), 
#                         'config', 'ubot_controllers.yaml')
#         ],
#         parameters=[{'use_sim_time': True}]
#     )
    
#     load_diff_drive_controller = Node(
#         package="controller_manager",
#         executable="spawner",
#         arguments=[
#             "diff_drive_controller", 
#             "--param-file", 
#             os.path.join(get_package_share_directory('ubot_bringup'), 
#                         'config', 'ubot_controllers.yaml')
#         ],
#         parameters=[{'use_sim_time': True}]
#     )
    
#     # 6. Twist Stamper (for teleop compatibility)
#     node_twist_stamper = Node(
#         package='twist_stamper',
#         executable='twist_stamper',
#         parameters=[{
#             'use_sim_time': True,
#             'frame_id': 'base_footprint'
#         }],
#         remappings=[
#             ('/cmd_vel_in', '/cmd_vel'),
#             ('/cmd_vel_out', '/diff_drive_controller/cmd_vel'),
#         ]
#     )
    
#     return LaunchDescription([
#         node_robot_state_publisher,
#         gazebo,
#         bridge,
#         spawn_entity,
#         node_twist_stamper,
#         RegisterEventHandler(
#             event_handler=OnProcessExit(
#                 target_action=spawn_entity,
#                 on_exit=[load_joint_state_broadcaster, load_diff_drive_controller],
#             )
#         ),
#     ])


