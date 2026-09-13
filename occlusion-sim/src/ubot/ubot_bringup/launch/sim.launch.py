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
            ' use_gazebo:=true'
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
        DeclareLaunchArgument('spawn_x', default_value='272.0'),
        DeclareLaunchArgument('spawn_y', default_value='-136.0'),
        DeclareLaunchArgument('spawn_z', default_value='3.45'),
        DeclareLaunchArgument('spawn_yaw', default_value='2.40'),
        # The raceway world is expensive. Turning these off leaves just Gazebo + the robot,
        # which is usually what you want for occlusion capture, and makes startup reliable:
        #   ros2 launch ubot_bringup sim.launch.py use_slam:=false use_nav2:=false use_rviz:=false
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        # headless:=true runs the Gazebo server only (no GUI window). Sensors still work.
        # Much lighter, and the only way to run on a machine without a display.
        DeclareLaunchArgument('headless', default_value='false'),
        # start_gazebo:=false attaches to a Gazebo that is ALREADY running, e.g. one you
        # started by hand with:  gz sim -r sonoma_occlusion.sdf
        # Leaving this true while a server is already up starts a SECOND server: both
        # advertise the same world name, the robot is spawned into one of them while you
        # watch the other, and it looks like the spawn silently failed.
        DeclareLaunchArgument('start_gazebo', default_value='true'),
    ]

    #    Gazebo finds model://apartment through GZ_SIM_RESOURCE_PATH, which must point at
    #    the directory CONTAINING the model folder (…/share/ubot_bringup/models).
    gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('ubot_bringup'), 'models'))

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
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
        ],
        remappings=[
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

    # 8. RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    nav2_params_path = os.path.join(
            get_package_share_directory('ubot_bringup'),
            'config',
            'sim_nav2_params.yaml'
        )

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


