import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
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
    
    #2. Gazebo
    world_file = os.path.join(
        get_package_share_directory('ubot_bringup'),
        'worlds',
        'sonoma_occlusion.sdf'
    )

    # 2. Modify the gazebo launch description
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': f'-r -v 4 {world_file}'}.items(),
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
    
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description', 
            '-name', 'ubot',
            '-world', 'sensors', 
            '-x', '272.0',
            '-y', '-136.0',
            '-z', '3.45',
            '-Y', '2.40'
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
                        'config', 'sim_ubot_controllers.yaml')
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
                        'config', 'sim_ubot_controllers.yaml')
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
        }.items()
    )

    # 8. RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}]
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

    )

    return LaunchDescription([
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


