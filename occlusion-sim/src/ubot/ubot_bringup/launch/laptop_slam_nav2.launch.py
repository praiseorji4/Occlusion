import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    ubot_bringup_dir = get_package_share_directory('ubot_bringup')
    ubot_description_dir = get_package_share_directory('ubot_description')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    # SLAM parameters
    slam_params_file = os.path.join(ubot_bringup_dir, 'config', 'mapper_params_online_async.yaml')
    
    # RViz config
    rviz_config_file = os.path.join(ubot_description_dir, 'rviz', 'slam.rviz')

    # Nav2 params
    nav2_params_path = os.path.join(ubot_bringup_dir, 'config', 'nav2_params.yaml')

    # SLAM Toolbox
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': slam_params_file,
            'use_sim_time': 'false'
        }.items()
    )

    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': False}]
    )

    # Nav2 - use_sim_time: FALSE
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': nav2_params_path 
        }.items(),
    )
    
    return LaunchDescription([
        slam_toolbox,
        rviz_node, 
        nav2
    ])