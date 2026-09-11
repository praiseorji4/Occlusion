import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. PATH CONFIGURATION
    # We use 'ubot_description' package
    pkg_share = get_package_share_directory("ubot_description")
    
    # Path to the xacro file we created
    model_path = os.path.join(pkg_share, "urdf", "body" ,"ubot_robot.urdf.xacro")
    
    # Path to the Rviz config (We will create this empty file next)
    rviz_config_path = os.path.join(pkg_share, "rviz", "display.rviz")

    # 2. DECLARE ARGUMENTS
    model_arg = DeclareLaunchArgument(
        name="model", 
        default_value=model_path,
        description="Absolute path to robot urdf file"
    )

    # 3. PROCESS URDF
    # We keep ParameterValue for safety.
    # We removed "is_sim" for now to avoid warnings with the basic box model.
    robot_description = ParameterValue(Command([
        "xacro ",
        LaunchConfiguration("model")
    ]), value_type=str)

    # 4. DEFINE NODES
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        # This loads the saved view configuration
        arguments=["-d", rviz_config_path]
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])