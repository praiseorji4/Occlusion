# Raw fusion2urdf export, 2026-09-11 — provenance only, DO NOT USE

Kept to show what the exporter produced. Two reasons not to build on it:

1. **Every joint origin is wrong** by a constant (-16.4, +3.4, +19.7) mm. The meshes
   were correct, so the robot looked right in RViz while each wheel turned about an
   axle 26 mm off-centre. Corrected values live in `ubot_description/urdf/`.
2. It is a **ROS 1 (catkin)** package: `package.xml` format 2, a catkin
   `CMakeLists.txt`, and `.launch` XML files that ROS 2 cannot run.

Its `meshes/` are not here: byte-identical to `ubot_description/meshes/inspection/*.stl`,
renamed there (`base_link.stl` -> `base.stl`, `wheel_fl_1.stl` -> `wheel_front_left.stl`).
