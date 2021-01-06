# Wheel
ROS2 package that develops wheeled mobile robot capabilities


## Simple Usage
- Control a mobile robot
    ``` bash
    ros2 launch wheel wakeup.launch.py
    ros2 launch wheel getup.launch.py
    ros2 launch wheel play.launch.py
    ```
- Simulate mobile robots
    ``` bash
    ros2 launch wheel learn_regression.launch.py
    ```


## Project Structure
- src/wheel: Main ROS2 package
- src/wheel_navigation: Navigation with deep reinforcement learning
- unity/Wheel: Unity project for simulation


## Setup
- Ubuntu 18.04
    - CUDA 10.2
- Python 3.8.3
    - torch 1.7.0
- ROS2 dashing
    - [turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/ros2_setup/)
    - [rplidar_ros](https://github.com/allenh1/rplidar_ros.git) 2.0.0
    - [ros2_intel_realsense](https://github.com/intel/ros2_intel_realsense)
    - [cartographer_ros](https://google-cartographer-ros.readthedocs.io/en/latest/compilation.html#building-installation)
- Unity 2019.4.15f1
    - [ml-agents](https://github.com/Unity-Technologies/ml-agents.git) release_9_branch
    - [VSCode](https://assetstore.unity.com/packages/tools/utilities/vscode-45320?locale=ko-KR)
- Visual Studio Code
    - `Omnisharp: Use Global Mono` = always
    - `Omnisharp: Path` = lateset

``` bash
git clone https://github.com/finiel/wheel.git
cd wheel
colcon build --symlink-install
source install/local_setup.bash
```
