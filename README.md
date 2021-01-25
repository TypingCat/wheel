# Wheel
ROS2 package that develops wheeled mobile robot capabilities using reinforcement learning


## [First step: Go to target](https://github.com/finiel/wheel/issues/23)
Moving toward the target is the basic ability of a mobile robot. The first step is to train the agent to move the robot to the target. The robot simply needs to go straight toward the target, and there are no special obstacles. The real problem is building simulation environments and creating learning algorithms. See issue [#4](https://github.com/finiel/wheel/issues/4) for details.

The following is a simple demo video of the [regression](https://github.com/finiel/wheel/issues/24). These 60 agents are learning to move to the target. If an algorithm succeeds in this step, it is ready to go to the next step.

![regression](https://user-images.githubusercontent.com/16618451/105570210-0fb09480-5d8b-11eb-9833-b22f0722a062.gif)


## Simple Usage
- Control a mobile robot
    ``` bash
    ros2 launch wheel wakeup.launch.py
    ros2 launch wheel getup.launch.py
    ros2 launch wheel play.launch.py
    ```
- Simulate mobile robots
    ``` bash
    ros2 run wheel_navigation regression
    ros2 run wheel_navigation spg
    ```


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
