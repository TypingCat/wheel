# Wheel
ROS2 package that develops wheeled mobile robot capabilities using reinforcement learning

Moving toward the target is the basic ability of a mobile robot. The first step of this project is to train the agent to move the robot to the target. There are no obstacles, and the robot simply needs to go straight toward the target. The point is to implement algorithms and its environment for the agent to learn rules.

The following is a simple demo video: There are 60 agents in Unity(right), and the acquired observations are published to ROS2. This returns next best actions which predicted by Multi-Layered Perceptron(MLP). At the same time, MLP is trained when enough data is accumulated. Its loss decreases over time(left).

![regression](https://user-images.githubusercontent.com/16618451/105570210-0fb09480-5d8b-11eb-9833-b22f0722a062.gif)


## Algorithms
### [Regression](https://github.com/TypingCat/wheel/issues/24)
Regression is a subfield of supervised machine learning, not reinforcement learning. Nevertheless, it was implemented for learning environmental testing or use as a performance comparisons. This allowed the initial structure to be improved to make the system work. See issue [#4](https://github.com/TypingCat/wheel/issues/4) for details. The performance is good enough to learn to move to the target in 40 seconds, but it requires supervisor. In other words, problems that cannot be solved by classic path planning cannot be solved by regression.

### [Simplest Policy Gradient(SPG)](https://github.com/TypingCat/wheel/issues/14)
The purpose of reinforcement learning is to choose policies that maximize rewards. One of the reinforcement learning approaches, on-policy gradient, directly learns policy using fresh observations. Various techniques should be applied to policy gradient, but SPG implements only the core of them. It simply predicts linear and angular velocity from observations. This algorithm evolves into the following VPG and PPO.

### [Vanilla Policy Gradient(VPG)](https://github.com/TypingCat/wheel/issues/27)
The main idea of VPG is to learn behaviors that cause good rewards. Add value network and apply [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). This decays the reward over time. It changes the learning objective of the agent: in SPG, the reward is the same as long as the robot reaches its target. On the other hand, VPG needs to reach its target within a short time to achieve high rewards. Therefore the learning objective is changed from the collision avoidance path to the optimal path.

### [Proximal Policy Optimization(PPO)](https://github.com/TypingCat/wheel/issues/29)
Sample efficiency is an important issue for reinforcement learning. Trust Region Policy Optimization(TRPO) considers it using KL-divergence, but it's complex to solve. PPO solves this problem with first-order method that use a simple trick: clip. Basically the ratio of logp and old logp is used as weight, and it has a limit to prevent a big difference. On the other hand, one sample is repeatedly learned until KL-divergence changes by a certain value. Thanks to these features, PPO uses the information in the sample more efficiently than VPG.


## Simple Usage
- Control a mobile robot
    ``` bash
    ros2 launch wheel wakeup.launch.py
    ros2 launch wheel getup.launch.py
    ros2 launch wheel play.launch.py
    ```
- Train agents
    ``` bash
    ros2 run wheel_navigation regression
    ros2 run wheel_navigation spg
    ros2 run wheel_navigation vpg
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
git clone https://github.com/TypingCat/wheel.git
cd wheel
colcon build --symlink-install
source install/local_setup.bash
```
