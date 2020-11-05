using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using rclcs;


public class KeyboardTeleop : MonoBehaviourRosNode
{
    public string NodeName = "keyboard_teleop";
    public string CommandVelocityTopic = "cmd_vel";

    public float MaxLinearVelocity = 0.5f;
    public float MaxAngularVelocity = 1.0f;
    public float PublishingFrequency = 30.0f;

    protected override string nodeName { get { return NodeName; } }

    private Publisher<geometry_msgs.msg.Twist> cmdVelPublisher;
    private geometry_msgs.msg.Twist cmdVelMsg;
    private geometry_msgs.msg.Twist cmdVelMsg_;

    protected override void StartRos()
    {
        cmdVelPublisher = node.CreatePublisher<geometry_msgs.msg.Twist>(CommandVelocityTopic);

        cmdVelMsg = new geometry_msgs.msg.Twist();
        cmdVelMsg_ = new geometry_msgs.msg.Twist();
        StartCoroutine("PublishCommandVelocity");
    }

    // Publish ROS2 command velocity using keyboard
    IEnumerator PublishCommandVelocity()
    {
        while(true) {
            yield return new WaitForSeconds(1.0f / PublishingFrequency);

            cmdVelMsg.Linear.X = Input.GetAxis("Vertical") * MaxLinearVelocity;
            cmdVelMsg.Angular.Z = -Input.GetAxis("Horizontal") * MaxAngularVelocity;

            if(cmdVelMsg_.Linear.X != 0 || cmdVelMsg_.Angular.Z != 0 ||
                cmdVelMsg.Linear.X != 0 || cmdVelMsg.Angular.Z != 0) {
                cmdVelPublisher.Publish(cmdVelMsg);
            }

            cmdVelMsg_.Linear.X = cmdVelMsg.Linear.X;
            cmdVelMsg_.Angular.Z = cmdVelMsg.Angular.Z;
        }
    }
}
