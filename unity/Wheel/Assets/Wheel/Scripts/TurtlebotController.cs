using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using rclcs;

[RequireComponent(typeof(Rigidbody))]
public class TurtlebotController : MonoBehaviourRosNode
{
    public string NodeName = "turtlebot_controller";
    public string CommandVelocityTopic = "cmd_vel";

    public Rigidbody BaseRigidbody;

    private Vector3 commandVelocityLinear = Vector3.zero;
    private Vector3 commandVelocityAngular = Vector3.zero;

    protected override string nodeName { get { return NodeName; } }

    private Subscription<geometry_msgs.msg.Twist> commandVelocitySubscription;

    protected override void StartRos()
    {
        commandVelocitySubscription = node.CreateSubscription<geometry_msgs.msg.Twist>(
            CommandVelocityTopic,
            (msg) =>
            {
                commandVelocityLinear = msg.Linear.Ros2Unity(); 
                commandVelocityAngular = msg.Angular.Ros2Unity();
            });
    }

    private void Start()
    {
        if (BaseRigidbody == null)
        {
            BaseRigidbody = GetComponent<Rigidbody>();
        }

        commandVelocityLinear = Vector3.zero;
        commandVelocityAngular = Vector3.zero;
    }

    private void FixedUpdate()
    {
        SpinSome();

        Vector3 deltaPosition = commandVelocityLinear * Time.deltaTime;
        deltaPosition = BaseRigidbody.transform.TransformDirection(deltaPosition);
        Quaternion deltaRotation = Quaternion.Euler(-commandVelocityAngular * Mathf.Rad2Deg * Time.deltaTime);

        BaseRigidbody.MovePosition(BaseRigidbody.position + deltaPosition);
        BaseRigidbody.MoveRotation(BaseRigidbody.rotation * deltaRotation);
    }
}
