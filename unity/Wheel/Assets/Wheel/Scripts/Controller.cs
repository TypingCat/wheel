using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using rclcs;

[RequireComponent(typeof(Rigidbody))]
public class Controller : CustomBehaviourRosNode
{
    public string NodeName = "turtlebot_controller";
    public string CommandVelocityTopic = "cmd_vel";

    public TurtlebotAgent agent;
    public Rigidbody BaseRigidbody;

    private Vector3 commandVelocityLinear = Vector3.zero;
    private Vector3 commandVelocityAngular = Vector3.zero;

    protected override string nodeName { get { return NodeName; } }

    private Subscription<geometry_msgs.msg.Twist> commandVelocitySubscription;

    protected override void CreateRosNode()
    {        
        getSharedContext();
        node = new Node(nodeName, context, "agent_" + agent.id);
    }
    
    protected override void StartRos()
    {
        commandVelocitySubscription =
            node.CreateSubscription<geometry_msgs.msg.Twist>(CommandVelocityTopic, (msg) => {
                commandVelocityLinear = msg.Linear.Ros2Unity(); 
                commandVelocityAngular = msg.Angular.Ros2Unity();
            });
    }

    private void Start()
    {
        if(BaseRigidbody == null) {
            BaseRigidbody = GetComponent<Rigidbody>();
        }

        commandVelocityLinear = Vector3.zero;
        commandVelocityAngular = Vector3.zero;
    }

    // Update robot position by velocity
    private void FixedUpdate()
    {
        SpinSome();

        Vector3 deltaPosition = commandVelocityLinear * Time.fixedDeltaTime;
        deltaPosition = BaseRigidbody.transform.TransformDirection(deltaPosition);
        Quaternion deltaRotation = Quaternion.Euler(-commandVelocityAngular * Mathf.Rad2Deg * Time.fixedDeltaTime);

        BaseRigidbody.MovePosition(BaseRigidbody.position + deltaPosition);
        BaseRigidbody.MoveRotation(BaseRigidbody.rotation * deltaRotation);
    }

    public void SetVelocity(float linearVelocity, float angularVelocity)
    {
        commandVelocityLinear = new Vector3(0, 0, linearVelocity);
        commandVelocityAngular = new Vector3(0, angularVelocity, 0);
    }
}
