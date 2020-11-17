using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class TurtlebotAgent : Agent
{
    void Start()
    {
        Debug.Log("Start");
    }

    public GameObject target;
    public GameObject plane;

    // private int episodeStep;
    // public int maxEpisodeStep = 2000;
    public override void OnEpisodeBegin()
    {
        Debug.Log("OnEpisodeBegin");

        // episodeStep = 0;

        // float x = plane.transform.localScale.x;
        // float y = plane.transform.localScale.y;
        // float z = plane.transform.localScale.z;
        // Debug.Log(x.ToString() + y.ToString() + z.ToString());

        // // Respawn target
        // target.transform.localPosition = new Vector3(
        //     4*Random.value - 2,
        //     0.1f,
        //     4*Random.value - 2);
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        Debug.Log("CollectObservations");
        
        float empty = 1.2f;
        sensor.AddObservation(empty);

        // Observe relative pose of target
        // float distanceToTarget = Vector3.Distance(
        //     this.transform.localPosition, target.transform.localPosition);
        // float angleToTarget = Mathf.Rad2Deg * Mathf.Atan2(
        //     target.transform.localPosition.x - this.transform.localPosition.x,
        //     target.transform.localPosition.z - this.transform.localPosition.z)
        //     - this.transform.localEulerAngles.y;
        // angleToTarget = ((angleToTarget + 540) % 360 - 180) * Mathf.Deg2Rad;

        // sensor.AddObservation(distanceToTarget);
        // sensor.AddObservation(angleToTarget);
        // sensor.AddObservation(Mathf.Sin(angleToTarget));
        // sensor.AddObservation(Mathf.Cos(angleToTarget));
    }

    // Move, rewarding, and check episode end
    // public float maxLinearVelocity = 0.88f;    // 0.22 [m/s]
    // public float maxAngulerVelocity = 2.84f;   // 2.84 [rad/s]
    public override void OnActionReceived(float[] vectorAction)
    {
        Debug.Log("OnActionReceived");

        // episodeStep += 1;

        // Move robot body
        // Vector3 deltaPosition = this.transform.TransformDirection(
        //     new Vector3(0, 0, maxLinearVelocity * vectorAction[1] * Time.deltaTime));
        // Quaternion deltaRotation = Quaternion.Euler(
        //     new Vector3(0, maxAngulerVelocity * vectorAction[0] * Time.deltaTime * 180 / Mathf.PI, 0));
        // this.GetComponent<Rigidbody>().MovePosition(this.transform.position + deltaPosition);
        // this.GetComponent<Rigidbody>().MoveRotation(this.transform.rotation * deltaRotation);
        
        // Rewarding
        float reward = -0.05f;
        // float distanceToTarget = Vector3.Distance(
        //     this.transform.localPosition, target.transform.localPosition);
        // if(distanceToTarget < 0.3f) reward += 15;
        // if(countCollisionEnter > 0) reward -= 20;      
        // if(Mathf.Abs(vectorAction[0]) > 0.5f) reward -= 0.15f * Mathf.Abs(vectorAction[0]);
        // if(vectorAction[1] < -0.5f) reward += 0.15f * vectorAction[1];
        AddReward(reward);

        // Check episode end conditions
        // if(episodeStep > maxEpisodeStep) EndEpisode();
        // else if(distanceToTarget < 0.3f) EndEpisode();
        // else if(countCollisionEnter > 0) {
        //     countCollisionEnter = 0;
        //     this.transform.localPosition = Vector3.zero;
        //     EndEpisode();
        // }
    }

    // Send action with keyboard
    // public override void Heuristic(float[] actionsOut)
    // {
    //     actionsOut[0] = Input.GetAxis("Horizontal");
    //     actionsOut[1] = Input.GetAxis("Vertical");
    // }
}