using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class TurtlebotAgent : Agent
{
    public GameObject target;
    public GameObject plane;
    public Transform baseFootprint;
    public Transform baseLinkCollision;
    public Transform baseScan;
    public LaserScanner laserScanner;
    public float collisionDistance;

    void Start()
    {
        Debug.Log("Start");

        // Initialize parameters
        preRobotPose = GetPose(baseFootprint);
        scanPose = new Vector2(
            baseLinkCollision.localPosition.z - baseScan.localPosition.z,
            -(baseLinkCollision.localPosition.x - baseScan.localPosition.x));
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("OnEpisodeBegin");

        respawnTarget(1);
    }

    // Randomly respawn target in the plane
    private void respawnTarget(float margin)
    {
        Vector3 respawnRadius = new Vector3(
            (plane.transform.localScale.x * 10) / 2 - margin,
            0,
            (plane.transform.localScale.z * 10) / 2 - margin
        );
        target.transform.localPosition = new Vector3(
            (respawnRadius.x * 2 * Random.value) - respawnRadius.x,
            0,
            (respawnRadius.z * 2 * Random.value) - respawnRadius.z
        );
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        Debug.Log("CollectObservations");
        
        // Observe robot dynamics
        Vector3 robotPose = GetPose(baseFootprint);
        Vector2 robotVelocity = GetVelocity(robotPose);
        sensor.AddObservation(robotPose);
        sensor.AddObservation(robotVelocity);

        // Observe relative pose of target
        Vector2 targetPose = GetRelativePose(robotPose, GetPose(target.transform));
        sensor.AddObservation(targetPose);

        // Observe scan ranges
        float[] scan = laserScanner.Scan().Ranges;
        sensor.AddObservation(scan);

        // float[] s = new float[laserScanner.numLines];
        Vector2[] sense = ChangeScanOrigin2Base(scan);
        
        
        // Check episode end conditions
        float closestObstacleDistance = Mathf.Min(scan);
        float distanceToTarget = targetPose.x;


        // if(episodeStep > maxEpisodeStep) EndEpisode();
        // else if(distanceToTarget < 0.3f) EndEpisode();
        // else if(countCollisionEnter > 0) {
        //     countCollisionEnter = 0;
        //     this.transform.localPosition = Vector3.zero;
        //     EndEpisode();
        // }

        // Log
        preRobotPose = robotPose;
        // Debug.Log(robotPose);
        // Debug.Log(robotVelocity);
        // Debug.Log(targetPose);
    }

    // Get 2D pose in the right-handed coordinate system
    private Vector3 GetPose(Transform tf)
    {
        return new Vector3(
            tf.localPosition.x,
            tf.localPosition.z,
            -((tf.localEulerAngles.y + 450) % 360 - 180) * Mathf.Deg2Rad
        );
    }

    // Get linear and angular velocity
    private Vector3 preRobotPose;
    private Vector2 GetVelocity(Vector3 pose)
    {
        return new Vector2(
            Vector3.Distance(preRobotPose, pose) / (Time.deltaTime * 10),
            (pose.z - preRobotPose.z) / (Time.deltaTime * 10)
        );
    }

    // Get relative pose from p0 to p1 in the cylindrical coordinate system
    private Vector2 GetRelativePose(Vector3 p0, Vector3 p1)
    {
        float distance = Vector3.Distance(p0, p1);
        float angle = Mathf.Atan2(p1.y - p0.y, p1.x - p0.x) - p0.z;
        angle = ((angle + (3 * Mathf.PI)) % (2 * Mathf.PI) - Mathf.PI);
        return new Vector2(distance, angle);
    }

    Vector2 scanPose;
    private Vector2[] ChangeScanOrigin2Base(float[] scan)
    {
        Vector2[] sense = new Vector2[laserScanner.numLines];
        for(int i = 0; i < laserScanner.numLines; i++) {
            float angle = ((laserScanner.ApertureAngle / 2) - (i * laserScanner.AngularResolution)) * Mathf.Deg2Rad;
            sense[i].x = scan[i] * Mathf.Cos(angle) - scanPose.x;
            sense[i].y = scan[i] * Mathf.Sin(angle) - scanPose.y;
        }
        return sense;
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        // Debug.Log("OnActionReceived");
    }
}