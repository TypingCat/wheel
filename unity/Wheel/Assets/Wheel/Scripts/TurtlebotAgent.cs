using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class TurtlebotAgent : Agent
{
    public string id = "";
    public float collisionMargin = 2;   // robotRadius is not enough to cover huge wheels
    public float maxLinearVelocity = 0.5f;
    public float maxAngularVelocity = 1;

    public GameObject target;
    public GameObject plane;
    public GameObject baseFootprint;
    public GameObject baseLink;
    public GameObject baseScan;

    private Rigidbody robotBody;
    private Transform baseLinkCollisions;
    private LaserScanner laserScanner;
    private Controller controller;

    private float targetRadius;
    private float robotRadius;

    void Start()
    {
        // Connect with components
        robotBody = baseFootprint.GetComponentInChildren<Rigidbody>();
        baseLinkCollisions = baseLink.transform.Find("Collisions");
        laserScanner = baseScan.GetComponentInChildren<LaserScanner>();
        controller = baseFootprint.GetComponentInChildren<Controller>();

        // Initialize parameters
        preRobotPose = GetPose(robotBody);
        scanPose = new Vector2(     // The transform base_scan in on the base_link 
            baseLinkCollisions.transform.localPosition.z - baseScan.transform.localPosition.z,
            -(baseLinkCollisions.transform.localPosition.x - baseScan.transform.localPosition.x));
        targetRadius = (target.transform.localScale.x + target.transform.localScale.z) / 4;
        robotRadius = (baseLinkCollisions.transform.localScale.x + baseLinkCollisions.transform.localScale.z) / 4;
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("Start new episode");
    }

    private void respawn(GameObject obj, float margin)
    {
        Vector3 respawnArea = new Vector3(
            (plane.transform.localScale.x * 10) / 2 - margin,
            0,
            (plane.transform.localScale.z * 10) / 2 - margin
        );
        obj.transform.localPosition = new Vector3(
            (respawnArea.x * 2 * Random.value) - respawnArea.x,
            0,
            (respawnArea.z * 2 * Random.value) - respawnArea.z
        );
    }
    private void respawn(Rigidbody rb, float margin)
    {
        Vector3 respawnArea = new Vector3(
            (plane.transform.localScale.x * 10) / 2 - margin,
            0,
            (plane.transform.localScale.z * 10) / 2 - margin
        );
        rb.position = new Vector3(
            (respawnArea.x * 2 * Random.value) - respawnArea.x,
            0,
            (respawnArea.z * 2 * Random.value) - respawnArea.z
        );
        rb.rotation = Quaternion.Euler(0, 720 * Random.value - 360, 0);
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe scan ranges
        float[] scan = laserScanner.Scan().Ranges;
        sensor.AddObservation(scan);

        // Observe robot dynamics
        Vector3 robotPose = GetPose(robotBody);
        Vector2 robotVelocity = GetVelocity(robotPose);
        sensor.AddObservation(new Vector2(
            robotVelocity.x / maxLinearVelocity,    // Normalized velocity
            robotVelocity.y / maxAngularVelocity
        ));

        // Observe relative pose of target
        Vector2 targetPosition = GetRelativePos2D(robotPose, GetPose(target.transform));
        sensor.AddObservation(targetPosition);

        // Check episode end conditions
        float[] sense = ShiftScanOrigin(scan);
        float distanceToObstacle = Mathf.Min(sense);
        float distanceToTarget = targetPosition.x;
        if(distanceToObstacle < robotRadius * collisionMargin) {
            Debug.Log("Collides with an obstacle");
            respawn(robotBody, 2);
            EndEpisode();
        }
        else if(distanceToTarget < robotRadius + targetRadius) {
            Debug.Log("Reaches the target");
            respawn(target, 1);
            EndEpisode();
        }

        // Backup
        preRobotPose = robotPose;
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
    private Vector3 GetPose(Rigidbody rb)
    {
        return new Vector3(
            rb.position.x,
            rb.position.z,
            -((rb.rotation.eulerAngles.y + 450) % 360 - 180) * Mathf.Deg2Rad
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

    // Get relative position from p0 to p1 in the cylindrical coordinate system
    private Vector2 GetRelativePos2D(Vector3 p0, Vector3 p1)
    {
        float dx = p1.x - p0.x;
        float dy = p1.y - p0.y;
        float distance = Mathf.Sqrt(Mathf.Pow(dx, 2) + Mathf.Pow(dx, 2));
        float angle = Mathf.Atan2(dy, dx) - p0.z;
        angle = ((angle + (3 * Mathf.PI)) % (2 * Mathf.PI) - Mathf.PI);
        return new Vector2(distance, angle);
    }

    // Change scan origin from base_scan to base_link
    Vector2 scanPose;
    private float[] ShiftScanOrigin(float[] scan)
    {
        float[] sense = new float[laserScanner.numLines];
        for(int i = 0; i < laserScanner.numLines; i++) {
            float angle = ((laserScanner.ApertureAngle / 2) - (i * laserScanner.AngularResolution)) * Mathf.Deg2Rad;
            sense[i] = Mathf.Sqrt(Mathf.Pow(scan[i] * Mathf.Cos(angle) - scanPose.x, 2) + Mathf.Pow(scan[i] * Mathf.Sin(angle) - scanPose.y, 2));
        }
        return sense;
    }

    // Send command velocity
    public override void OnActionReceived(float[] vectorAction)
    {
        float linearVelocity = maxLinearVelocity * vectorAction[0];
        float angularVelocity = maxAngularVelocity * vectorAction[1];
        controller.SetVelocity(linearVelocity, angularVelocity);
    }
}