using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class PedestrianController : MonoBehaviour {

    PedVisionCamera pedCam;

    public float personalArea = 0.4f;
    public float height = 1.8f;
    public float speedComfort = 1.5f;

    [HideInInspector]
    public float speedDesired = 0.0f;
    [HideInInspector]
    public float accMax = 1.0f;

    float distToGoal;
    [HideInInspector]
    public float orientation = 0.0f;
    [HideInInspector]
    public float lookOrientation;

    [HideInInspector]
    public Vector2 position;
    [HideInInspector]
    public Vector2 orientationVec;
    [HideInInspector]
    public Vector2 velocity;
    [HideInInspector]
    public Vector2 velocityComfort;
    Vector2 velocityDesired;
    [HideInInspector]
    public Vector2 dirToGoal;

    float ttc;
    [HideInInspector]
    public float ttg;
    float alphaDotGoal;
    float thetaDotOld = 0.0f;

    float waitTime = 0.0f;
    bool waiting = false;

    Vector2 goalPoint;

    private void Start()
    {
        position = new Vector2(transform.position.x, transform.position.z);
        velocity = new Vector2(0.0f, 0.0f);

        goalPoint = new Vector2(-position.x, -position.y);

        personalArea = 0.4f;
        height = 1.6f;
        speedComfort = 1.5f;
        accMax = 1.0f;
        orientationVec = new Vector2(0.0f, 0.0f);
        thetaDotOld = 0.0f;

        speedComfort = Random.Range(1.3f, 1.6f);
        pedCam = GetComponent<PedVisionCamera>();

        orientationVec = goalPoint - position;
        orientationVec.Normalize();
        orientation = Mathf.Atan2(orientationVec.y, orientationVec.x);
    }

    public void SolveInteraction()
    {
        pedCam.SolveInteraction();
    }

    void computeComfortVelocity()
    {
        if (goalPoint != null)
        {
            dirToGoal = goalPoint - position;
            distToGoal = (float)dirToGoal.magnitude;

            Vector2 direction = dirToGoal;
            direction.Normalize();
            velocityComfort = direction * speedComfort;
        }
        else
        {
            distToGoal = 0.0f;
            velocityComfort = new Vector2(0.0f, 0.0f);
        }
    }

    // computes velocity avoiding collisions
    public void updateVelocity(float thetaMin, float thetaMax, float lttc, float dttc, bool goFirstMin, bool goFirstMax)
    {
        float dt = 0.16f;
        float speed = speedComfort;
        float thetaDot = 0.0f;
        ttc = lttc;
        bool goFirst = goFirstMin;

        // velocity to the goal
        computeComfortVelocity();
        computeGoalParams();

        if (distToGoal < 0.2)
        {
            // at the goal, stop
            velocityDesired = new Vector2(0.0f, 0.0f);
        }
        else if (ttc >= 0 && ttc < 10.0)
        {
            if (waitTime > 0.0f)
            {
                waitTime -= dt;
            }
            else
            {
                float speedCurrent = velocity.magnitude;
                if (!waiting)
                {
                    float ttcTreshlod = 3.0f; // Threshold for speed adaptation
                                              // Chose turning side according to the goal position
                    thetaDot = thetaMin;
                    if (thetaMin * alphaDotGoal > 0 && Mathf.Abs(thetaMin) < Mathf.Abs(alphaDotGoal))
                    {
                        thetaDot = alphaDotGoal;
                    }
                    else if (thetaMax * alphaDotGoal > 0 && Mathf.Abs(thetaMax) < Mathf.Abs(alphaDotGoal))
                    {
                        thetaDot = alphaDotGoal;
                    }
                    else if (ttc > 0)
                    {
                        float t1 = Mathf.Abs(thetaMin - alphaDotGoal);
                        float t2 = Mathf.Abs(thetaMax - alphaDotGoal);

                        if ((t1 < 0.08f && t2 < 0.08f) || t1 <= t2)
                        {
                            thetaDot = thetaMin;
                        }
                        else
                        {
                            thetaDot = thetaMax;
                            goFirst = goFirstMax;
                        }
                    }

                    // giving way, slow down
                    if (ttc < ttcTreshlod && !goFirst)
                    {
                        speed = speedComfort * ((1 - Mathf.Pow(2.718f, -(ttc * ttc) * 0.4f)));
                        if (speed < 0.5f) speed = 0.0f;
                    }

                    if (speed == 0.0 && speedCurrent == 0.0)
                    {
                        // stop and wait
                        waitTime = 0.1f + 0.1f * (Random.value % 6);
                        waiting = true;
                    }
                    else if (speedCurrent > 0.0)
                    {
                        float tv = 0.5f;
                        float thetaDiff = Mathf.Max(-tv, Mathf.Min(tv, thetaDot - thetaDotOld));
                        thetaDot = thetaDotOld + thetaDiff;
                    }

                }
                else
                {
                    // recover from stop
                    float t1 = Mathf.Abs(thetaMin - alphaDotGoal);
                    float t2 = Mathf.Abs(thetaMax - alphaDotGoal);

                    thetaDot = thetaMin;
                    thetaDot = ((t1 < 0.01 && t2 < 0.01) || t1 <= t2) ? thetaMin : thetaMax;
                    waiting = false;
                    speed = 0.001f;
                }

                // prevents going backwards
                Vector2 newDir = new Vector2(Mathf.Cos(orientation + thetaDot), Mathf.Sin(orientation + thetaDot));
                float dot = Vector2.Dot(dirToGoal, newDir);
                if (speedCurrent != 0 && dot < -0.8 && (ttc != 0))
                {
                    thetaDot = 0;
                }
                thetaDotOld = thetaDot;

                Matrix2 rmat = Matrix2.IdentityMatrix();
                rmat.MakeRotation(thetaDot);

                velocityDesired = orientationVec * rmat;
                velocityDesired = velocityDesired * speed;
            }
        }
        else
        {
            // No collision, use comfort velocity
            waiting = false;
            waitTime = 0;
            velocityDesired = velocityComfort;
            thetaDotOld = 0;
        }

        updateVelocity(velocityDesired);
    }

    // computes desired velocity
    public void updateVelocity(Vector2 velocityNew)
    {
        float clippedSpeed;

        float slowingDistance = Mathf.Max(1.0f, 1.3f * velocity.magnitude);
        float speed = velocityNew.magnitude;
        clippedSpeed = Mathf.Min(speed, (float)(speed * Mathf.Pow(distToGoal / slowingDistance, 2.0f)));

        velocityDesired = velocityNew * clippedSpeed;
    }

    // entry point agent update
    public void pedestrianUpdate()
    {
        float dt = Time.deltaTime;

        Vector2 steering = velocityDesired - velocity;
        speedDesired = velocityDesired.magnitude;

        if (speedDesired != 0 && speedDesired < 0.002)
        {
            orientationVec = velocityDesired;
            orientationVec.Normalize();
            velocity.Set(0,0);
        }
        else if (!((speedDesired == 0 && steering.magnitude < 0.1)))
        {
            Vector2 acceleration;
            float accm = accMax;

            if (speedDesired == 0)
                accm = 2 * accMax;

            if (steering.magnitude > dt * accm)
            {
                steering.Normalize();
                acceleration = steering * dt * accm;
            }
            else
            {
                acceleration = steering;
            }
            // update velocity
            velocity = velocity + acceleration;

            if (velocity.magnitude > 0)
            {
                orientationVec = velocity;
                orientationVec.Normalize();
            }
        }
        else   // stop
        {
            velocity.Set(0,0);
        }

        orientation = Mathf.Atan2(orientationVec.y, orientationVec.x);
        lookOrientation = orientation;
     
        Vector2 trans = velocity * dt;
        transform.localPosition += new Vector3(trans.x, 0, trans.y);
        position = new Vector2(transform.position.x, transform.position.z);
    }

    void computeGoalParams()
    {
        ttg = 1e30f;
        Vector2 dir12;
        Vector2 relPosGoal;

        relPosGoal = goalPoint - position;

        dir12 = relPosGoal;
        dir12.Normalize();

        Vector2 velComposed = new Vector2(0.0f, 0.0f) - (orientationVec * speedComfort);

        Vector2 dir12n = relPosGoal + velComposed;
        dir12n.Normalize();

        float alphaC = Mathf.Atan2(dir12.y, dir12.x);
        float alphaN = Mathf.Atan2(dir12n.y, dir12n.x);

        /// alphaDot
        alphaDotGoal = alphaN - alphaC;

        if (alphaDotGoal > 3.14)
            alphaDotGoal -= 6.28f;
        else if (alphaDotGoal < -3.14)
            alphaDotGoal += 6.28f;

        /// SpeedAlpha (ttg)
        velComposed = new Vector2(0.0f, 0.0f) - velocity;
        float speedAlpha = Vector2.Dot(velComposed, -dir12);

        if (speedAlpha > 0.0)
            ttg = distToGoal / speedAlpha;


        float d = Vector2.Dot(relPosGoal, orientationVec);
        if (d <= 0.0)
        {
            alphaDotGoal += (alphaDotGoal > 0.0) ? 1.57f : -1.57f;
        }
    }
}