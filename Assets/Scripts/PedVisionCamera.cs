using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PedVisionCamera : MonoBehaviour {

    public int a_id;
    public RenderTexture Target;
    public GameObject PedCamPrefab;
    private GameObject PedCamObj;
    private Camera PedCam;

    public Material pedestrianMaterial;

    private ProceduralCone pedProxy;
    private List<Matrix4x4> idents;

    void Start ()
    {
        int pixelHeight = 48;
        int pixelWidth = 256;
        float normailizedHeight = (float) pixelHeight / (float) Target.height;
        float normailizedWidth = (float) pixelWidth / (float) Target.width;

        PedCamObj = (GameObject)Instantiate(PedCamPrefab);
        PedCam = PedCamObj.GetComponent<Camera>();

        Rect camRect = new Rect(0, a_id*normailizedHeight, normailizedWidth, normailizedHeight);

        PedCam.transform.position = transform.position;
        PedCam.transform.rotation = transform.rotation;
        PedCam.rect = camRect;
        PedCam.targetTexture = Target;

        PedCam.enabled = false;
        int PedCount = GameObject.FindGameObjectsWithTag("Pedestrian").Length;

        pedProxy = new ProceduralCone(new Vector3(0, 0, 0));

        idents = Enumerable.Repeat(Matrix4x4.identity, PedCount).ToList();
    }
	
	public void SolveInteraction () {
        PedestrianController agent = GetComponent<PedestrianController>();

        PedCam.transform.position = new Vector3(agent.position.x, agent.height, agent.position.y);
        float sceneroty = 90 - agent.lookOrientation * 57.29577951f;
        Quaternion rot = Quaternion.AngleAxis(sceneroty, new Vector3(0, 1, 0));
        PedCam.transform.rotation = rot;

        float o = agent.orientation;
        float rotation = -o + 1.57f;

        Matrix2 rmat = Matrix2.IdentityMatrix();
        rmat.MakeRotation(rotation);

        Vector4 _g_agent_rot_mat = new Vector4(rmat[0,0], rmat[0,1], rmat[1,0], rmat[1,1]);

        float speed = agent.velocity.magnitude;
        if (speed == 0)
        {
            speed = 0.1f; 
        }

        Vector2 _g_agent_velocity = (agent.orientationVec * speed) * rmat;
        Vector2 _g_agent_velocity_goal = (agent.dirToGoal * agent.speedComfort) * rmat;

        float _g_agent_pa = agent.personalArea;
        float _g_agent_ttg = agent.ttg;
        int _g_agent_id = 0;
        int _g_agent_aid = GetComponent<PedVisionCamera>().a_id;

        pedestrianMaterial.SetVector("_g_agent_rot_mat", _g_agent_rot_mat);
        pedestrianMaterial.SetVector("_g_agent_velocity", _g_agent_velocity);
        pedestrianMaterial.SetVector("_g_agent_velocity_goal", _g_agent_velocity_goal);
        pedestrianMaterial.SetFloat("_g_agent_pa", _g_agent_pa);
        pedestrianMaterial.SetFloat("_g_agent_ttg", _g_agent_ttg);
        pedestrianMaterial.SetInt("_g_agent_id", _g_agent_id);
        pedestrianMaterial.SetInt("_g_agent_aid", _g_agent_aid);

        Graphics.DrawMeshInstanced(pedProxy.Mesh,
            0,
            pedestrianMaterial,
            idents,
            null,
            UnityEngine.Rendering.ShadowCastingMode.On,
            false,
            LayerMask.NameToLayer("Pedestrians"));

        PedCam.Render();
    }
}