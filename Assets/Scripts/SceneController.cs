using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;

public class SceneController : MonoBehaviour {

    public int PedestriansCount;
    public GameObject PedPrefab;

    private GameObject[] Pedestrians;
    private int pedStart, pedStop;

    private Vector3[] StartPositions;
     
    private Vector3 StartRotation = new Vector3(0, -90, 0);

    private Vector2[] Positions;
    private Vector2[] Velocities;

    public Material AgentViewMaterial;

    public RenderTexture RenderTarget;
    private Texture2D RenderTargetTex;
    int agentViewTexHeight = 48;

    //CUDA Related Variables
    public string[] Cudafiles;
    public string[] CompileOption;

    CudaDeviceVariable<float4> d_idata;
    CudaDeviceVariable<float> d_odata;
    CudaDeviceVariable<float> d_result_data;

    Color[] h_idata;
    float4[] h_idata_float4;
    float[,] h_result_data;

    int dataSize, resultSize;

    protected CudaKernel[] cudaKernel;
    protected CudaContext ctx;


    void InitAgentsOpposite()
    {
        StartPositions = new Vector3[] {
            new Vector3(9, 2, -8),
            new Vector3(9, 2, -4),
            new Vector3(9, 2, 0),
            new Vector3(9, 2, 4),
            new Vector3(9, 2, 8),
            new Vector3(-9, 2, -8),
            new Vector3(-9, 2, -4),
            new Vector3(-9, 2, 0),
            new Vector3(-9, 2, 4),
            new Vector3(-9, 2, 8)};
    }

    void InitAgentsCircle()
    {
        float pa = 0.4f;
        float l = (pa * 2 + 1.0f) * PedestriansCount;
        float r = Mathf.Max(l / (2 * 3.14f), 10.0f);

        float alpha = 0.0f;
        float delta = 2 * 3.14159f / PedestriansCount;

        StartPositions = new Vector3[PedestriansCount];

        for (int i = 0; i < PedestriansCount; ++i, alpha += delta)
        {
            StartPositions[i] = new Vector3(r * Mathf.Cos(alpha), 2,  r * Mathf.Sin(alpha));
        }
    }

    void Awake () {
        Pedestrians = new GameObject[PedestriansCount];
        Material red = Resources.Load("Red", typeof(Material)) as Material;
        Material green = Resources.Load("Green", typeof(Material)) as Material;

        InitAgentsCircle();

        for (int i = 0; i < PedestriansCount; i++)
        {
            Pedestrians[i] = (GameObject)Instantiate(PedPrefab, StartPositions[i], Quaternion.Euler(StartRotation));
            MeshRenderer mesh = Pedestrians[i].GetComponent<MeshRenderer>();
            if (i < (PedestriansCount / 2)) {
                mesh.material = green; 
            }
            else {
                mesh.material = red;
            }
            PedVisionCamera PedCam = Pedestrians[i].GetComponent<PedVisionCamera>();
            PedCam.a_id = i;
        }

        InitializeCUDA();
        RenderTargetTex = new Texture2D(RenderTarget.width, RenderTarget.height,TextureFormat.RGBAFloat,false);
        pedStart = 0;
        pedStop = Pedestrians.Length;

        dataSize = (RenderTargetTex.width * agentViewTexHeight) * 5;
        resultSize = Pedestrians.Length * 5;

        d_idata = new CudaDeviceVariable<float4>(RenderTargetTex.width * RenderTargetTex.height);
        d_odata = new CudaDeviceVariable<float>(dataSize);
        d_result_data = new CudaDeviceVariable<float>(resultSize);
        h_result_data = new float[Pedestrians.Length, 5];

        Positions = new Vector2[Pedestrians.Length + 1];
        Velocities = new Vector2[Pedestrians.Length + 1];

        h_idata = RenderTargetTex.GetPixels();
        h_idata_float4 = new float4[h_idata.Length];
    }

    private void Update()
    {
        SolveInteractions();

        for (int i = 0; i < Pedestrians.Length; ++i)
        {
            Pedestrians[i].GetComponent<PedestrianController>().pedestrianUpdate();
        }
    }

    void SolveInteractions()
    {
        ComputeBuffer positionBuffer = new ComputeBuffer(Pedestrians.Length + 1, 8);
        ComputeBuffer velocityBuffer = new ComputeBuffer(Pedestrians.Length + 1, 8);
        int j = 0;
        for (j = 0; j < Pedestrians.Length; ++j)
        {
            Positions[j] = Pedestrians[j].GetComponent<PedestrianController>().position;
            Velocities[j] = Pedestrians[j].GetComponent<PedestrianController>().velocity;
        }
        Positions[j] = new Vector2(0,0);
        Velocities[j] = new Vector2(0, 0);

        positionBuffer.SetData(Positions);
        AgentViewMaterial.SetBuffer("positionBuffer", positionBuffer);
        velocityBuffer.SetData(Velocities);
        AgentViewMaterial.SetBuffer("velocityBuffer", velocityBuffer);


        //1. First Render From Every Pedestrians PoV
        for (int i = 0; i < Pedestrians.Length; ++i)
        {
            Pedestrians[i].GetComponent<PedestrianController>().SolveInteraction();
        }
        
        //2. Now Process the Resultant Texture
        if (RenderTargetTex && RenderTarget.IsCreated())
        {
            RenderTexture.active = RenderTarget;
            RenderTargetTex.ReadPixels(new Rect(0, 0, RenderTarget.width, RenderTarget.height), 0, 0);
            RenderTargetTex.Apply();

            // Launch CUDA Kernel
            uint textureSize = (uint)(RenderTargetTex.width * agentViewTexHeight);
            int threadsPerBlock = 1024;
            float threadsPerBlockInv = 1.0f / (float)threadsPerBlock;
            int blocksPerGrid = (int)((textureSize + threadsPerBlock - 1) * threadsPerBlockInv);

            /*******************************************************************/
            /************************copyReductionKernel************************/
            /*******************************************************************/
            dim3 block = new dim3(threadsPerBlock, 1, 1);
            dim3 grid = new dim3(blocksPerGrid, 1, 1);

            uint shMemeSize = (uint)(block.x * 5 * sizeof(float));  // size of shared memory
            uint offset, currentNumData, currentNumBlocks;

            h_idata = RenderTargetTex.GetPixels();
            for (int i = 0; i < h_idata.Length; i++)
            {
                h_idata_float4[i] = new float4(h_idata[i].r,
                    h_idata[i].g,
                    h_idata[i].b,
                    h_idata[i].a);
            }

            d_idata.CopyToDevice(h_idata_float4);

            for (int i = pedStart; i < pedStop; i++)
            {
                offset = (uint)(i - pedStart) * (textureSize);
                currentNumData = textureSize;
                currentNumBlocks = (uint)blocksPerGrid;
                grid.x = currentNumBlocks;

                cudaKernel[0].BlockDimensions = block;
                cudaKernel[0].GridDimensions = grid;
                cudaKernel[0].DynamicSharedMemory = shMemeSize;
                cudaKernel[0].Run(d_idata.DevicePointer, d_odata.DevicePointer, 5, textureSize, offset);
                //Debug.Log("1: CUDA kernel launch with " + blocksPerGrid + " blocks of " + threadsPerBlock + " threads\n");

                /*******************************************************************/
                /**************************reductionKernel**************************/
                /*******************************************************************/
                currentNumData = currentNumBlocks;
                currentNumBlocks = (uint)((currentNumData + threadsPerBlock - 1) * threadsPerBlockInv);

                for (; currentNumData > 1;)
                {
                    // perform reduction to get one pixel
                    grid.x = currentNumBlocks;
                    cudaKernel[1].BlockDimensions = block;
                    cudaKernel[1].GridDimensions = grid;
                    cudaKernel[1].DynamicSharedMemory = shMemeSize;
                    cudaKernel[1].Run(d_odata.DevicePointer, 5, currentNumData);
                    //Debug.Log("2: CUDA kernel launch with " + blocksPerGrid + " blocks of " + threadsPerBlock + " threads\n");

                    currentNumData = currentNumBlocks;
                    currentNumBlocks = (uint)((currentNumData + threadsPerBlock - 1) * threadsPerBlockInv);
                }

                d_result_data.CopyToDevice(d_odata, 0, 5 * i * sizeof(float), 5 * sizeof(float));
            }

            d_result_data.CopyToHost(h_result_data);
            RenderTexture.active = null;
        }

        //3. Now Calculate Agent Veclocity Based on Results from Step 2.
        for (int i = 0; i < Pedestrians.Length; ++i)
        {
            float thetaMax = 0;
            float thetaMin = 0;
            float ttcMin = 0;
            float dttcMin = 0;
            bool goFirstMin = true;
            bool goFirstMax = true;

            computeParams(i, ref thetaMax, ref thetaMin, ref ttcMin, ref dttcMin, ref goFirstMin, ref goFirstMax);

            // compute new velocity
            Pedestrians[i].GetComponent<PedestrianController>().updateVelocity(thetaMin, thetaMax, ttcMin, dttcMin, goFirstMin, goFirstMax);
        }
    }

    void computeParams(int k, ref float thetaMax, ref float thetaMin, ref float ttcMin, ref float dttcMin, ref bool goFirstMin, ref bool goFirstMax)
    {
        float thetaDotPlus = 0;
        float thetaDotMinus = 0;
        bool goFirstL = true;
        bool goFirstR = true;
        goFirstMin = true;
        goFirstMax = true;
        thetaMin = 0;
        thetaMax = 0;
        ttcMin = -10.0f;

        float ttcValue = h_result_data[k,2];        // ttc
        float thetaDotValue1 = h_result_data[k,0];  // thetaDot
        float thetaDotValue2 = h_result_data[k,1];  // thetaDot

        if (ttcValue >= 0)
        {
            if (thetaDotValue1 < 10)
            {
                thetaDotMinus = thetaDotValue1;
                thetaDotPlus = thetaDotValue2;
            }
            ttcMin = ttcValue;
            goFirstR = (h_result_data[k,3] > 0) ? true : false;
            goFirstL = (h_result_data[k,4] > 0) ? true : false;
        }

        if (Mathf.Abs(thetaDotMinus) > Mathf.Abs(thetaDotPlus))
        {
            thetaMin = thetaDotPlus;
            thetaMax = thetaDotMinus;
            goFirstMin = goFirstL;
            goFirstMax = goFirstR;
        }
        else
        {
            thetaMin = thetaDotMinus;
            thetaMax = thetaDotPlus;
            goFirstMin = goFirstR;
            goFirstMax = goFirstL;
        }
    }

    protected void InitializeCUDA()
    {
        string[] filetext = new string[Cudafiles.Length];
        cudaKernel = new CudaKernel[Cudafiles.Length];
        ctx = new CudaContext(0);

        for (int i = 0; i < Cudafiles.Length; ++i)
        {
            filetext[i] = File.ReadAllText(Application.dataPath + @"\Scripts\CUDA\" + Cudafiles[i] + ".cu");
            Debug.Log(filetext[i]);

            CudaRuntimeCompiler rtc = new CudaRuntimeCompiler(filetext[i], Cudafiles[i]);
            rtc.Compile(CompileOption);
            Debug.Log(rtc.GetLogAsString());

            byte[] ptx = rtc.GetPTX();
            rtc.Dispose();

            cudaKernel[i] = ctx.LoadKernelPTX(ptx, Cudafiles[i]);
        }
    }

    void OnDestroy()
    {
        d_idata.Dispose();
        d_odata.Dispose();
        d_result_data.Dispose();
        ctx.Dispose();
    }
}
