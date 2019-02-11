/*
 * Copyright 2014.  All rights reserved.
 *
 * CUDA Kernel Device code
 * Rowan Hughes
 */

#define SDATA(index) sdata[index]
#define SMEM(X, Y) sdata[(Y)*bw+(X)]

extern "C" __global__ void
copyReductionKernel(float4* g_idata, float* g_odata, int chanels, int sizeData, int offset)
{
	// shared memory
	// the size is determined by the host application
	extern  __shared__  float sdata[];
	const int dId = (blockDim.x * blockIdx.x + threadIdx.x);
	if(dId >= sizeData) return;

    const unsigned int tIdC = chanels * threadIdx.x;
    const int pixelIdIn  = offset + dId;
	const int tIdMax = sizeData - blockDim.x*blockIdx.x;

    // load data from global to shared memory
    float4 ldata = g_idata[pixelIdIn]; //[thetaDot1, thetaDot2, ttc, first]

    if(ldata.z >= 0 && ldata.x != 0 )
    {
        // it is a pixel belonging to an object
        SDATA(tIdC  ) = ldata.x;    // thetaDot1
        SDATA(tIdC+1) = ldata.y;    // thetaDot2
        SDATA(tIdC+2) = ldata.z;    // ttc
        SDATA(tIdC+3) = 1;          // go first if taking  thetaDot1
        SDATA(tIdC+4) = 1;          // go first if taking  thetaDot2

        if(ldata.z < 3 && ldata.w <= 0) // ttc < 3s and giving a way
        {
            if(abs(ldata.x) < abs(ldata.y))
            {
                SDATA(tIdC+3) = -1; // go second if taking  thetaDot1 => slow down
            }
            else
            {
                SDATA(tIdC+4) = -1; // go second if taking  thetaDot2 => slow down
            }
        }
    }
    else
    {
        // it is a background pixel
        SDATA(tIdC+2) = -1;
    }
    __syncthreads();

    // perform reduction
	for (unsigned int i=blockDim.x*0.5; i>0; i>>=1)
	{
        if(threadIdx.x < i && (threadIdx.x + i < tIdMax))
        {
            int ic = chanels*i+tIdC;

            if(SDATA(ic+2) >= 0)    // if ttc2 >= 0
            {
                if(SDATA(tIdC+2) >= 0)  // if ttc1 >= 0
                {
                    SDATA(tIdC  ) = min(SDATA(tIdC  ), SDATA(ic  ));
                    SDATA(tIdC+1) = max(SDATA(tIdC+1), SDATA(ic+1));
                    SDATA(tIdC+2) = min(SDATA(tIdC+2), SDATA(ic+2));
                    SDATA(tIdC+3) = min(SDATA(tIdC+3), SDATA(ic+3));
                    SDATA(tIdC+4) = min(SDATA(tIdC+4), SDATA(ic+4));
                }
                else
                {
                    SDATA(tIdC  ) = SDATA(ic);
                    SDATA(tIdC+1) = SDATA(ic+1);
                    SDATA(tIdC+2) = SDATA(ic+2);
                    SDATA(tIdC+3) = SDATA(ic+3);
                    SDATA(tIdC+4) = SDATA(ic+4);
                }
            }
        }
        __syncthreads();
	}

	// write data to global memory
	if(threadIdx.x==0)
	{
		int bc = chanels*blockIdx.x;

		g_odata[bc]   = SDATA(0);
		g_odata[bc+1] = SDATA(1);
		g_odata[bc+2] = SDATA(2);
		g_odata[bc+3] = SDATA(3);
		g_odata[bc+4] = SDATA(4);
	}
}