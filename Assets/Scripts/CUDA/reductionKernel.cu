/*
 * Copyright 2014.  All rights reserved.
 *
 * CUDA Kernel Device code
 * Rowan Hughes
 */

#define SDATA( index)  sdata[index]

extern "C" __global__ void
reductionKernel( float* g_idata, int chanels, int sizeData)
{

	// shared memory
	// the size is determined by the host application
	extern  __shared__  float sdata[];   
    const unsigned int dId = (blockIdx.x * blockDim.x + threadIdx.x);
	if(dId >= sizeData) return;

	const unsigned int tIdC = chanels * threadIdx.x;
	const unsigned int pixelId = chanels * dId;
	const unsigned int tIdMax = sizeData - blockDim.x*blockIdx.x;

    float ttc = g_idata[pixelId+2];

    // TODO: perform reduction on loading
    if(ttc >= 0)
    {
        // it is a pixel belonging to an object
        SDATA(tIdC  ) = g_idata[pixelId];
        SDATA(tIdC+1) = g_idata[pixelId+1];
        SDATA(tIdC+2) = ttc;
        SDATA(tIdC+3) = g_idata[pixelId+3];
        SDATA(tIdC+4) = g_idata[pixelId+4];
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
                    // take minimum/maximum
                    SDATA(tIdC  ) = min(SDATA(tIdC  ), SDATA(ic  ));
                    SDATA(tIdC+1) = max(SDATA(tIdC+1), SDATA(ic+1));
                    SDATA(tIdC+2) = min(SDATA(tIdC+2), SDATA(ic+2));
                    SDATA(tIdC+3) = min(SDATA(tIdC+3), SDATA(ic+3));
                    SDATA(tIdC+4) = min(SDATA(tIdC+4), SDATA(ic+4));
                }
                else
                {
                    // replace background pixel (tIdC) with object pixel (ic)
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
	    g_idata[bc]   = SDATA(0);
	    g_idata[bc+1] = SDATA(1);
	    g_idata[bc+2] = SDATA(2);
	    g_idata[bc+3] = SDATA(3);
	    g_idata[bc+4] = SDATA(4);
	}	
}