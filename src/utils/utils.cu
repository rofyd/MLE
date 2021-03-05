#include "src/utils/utils.cuh"




__global__ void addPerturbation(letype* f, const dim3 pos)
{
	f[index(pos.x, pos.y, pos.z)] = 10.0f;
}

__global__ void calcMean2D(letype* f, letype* mean2D)
{
	//coordinates on the grid
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    letype res = 0.0;
    letype c = 0.0;
    letype w = 0.0;
    letype t = 0.0;

    for(int z = 0; z < N; z++)
    {
    	
    	w = f[index(x, y, z)] - c;
    	t = res + w;
    	c = (t - res) - w;
    	res = t;
    	
    	//res += f[index(x, y, z)];
    }
    mean2D[index(x, y, 0)] = res;
}

__global__ void calcMean1D(letype* f, letype* mean1D)
{
	//coordinates on the grid
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	double res = 0.0;
	letype c = 0.0;
    letype w = 0.0;
    letype t = 0.0;

    for(int y = 0; y < N; y++)
    {
    	w = f[index(x, y, 0)] - c;
    	t = res + w;
    	c = (t - res) - w;
    	res = t;
    	
    	//res += f[index(x, y, 0)];
    }
    mean1D[index(x, 0, 0)] = res;
}



/*letype calcMeanFrom2D(letype* mean2D)
{
	gpuErrchk(cudaDeviceSynchronize());
	static cudaMemGC<letype> mean1D(N);

	//printf("MEAN: %.20f\n", mean2D[20]);
	
	const dim3 numBlocksMean1D(N / tileSize, 1, 1);
	const dim3 threadsPerBlockMean1D(tileSize, 1, 1);

	if(threadsPerBlockMean1D.x * threadsPerBlockMean1D.y * threadsPerBlockMean1D.z > 1024 || threadsPerBlockMean1D.z > 64)
	{
		printf("%s\n", "ERROR: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
		abort();	
	}

	int device = -1;
	cudaGetDevice(&device);

	cudaMemPrefetchAsync(mean1D.data, N*sizeof(letype), device, NULL);
	cudaMemPrefetchAsync(mean2D, N*N*sizeof(letype), device, NULL);
	
	calcMean1D<<<numBlocksMean1D, threadsPerBlockMean1D>>>(mean2D, mean1D.data);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//printf("### 1D %.20f\n", mean1D.data[10]);

	double res_o = 0.0; //TODO just change this back to letype

    letype c = 0.0;
    letype w = 0.0;
    letype t = 0.0;

	for(int i = 0; i < N; i++)
	{
    	w = mean1D.data[i] - c;
    	t = res_o + w;
    	c = (t - res_o) - w;
    	res_o = t;
	}

	return res_o;
}*/

void prefetchAsyncFields(letype** f, letype** fd, int destDevice)
{
	const size_t gridsize = N*N*N;

	// prefetch data
	for(int fld = 0; fld < nflds; fld++)
	{
		cudaMemPrefetchAsync(f[fld],   gridsize * sizeof(letype), destDevice, NULL);
		cudaMemPrefetchAsync(fd[fld],  gridsize * sizeof(letype), destDevice, NULL);
	}

	gpuErrchk(cudaPeekAtLastError());
}

void prefetchAsyncGWFields(letype*** h, letype*** hd, letype*** EMT, int destDevice)
{
	const size_t gridsize = N*N*N;

	if constexpr(sgw)
	{
		for(int fld = 0; fld < nflds; fld++)
		{
			for(int gwfld = 0; gwfld < 6; gwfld++)
			{
				cudaMemPrefetchAsync(h[fld][gwfld],    gridsize * sizeof(letype), destDevice, NULL);
				cudaMemPrefetchAsync(hd[fld][gwfld],   gridsize * sizeof(letype), destDevice, NULL);
				cudaMemPrefetchAsync(EMT[fld][gwfld],  gridsize * sizeof(letype), destDevice, NULL);
			}
		}
	}

	gpuErrchk(cudaPeekAtLastError());
}