#include "src/output/gw_output.cuh"

__constant__ letype d_H = 0.0;

__constant__ letype d_a   = 0.0;
__constant__ letype d_asr = 0.0;

__constant__ cufft_type* d_f[6];
__constant__ fTT_type* d_tf[6];

__global__ void adjustDerivativeValues(letype* h, letype* hd, const letype meanAcc, const int direction)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

   	hd[index(x, y, z)] -= direction * rescale_r * d_H * (h[index(x, y, z)]);
}

void startAdjustDerivativeValues(letype** h, letype** hd, const int gwfld, const int direction)
{
	letype H = ad/a;
	cudaMemcpyToSymbol(d_H, &H, sizeof(letype));

    double asr = pow(a, rescale_s - rescale_r);

    cudaMemcpyToSymbol(d_a,   &a,   sizeof(letype));
    cudaMemcpyToSymbol(d_asr, &asr, sizeof(letype));

	const dim3 numBlocks(N / tileSize, N / tileSize, N / tileSize * 4);
	const dim3 threadsPerBlock(tileSize, tileSize, tileSize / 4);

	if(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z > 1024 || threadsPerBlock.z > 64)
	{
		printf("%s\n", "ERROR in startAdjustDerivativeValues: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
		abort();	
	}

    //printf("hd before:\t%.20f, h before:\t%.20f, value:\t%.20f\n", hd[0][index(10, 20, 30)], h[0][index(10, 20, 30)], rescale_r * ad/a);

    for(int i = 0; i < 6; i++)
    {
       cudaMemPrefetchAsync(h[i],  gridsize * sizeof(letype), 0, NULL);
       cudaMemPrefetchAsync(hd[i], gridsize * sizeof(letype), 0, NULL);
	   adjustDerivativeValues<<<numBlocks, threadsPerBlock>>>(h[i], hd[i], hmean_acc[gwfld][i], direction);
    }
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    //printf("hd after:\t%.20f, h after:\t%.20f, value:\t%.20f\n", hd[0][index(10, 20, 30)], h[0][index(10, 20, 30)], rescale_r * ad/a);
}


__device__ inline double keff(int i, const int nVals[])
{
    return 2.0 * M_PI * nVals[i] / (double)L; // continuum projector
 
    double res = 0.0;

    #pragma unroll
    for(int k = 1; k <= cfdHaloSize; k++)
        res += cu_cfd_stencil_d[cfdHaloSize + k] * sinpi(2.0 * k * nVals[i] / (double)N);

    return 2.0 * res / dx; // modified projector
}

__device__ inline double keffabs2(const int nVals[])
{
    double res = 0.0;
    for(int i = 0; i < NDIMS; i++)
        res += keff(i, nVals) * keff(i, nVals);
    
    return res;
}

__device__ inline double P0(int i, int j, const int nVals[])
{
    if(nVals[i] == 0 || nVals[i] == N/2 || nVals[i] == -N/2)
        if(nVals[j] == 0 || nVals[j] == N/2 || nVals[j] == -N/2)
        {
            if(i == j)
                return 0.0;
            return 0.0;
        }

    double res = 0.0;
    if(i == j)
        res += 1.0;

    res -= keff(i, nVals) * keff(j, nVals) / keffabs2(nVals);
    return res;
}

__device__ inline double lam0(int i, int j, int l, int m, const int nVals[])
{
    return P0(i, l, nVals) * P0(j, m, nVals) - 0.5 * P0(i, j, nVals) * P0(l, m, nVals);
}

const double traceMargin = 10e-4;

__global__ void testTTProj()
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    const int nVals[3] = {x - N/2, y - N/2, z - N/2};

    double val_r = 0.0;
    double val_c = 0.0;

    //check tracelessness
    for(int i = 0; i < 3; i++)
    {
        val_r += d_tf[indexTensor(i, i)][indexFT(x, y, z)].x;
        val_c += d_tf[indexTensor(i, i)][indexFT(x, y, z)].y;
    }
    if(abs(val_r) > traceMargin || abs(val_c) > traceMargin)
    {
        printf("ERROR: trace is too large at %d %d %d with values %e %e\n", nVals[0], nVals[1], nVals[2], val_r, val_c);
    }

    int ii = 0;
    int ji = 0;

    //check transversality
    for(int j = 0; j < 3; j++)
    {
        val_r = 0.0;
        val_c = 0.0;

        for(int i = 0; i < 3; i++)
        {
            ii = i;
            ji = j;

            if(j > i)
            {
                ii = j;
                ji = i;
            }

            val_r += keff(i, nVals) * d_tf[indexTensor(ii, ji)][indexFT(x, y, z)].x;
            val_c += keff(i, nVals) * d_tf[indexTensor(ii, ji)][indexFT(x, y, z)].y;
        }


        if(abs(val_r) > traceMargin || abs(val_c) > traceMargin)
        {
            printf("ERROR: transversality test failed at %d %d %d with values %e %e\n", nVals[0], nVals[1], nVals[2], val_r, val_c);
        }
    }

    //check
    for(int j = 0; j < 3; j++)
    {
        val_r = 0.0;
        val_c = 0.0;

        for(int i = 0; i < 3; i++)
        {
            ii = i;
            ji = j;

            if(j > i)
            {
                ii = j;
                ji = i;
            }
            
            val_r = d_tf[indexTensor(ii, ji)][indexFT(x, y, z)].x - d_tf[indexTensor(ii, ji)][indexFT(N - x, N - y, N - z)].x;
            val_c = d_tf[indexTensor(ii, ji)][indexFT(x, y, z)].y + d_tf[indexTensor(ii, ji)][indexFT(N - x, N - y, N - z)].y;

            if(abs(val_r) > traceMargin || abs(val_c) > traceMargin)
            {
                printf("ERROR: conjugate test failed at %d %d %d with values %e %e\n", nVals[0], nVals[1], nVals[2], val_r, val_c);
            }
        }
    }

}

void startTestTTProj()
{
    const dim3 numBlocks(2 * N / tileSize, 2 * N / tileSize, N / tileSize * 4);
    const dim3 threadsPerBlock(tileSize / 2, tileSize / 2, tileSize / 4);

    if(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z > 1024 || threadsPerBlock.z > 64)
    {
        printf("%s\n", "ERROR in startTTProj: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
        abort();    
    }

    //testTTProj<<<numBlocks, threadsPerBlock>>>();
    printf("Sarting the TT projection test\n");
    //testTTProj<<<1, 10>>>();
    testTTProj<<<numBlocks, threadsPerBlock>>>();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void TTProj(cufft_type** f, fTT_type** tf)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    const int nVals[3] = {x - N/2, y - N/2, z - N/2};
	
    int li = 0;
    int mi = 0;

    int ii = 0;
    int ji = 0;

    double tf_temp_r[NDIMS * NDIMS];
    double tf_temp_c[NDIMS * NDIMS];

    for(int i = 0; i < 3; i++)
    	for(int j = 0; j < 3; j++)
	{
        ii = i;
        ji = j;
        if(j > i)
        {
            ii = j;
            ji = i;
        }

        tf_temp_r[indexTensor(ii, ji)] = 0.0;
		tf_temp_c[indexTensor(ii, ji)] = 0.0;
        
		for(int l = 0; l < 3; l++)
			for(int m = 0; m < 3; m++)
		{
			if(l >= m)
			{
				li = l;
				mi = m;
			}
			else
			{	
				li = m;
				mi = l;
			}
			
			tf_temp_r[indexTensor(ii, ji)] += lam0(i, j, l, m, nVals) * f[indexTensor(li, mi)][indexFT(x, y, z)].x;
			tf_temp_c[indexTensor(ii, ji)] += lam0(i, j, l, m, nVals) * f[indexTensor(li, mi)][indexFT(x, y, z)].y;
		}

        tf[indexTensor(ii, ji)][indexFT(x, y, z)].x = tf_temp_r[indexTensor(ii, ji)];
        tf[indexTensor(ii, ji)][indexFT(x, y, z)].y = tf_temp_c[indexTensor(ii, ji)];
	}
}

void startTTProj(cufft_type** f, fTT_type** tf)
{
    gpuErrchk(cudaMemcpyToSymbol(d_f,   f, 6 * sizeof(cufft_type*)));
    gpuErrchk(cudaMemcpyToSymbol(d_tf, tf, 6 * sizeof(fTT_type*)));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


	const dim3 numBlocks(2 * N / tileSize, 2 * N / tileSize, N / tileSize * 4);
	const dim3 threadsPerBlock(tileSize / 2, tileSize / 2, tileSize / 4);

	if(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z > 1024 || threadsPerBlock.z > 64)
	{
		printf("%s\n", "ERROR in startTTProj: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
		abort();	
	}

	int device = -1;
	cudaGetDevice(&device);

    const size_t gridsize_ft = (N + 1) * (N + 1) * (N + 1);

	for(int fld = 0; fld < 6; fld++)
	{
		gpuErrchk(cudaMemPrefetchAsync(f[fld],  gridsize_ft * sizeof(cufft_type), device, NULL));
		gpuErrchk(cudaMemPrefetchAsync(tf[fld], gridsize_ft * sizeof(fTT_type), device, NULL));
	}

	TTProj<<<numBlocks, threadsPerBlock>>>(f, tf);		
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__global__ void addGWFields(int gwnflds, fTT_type*** hdkTT)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    for(int i = 0; i < 6; i++)
    {
	    hdkTT[gwnflds - 1][i][indexFT(x, y, z)].x = 0.0;
		hdkTT[gwnflds - 1][i][indexFT(x, y, z)].y = 0.0;
    }

    for(int i = 0; i < 6; i++)
    {
	    for(int gwfld = 0; gwfld < gwnflds - 1; gwfld++)
	    {
	    	hdkTT[gwnflds - 1][i][indexFT(x, y, z)].x += hdkTT[gwfld][i][indexFT(x, y, z)].x;
			hdkTT[gwnflds - 1][i][indexFT(x, y, z)].y += hdkTT[gwfld][i][indexFT(x, y, z)].y;
	    }
	}
}

void startAddGWFields(int gwnflds, fTT_type*** hdkTT)
{
	const dim3 numBlocks(2 * N / tileSize, 2 * N / tileSize, N / tileSize * 4);
	const dim3 threadsPerBlock(tileSize / 2, tileSize / 2, tileSize / 4);

	if(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z > 1024 || threadsPerBlock.z > 64)
	{
		printf("%s\n", "ERROR in startAddGWFields: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
		abort();	
	}

	addGWFields<<<numBlocks, threadsPerBlock>>>(gwnflds, hdkTT);
    gpuErrchk(cudaPeekAtLastError());
}