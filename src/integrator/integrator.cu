#include "integrator.cuh"

// declare cuda variable counterparts
__constant__ letype d_dvdf_params[num_dvdf_params];
__constant__ letype d_pot_params[num_pot_params];

__constant__ letype d_a = 0.0; // scale factor saved in constant memory
__constant__ letype d_aterm = 0.0; // a-term
__constant__ letype d_laplnorm = 0.0; // lapl norm factor
__constant__ letype d_asr = 0.0;

__constant__ letype* d_f[nflds];
__constant__ letype* d_fd[nflds];

__constant__ letype* d_EMT[nflds][6];

// copy field pointer to constant memory for easier handling
void copyFieldsToConstantMemory()
{
	gpuErrchk(cudaMemcpyToSymbol(d_f,  f,  nflds * sizeof(letype*)));
	gpuErrchk(cudaMemcpyToSymbol(d_fd, fd, nflds * sizeof(letype*)));

	if constexpr(sgw)
	{
		gpuErrchk(cudaMemcpyToSymbol(d_EMT, EMT, nflds * 6 * sizeof(letype*)));
	}
}

// project the field values of f to a 2d grid
__global__ void caclMeanGrad2D(const letype* __restrict__ f, letype* __restrict__ mean2D)
{
	//here we save the tiles of the x-y-slice that also includes the halo
	volatile __shared__ letype tile[numThreadsGWnum][numThreadsGWnum][2*isoHaloSize + 1];

	//coordinates on the grid
	const int x = mod(blockIdx.x * blockDim.x + threadIdx.x - isoHaloSize - blockIdx.x * 2 * isoHaloSize, N);
    const int y = mod(blockIdx.y * blockDim.y + threadIdx.y - isoHaloSize - blockIdx.y * 2 * isoHaloSize, N);
    int z = 0;

    //coordinates on the tile
    const int xt = threadIdx.x;
    const int yt = threadIdx.y;

    //check if this thread is in the inner tile and has to do calculations
    const bool doCalculations = !(xt < isoHaloSize || yt < isoHaloSize || xt - isoHaloSize >= tileSizeGW || yt - isoHaloSize >= tileSizeGW);

    //xyOffset for grid coord
    const int xyOffset = y * N + x;

	//in this array the old and upcoming values are saved, such that zPen[0] -> most old one
	letype zPen[2*isoHaloSize + 1];

    //load old and upcoming values
	for(int i = 0; i < 2*isoHaloSize + 1; i++)
		zPen[i] = f[mod(-isoHaloSize + i, N) * N * N + xyOffset];

    //load the values for the tile
    for(int i = 0; i < 2*isoHaloSize + 1; i++)
    	tile[yt][xt][i] = zPen[i];

	letype meanTemp = 0.0;
	letype res = 0.0;

	if(doCalculations)
		mean2D[xyOffset] = 0.0;

    letype st_prevals_sq[NDIMS * isoHaloSize + 1] = {0.0};

    //make sure that the tile is completely initialized before continuing
	__syncthreads();

    //now the initialization finished and the fun can begin
	while(z < N)
	{
		res = 0.0;
		//check if this thread has to do calculations
		if(doCalculations)
		{
			for(int i = 0; i < NDIMS * isoHaloSize + 1; i++)
				st_prevals_sq[i] = 0.0;

			for(int i = 0; i < 2*isoHaloSize + 1; i++)
				for(int j = 0; j < 2*isoHaloSize + 1; j++)
					for(int k = 0; k < 2*isoHaloSize + 1; k++)
						st_prevals_sq[stencilIndex(i, j, k)] += cu_pow2(tile[yt - isoHaloSize + i][xt - isoHaloSize + j][k] - zPen[isoHaloSize]);	

			for(int i = NDIMS * isoHaloSize; i >= 0; i--)
				res += stencil[i] * st_prevals_sq[i];
		}

		meanTemp += res;

		//make sure that all threads are ready before starting to shift the data
		__syncthreads();

		//increase the z index
		z++;

		//shift the data: tile gets next value from zPen and zPen gets shifted older while also obtaining the next value
		for(int i = 0; i < 2*isoHaloSize; i++)
			zPen[i] = zPen[i + 1];

		zPen[2*isoHaloSize] = f[mod(z + isoHaloSize, N) * N * N + xyOffset];

    	for(int i = 0; i < 2*isoHaloSize + 1; i++)
			tile[yt][xt][i] = zPen[i];

		//make sure that all threads are ready before moving on with the calculations
		__syncthreads();
	}

	if(doCalculations)
		mean2D[xyOffset] = 0.5 * meanTemp;
}

// calculates the mean grad energy for all fields f
void startCalcMeanGradEnergy(letype** f, letype curr_gradEnergy[nflds])
{
	static cudaMemGC<letype> meanLapl2D(N*N);

	const dim3 numBlocks(N / tileSizeGW, N / tileSizeGW, 1);
	const dim3 threadsPerBlock(numThreadsGWnum, numThreadsGWnum, 1);

	for(int fld = 0; fld < nflds; fld++)
	{
		cudaMemPrefetchAsync(f[fld],  gridsize * sizeof(letype), 0, NULL);
		cudaMemPrefetchAsync(meanLapl2D.data, N * N * sizeof(letype), 0, NULL);

		caclMeanGrad2D<<<numBlocks, threadsPerBlock>>>(f[fld], meanLapl2D.data);

		gpuErrchk(cudaPeekAtLastError());
		reduce<N * N>(meanLapl2D.data, &curr_gradEnergy[fld]);
		gpuErrchk(cudaPeekAtLastError());
	}
}

// kernel to calculate square gradient at all points
__global__ void calcLaplacian(const letype* __restrict__ f, letype* __restrict__ fd, const letype dtime, int fld)
{
	//here we save the tiles of the x-y-slice that also includes the halo
	volatile __shared__ letype tile[numThreads][numThreads][2*isoHaloSize + 1];

	//coordinates on the grid
	const int x = mod(blockIdx.x * blockDim.x + threadIdx.x - haloSize - blockIdx.x * 2 * haloSize, N);
    const int y = mod(blockIdx.y * blockDim.y + threadIdx.y - haloSize - blockIdx.y * 2 * haloSize, N);
    int z = 0;

    //coordinates on the tile
    const int xt = threadIdx.x;
    const int yt = threadIdx.y;

    //check if this thread is in the inner tile and has to do calculations
    const bool doCalculations = !(xt < haloSize || yt < haloSize || xt - haloSize >= tileSize || yt - haloSize >= tileSize);

    //xyOffset for grid coord
    const int xyOffset = y * N + x;

	//in this array the old and upcoming values are saved, such that zPen[0] -> most old one
	letype zPen[2*cfdHaloSize + 1];

    //load old and upcoming values
	for(int i = 0; i < 2*cfdHaloSize + 1; i++)
		zPen[i] = f[mod(-cfdHaloSize + i, N) * N * N + xyOffset];

    //load the values for the tile
    for(int i = 0; i < 2*isoHaloSize + 1; i++)
    	tile[yt][xt][i] = zPen[cfdHaloSize - isoHaloSize + i];

	letype res = 0.0;

	letype cfd_dirs[3];

    letype st_prevals[NDIMS * isoHaloSize + 1] = {0.0}; // assumes cube like stencil

    //make sure that the tile is completely initialized before continuing
	__syncthreads();

    //now the initialization finished and the fun can begin
	while(z < N)
	{
		//check if this thread has to do calculations
		if(doCalculations)
		{
			res = 0.0;

			for(int i = 0; i < NDIMS * isoHaloSize + 1; i++)
				st_prevals[i] = 0.0;

			for(int i = 0; i < 2*isoHaloSize + 1; i++)
				for(int j = 0; j < 2*isoHaloSize + 1; j++)
					for(int k = 0; k < 2*isoHaloSize + 1; k++)
						st_prevals[stencilIndex(i, j, k)] += tile[yt - isoHaloSize + i][xt - isoHaloSize + j][k];


			for(int i = NDIMS * isoHaloSize; i >= 0; i--)
			{
				res += st_prevals[i] * stencil[i];
			}

			if constexpr(sgw)
			{		
				cfd_dirs[0] = 0.0;
				cfd_dirs[1] = 0.0;
				cfd_dirs[2] = 0.0;
		
				for(int i = cfdHaloSize; i >= 1; i--)
				{
					cfd_dirs[1] += (tile[yt + i][xt][isoHaloSize] - tile[yt - i][xt][isoHaloSize]) * cu_cfd_stencil[cfdHaloSize + i];
					cfd_dirs[0] += (tile[yt][xt + i][isoHaloSize] - tile[yt][xt - i][isoHaloSize]) * cu_cfd_stencil[cfdHaloSize + i];
					cfd_dirs[2] += (zPen[cfdHaloSize + i] - zPen[cfdHaloSize - i]) * cu_cfd_stencil[cfdHaloSize + i];
				}

				for(int i = 0; i < 3; i++)
					for(int j = 0; j <= i; j++)
						d_EMT[fld][indexTensor(i, j)][index(x, y, z)] = cfd_dirs[i] * cfd_dirs[j];
			}

			res *= d_laplnorm;

			//res = 0.0; // TODO remove this. It is only there for testing purposes.

			fd[z*N*N + xyOffset] += dtime * (res + d_aterm * zPen[cfdHaloSize] - d_dvdf(fld, d_dvdf_params, d_f, d_a, x, y, z));
		}

		//make sure that all threads are ready before starting to shift the data
		__syncthreads();

		//increase the z index
		z++;

		//shift the data: tile gets next value from zPen and zPen gets shifted older while also obtaining the next value
		for(int i = 0; i < 2*cfdHaloSize; i++)
			zPen[i] = zPen[i + 1];

		zPen[2*cfdHaloSize] = f[mod(z + cfdHaloSize, N) * N * N + xyOffset];

    	for(int i = 0; i < 2*isoHaloSize + 1; i++)
			tile[yt][xt][i] = zPen[cfdHaloSize - isoHaloSize + i];

		//make sure that all threads are ready before moving on with the calculations
		__syncthreads();
	}
}

// calculates the derivative for all fields
void startCalcLaplacian(letype dtime, letype curr_gradEnergy[nflds])
{
	double laplnorm = 1.0 / pw2(dx) / pow(a, 2.0 * rescale_s + 2.0); // Set coefficient for laplacian term in equations of motion. The dx^2 converts the output of lapl() to a laplacian and the scale factor term accounts for model dependent rescalings of the equations of motion.


	letype l_laplnorm = (letype) laplnorm;
	letype l_a = (letype) a;
	letype l_aterm = (letype) aterm;

	gpuErrchk(cudaMemcpyToSymbolAsync(d_a, &l_a, sizeof(d_a), 0, cudaMemcpyHostToDevice, 0));
	gpuErrchk(cudaMemcpyToSymbolAsync(d_aterm, &l_aterm, sizeof(d_aterm), 0, cudaMemcpyHostToDevice, 0));
	gpuErrchk(cudaMemcpyToSymbolAsync(d_laplnorm, &l_laplnorm, sizeof(d_laplnorm), 0, cudaMemcpyHostToDevice, 0));

	const dim3 numBlocksLapl(N / tileSize, N / tileSize, 1);
	const dim3 threadsPerBlockLapl(numThreads, numThreads, 1);

	if(threadsPerBlockLapl.x * threadsPerBlockLapl.y * threadsPerBlockLapl.z > 1024 || threadsPerBlockLapl.z > 64)
	{
		printf("%s\n", "ERROR in startCalcLaplacian: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
		abort();
	}

	gpuErrchk(cudaMemcpyToSymbolAsync(d_dvdf_params, dvdf_params, num_dvdf_params * sizeof(letype), 0, cudaMemcpyHostToDevice, 0));

//printf("FIELD TEST1: %.20f\n", f[0][index(1, 1, 1)]);

	for(int fld = 0; fld < nflds; fld++)
	{
		cudaMemPrefetchAsync(f[fld],  gridsize * sizeof(letype), 0, NULL);
		cudaMemPrefetchAsync(fd[fld], gridsize * sizeof(letype), 0, NULL);
		gpuErrchk(cudaPeekAtLastError());

		if constexpr(sgw)
		{
			for(int gwfld = 0; gwfld < 6; gwfld++)
			{
				cudaMemPrefetchAsync(EMT[fld][gwfld],  gridsize * sizeof(letype), 0, NULL);
			}
		}
		gpuErrchk(cudaPeekAtLastError());

		calcLaplacian<<<numBlocksLapl, threadsPerBlockLapl>>>(f[fld], fd[fld], dtime, fld);
		gpuErrchk(cudaPeekAtLastError());
	}
}

// calculates the derivatives for all gw fields
__global__ void evolveGWd(const letype* __restrict__ h, letype* __restrict__ hd, const letype* __restrict__ EMT, const letype dtime, const letype meanAcc)
{
	//here we save the tiles of the x-y-slice that also includes the halo
	volatile __shared__ letype tile[numThreadsGWnum][numThreadsGWnum][2*isoHaloSize + 1];
	//__shared__ letype tile[6 numThreadsGWnum * numThreadsGWnum * (2*isoHaloSize + 1)];

	//coordinates on the grid
	const int x = mod(blockIdx.x * blockDim.x + threadIdx.x - isoHaloSize - blockIdx.x * 2 * isoHaloSize, N);
    const int y = mod(blockIdx.y * blockDim.y + threadIdx.y - isoHaloSize - blockIdx.y * 2 * isoHaloSize, N);
    int z = 0;

    //coordinates on the tile
    const int xt = threadIdx.x;
    const int yt = threadIdx.y;

    //check if this thread is in the inner tile and has to do calculations
    const bool doCalculations = !(xt < isoHaloSize || yt < isoHaloSize || xt - isoHaloSize >= tileSizeGW || yt - isoHaloSize >= tileSizeGW);

    //xyOffset for grid coord
    const int xyOffset = y * N + x;

    //load the values for the tile
    for(int i = 0; i < 2*isoHaloSize + 1; i++)
    	tile[yt][xt][i] = h[mod(-isoHaloSize + i, N) * N * N + xyOffset];

	letype res = 0.0;
	letype st_prevals[NDIMS * isoHaloSize + 1] = {0.0}; // assumes cube like stencil

    //make sure that the tile is completely initialized before continuing
	__syncthreads();

   //now the initialization finished and the fun can begin
	while(z < N)
	{
		//check if this thread has to do calculations
		if(doCalculations)
		{
			res = 0.0;

			for(int i = 0; i < NDIMS * isoHaloSize + 1; i++)
				st_prevals[i] = 0.0;

			for(int i = 0; i < 2*isoHaloSize + 1; i++)
				for(int j = 0; j < 2*isoHaloSize + 1; j++)
					for(int k = 0; k < 2*isoHaloSize + 1; k++)
			{ 
				st_prevals[stencilIndex(i, j, k)] += tile[yt - isoHaloSize + i][xt - isoHaloSize + j][k];
			}

			for(int i = NDIMS * isoHaloSize; i >= 0; i--)
				res += st_prevals[i] * stencil[i];

			res *= d_laplnorm;

			hd[z*N*N + xyOffset] += dtime * (res + d_aterm * (tile[yt][xt][isoHaloSize] + meanAcc) + d_asr * EMT[z*N*N + xyOffset]);
		}

		//make sure that all threads are ready before starting to shift the data
		__syncthreads();

		//increase the z index
		z++;

		//shift the data: tile gets next value from zPen and zPen gets shifted older while also obtaining the next value
    	for(int i = 0; i < 2*isoHaloSize; i++)
			tile[yt][xt][i] = tile[yt][xt][i + 1];

		tile[yt][xt][2*isoHaloSize] = h[mod(z + isoHaloSize, N) * N * N + xyOffset];

		//make sure that all threads are ready before moving on with the calculations
		__syncthreads();
	}
}

// starts the kernel to calculate the derivative for all gw fields
void startEvolveGWd(letype** h, letype** hd, letype** EMT, letype dtime, const int gwfld)
{
	const letype asr = 16.0 * M_PI * pow(a, -2.0 * rescale_s - rescale_r - 2.0) / pw2(dx);

	cudaMemcpyToSymbolAsync(d_asr, &asr, sizeof(asr), 0, cudaMemcpyHostToDevice, 0);

	const dim3 numBlocksLapl(N / tileSizeGW, N / tileSizeGW, 1);
	const dim3 threadsPerBlockLapl(numThreadsGWnum, numThreadsGWnum, 1);

	if(threadsPerBlockLapl.x * threadsPerBlockLapl.y * threadsPerBlockLapl.z > 1024 || threadsPerBlockLapl.z > 64)
	{
		printf("%s\n", "ERROR in startEvolveGWd: Total number of threads in a block may not exceed 1024. The number of threads in the z-direction may not exceed 64\n");
		abort();
	}

	for(int fld = 0; fld < 6; fld++)
	{
		cudaMemPrefetchAsync(h[fld],  gridsize * sizeof(letype), 0, NULL);
		cudaMemPrefetchAsync(hd[fld], gridsize * sizeof(letype), 0, NULL);

		evolveGWd<<<numBlocksLapl, threadsPerBlockLapl>>>(h[fld], hd[fld], EMT[fld], dtime, hmean_acc[gwfld][fld]);
		gpuErrchk(cudaPeekAtLastError());
	}
}

// leap frog: adjust field values by their derivative
__global__ void leapFrog(letype* __restrict__ f, const letype* __restrict__ fd, const letype dtime)
{
	//coordinates on the grid
	const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

   	f[x] += fd[x] * dtime;
}

// leap frog: adjust field values by their derivative and account for the mean value
__global__ void leapFrogMean(letype* __restrict__ f, const letype* __restrict__ fd, const letype dtime, const letype meanAcc)
{
	//coordinates on the grid
	const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

   	f[x] += (fd[x] + meanAcc) * dtime;
}

// function to start leapFrogMean kernel
void startLeapFrogMean(int gwfld, int amt, letype** field, letype** fieldd, letype dtime)
{
	const int blockSize = 16 * tileSize;
	const int numBlocks = gridsize / blockSize;

	for(int fld = 0; fld < amt; fld++)
	{
		cudaMemPrefetchAsync(field[fld],  gridsize * sizeof(letype), 0, NULL);
		cudaMemPrefetchAsync(fieldd[fld], gridsize * sizeof(letype), 0, NULL);

		leapFrogMean<<<numBlocks, blockSize>>>(field[fld], fieldd[fld], dtime, hdmean_acc[gwfld][fld]);
		gpuErrchk(cudaPeekAtLastError());
	}
}

// function to start leapFrog kernel
void startLeapFrog(int amt, letype** field, letype** fieldd, letype dtime)
{
	const int blockSize = 16 * tileSize;
	const int numBlocks = gridsize / blockSize;

	for(int fld = 0; fld < amt; fld++)
	{
		cudaMemPrefetchAsync(field[fld],  gridsize * sizeof(letype), 0, NULL);
		cudaMemPrefetchAsync(fieldd[fld], gridsize * sizeof(letype), 0, NULL);

		leapFrog<<<numBlocks, blockSize>>>(field[fld], fieldd[fld], dtime);
		gpuErrchk(cudaPeekAtLastError());
	}
}

// calculates the mean potential energy
template <unsigned int blockSize>
__global__ void cu_pot_reduce(const int term, letype* __restrict__ reduct, const size_t array_len)
{
    extern volatile __shared__ letype sdata[];
    size_t  tid        = threadIdx.x,
            gridSize   = blockSize * gridDim.x,
            i          = blockIdx.x * blockSize + tid;
    sdata[tid] = 0;
    while (i < array_len)
    {
		sdata[tid] += cu_singlePotEnergy(term, d_pot_params, d_f, i);
        i += gridSize; 
    }

    __syncthreads();
    if (blockSize >= 512)
        { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)
        { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)
        { if (tid <  64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
    if (tid < 32)
        { if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
          if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
          if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
          if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
          if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
          if (blockSize >= 2)  sdata[tid] += sdata[tid + 1]; }
    if (tid == 0) reduct[blockIdx.x] = sdata[0];
}

// function to start the kernel to calculate the mean potential energy
void meanPot2(letype** f)
{
	gpuErrchk(cudaMemcpyToSymbolAsync(d_pot_params, pot_params, num_pot_params * sizeof(letype), 0, cudaMemcpyHostToDevice, 0));
	gpuErrchk(cudaPeekAtLastError());

	const int blockSize = 8 * tileSize;
	const int gridSize = gridsize / blockSize;

	int smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(letype) : blockSize * sizeof(letype);


	static cudaMemGC<letype> temp(gridSize);
	cudaMemPrefetchAsync(temp.data, gridSize*sizeof(letype), 0, NULL);
	gpuErrchk(cudaPeekAtLastError());

	for(int fld = 0; fld < nflds; fld++)
	{
		cudaMemPrefetchAsync(f[fld],  gridsize*sizeof(letype), 0, NULL);
	}

	for(int term = 0; term < num_potential_terms; term++)
	{
		cu_pot_reduce<blockSize><<<gridSize, blockSize, smemSize>>>(term, temp.data, gridsize);
		gpuErrchk(cudaPeekAtLastError());

		//reduce(temp.data, &curr_potEnergy[term], gridSize);
		cu_reduce<blockSize><<<1, blockSize, smemSize>>>(temp.data, &curr_potEnergy[term], gridSize);
		gpuErrchk(cudaPeekAtLastError());
	}
}

// calcualtes the mean value for all gw fields
void calchmean()
{
	for(int fld = 0; fld < nflds; fld++)
	{
		for(int i = 0; i < 6; i++)
		{
			reduce<gridsize>(h[fld][i], &hmean[fld][i]);
			gpuErrchk(cudaPeekAtLastError());
		}
	}
}

// calculates the mean derivative value for all gw fields
void calchdmean()
{
	for(int fld = 0; fld < nflds; fld++)
	{
		for(int i = 0; i < 6; i++)
		{
			reduce<gridsize>(hd[fld][i], &hdmean[fld][i]);
			gpuErrchk(cudaPeekAtLastError());
		}
	}
}

// substract the mean value of gw fields
__global__ void adjusthwithmean(letype* __restrict__ h, letype mean, double* __restrict__ acc)
{
	const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

	//printf("%d %d %d\n", blockDim.x, blockDim.y, blockDim.z);

   	h[x] -= mean / gridsize;

   	if(x == 0)
   		*acc += mean / gridsize;
}

// start kernel to adjust value of gw fields by their mean value
void startadjusth()
{
	const int blockSize = 16 * tileSize;
	const int numBlocks = gridsize / blockSize;

	for(int fld = 0; fld < nflds; fld++)
	{
		for(int i = 0; i < 6; i++)
		{
			cudaMemPrefetchAsync(h[fld][i],  gridsize*sizeof(letype), 0, NULL);
			adjusthwithmean<<<numBlocks, blockSize>>>(h[fld][i], hmean[fld][i], &hmean_acc[fld][i]);
		}
		gpuErrchk(cudaPeekAtLastError());
	}
}

// substract the mean value of gw derivative fields
__global__ void adjusthdwithmean(letype* __restrict__ h, letype mean, double* __restrict__ acc)
{
	const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

	//printf("%d %d %d\n", blockDim.x, blockDim.y, blockDim.z);

   	h[x] -= mean / gridsize;

   	if(x == 0)
   		*acc += mean / gridsize;
}

// start kernel to adjust value of gw derivative fields by their mean value
void startadjusthd()
{
	const int blockSize = 16 * tileSize;
	const int numBlocks = gridsize / blockSize;

	for(int fld = 0; fld < nflds; fld++)
	{
		for(int i = 0; i < 6; i++)
		{
			cudaMemPrefetchAsync(hd[fld][i],  gridsize*sizeof(letype), 0, NULL);
			adjusthdwithmean<<<numBlocks, blockSize>>>(hd[fld][i], hdmean[fld][i], &hdmean_acc[fld][i]);
		}
		gpuErrchk(cudaPeekAtLastError());
	}
}