#pragma once

#include "variables/parameters.cuh"
#include <stdio.h>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//garbage collected cuda memory
template <typename T>
class cudaMemGC
{
public:
	cudaMemGC(size_t size)
	{
		gpuErrchk(cudaMallocManaged(&data, size * sizeof(T)));
		cudaMemPrefetchAsync(data,  size * sizeof(T), 0, NULL);
	}

	~cudaMemGC()
	{
		//printf("DEAD\n");
		gpuErrchk(cudaFree(data));
	}

	T* data;
};

//sets the value of f at the position pos to 10.0f
__global__ void addPerturbation(letype* f, const dim3 pos);

//calculates the mean value of f
letype calcMeanFrom2D(letype* f);

__global__ void calcMean2D(letype* f, letype* mean2D);
__global__ void calcMean1D(letype* f, letype* mean1D);

/*template <size_t length>
void reduce(const letype* data, letype* res);*/



void prefetchAsyncFields(letype** f, letype** fd, int destDevice);
void prefetchAsyncGWFields(letype*** h, letype*** hd, letype*** EMT, int destDevice);

// taken from here https://forums.developer.nvidia.com/t/understanding-and-adjusting-mark-harriss-array-reduction/64271/9
template <unsigned int blockSize>
__global__ void cu_reduce(const letype* __restrict__ array_in, letype* __restrict__ reduct, const size_t array_len)
{
    extern volatile __shared__ letype sdata[];
    size_t  tid        = threadIdx.x,
            gridSize   = blockSize * gridDim.x,
            i          = blockIdx.x * blockSize + tid;
    sdata[tid] = 0;
    while (i < array_len)
    { 
    	sdata[tid] += array_in[i];
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

template <size_t length>
void reduce(const letype* data, letype* res)
{
	const int blockSize = tileSize;
	const int gridSize = length / blockSize;

	int smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(letype) : blockSize * sizeof(letype);

	static cudaMemGC<letype> meantemp(gridSize);
	cudaMemPrefetchAsync(meantemp.data, gridSize*sizeof(letype), 0, NULL);
	
	gpuErrchk(cudaPeekAtLastError());
	cudaMemPrefetchAsync(data, length * sizeof(letype), 0, NULL);
	gpuErrchk(cudaPeekAtLastError());
	cudaMemPrefetchAsync(res, 1 * sizeof(letype), 0, NULL);
	gpuErrchk(cudaPeekAtLastError());

	cu_reduce<blockSize><<<gridSize, blockSize, smemSize>>>(data, meantemp.data, length);
	gpuErrchk(cudaPeekAtLastError());

	cu_reduce<blockSize><<<1, blockSize, smemSize>>>(meantemp.data, res, gridSize);
	gpuErrchk(cudaPeekAtLastError());
}

// proper mod function (only works if a >= -b)
__host__ __device__ inline int mod(int a, int b) 
{ 
	return (a + b) % b; 
}


class Position
{
public:
	Position(int x, int y, int z)
	:
	x(x),
	y(y),
	z(z)
	{}

	int x;
	int y;
	int z;
};

//calculates the array index from the grid position
__host__ __device__ inline int index(int x, int y, int z)
{
	return (z * N + y) * N + x; 
}

__host__ __device__ inline int indexFT(int x, int y, int z)
{
	return (z * (N + 1) + y) * (N + 1) + x; //TODO make order of indexing better
}

__host__ __device__ inline int index(int x, int y)
{
	return y * N + x; 
}

__host__ __device__ inline int index(int x)
{
	return x; 
}

//calculate tensor index. i should be larger than j!
__host__ __device__ inline int indexTensor(int i, int j)
{
	return i + 3 * j - j * (j + 1) / 2;
}

//calculates the array index for the stencil from the relative grid position
__host__ __device__ inline int stencilIndex(int x, int y, int z)
{
	return abs(x - isoHaloSize) + abs(y - isoHaloSize) + abs(z - isoHaloSize);
}


__device__ inline letype cu_pow2(const letype x)
{
	return x * x;
}

__device__ inline letype cu_pow3(const letype x)
{
	return x * x * x;
}

__device__ inline letype cu_pow4(const letype x)
{
	return cu_pow2(x) * cu_pow2(x);
}

__device__ inline letype cu_pow5(const letype x)
{
	return cu_pow4(x) * x;
}

__device__ inline letype cu_pow6(const letype x)
{
	return cu_pow3(x) * cu_pow3(x);
}

__device__ inline letype cu_pow(const letype x, const letype y)
{
#ifdef USE_FLOAT
	return powf(x, y);
#else
	return pow(x, y);
#endif
}

__device__ inline letype cu_abs(const letype x)
{
#ifdef USE_FLOAT
	return fabsf(x);
#else
	return fabs(x);
#endif
}

__device__ inline letype cu_sin(const letype x)
{
#ifdef USE_FLOAT
	return sinf(x);
#else
	return sin(x);
#endif
}

// iso C
__constant__ const letype stencil[] = {-64.0/15.0, 7.0/15.0, 1.0/10.0, 1.0/30.0};

// my personal stencil :)
//__constant__ const double stencil[] = {-177.0/20.0, 52.0/25.0, -53.0/200.0, -1.0/600.0, 49.0/1200.0, 0.0, -83.0/2400.0};
//__constant__ const double stencil_inv[] = {-20.0/177.0, 25.0/52.0, -200.0/53.0, -600.0, 1200.0/49.0, INFINITY, -2400.0/83.0};

// second order nearest neighborh
//__constant__ const float stencil[] = {-6.0, 1.0, 0.0, 0.0};

/*
// 6th order
const letype cfd_stencil[] = {-1.0/60.0, 3.0/20.0, -3.0/4.0, 0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
__constant__ const letype cu_cfd_stencil[] = {-1.0/60.0, 3.0/20.0, -3.0/4.0, 0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
__constant__ const double cu_cfd_stencil_d[] = {-1.0/60.0, 3.0/20.0, -3.0/4.0, 0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
*/



// 4th order
const letype cfd_stencil[] = {1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0};
__constant__ const letype cu_cfd_stencil[] = {1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0};
__constant__ const double cu_cfd_stencil_d[] = {1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0};


/*
// 10th order
const letype cfd_stencil[] = {-1.0/1260.0, 5.0/504.0, -5.0/84.0, 5.0/21.0, -5.0/6.0, 0.0, 5.0/6.0, -5.0/21.0, 5.0/84.0, -5.0/504.0, 1.0/1260.0};
__constant__ const letype cu_cfd_stencil[] = {-1.0/1260.0, 5.0/504.0, -5.0/84.0, 5.0/21.0, -5.0/6.0, 0.0, 5.0/6.0, -5.0/21.0, 5.0/84.0, -5.0/504.0, 1.0/1260.0};
__constant__ const double cu_cfd_stencil_d[] = {-1.0/1260.0, 5.0/504.0, -5.0/84.0, 5.0/21.0, -5.0/6.0, 0.0, 5.0/6.0, -5.0/21.0, 5.0/84.0, -5.0/504.0, 1.0/1260.0};
*/

/*
// 8th order
const letype cfd_stencil[] = {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0};
__constant__ const letype cu_cfd_stencil[] = {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0};
__constant__ const double cu_cfd_stencil_d[] = {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0};
*/

// test stencil for lapl
//__constant__ const letype stencil[] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};

//__constant__ const letype stencil_test[] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};

//__constant__ const letype cu_cfd_stencil_inv[] = {280.0, -105.0/4.0, 5.0, -5.0/4.0, 0.0, 5.0/4.0, -5.0, 105.0/4.0, -280.0};

/*// 2nd order
const letype cfd_stencil[] = {-0.5, 0.0, 0.5};
__constant__ const letype cu_cfd_stencil[] = {-0.5, 0.0, 0.5};
__constant__ const double cu_cfd_stencil_d[] = {-0.5, 0.0, 0.5};*/