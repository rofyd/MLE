#pragma once

#include "variables/parameters.cuh"
#include "src/utils/utils.cuh"

#include <cuda.h>
#include <cufft.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <iostream>


#ifdef FFT_FLOAT
    using cufft_type = cufftComplex;
    #define REAL2COMPLEX CUFFT_R2C
#else
    using cufft_type = cufftDoubleComplex;
    #define REAL2COMPLEX CUFFT_D2Z
#endif

using fTT_type = cufftDoubleComplex;

template <typename T = letype, bool isFT = false>
class field
{
public:
    field(bool init = true)
    {
        if(!init)
            return;
        
        dim = NDIMS;

        latticeSize = N;

        if(latticeSize % 2 == 0 && isFT)
            latticeSize++;

        numEntries = latticeSize;
        for(int i = 0; i < NDIMS - 1; i++)
            numEntries *= latticeSize;


        //entries = new T[numEntries]();
        cudaMallocManaged(&entries, numEntries * sizeof(T));
    }

    T* getData()
    {
        return entries;
    }

    //initializing the field to be in momentum space
    void init(cufft_type* initVals, size_t length)
    {
        this->fill(initVals, length);
    }

    ~field()
    {
        cudaFree(entries);
        //delete[] entries;
    }

#if NDIMS == 1
    T& operator ()(int k)
    {
        return this->at(0, 0, k);
    }
#elif NDIMS == 2
    T& operator ()(int j, int k)
    {
        return this->at(0, j, k);
    }
#elif NDIMS == 3
    __host__ __device__ T& operator ()(int i, int j, int k)
    {
        return this->at(i, j, k);
    }
#endif

    T& operator ()(int* indices)
    {
        int shift = 0;
        if constexpr(isFT)
            shift = -N/2;

        if constexpr(NDIMS == 1)
            return this->at(shift, shift, indices[0]);
        else if constexpr(NDIMS == 2)
            return this->at(shift, indices[1], indices[0]);
        else if constexpr(NDIMS == 3)
            return this->at(indices[2], indices[1], indices[0]);
    }

    cufft_type* fourierTransform(letype* in, cufft_type* out)
    {

        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //cufft_type* out;
        cufftHandle plan;
        //fftw_plan_with_nthreads(1);

        
        if constexpr(NDIMS == 1)
            cufftPlan1d(&plan, latticeSize, REAL2COMPLEX, 1);
        else if constexpr(NDIMS == 2)
            cufftPlan2d(&plan, latticeSize, latticeSize, REAL2COMPLEX);
        else if constexpr(NDIMS == 3)
        {
           cufftPlan3d(&plan, N, N, N, REAL2COMPLEX);
        }

        gpuErrchk(cudaMemPrefetchAsync(in, N * N * N * sizeof(letype), 0, NULL));
        //gpuErrchk(cudaMemPrefetchAsync(out, (N + 1) * (N + 1) * (N + 1) * sizeof(cufft_type), 0, NULL));


#ifdef FFT_FLOAT
        cufftExecR2C(plan, (cufftReal*)in, out);
        //printf("FLOAT ACTION\n");
#else
        cufftExecD2Z(plan, (cufftDoubleReal*)in, out);
#endif        


        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        //printf("TEST %f\n", out[2].y);
        /*
        field<cufft_type>* fT = new field<cufft_type>();
        fT->init(out, numEntries / latticeSize * ( latticeSize / 2 + 1 ));
        */
        cufftDestroy(plan);        
        //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "FT Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
       
        //fftw_free(out);

        return out;
    }

#if NDIMS == 1
    void printEntries()
    {
        for(int k = 0; k < latticeSize; k++)
        {
            if constexpr(isFT)
            {
                printEntry((*this)(k - N/2));
            }
            else
            {
                printEntry((*this)(k));
            }
        }
        printf("\n");
    }
#elif NDIMS == 2
    void printEntries()
    {
        for(int j = 0; j < latticeSize; j++)
        {
            for(int k = 0; k < latticeSize; k++)
            {
                if constexpr(isFT)
                {
                    printEntry((*this)(j - N/2, k - N/2));
                }
                else
                {
                    printEntry((*this)(j, k));
                }
            }
            printf("\n");
        }
        printf("\n\n");
    }
#elif NDIMS == 3
    void printEntries()
    {
        for(int i = 0; i < latticeSize; i++)
        {
            for(int j = 0; j < latticeSize; j++)
            {
                for(int k = 0; k < latticeSize; k++)
                {
                    if constexpr(isFT)
                    {
                        printEntry((*this)(i - N/2, j - N/2, k - N/2));
                    }
                    else
                    {
                        printEntry((*this)(i, j, k));
                    }
                }
                printf("\n");
            }
            printf("\n\n");
        }
    }
#endif

    // returns |f_k|^2 averaged over the points with n dk < k < (n + 1) dk
    void bin(std::vector<T>& bins)
    {
        size_t binNum = sqrt(NDIMS) * N / 2 + 1;
        bins.resize(binNum);

        std::fill(bins.begin(), bins.end(), 0.0);

        std::vector<size_t> amtInBin;
        amtInBin.resize(binNum);

        std::fill(amtInBin.begin(), amtInBin.end(), 0);

        size_t pos = 0;
        //cufft_type val;

        int i = 0;
        int j = 0;
        int k = 0; 

#if NDIMS > 2
        for(i = -N/2 + 1; i <= N/2; i++)
#endif
#if NDIMS > 1
            for(j = -N/2 + 1; j <= N/2; j++)
#endif
                for(k = -N/2 + 1; k <= N/2; k++)
        {
            pos = sqrt(i*i + j*j + k*k);
            
            bins[pos] += this->at(i, j, k);// * this->at(i, j, k);
            amtInBin[pos]++;
        }

        for(size_t i = 0; i < binNum; i++)
            if(amtInBin[i] > 0)
                bins[i] = bins[i] / (double)amtInBin[i];

/*        for(const auto & val : amtInBin)
            printf("%ld\n", val);*/
    }

    inline void printEntry(letype val)
    {
        printf("%f ", val);
    }

    inline void printEntry(cufft_type val)
    {
        printf("%c%.2f%c%.2fi | ", val.x < 0 ? '\0' : ' ', val.x, val.y < 0.0 ? '\0' : '+',val.y);
    }

private:

    __host__ __device__ inline T& at(int i, int j, int k)
    {
        size_t pos = 0;

        if constexpr(isFT)
        {
            i += N / 2;
            j += N / 2;
            k += N / 2;
        }

        if(i < 0 || j < 0 || k < 0)
            printf("hmmm %d %d %d\n", i, j, k);

        if constexpr(NDIMS < 3)
            i = 0;
        if constexpr(NDIMS < 2)
            j = 0;

        pos = i + latticeSize * j + latticeSize * latticeSize * k;
        //pos = k + latticeSize * j + latticeSize * latticeSize * i;

        //if(pos >= numEntries)
        //    printf("out of bounds %ld\t%ld\n", pos, numEntries); //TODO: delete this line for release
        return entries[pos % numEntries];
    }

    void fill(cufft_type* initVals, size_t length)
    {
        for(size_t i = 0; i < numEntries; i++){
            entries[i].x = 0.0;
            entries[i].y = 0.0;
        }

        //size_t pos = 0;

        /*
        int ia = 0;
        int ja = 0;
        int ka = 0;
        */

/*        letype rescale = pow(N, -NDIMS / 2);

        rescale = pow(N, -1/2); //TODO why this value and not the above one?

        size_t shift = latticeSize - 1 - N/2;*/

        // easy indexing function for access of the data produced by cufft/fftw
        auto indexFT2 = [](int x, int y, int z)
        {
            return (x * N + y) * (N / 2 + 1) + z; 
        };

        int ii = 0;
        int ji = 0;
        int ki = 0;

        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                for(int k = 0; k <= N/2; k++)
        {
            //wrap momenta values to the range of -N/2 + 1, ..., N/2
            ii = i > N/2 ? i - N : i;
            ji = j > N/2 ? j - N : j;
            ki = k > N/2 ? k - N : k;

            this->at(ki, ji, ii).x = initVals[indexFT2(i, j, k)].x;
            this->at(ki, ji, ii).y = initVals[indexFT2(i, j, k)].y;
        }

        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                for(int k = N/2; k < N; k++)
        {
            ii = i > N/2 ? i - N : i;
            ji = j > N/2 ? j - N : j;
            ki = k > N/2 ? k - N : k;

            // we need to do the "? : " because we shouldnt map N/2 -> N/2, instead N/2 already corresponds to the position for the conjugate
            this->at(ki, ji, ii).x =  this->at(k == N/2 ? ki : -ki, j == N/2 ? ji : -ji, i == N/2 ? ii : -ii).x;
            this->at(ki, ji, ii).y = -this->at(k == N/2 ? ki : -ki, j == N/2 ? ji : -ji, i == N/2 ? ii : -ii).y;
        }

        //#pragma omp parallel for collapse(NDIMS)
/*#if NDIMS > 2
        for(int i = 0; i < latticeSize; i++)
        {
#endif
#if NDIMS > 1
            for(int j = 0; j < latticeSize; j++)
            {
#endif
                for(int k = latticeSize / 2; k < latticeSize; k++)
                {
                    int ia = (i + latticeSize / 2) % latticeSize - N/2;
                    int ja = (j + latticeSize / 2) % latticeSize - N/2;
                    int ka = k - N/2; 

                    //printf("%d, %d, %d\n", ia, ja, k);
                    if(N % 2 == 0 && (ia == -N/2 || ja == -N/2))
                    {
                        this->at(ia, ja, ka).x = this->at(ia == -N/2 ? shift : ia, ja == -N/2 ? shift : ja, ka).x;
                        this->at(ia, ja, ka).y = this->at(ia == -N/2 ? shift : ia, ja == -N/2 ? shift : ja, ka).y;  
                    }
                    else
                    {
                        int posi = ia < 0 ? ia + N : ia;
                        int posj = ja < 0 ? ja + N : ja;
                        int posk = ka < 0 ? ka + N : ka;
                    
                        size_t pos2 = posk + (N/2 + 1) * posj + (N/2 + 1)*N * posi;
                        this->at(ia, ja, ka).x =  rescale * initVals[pos2].x;
                        this->at(ia, ja, ka).y = -rescale * initVals[pos2].y; //TODO why is there a minus sign here
                        
                    }
                }
#if NDIMS > 1
            }
#endif
#if NDIMS > 2
        }   
#endif

        //#pragma omp parallel for collapse(NDIMS)
#if NDIMS > 2
        for(int i = 0; i < latticeSize; i++)
        {
#endif
#if NDIMS > 1
            for(int j = 0; j < latticeSize; j++)
            {
#endif
                for(int k = 0; k < latticeSize / 2; k++)
                {
                    int ia = (i + latticeSize/2) % latticeSize - N/2;
                    int ja = (j + latticeSize/2) % latticeSize - N/2;
                    int ka = k - N/2; //TODO set values for ia, ja, ka for 1d and 2d

                    this->at(ia, ja, ka).x =  this->at( -ia, -ja, -ka).x;
                    this->at(ia, ja, ka).y = -this->at( -ia, -ja, -ka).y;
                }
#if NDIMS > 1
            }
#endif
#if NDIMS > 2
        }
#endif*/
    }

    size_t dim;
    size_t latticeSize;

    size_t numEntries;
    T* entries = nullptr;
};


template <typename T>
class symmTensor
{
public:
    symmTensor(bool init = true)
    {
        if(!init)
            return;

        dim = NDIMS;
        indEntries = dim * (dim + 1) / 2;
        latticeSize = N;

        totalLatticeSize = latticeSize;
        for(int i = 0; i < NDIMS - 1; i++)
            totalLatticeSize *= latticeSize;

        entries = new T[indEntries]();
    }

    ~symmTensor()
    {
        if(out != nullptr)
            cudaFree(out);
        //cudaFree(entries);
        delete[] entries;
    }


    T** getEntries()
    {
        return &entries;
    }

    void projTT(symmTensor<field<cufft_type, true>>& target)
    {
        const double dx = (double)L / (double)N; //TODO make this right deltax -> dx

        auto keff = [dx](int i, int nVals[]){
            double res = 0.0;

            #pragma unroll
            for(int k = 1; k <= cfdHaloSize; k++)
                res += cfd_stencil[cfdHaloSize + k] * sin(2.0 * M_PI * k * nVals[i] / (double)N);

            return 2.0 * res / dx;
        };

        auto keffabs2 = [&keff](int nVals[]){
            double res = 0.0;
            for(int i = 0; i < NDIMS; i++)
                res += keff(i, nVals) * keff(i, nVals);
            
            return res;
        };

        auto P0 = [&keff, &keffabs2](int i, int j, int nVals[]){
            if(nVals[i] == 0 || nVals[i] == N/2 || nVals[i] == -N/2)
                if(nVals[j] == 0 || nVals[j] == N/2 || nVals[j] == -N/2)
                {
                    if(i == j)
                        return (letype)1.0;
                    return (letype)0.0;
                }

            double res = (letype)0.0;
            if(i == j)
                res += (letype)1.0;

            res -= keff(i, nVals) * keff(j, nVals) / keffabs2(nVals);
            return res;
        };

        auto lam0 = [&P0](int i, int j, int l, int m, int nVals[]){
            return P0(i, l, nVals) * P0(j, m, nVals) - 0.5 * P0(i, j, nVals) * P0(l, m, nVals);
        };

        cufft_type res;

            //loop over space time indices for the final tensor

        //we have to first loop over the grid
        //#pragma omp parallel for collapse(NDIMS)
        for(int x = -N/2; x < N/2 + 1; x++)
#if NDIMS > 1
            for(int y = -N/2; y < N/2 + 1; y++)
#endif            
#if NDIMS == 3
                for(int z = -N/2; z < N/2 + 1; z++)
#endif
        {
            int nVals[3] = {0};
            nVals[0] = x;
#if NDIMS > 1
            nVals[1] = y;
#endif
#if NDIMS == 3
            nVals[2] = z;
#endif
            
            for(int i = 0; i < NDIMS; i++)
            {
                for(int j = 0; j <= i; j++)
                {
                    target(i, j)(nVals).x = 0.0;
                    target(i, j)(nVals).y = 0.0;
                    //loop over space time indices to calculate the tensor product
                    //#pragma omp parallel for
                    for(int l = 0; l < NDIMS; l++)
                    {
                        for(int m = 0; m < NDIMS; m++)
                        {
                            target(i, j)(nVals).x += lam0(i, j, l, m, nVals) * (*this)(l, m)(nVals).x;
                            target(i, j)(nVals).y += lam0(i, j, l, m, nVals) * (*this)(l, m)(nVals).y;
                        }
                    }
                }
            }
        }
    }

    /*
    __host__ void fourierTransform(symmTensor<field<cufft_type, true>>& target)
    {
        size_t FT_size = latticeSize;
        if(latticeSize % 2 == 0)
            FT_size++;

        size_t FT_totalSize = FT_size;
        for(int i = 0; i < NDIMS - 1; i++)
            FT_totalSize *= FT_size;

        if(out == nullptr)
            cudaMallocManaged(&out, sizeof(cufft_type) * (FT_totalSize / FT_size * ( FT_size / 2 + 1 )));
        
        //symmTensor<field<cufft_type, true>> fT = symmTensor<field<cufft_type, true>>();
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                cufft_type* fTvalues = (*this)(i, j).fourierTransform(out);
                //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                target(i, j).init(fTvalues, (totalLatticeSize / latticeSize * ( latticeSize / 2 + 1 )));
                       
                //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                //std::cout << "FILL Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
            }
        }
        //return fT;
    }
    */


    void fourierTransform(letype** target)
    {
        size_t FT_size = latticeSize;
        if(latticeSize % 2 == 0)
            FT_size++;

        size_t FT_totalSize = FT_size;
        for(int i = 0; i < NDIMS - 1; i++)
            FT_totalSize *= FT_size;

        if(out == nullptr)
            cudaMallocManaged(&out, sizeof(cufft_type) * (FT_totalSize / FT_size * ( FT_size / 2 + 1 )));
        cudaDeviceSynchronize();


        //symmTensor<field<cufft_type, true>> fT = symmTensor<field<cufft_type, true>>();
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                cufft_type* fTvalues = (*this)(i, j).fourierTransform(target[indexTensor(i, j)], out);
                
                cudaDeviceSynchronize();
                //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                
                (*this)(i, j).init(fTvalues, (totalLatticeSize / latticeSize * ( latticeSize / 2 + 1 )));
                       
                //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                //std::cout << "FILL Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
            }
        }
        //return fT;
    }

    void bin(symmTensor<std::vector<letype>>& target)
    {
        //symmTensor<std::vector<letype>> binned = symmTensor<std::vector<letype>>();

        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j <=i; j++)
                (*this)(i, j).bin(target(i, j));
        }

        //return binned;
    }

    __host__ __device__ T& operator ()(int i, int j)
    {
        int temp = i;
        if(j > i)
        {
            i = j;
            j = temp;
        }
        int pos = i + dim * j;
        pos -= j * (j + 1) / 2;
        if(pos >= indEntries)
            return entries[0];
        return entries[pos];
    }

    void printInfo()
    {
        printf("Space-Time Dimension: %ld\n", dim);
        printf("Independent Entries: %ld\n", indEntries);
        printf("Lattice Points per space Dimension: %ld\n", latticeSize);
        printf("Total Lattice Points: %ld\n", totalLatticeSize);
    }

    void printEntries()
    {
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                printf("### (i, j) = (%d, %d) ###\n", i, j);
                (*this)(i, j).printEntries();
            }
        }
    }

private:
    size_t dim;
    size_t indEntries;
    size_t latticeSize;
    size_t totalLatticeSize;

    cufft_type* out = nullptr;

    T* entries;
};
