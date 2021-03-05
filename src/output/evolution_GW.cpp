#pragma once

#include <math.h>
#include <omp.h>

#include "src/latticeeasy.cuh"
#include "src/utils/stencil.h"
//#include "symmTensor.h"

/*
void prepare_emt(letype dt)
{
	auto partial = [](int i, int k, int l, int m){
		letype resize = 1 / dx;
		return resize * Stencil::fd_central<letype, 0, NDIMS, N>(f[0], Position(k, l, m), i);
	};
	
	//#pragma omp parallel for collapse(4)

	#pragma omp parallel for collapse(NDIMS + 1)
#if NDIMS > 2
		for(int k = 0; k < N; k++)
#endif
#if NDIMS > 1	
			for(int l = 0; l < N; l++)
#endif
				for(int m = 0; m < N; m++)
		{
		for(int i = 0; i < NDIMS; i++)
			for(int j = 0; j <= i; j++)
		{
#if NDIMS == 1
			EMT_pos(i, j)(m)
#elif NDIMS == 2
			EMT_pos(i, j)(l, m)
#elif NDIMS == 3
			EMT_pos(i, j)(k, l, m)
#endif
			= partial(i, k, l, m) * partial(j, k, l, m);
		}
	}
}

void evolve_h(letype dt)
{
	letype kabs2_norm = pow(2.0 * M_PI / L, 2.0);
	letype rescaled_a = pow(a, -2.0 * rescale_s - 2.0);

	#pragma omp parallel for collapse(NDIMS + 1)
#if NDIMS > 2
		for(int k = -N/2; k <= N/2; k++)
#endif
#if NDIMS > 1
			for(int l = -N/2; l <= N/2; l++)
#endif
				for(int m = -N/2; m <= N/2; m++)
		{

			for(int i = 0; i < NDIMS; i++)
				for(int j = 0; j <= i; j++)
			{
				letype kabs2 = (l*l + m*m + k*k) * kabs2_norm;
#if NDIMS == 2
				dh_mom(i, j)(l, m).x += dt * (2.0 / rescale_A * rescaled_a * pow(a, -rescale_r) * EMT_mom(i, j)(l, m).x - (rescaled_a * kabs2 - aterm) * h_mom(i, j)(l, m).x);
				dh_mom(i, j)(l, m).y += dt * (2.0 / rescale_A * rescaled_a * pow(a, -rescale_r) * EMT_mom(i, j)(l, m).y - (rescaled_a * kabs2 - aterm) * h_mom(i, j)(l, m).y);
				h_mom(i, j)(l, m).x += dt * dh_mom(i, j)(l, m).x;
				h_mom(i, j)(l, m).y += dt * dh_mom(i, j)(l, m).y;
#elif NDIMS == 3
				dh_mom(i, j)(k, l, m).x += dt * (2.0 / rescale_A * rescaled_a * pow(a, -rescale_r) * EMT_mom(i, j)(k, l, m).x - (rescaled_a * kabs2 - aterm) * h_mom(i, j)(k, l, m).x);
				dh_mom(i, j)(k, l, m).y += dt * (2.0 / rescale_A * rescaled_a * pow(a, -rescale_r) * EMT_mom(i, j)(k, l, m).y - (rescaled_a * kabs2 - aterm) * h_mom(i, j)(k, l, m).y);
				h_mom(i, j)(k, l, m).x += dt * dh_mom(i, j)(k, l, m).x;
				h_mom(i, j)(k, l, m).y += dt * dh_mom(i, j)(k, l, m).y;
#endif
			}
		}
}
*/

template <typename T>
void calc_GW_spectrum(FILE* file, symmTensor<field<T, true>>& TT)
{
	field<double, true> drhoGW = field<double, true>();

	double kabs3_norm = pow(2.0 * M_PI / (double)L, 3.0);//  * M_PI / pow(L, 3.0);





	//#pragma omp parallel for collapse(NDIMS)
#if NDIMS > 2
	for(int k = -N/2 + 1; k <= N/2; k++)
#endif
#if NDIMS > 1
		for(int l = -N/2 + 1; l <= N/2; l++)
#endif
			for(int m = -N/2 + 1; m <= N/2; m++)
	{
		drhoGW(k, l, m) = 0.0;
		double kabs3 = pow(l*l + m*m + k*k, 1.5) * kabs3_norm;

		for(int i = 0; i < NDIMS; i++)
			for(int j = 0; j < NDIMS; j++)
			{
				drhoGW(k, l, m) += pow(TT(i, j)(k, l, m).x, 2.0) + pow(TT(i, j)(k, l, m).y, 2.0);

/*				if(abs(TT(i, j)(k, l, m).x) > 1)
					printf("its arrived %d %d %d\n", k, l, m);*/
			}
		drhoGW(k, l, m) *= 1;//kabs3; // adjusted GW binning
	}

	//printf("\t%f\n", pow((f[0][0][2] - f[0][0][0]) *0.5 /dx, 2));
	//printf("\t%f %f\n", dh_mom(0, 0)(1, 1)[0],dh_mom(0, 0)(1, 1)[1]);
	
	std::vector<double> bins;
	drhoGW.bin(bins);

	auto it = std::max_element(bins.begin(), bins.end());
	double max = *it;

	if(it == bins.end())
		printf("HOLYLYLALAS\n");

	//printf("   %ld\t%.15f %.15f\n", std::distance(bins.begin(), it), max, bins[1]);
	/*
	if(max > 0.00001){
	printf("\n%f\n\n", max);
    */

	//output time, scale factor and Hubble parameter (the latter is already converted to physical units)
	fprintf(file, "%f %f %e ", t, a, ad * rescale_B * pow(a, rescale_s - 1.0));

    for(const auto& v : bins)
      fprintf(file, "%.6e ", v);

    fprintf(file, "\n");
    fflush(file);
}
