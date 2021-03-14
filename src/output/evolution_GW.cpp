#pragma once

#include <math.h>
#include <omp.h>

#include "src/latticeeasy.cuh"
#include "src/utils/stencil.h"

template <typename T>
void calc_GW_spectrum(FILE* file, symmTensor<field<T, true>>& TT)
{
	field<double, true> drhoGW = field<double, true>();

	double kabs3_norm = pow(2.0 * M_PI / (double)L, 3.0);//  * M_PI / pow(L, 3.0);

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
			}
		drhoGW(k, l, m) *= 1; // adjusted GW binning
	}

	std::vector<double> bins;
	drhoGW.bin(bins);

	auto it = std::max_element(bins.begin(), bins.end());
	double max = *it;

	//output time, scale factor and Hubble parameter (the latter is already converted to physical units)
	fprintf(file, "%f %f %e ", t, a, ad * rescale_B * pow(a, rescale_s - 1.0));

    for(const auto& v : bins)
      fprintf(file, "%.6e ", v);

    fprintf(file, "\n");
    fflush(file);
}
