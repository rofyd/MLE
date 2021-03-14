#pragma once

/*
This file contains the global variable declarations, function declarations, 
and some definitions used in many of the routines. The global variables are 
defined in the file latticeeasy.cpp.
*/

#ifndef _LATTICEEASYHEADER_
#define _LATTICEEASYHEADER_

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <float.h>
#include <cuda.h>

#include "src/utils/utils.cuh"
#include "src/utils/symmTensor.cuh"

/////////////////////////////////INCLUDE ADJUSTABLE PARAMETERS///////////////////
#include "variables/parameters.cuh"
#include "src/integrator/integrator.cuh"

const double pi = (double)(2.*asin(1.));
inline double pw2(double x) {return x*x;} // Useful macro for squaring letypes

/////////////////////////////////GLOBAL DYNAMIC VARIABLES////////////////////////
extern double t,t0; // Current time and initial time (t0=0 unless the run is a continuation of a previous one)
extern double a, ad, ad2, aterm; // Scale factor and its derivatives (aterm is a combination of the others used in the equations of motion)
extern double hubble_init; // Initial value of the Hubble constant
extern int run_number; // 0 for a first run, 1 for a continuation of a "0" run, etc.. Stored in the grid image (see checkpoint() function).
extern int no_initialization; // If this variable is set to 1 by the model file then the fields will not be initialized in the normal way.
extern letype rescaling; // Rescaling for output. This is left as 1 unless the model file modifies it.
extern char ext_[500]; // Extension for filenames - set once and used by all functions
extern int nfldsout; // Number of fields to output
extern char mode_[]; // Mode in which to open files, i.e. write ("w") or append ("a+"). Depends on the variable continue_run and on whether a previous grid image was found.

/////////////////////////////////NON-ADJUSTABLE VARIABLES////////////////////////
const double dx = L/(double)N; // Distance between adjacent gridpoints
const double dx_inv = N / (double)L;
const double dx_inv_sq = (N / (double)L) * (N / (double)L);

extern letype* f[nflds]; // pointer to field values
extern letype* fd[nflds]; // pointer to field derivative values

extern letype* h[nflds][6]; // pointer to field values of tensor perturbation
extern letype* hd[nflds][6]; // pointer to field derivative values of tensor perturbation

extern cufft_type*** hdk; // pointer to fourier transform of tensor perturbation
extern fTT_type*** hdkTT; // pointer to TT projection of tensor perturbation in fourier space

extern letype* EMT[nflds][6]; // pointer to energy-momentum tensor (EMT)

extern cufft_type*** EMTk; // pointer to EMT in fourier space
extern fTT_type*** EMTkTT; // pointer to TT projection of EMT in fourier space


/////////////////////////////////DIMENSIONAL SPECIFICATIONS//////////////////////
#if NDIMS == 1
extern letype f[nflds][N], fd[nflds][N]; // Field values and derivatives
#define FIELD(fld) f[fld][index(i, 0, 0)]
#define FIELDD(fld) fd[fld][index(i, 0, 0)]
#define FIELDPOINT(fld, i, j, k) f[fld][index(k, 0, 0)] //This define is used to calculate the slices in output.cpp
#define LOOP for(i = 0; i < N; i++)
#define INDEXLIST int i, ...
#define DECLARE_INDICES int i;


#elif NDIMS == 2
const int gridsize = N*N; // Number of spatial points in the grid
#define FIELD(fld) f[fld][index(i, j, 0)]
#define FIELDD(fld) fd[fld][index(i, j, 0)]
#define FIELDPOINT(fld, i, j, k) f[fld][index(j, k, 0)]
#define LOOP for(i = 0; i < N; i++) for(j = 0; j < N; j++)
#define INDEXLIST int i, int j, ...
#define DECLARE_INDICES int i, j;


#elif NDIMS == 3
const int gridsize = N*N*N; // Number of spatial points in the grid
#define FIELD(fld) f[fld][index(i, j, k)]
#define FIELDD(fld) fd[fld][index(i, j, k)]
#define FIELDPOINT(fld, i, j, k) f[fld][index(i, j, k)]
#define LOOP for(i = 0; i < N; i++) for(j = 0; j < N; j++) for(k = 0; k < N; k++)
#define INDEXLIST int i, int j, int k
#define DECLARE_INDICES int i, j, k;
#endif

/////////////////////////////////INCLUDE SPECIFIC MODEL//////////////////////////
#include "variables/model.cuh"

extern letype* curr_gradientEnergy;//[nflds];
extern letype* curr_potEnergy;//[num_potential_terms];

extern letype* hmean[6]; // mean value of each field of tensor perturbation
extern double* hmean_acc[6]; // accumulated mean value of each field of tensor perturbation

extern letype* hdmean[6]; // mean value of each derivative field of tensor perturbation
extern double* hdmean_acc[6]; // accumulated mean value of each derivative field of tensor perturbation

extern letype dvdf_params[]; // parameter that are used for the calculation of the potential derivative
extern letype pot_params[]; // parameter that are used for the calculation of the potential

// energy momentum tensor in position space
//extern symmTensor<field<letype, false>> EMT_pos;

// energy momentum tensor in momentum space
extern symmTensor<field<cufft_type, true>>* EMT_mom;
extern symmTensor<field<fTT_type, true>>* EMT_mom_TT;

// tensor perturbation in momentum space
//extern symmTensor<field<cufft_type, true>> h_mom;
extern symmTensor<field<cufft_type, true>>* dh_mom;
extern symmTensor<field<fTT_type, true>>* dh_mom_TT;

// binned things
extern symmTensor<std::vector<letype>> bins;

void prepare_emt(letype dt);
void evolve_h(letype dt);
/*
template <typename T>
void calc_GW_spectrum(FILE* file, symmTensor<field<T, true>>& TT);*/

#include "src/output/evolution_GW.cpp"

/////////////////////////////////FUNCTION DECLARATIONS///////////////////////////
// initialize.cpp
void initialize(); // Set initial parameters and field values
// evolution.cpp
letype gradient_energy(int fld); // Calculate the gradient energy, <|Grad(f)|^2>=<-f Lapl(f)>, of a field
void evolve_scale(letype d); // Calculate the scale factor and its derivatives
void evolve_fields(letype d); // Advance the field values and scale factor using the first derivatives
void evolve_derivs(letype d); // Calculate second derivatives of fields and use them to advance first derivatives. Also calls evolve_scale().
// output.cpp
void output_parameters(); // Output information about the run parameters
void save(int force); // Calculate and save quantities (means, variances, spectra, etc.)
// ffteasy.cpp
void fftr1(letype f[], int N, int forward); // Do a Fourier transform of a 1D array of real numbers. Used when NDIMS=1.
void fftrn(letype f[], letype fnyquist[], int ndims, int size[], int forward); // Do a Fourier transform of an ndims dimensional array of real numbers

#endif // End of conditional for definition of _LATTICEEASYHEADER_ macro






