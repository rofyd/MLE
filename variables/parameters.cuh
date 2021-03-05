#pragma once

#include <vector>
/*
This file contains all the adjustable parameters for running the latticeeasy program. Once a particular model has been defined all runs can be done by changing the parameters in this file and recompiling. See the documentation for how to define new models.
*/


//set the type of precision you want to use
#define USE_FLOAT
//#define USE_DOUBLE


#ifdef USE_FLOAT
  using letype = float;
  #define FFT_FLOAT
#else
  using letype = double;
#endif


const double pmass = 9.97356e-7;//5.e-6; //mass of the inflaton
const double param_1st = 0.015;
const double param_2nd = 0.006;

const double res_r = 3.0 / 2.0;

// CUDA parameter for iterations
__constant__ const int tileSize = 16;
__constant__ const int isoHaloSize = 1; //halo size for isotropic stencil
__constant__ const int cfdHaloSize = 1; //halo size for central finite difference, has to be at least isoHaloSize and derivative should have the same order
__constant__ const int haloSize = cfdHaloSize; //halo size

__constant__ const int numThreads = tileSize + 2*haloSize;

__constant__ const int tileSizeGW = 16;
__constant__ const int numThreadsGWnum = tileSizeGW + 2*isoHaloSize;

// Adjustable run parameters
#define NDIMS 3
const int N = 512; // Number of points along each edge of the cubical lattice
const int nflds = 1;  //Number of fields
const int gwnflds = nflds == 1 ? 1 : nflds + 1;
const double L = 300; //19.05// Size of box (i.e. length of each edge) in rescaled distance units
const letype dt = 0.1; // Size of time step
const double tf = 10e10;//0.2; // Final time
const double af = 13.45; // Final scale factor
const std::vector<double> a_saves = {1.7, 2.5, 3.35, 4.0, 6.7};//{1.001, 1.6, 2.3, 3.0, 3.35, 4.0, 5.0, 5.4, 5.8, 6.3, 6.7, 7.3, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.45}; // scale factors at which to save quantities

const int seed = 642795; // Random number seed. Should be a positive integer
const double initfield[] = {1.0}; // Initial values of the fields in program units. All nonspecified values are taken to be zero.
const double initderivs[] = {0.719724};//  Initial values of the field derivatives in program units. All nonspecified values are taken to be zero.
const int expansion = 2; // Whether to use no expansion (0), power-law expansion (1), or self-consistent expansion (2)
const letype expansion_power = .5; // Power of t in power law expansion. Only used when expansion=1.. Set to .5 for radiation or .67 for matter domination.
const letype kcutoff = 0; // Momentum for initial lowpass filter. Set to 0 to not filter

// If and how to continue previous runs.
// If no grid image is available in the run directory then a new run will be started irrespective of continue_run.
// 0=Start a new run at t=0. 1=Continue old run, appending new data to old data files. 2=Continue old run, creating new data files for all output. (Old ones will not in general be overwritten.)
const int continue_run = 0;

// Variables controlling output
const int noutput_flds = 1; // Number of fields to output information about. All fields will be output nflds if noutput=0 or noutput>nflds
const char alt_extension[] = ""; // Optional alternative extension for output files (default is the run number)
const int print_interval = 1; // Interval in seconds between successive outputs of time
const int screen_updates = 1; // Set to 1 for time to be periodically output to screen (0 otherwise)
const letype checkpoint_interval = tf;//10; // How often to output a grid image and perform infrequent calculations (see list below). Only done at end if checkpoint_interval=0.
const int noutput_times = 100000000;//5000;//(int)(tf / checkpoint_interval); // Number of times at which to calculate and save output variables
const letype store_lattice_times[] = {0.}; // An optional list of times at which to close the grid image file and open a new one.

// Gravitational waves
const int sgw = 0; // Output gravitational waves spectrum
const int sast = 0; // Output anisotropic stress tensor
const int sgwnonTT = 0; // Output gw spectrum without TT projecting
const int sastnonTT = 0; // Output AST without TT projecting

// Output oscillon data
const int soscillon = 1;
const float overdensity = 5.0;

// The variables s<name> control what will be saved (1=save, 0=don't save)
const int smeansvars = 1; // Output means and variances. This function is also used to check for exponential instability, so it is generally wise to leave it on.
const int sexpansion = 1; // Output scale factor, Hubble constant, and a'' (This is ignored except in self-consistent expansion.)
const int smodel = 0; // Call model-specific output functions. This must be on for the model file to set the rescaling for the other output functions.

// The following calculations are performed at intervals given by checkpoint_interval
const letype t_start_output = 0.; // Time to start doing these calculations. This can be reset for any individual calculation.
const int scheckpoint = 0; // Save an image of the grid
  const letype tcheckpoint = t_start_output;

// Output power spectra
const int sspectra = 1;
  const letype tspectra = t_start_output;

// Output components of energy density
const int senergy = 1;
  const letype tenergy = t_start_output;

// Output histograms of fields using nbins as the number of bins
const int shistograms = 0;
  const letype thistograms = t_start_output;
  const int nbins = 256; // Number of bins
  const letype histogram_min = 0., histogram_max = 0.; // Upper and lower limits of the histograms. To use all current field values set these two to be equal.

// Output two dimensional histograms of fields
const int shistograms2d = 0;
  const letype thistograms2d = t_start_output;
  const int nbins2d = 10, nbins0 = nbins2d, nbins1 = nbins2d; // Number of bins in each field direction
  const letype histogram2d_min = 0., histogram2d_max = 0.; // Upper and lower limits of the histograms. To use all current field values set these two to be equal.
  const int hist2dflds[] = {0, 1}; // Pairs of fields to be evaluated. This array should always contain an even number of integers.

// Output the field values on a slice through the lattice
const int sslices = 0;
  const letype tslices = t_start_output;
  const int slicedim = 1; // Dimensions of slice to be output. (If slicedim>=NDIMS the whole lattice will be output.) Warning: If slicedim=3 the resulting file may be very large.
  const int slicelength = N, sliceskip = 1; // The slices will use every <sliceskip> point up to a total of <slicelength>.  Set length=N and skip=1 to output all points in the slice.
  const int sliceaverage = 1; // If sliceskip>1 and sliceaverage=1 the slices will contain averages over all the field values in between sampled points.  Otherwise they will just use the sampled values.

