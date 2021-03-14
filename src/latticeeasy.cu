/*
LATTICEEASY consists of the C++ files ``latticeeasy.cpp,''
``initialize.cpp,'' ``evolution.cpp,'' ``output.cpp,''
``latticeeasy.h,'' ``parameters.h,''. (The distribution also includes
the file ffteasy.cpp but this file is distributed separately and
therefore not considered part of the LATTICEEASY distribution in what
follows.) LATTICEEASY is free. We are not in any way, shape, or form
expecting to make money off of these routines. We wrote them for the
sake of doing good science and we're putting them out on the Internet
in case other people might find them useful. Feel free to download
them, incorporate them into your code, modify them, translate the
comment lines into Swahili, or whatever else you want. What we do want
is the following:
1) Leave this notice (i.e. this entire paragraph beginning with
``LATTICEEASY consists of...'' and ending with our email addresses) in
with the code wherever you put it. Even if you're just using it
in-house in your department, business, or wherever else we would like
these credits to remain with it. This is partly so that people can...
2) Give us feedback. Did LATTICEEASY work great for you and help
your work?  Did you hate it? Did you find a way to improve it, or
translate it into another programming language? Whatever the case
might be, we would love to hear about it. Please let us know at the
email address below.
3) Finally, insofar as we have the legal right to do so we forbid
you to make money off of this code without our consent. In other words
if you want to publish these functions in a book or bundle them into
commercial software or anything like that contact us about it
first. We'll probably say yes, but we would like to reserve that
right.

For any comments or questions you can reach us at
gfelder@email.smith.edu
Igor.Tkachev@cern.ch

Enjoy LATTICEEASY!

Gary Felder and Igor Tkachev
*/

#include <chrono> // for benchmarking
#include <string>
#include <queue>

#include "latticeeasy.cuh"
  
letype* f[nflds];
letype* fd[nflds];

letype* h[nflds][6];
letype* hd[nflds][6];

cufft_type*** hdk = nullptr;
fTT_type*** hdkTT = nullptr;

cufft_type*** EMTk = nullptr;
fTT_type*** EMTkTT = nullptr;

letype* EMT[nflds][6];

letype* curr_gradientEnergy;//[nflds];
letype* curr_potEnergy;//[num_potential_terms];

letype dvdf_params[num_dvdf_params];
letype pot_params[num_pot_params];

letype* hmean[6];
double* hmean_acc[6];

letype* hdmean[6];
double* hdmean_acc[6];

// energy momentum tensor in momentum space
symmTensor<field<cufft_type, true>>* EMT_mom = new symmTensor<field<cufft_type, true>>[gwnflds]();
symmTensor<field<fTT_type, true>>* EMT_mom_TT = new symmTensor<field<fTT_type, true>>[gwnflds]();

// tensor perturbation in momentum space
symmTensor<field<cufft_type, true>>* dh_mom = new symmTensor<field<cufft_type, true>>[gwnflds]();
symmTensor<field<fTT_type, true>>* dh_mom_TT = new symmTensor<field<fTT_type, true>>[gwnflds]();

// binned things
symmTensor<std::vector<letype>> bins = symmTensor<std::vector<letype>>();


double t, t0; // Current time and initial time (t0=0 unless the run is a continuation of a previous one)
double a = 1.0, ad = 0.0, ad2 = 0.0, aterm = 0.0; // Scale factor and its derivatives (aterm is a combination of the others used in the equations of motion). Values are initialized to their defaults for the case of no expansion.
double hubble_init= 0.0; // Initial value of the Hubble constant
int run_number; // 0 for a first run, 1 for a continuation of a "0" run, etc.. Stored in the grid image (see checkpoint() function).
int no_initialization = 0; // If this variable is set to 1 by the model file then the fields will not be initialized in the normal way.
char mode_[10] = "w"; // Mode in which to open files, i.e. write ("w") or append ("a+"). Depends on the variable continue_run and on whether a previous grid image was found.
letype rescaling = 1.0; // Rescaling for output. This is left as 1 unless the model file modifies it.
char ext_[500] = "_0.dat"; // Extension for filenames - set once and used by all output functions
int nfldsout; // Number of fields to output
letype model_vars[num_model_vars]; // Model-specific variables


int main()
{
  printf("Precision is: %s.\n", typeid(letype) == typeid(float) ? "float" : "double");

  cudaMemGC<letype> res(6);

  // allocate memory for all fields
  gpuErrchk(cudaMallocManaged(&curr_gradientEnergy,  nflds * sizeof(letype)));
  gpuErrchk(cudaMallocManaged(&curr_potEnergy,  num_potential_terms * sizeof(letype)));

  for(int fld = 0; fld < nflds; fld++)
  {
    gpuErrchk(cudaMallocManaged(&f[fld],  gridsize * sizeof(letype)));
    gpuErrchk(cudaMallocManaged(&fd[fld], gridsize * sizeof(letype)));
  }

  if constexpr(sgw)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++)
    {
      gpuErrchk(cudaMallocManaged(&hmean[gwfld],      6 * sizeof(letype)));
      gpuErrchk(cudaMallocManaged(&hmean_acc[gwfld],  6 * sizeof(double)));

      gpuErrchk(cudaMallocManaged(&hdmean[gwfld],      6 * sizeof(letype)));
      gpuErrchk(cudaMallocManaged(&hdmean_acc[gwfld],  6 * sizeof(double)));

      for(int fld = 0; fld < 6; fld++)
      {
        gpuErrchk(cudaMallocManaged(&h[gwfld][fld],   gridsize * sizeof(letype)));
        gpuErrchk(cudaMallocManaged(&hd[gwfld][fld],  gridsize * sizeof(letype)));
        gpuErrchk(cudaMallocManaged(&EMT[gwfld][fld], gridsize * sizeof(letype)));

        gpuErrchk(cudaMemset(h[gwfld][fld],   0, gridsize*sizeof(letype)));
        gpuErrchk(cudaMemset(hd[gwfld][fld],  0, gridsize*sizeof(letype)));
        gpuErrchk(cudaMemset(EMT[gwfld][fld], 0, gridsize*sizeof(letype)));
      }
    }

    gpuErrchk(cudaMallocManaged(&hdk,   gwnflds * sizeof(cufft_type**)));
    gpuErrchk(cudaMallocManaged(&hdkTT, gwnflds * sizeof(fTT_type**)));

    gpuErrchk(cudaMallocManaged(&EMTk,   gwnflds * sizeof(cufft_type**)));
    gpuErrchk(cudaMallocManaged(&EMTkTT, gwnflds * sizeof(fTT_type**)));

    cudaMemPrefetchAsync(hdk,   gwnflds * sizeof(letype**), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(hdkTT, gwnflds * sizeof(fTT_type**), cudaCpuDeviceId, NULL);

    for(int gwfld = 0; gwfld < gwnflds; gwfld++)
    {
      gpuErrchk(cudaMallocManaged(&hdk[gwfld],   6 * sizeof(cufft_type*)));
      gpuErrchk(cudaMallocManaged(&hdkTT[gwfld], 6 * sizeof(fTT_type*)));

      gpuErrchk(cudaMallocManaged(&EMTk[gwfld],   6 * sizeof(cufft_type*)));
      gpuErrchk(cudaMallocManaged(&EMTkTT[gwfld], 6 * sizeof(fTT_type*)));
    }
  
    for(int i = 0; i < nflds; i++)
    {
      for(int j = 0; j < 6; j++)
      {
        hmean_acc[i][j] = 0.0;
        hdmean_acc[i][j] = 0.0;
      }
    }

  }

  prefetchAsyncFields(f, fd, cudaCpuDeviceId);
  // prefetchAsyncGWFields(h, hd, EMT, cudaCpuDeviceId);
  gpuErrchk(cudaDeviceSynchronize());

  // copy pointer to field to constant memory for easier access in kernels
  copyFieldsToConstantMemory();
  gpuErrchk(cudaPeekAtLastError()); 
  gpuErrchk(cudaDeviceSynchronize());

  // copy the values of the scale factor at which to calculate an output to a queue for easier handling
  std::queue<double> a_queue;
  for (const auto& v : a_saves)
    a_queue.push(v);


  int numsteps = 0, output_interval = 0; // Quantities for counting how often to calculate and output derived quantities
  FILE *output_= fopen("output.txt","w"); // Outputs time. Used to remotely monitor progress
  int update_time; // Controls when to output time to output file and screen 

  omp_set_num_threads(2);

  if(seed<1) // The use of seed<1 turns off certain functions (random numbers, fourier transforms, gradients, and potential energy) and should only be used for debugging
    printf("Warning: The parameter seed has been set to %d, which will result in incorrect output. For correct output set seed to a positive integer.",seed);
  

  initialize(); // Set parameter values and initial conditions

  gpuErrchk(cudaPeekAtLastError());


  t = t0;

  output_interval = int(checkpoint_interval / dt);

  //prefetch fields before starting any kernels
  prefetchAsyncFields(f, fd, 0);
  //prefetchAsyncGWFields(h, hd, EMT, 0);
  
  // Take Initial Half Time Step if this is a new run
  if(run_number == 0)
    evolve_fields(0.5*dt);

  // precalculate the paramter that are necessary to evolve the fields
  prepareParamsForCalcLapl(dvdf_params);
  prepareParamsForPotEnergy(pot_params);


  update_time=time(NULL)+print_interval; // Set initial time for update
  while((t <= tf || tf == -1) && a <= af) // Main time evolution loop
  {
    gpuErrchk(cudaPeekAtLastError());
    evolve_derivs(dt); // evolve derivatives
    gpuErrchk(cudaPeekAtLastError());
    evolve_fields(dt); // evolve fields
    gpuErrchk(cudaPeekAtLastError());


    numsteps++;
    // check if it is time for output
    if(noutput_times != 0 && numsteps % output_interval == 0 && t < tf && a < af)
    {
      save(0); // Calculate and output grid-averaged quantities (means, variances, etc.)
    }

    // check if scale factor advance far enough for output
    if(a_queue.size() != 0)
    {
      if(a > a_queue.front()) // Save data at the specified values of the scale factor
      {
        save(1);
        a_queue.pop();
      }
    }

    if(time(NULL) >= update_time) // Print an update whenever elapsed time exceeds print_interval
    {
      if(screen_updates){ // This option determines whether or not to update progress on the screen
        printf("t = %f\ta = %f\n", t, a);
        fflush(stdout);
      }
      fprintf(output_, "%f\n", t); // Output progress to a file for monitoring progress
      fflush(output_); // Make sure output file is always up to date
      update_time += print_interval; // Set time for next update
    }
  } // End of main loop
  
  gpuErrchk(cudaDeviceSynchronize());
  printf("Saving final data\n");
  save(1); // Calculate and save quantities. Force infrequently calculated quantities to be calculated.
  output_parameters(); // Save run parameters and elapsed time
  fprintf(output_,"LATTICEEASY program finished\n");
  printf("LATTICEEASY program finished\n");


  // free all the memory that was allocated for cuda
  for(int fld = 0; fld < nflds; fld++)
  {
    gpuErrchk(cudaFree(f[fld]));
    gpuErrchk(cudaFree(fd[fld]));
  }

  if constexpr(sgw)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++)  
    {
      for(int fld = 0; fld < 6; fld++)
      {
        gpuErrchk(cudaFree(h[gwfld][fld]));
        gpuErrchk(cudaFree(hd[gwfld][fld]));
        gpuErrchk(cudaFree(EMT[gwfld][fld]));
      }
    }

    for(int gwfld = 0; gwfld < nflds; gwfld++)  
    {
      gpuErrchk(cudaFree(hdk[gwfld]));
      gpuErrchk(cudaFree(hdkTT[gwfld]));
    }

    if constexpr(gwnflds > 1)
     gpuErrchk(cudaFree(hdkTT[gwnflds - 1]));
  }


  gpuErrchk(cudaFree(hdk));
  gpuErrchk(cudaFree(hdkTT));

  //fclose(benchmarkFile);
  return(0);
}
