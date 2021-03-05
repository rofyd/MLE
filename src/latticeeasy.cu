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

// energy momentum tensor in position space
//symmTensor<field<letype, false>> EMT_pos = symmTensor<field<letype, false>>();

// energy momentum tensor in momentum space
symmTensor<field<cufft_type, true>>* EMT_mom = new symmTensor<field<cufft_type, true>>[gwnflds]();
symmTensor<field<fTT_type, true>>* EMT_mom_TT = new symmTensor<field<fTT_type, true>>[gwnflds]();

// tensor perturbation in momentum space
//symmTensor<field<cufft_type, true>> h_mom = symmTensor<field<cufft_type, true>>();
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

letype T_prepEMT = 0.0;
letype T_FT = 0.0;
letype T_evoh = 0.0;
letype T_projTT = 0.0;
letype T_calcGW = 0.0;
letype T_evods = 0.0;
letype T_evofs = 0.0;
letype T_total = 0.0;
letype T_totalRed = 0.0;

auto prevTime = std::chrono::high_resolution_clock::now();
FILE* benchmarkFile = fopen("benchmark.dat", "w");

void logTime(letype& value)
{
  auto currTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<letype> elapsed = currTime - prevTime;
  value += elapsed.count();

  //fprintf(benchmarkFile, "%s\t%f\n", name.c_str(), elapsed.count());
  prevTime = currTime;
}

void printTimeBM()
{
  fprintf(benchmarkFile, "%s\t%f\t%f\t%f\n", "prep EMT", T_prepEMT, T_prepEMT / T_total * 100.0, T_prepEMT / T_totalRed * 100.0);
  fprintf(benchmarkFile, "%s\t%f\t%f\t%f\n", "FT\t", T_FT, T_FT / T_total * 100.0, T_FT / T_totalRed * 100.0);
  fprintf(benchmarkFile, "%s\t%f\t%f\t%f\n", "evolve h", T_evoh, T_evoh / T_total * 100.0, T_evoh / T_totalRed * 100.0);
  fprintf(benchmarkFile, "%s\t%f\t%f\n", "projTT\t", T_projTT, T_projTT / T_total * 100.0);
  fprintf(benchmarkFile, "%s\t%f\t%f\n", "calc GWs", T_calcGW, T_calcGW / T_total * 100.0);
  fprintf(benchmarkFile, "%s\t%f\t%f\t%f\n", "evolve ds", T_evods, T_evods / T_total * 100.0, T_evods / T_totalRed * 100.0);
  fprintf(benchmarkFile, "%s\t%f\t%f\t%f\n", "evolve fs", T_evofs, T_evofs / T_total * 100.0, T_evofs / T_totalRed * 100.0);
  fprintf(benchmarkFile, "%s\t\t%f\t%f\n", "total: ", T_total, T_totalRed);
  fprintf(benchmarkFile, "%s\t%f\n", "slow down: ", T_totalRed / (T_evods + T_evofs));
}

//The following 2 lines are for debugging puproses only
//symmTensor<field<letype, false>> fieldV;
//symmTensor<field<cufft_type, true>> fieldVft;

/*
template <bool fT>
void copyTensorToDevice(symmTensor<field<letype, fT>>* d_tensor, symmTensor<field<letype, fT>>* h_tensor)
{
  gpuErrchk(cudaMallocManaged(&d_tensor, sizeof(symmTensor<field<letype, fT>>)));
  cudaMemcpy(d_tensor, h_tensor, sizeof(symmTensor<field<letype, fT>>), cudaMemcpyHostToDevice);

  field<letype, fT> tempField();
  
  for(int i = 0; i < NDIMS; i++)
    for(int j = 0; j <= i; j++)
  {
     printf("%d %d\n", i, j);
     cudaMemcpy(&(*d_tensor)(i, j), &tempField, sizeof(field<letype, fT>), cudaMemcpyHostToDevice);
  }
  
  symmTensor<field<letype, false>>* d_EMT_pos;
  copyTensorToDevice(d_EMT_pos, &EMT_pos);

  startGWLapl(*d_EMT_pos);
  printf("function over\n");
}
*/

int main()
{
  printf("Precision is: %s.\n", typeid(letype) == typeid(float) ? "float" : "double");

  cudaMemGC<letype> res(6);

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

/*    if constexpr(gwnflds > 1)
    {
        gpuErrchk(cudaMallocManaged(&hdkTT[gwnflds - 1], 6 * sizeof(fTT_type*)));
        gpuErrchk(cudaMallocManaged(&EMTkTT[gwnflds - 1], 6 * sizeof(fTT_type*)));
    }*/
  
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
  //prefetchAsyncGWFields(h, hd, EMT, cudaCpuDeviceId);
  gpuErrchk(cudaDeviceSynchronize());


  copyFieldsToConstantMemory();
  gpuErrchk(cudaPeekAtLastError()); 
  gpuErrchk(cudaDeviceSynchronize());

  //copy data to a queue for easier handling
  std::queue<double> a_queue;
  for (const auto& v : a_saves)
    a_queue.push(v);


/*
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < N; k++)
  {
    f[0][index(i, j, k)] = i*j*k;
  }


  startCalcLaplacian(1.0, curr_gradientEnergy);
  cudaDeviceSynchronize();
  for(int i = 0; i < 3; i++)
    for(int j = 0; j <= i; j++)
      printf("i %d j %d %f\n", i, j, EMT[0][indexTensor(i, j)][index(20, 10, 10)]);
  printf("\n");
  exit(1);*/


 /* int x = 5;
  int y = 60;
  int z = 12;

  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < N; k++)
  {
    h[0][0][index(i, j, k)] = 0.0;
    if(i == 10 && j == 10 && k == 10)
      h[0][0][index(i, j, k)] = 100.0;
  }


  dh_mom[0].fourierTransform(h[0]);
  gpuErrchk(cudaDeviceSynchronize());

  printf("%.20f %.20f\n", dh_mom[0](0, 0)(x, y, z).x, dh_mom[0](0, 0)(x, y, z).y);


  letype fnyquist[N][2*N];
  int arraysize[]={N,N,N};
  fftrn(h[0][0], (letype*) fnyquist, NDIMS, arraysize, 1);

  printf("%.20f %.20f\n", h[0][0][index(2 * z, y, x)], h[0][0][index(2 * z + 1, y, x)]);
  exit(1);  */

  /*
  //abort if 
  if constexpr(tf == 0.0 && af == 0.0)
  {
    printf("ERROR: tf and af should not simultaneously be set to 0. Aborting.\n");
    abort();
  }
  
  if constexpr(tf == 0.0)
    tf = 1e10;
  if constexpr(af == 0.0)
    af = 1e10;
  */

  int numsteps = 0, output_interval = 0; // Quantities for counting how often to calculate and output derived quantities
  FILE *output_= fopen("output.txt","w"); // Outputs time. Used to remotely monitor progress
  int update_time; // Controls when to output time to output file and screen 

  omp_set_num_threads(2);

  if(seed<1) // The use of seed<1 turns off certain functions (random numbers, fourier transforms, gradients, and potential energy) and should only be used for debugging
    printf("Warning: The parameter seed has been set to %d, which will result in incorrect output. For correct output set seed to a positive integer.",seed);
  
/*  printf("copying data to h and just checking\n");

  nfldsout = 1;
*/

  initialize(); // Set parameter values and initial conditions
/*
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < N; k++)
  {
   // f[0][index(i, j, k)] = N*N*N*sin(2.0 * M_PI * i * 20 / (double) N);
    hd[0][0][index(i, j, k)] = f[0][index(i, j, k)];
  }
  save(1);
  exit(1);*/
/*  cudaDeviceSynchronize();
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < N; k++)
  {
    f[0][index(i, j, k)]  = 1.0;
    fd[0][index(i, j, k)] = 0.0;
  }
*/
  gpuErrchk(cudaPeekAtLastError());




  t = t0;
/*  if constexpr(noutput_times == 0)
    output_interval = -1;
  else
    output_interval = (int)((tf - t0) / dt) / noutput_times + 1; // Set the interval between saves
*/

  output_interval = (int)(checkpoint_interval / dt);

  //prefetch fields before starting any kernels
  prefetchAsyncFields(f, fd, 0);
  //prefetchAsyncGWFields(h, hd, EMT, 0);

/*  printf("{");
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < N; k++)
  {
    printf("%.20f, ", f[0][index(i, j, k)]);  
  }
  printf("}\n");

  printf("{");
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < N; k++)
  {
    printf("%.20f, ", fd[0][index(i, j, k)]);  
  }
  printf("}\n");
  exit(1);*/
  
  // Take Initial Half Time Step if this is a new run
  if(run_number == 0)
    evolve_fields(0.5*dt);


  prepareParamsForCalcLapl(dvdf_params);
  prepareParamsForPotEnergy(pot_params);


  update_time=time(NULL)+print_interval; // Set initial time for update
  while(t <= tf && a <= af) // Main time evolution loop
  {
    gpuErrchk(cudaPeekAtLastError());
    evolve_derivs(dt);
    gpuErrchk(cudaPeekAtLastError());
    evolve_fields(dt);
    gpuErrchk(cudaPeekAtLastError());


    numsteps++;
    if(noutput_times != 0 && numsteps % output_interval == 0 && t < tf && a < af)
    {
      save(0); // Calculate and output grid-averaged quantities (means, variances, etc.)
    }

/*    cudaDeviceSynchronize();
    if(numsteps % 1000 == 0)
    {
      printf("%d   f[0][20]: %.20f fd[0][20]: %.20f\n", numsteps, f[0][20], fd[0][20]);
      printf("%d   h[0][20]: %.20f hd[0][20]: %.20f\n", numsteps, h[0][0][20], hd[0][0][20]);
      printf("%d   EMT[0][20]: %.20f a: %.20f t: %.10f\n", numsteps, EMT[0][0][20], a, t);
    }*/
    
/*    if(numsteps % 1000 == 0)
    {
    cudaDeviceSynchronize();
    printf("%.30f ", a);
    for(int i = 0; i < 6; i++)
      printf("%e ", hdmean[0][i] / gridsize);
    printf("\n"); 
    }*/

/*
    cudaDeviceSynchronize();
    if(numsteps % 1000 == 0)
    {
      for(int i = 0; i < 6; i++)
      {
        res.data[i] = 0.0;
        reduce(h[0][i], &res.data[i], gridsize);
      }

      cudaDeviceSynchronize();
      
      for(int i = 0; i < 6; i++)
      {
        printf("%d %.10f", res.data[i]);
      }      
      printf("\n");
    }*/


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
  }
  
  gpuErrchk(cudaDeviceSynchronize());
  printf("Saving final data\n");
  save(1); // Calculate and save quantities. Force infrequently calculated quantities to be calculated.
  output_parameters(); // Save run parameters and elapsed time
  fprintf(output_,"LATTICEEASY program finished\n");
  printf("LATTICEEASY program finished\n");

  //printTimeBM(); 



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
