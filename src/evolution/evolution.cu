/*
This file contains the functions for taking time steps to solve differential equations on the lattice.

The method used is a staggered leapfrog on a spatial grid with fixed time step. In other words the field values on the grid are stored in an array f at some time t and the first derivatives are stored in fd at a time t +/- 1/2 dt.

The functions here which should be called externally are gradient_energy, evolve_scale, evolve_fields and evolve_derivs.
(Only evolve_fields() and evolve_derivs() are called externally by the main function, but certain output functions call gradient_energy() and evolve_scale().)
The function gradient_energy(int fld) calculates the average gradient energy density of the field fld on the lattice.
The function evolve_scale(letype d) advances the value of the scale factor. The behavior of this function depends on the type of expansion being used. For no expansion (0) the function does nothing. For power-law expansion (1) the function calculates a and its derivatives at time t according to a preset formula. For consistent expansion (2) the function calculates the second derivative of a and uses it to advance adot by an interval d. In this case the scale factor itself is advanced in the function evolve_fields().
The function evolve_fields(letype d) advances the field values (f) and the scale factor (a) by a time interval d assuming that the field derivatives (fd) and the scale factor derivative (ad) are known at a time in the middle of that interval.
The function evolve_derivs(letype d) advances the field derivatives by a time interval d. It assumes f is known at the midpoint of that interval, uses it to calculate the second derivative, and uses that to advance the first derivatives. It also calls evolve_fields() to advance the scale factor.
*/

#include "src/latticeeasy.cuh"
#include "src/utils/stencil.h"
#include "src/integrator/integrator.cuh"
#include "src/utils/utils.cuh"

/////////////////////////////////////////////////////
// Externally called function(s)
/////////////////////////////////////////////////////

// Calculate the gradient energy, 1/2 <|Grad(f)|^2> = 1/2 <-f Lapl(f)>, of a field
/*letype gradient_energy(int fld)
{
  DECLARE_INDICES
  letype gradient = 0.0;
  double norm = 1.0/pw2(dx); // Converts the output of lapl() to an actual Laplacian



  cudaMemGC<letype> meanTemp(N*N);
  cudaMemsetAsync(meanTemp.data, 0, N*N*sizeof(letype));
  cudaMemPrefetchAsync(meanTemp.data,   N*N*sizeof(letype), cudaCpuDeviceId, NULL);

  //printf("TST: %.20f\n", meanTemp.data[10]);
  prefetchAsyncFields(f, fd, cudaCpuDeviceId);

  gpuErrchk(cudaDeviceSynchronize());
  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++)
      for(k = 0; k < N; k++)
  {
    //meanTemp.data[index(i, j, 0)] -= FIELD(fld) * Stencil::isotropic_lapl<letype, 3, NDIMS, N>(f[fld], Position(i, j, k), 1);
    meanTemp.data[index(i, j, 0)] += Stencil::isotropic_grad<letype, 3, NDIMS, N>(f[fld], Position(i, j, k));
  }


  cudaMemPrefetchAsync(meanTemp.data, N*N, 0, NULL);

  gradient = reduce(meanTemp.data, N*N);
  prefetchAsyncFields(f, fd, cudaCpuDeviceId);
  // norm converts the results of lapl() to an actual laplacian and gridsize converts the sum over the lattice to an average
  return(.5*gradient*norm/(double)gridsize);
}*/

// Calculate the scale factor and its derivatives
// Use d=0 to indicate that all quantities are known at the same time. Otherwise it's assumed that they are known at staggered times with time step d.
void evolve_scale(letype d)
{
  int fld = 0;
  double grad_energy = 0.0, pot_energy = 0.0; // Gradient and potential energies of fields
  double sfexponent, sfbase; // Model-dependent terms in power-law expansion
  double sfev1, sfev2, sfev3; // Model-dependent terms in self-consistent expansion

  if constexpr(expansion == 0) // In case of no expansion do nothing
    return;
  else if constexpr(expansion == 1) // In case of power-law expansion set all scale factor variables at time t using the model dependent parameter sfexponent.
  {
    sfexponent = expansion_power/(expansion_power*rescale_s+1.); // Exponent in power-law expansion expression for scale factor evolution
    sfbase = t*hubble_init/sfexponent + 1.; // Base of the exponent in power-law expansion expression for scale factor evolution
    a = pow(sfbase, sfexponent); // Scale factor
    ad = hubble_init/sfbase*a; // First derivative of scale factor
    ad2 = (sfexponent - 1.)/sfexponent*pw2(hubble_init)/pw2(sfbase)*a; // Second derivative of scale factor
    aterm = rescale_r*(rescale_s - rescale_r+2.) * pw2(ad/a) + rescale_r*ad2/a; // Term used in evolving fields
  }
  else if constexpr(expansion == 2) // In case of self-consistent expansion calculate sfdd
  {
    gpuErrchk(cudaPeekAtLastError());
    meanPot2(f);
    startCalcMeanGradEnergy(f, curr_gradientEnergy);
    gpuErrchk(cudaPeekAtLastError());
    
    cudaMemPrefetchAsync(curr_potEnergy,  num_potential_terms * sizeof(letype), cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(curr_gradientEnergy,  nflds * sizeof(letype), cudaCpuDeviceId, NULL);

    gpuErrchk(cudaPeekAtLastError()); 
    gpuErrchk(cudaDeviceSynchronize());

    //printf("grad %.20f\n", curr_gradientEnergy[0]);

    for(int term = 0; term < num_potential_terms; term++)
      curr_potEnergy[term] /= (letype) gridsize;



    sfev1 = rescale_s + 2.; // See documentation for an explanation of these terms in the evolution equation for a
    sfev2 = 2.*(rescale_r+rescale_s + 1.);
    sfev3 = 2.*(rescale_s + 1.);

    /*
    for(fld = 0; fld < nflds; fld++) // Sum gradient energy over all fields
      grad_energy += gradientEnergy(fld);
    */

    //printf("\tpot: %.20f grad: %.20f\n", curr_potEnergy[0], curr_gradientEnergy[0]);


    for(fld = 0; fld < nflds; fld++)
      grad_energy += curr_gradientEnergy[fld] * 0.5 * dx_inv_sq / (letype)gridsize;

    for(int term = 0; term < num_potential_terms; term++)
      pot_energy += curr_potEnergy[term];

    if(d == 0) // If all quantities are known at the same time calculate ad2 directly. (This option is called by the output routine while synchronizing values.)
      ad2 = -sfev1 * pw2(ad)/a + 8. * pi * pw2(rescale_A_inv) / pow(a, sfev2 - 1.) * (2. * grad_energy / 3. + pow(a, sfev3) * pot_energy);
    else // Otherwise use the leapfrog correction, and use ad2 to calculate aterm and advance ad.
    {
      ad2 = (-2. * ad - 2. * a / d / sfev1 * (1. - sqrt(1. + 2. * d * sfev1 * ad / a + 8. * pi * pw2(d * rescale_A_inv) * sfev1 / pow(a, sfev2) * (2. * grad_energy / 3. + pow(a, sfev3) * pot_energy)))) / d;
      ad += .5 * d * ad2; // Advance ad to time t for field evolution equations
      aterm = rescale_r * (rescale_s - rescale_r + 2.) * pw2(ad / a) + rescale_r * ad2 / a; // Term used in evolving fields
      ad += .5 * d * ad2; // Advance ad to time t+dt/2 for evolving the scale factor
    }
  }

  prepareParamsForCalcLapl(dvdf_params);
  prepareParamsForPotEnergy(pot_params);
}

// Advance the field values and scale factor using the first derivatives
void evolve_fields(letype d)
{


/*  cudaDeviceSynchronize();
  printf("LEAP \t %.10f beofre %.20f\n", t, f[0][index(10, 10, 10)]);*/

  // Advance time
  t += d;

  startLeapFrog(nflds, f, fd, d);
  gpuErrchk(cudaPeekAtLastError());
/*  cudaDeviceSynchronize();
  printf("LEAP \t %.10f after %.20f\n", t, f[0][index(10, 10, 10)]);*/

  if constexpr(sgw)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++) //we have to iterate up to nflds, not gwnflds, because the last gwfld is just the sum of all previous ones
    {
      startLeapFrogMean(gwfld, 6, h[gwfld], hd[gwfld], d);
    }
    gpuErrchk(cudaPeekAtLastError());
    
    calchmean();
    
  gpuErrchk(cudaDeviceSynchronize());
    startadjusth();
    gpuErrchk(cudaPeekAtLastError());

/*    calchdmean();
    startadjusthd();*/
    gpuErrchk(cudaPeekAtLastError());
  }

  gpuErrchk(cudaDeviceSynchronize());
  // In case of self-consistent expansion advance scale factor
  if(expansion == 2) 
    a += d*ad;
}


// Calculate second derivatives of fields and use them to advance first derivatives, and call evolve_scale to advance the scale factor
void evolve_derivs(letype d)
{
  letype laplnorm = 1. / pw2(dx) / pow(a, 2. * rescale_s + 2.); // Set coefficient for laplacian term in equations of motion. The dx^2 converts the output of lapl() to a laplacian and the scale factor term accounts for model dependent rescalings of the equations of motion.
  gpuErrchk(cudaPeekAtLastError());
  evolve_scale(d); // Calculate the scale factor and its derivatives

  /*
  startCalcLaplacian(f, fd, d);
  cudaDeviceSynchronize();
  */


/*  if constexpr(sgw)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++)
      startResetEMT(EMT[gwfld]);
  }*/
  
  startCalcLaplacian(d, curr_gradientEnergy); 
  gpuErrchk(cudaPeekAtLastError());
  
  if constexpr(sgw)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++) //we have to iterate up to nflds, not gwnflds, because the last gwfld is just the sum of all previous ones
    {
      startEvolveGWd(h[gwfld], hd[gwfld], EMT[gwfld], d, gwfld);
        gpuErrchk(cudaPeekAtLastError());
    }
  }
}