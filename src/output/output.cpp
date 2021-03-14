/*
This file contains the functions for calculating and outputting results from the lattice program (e.g. field averages and variances).

The output is stored in files named <name>_ext. By default ext is set to <run_number>.dat where run_number indicates how many times this run has been resumed from previous data. For example the file "means_0.dat" would hold the means of an original run. If that run were then continued to a later time the means of the new run would be stored in "means_1.dat". The default value of ext can also be reset in parameters.h
File buffers are flushed at regular intervals controlled by the global variable checkpoint_interval.

The functions here which should be called externally are output_parameters(letype time) and save(int force).
The function output_parameters() creates a file and writes to it the parameters used in the program, as well as storing the clock time that elapses during the run.
The function save() calls all the appropriate functions for calculating and saving derived quantities. Which quantities are saved is controlled by global constants contained in the file parameters.h. Certain quantities such as power spectra and histograms are calculated less often, with a frequency again controlled by the variable checkpoint_interval. If save() is called with force!=0 then all quantities are calculated and file buffers are flushed.
*/

#include "src/latticeeasy.cuh"
#include "src/utils/stencil.h"
#include "src/output/gw_output.cuh"

char name_[550]; // Filenames - set differently by each function to open output files

// Outputs the means of the first nfldsout fields to "means.dat" and the variances (<field^2>-<field>^2) to "variances.dat" as a function of time
// Note that field values are output in program - not physical - units. See documentation for the rescalings used.
void meansvars(int flush)
{
  static FILE *means_,*vars_;
  DECLARE_INDICES
  int fld;
  double av,av_sq,var;

  static int first=1;
  if(first) // Open output files
  {
    sprintf(name_,"means%s",ext_);
    means_= fopen(name_,mode_);
    sprintf(name_,"variance%s",ext_);
    vars_= fopen(name_,mode_);
    first = 0;
  }

  if(means_ == NULL || vars_ == NULL)
  {
    printf("ERROR in meansvars: failed to open means or vars file. Aborting\n");
    abort();
  }

  fprintf(means_, "%f", t);
  fprintf(vars_, "%f", t);

  for(fld=0;fld<nfldsout;fld++)
  {
    av=0.;
    var=0.;
    // Calculate field mean
    LOOP
      av += FIELD(fld);
    av = av/(letype)gridsize; // Convert sum to average
    av_sq = av*av; // Calculate mean squared for finding variance
    // Calculate variance
    LOOP
      var += pw2(FIELD(fld)) - av_sq;
    var = var/(letype)gridsize; // Convert sum to variance
    // Output means and variances to files
    fprintf(means_," %e",av*rescaling);
    fprintf(vars_," %e",var*pw2(rescaling));
    // Check for instability. See if the field has grown exponentially and become non-numerical at any point.
    if(av+FLT_MAX==av || (av!=0. && av/av!=1.)) // These two separate checks between them work on all the compilers I've tested
    {
      printf("Unstable solution developed. Field %d not numerical at t=%f\n",fld,t);
      output_parameters();
      fflush(means_);
      fflush(vars_);
      exit(1);
    }
  } // End of loop over fields

  fprintf(means_,"\n");
  fprintf(vars_,"\n");
  if(flush)
  {
    fflush(means_);
    fflush(vars_);
  }
}

// Outputs to "sf.dat" the time and the physical quantities a, adot/a (i.e. Hubble), and adotdot
void scale(int flush)
{
  static FILE *sf_;

  static int first=1;
  if(first) // Open output files
  {
    sprintf(name_,"sf%s",ext_);
    sf_=fopen(name_,mode_);
    first=0;
  }

  // Output a, H, and adotdot in physical units using model-dependent rescalings
  fprintf(sf_,"%f %f %e %e\n", t, a, ad * rescale_B * pow(a, rescale_s - 1.0), pw2(rescale_B) * pow(a, 2.0 * rescale_s) * (ad2 + rescale_s * pw2(ad) / a));
  if(flush)
    fflush(sf_);
}

// Calculates and outputs spectra for all output fields
// The files "spectra<field>" record for different momentum bins: momentum, number of grid points in the bin, omega_k^2, |f_k|^2, |f_k'|^2, n_k, and rho_k
// A separate file, "spectratimes.dat", records the times at which spectra are recorded

letype fnyquist[N][2*N], fdnyquist[N][2*N]; // Used by FFT routine to store modes with k=nyquist

void spectra()
{
 // printf("SPEC \t %.10f before %.20f\n", t, f[0][index(10, 10, 10)]);
  
  static FILE *spectra_[nflds],*spectratimes_; // Output files for power spectra and times at which spectra were taken
  int fld;
  const int maxnumbins=(int)(1.73205*(N/2))+1; // Number of bins (bin spacing=lattice spacing in Fourier space) = sqrt(NDIMS)*(N/2)+1. Set for 3D (i.e. biggest possible).
  int numpoints[maxnumbins]; // Number of points in each momentum bin
  double p[maxnumbins],f2[maxnumbins],fd2[maxnumbins],ndensity[maxnumbins],rhodensity[maxnumbins]; // Values for each bin: Momentum, |f_k|^2, |f_k'|^2, n_k, and rho_k
  int numbins=(int)(sqrt((letype)NDIMS)*(N/2))+1; // Actual number of bins for the number of dimensions
  double pmagnitude; // Total momentum (p) in units of lattice spacing, pmagnitude = Sqrt(px^2+py^2+pz^2). This also gives the bin index since bin spacing is set to equal lattice spacing.
  double dp=2.*pi/L; // Size of grid spacing in momentum space
  double fp2,fpd2; // Square magnitude of field (fp2) and derivative (fpd2) for a given mode
  double omega,omegasq; // Omega_p^2 = m^2 + p^2. Used for number and energy density
  double mass_sq[nflds]; // Effective square masses for fields. Used for calculating omega
  // Extra terms used to adjust for variable rescalings in the definitions of omega_p, n_p, and rho_p
  double extra_omega=pow(a,2.*rescale_s) * (a*ad2 + (1.+rescale_s)*pw2(ad));  // Extra term in omega
  double extra_fpd_1=(1.-rescale_r)*ad/a, extra_fpd_2=pow(a,2.+2.*rescale_s); // Terms used to modify f'_p
  double extra_np=pow(a,2.-2.*rescale_r)*pow(dx,6)/(pw2(rescale_A*rescale_B)*2.*L*L*L); // Term used to modify n_p
  double extra_rhop=rescale_B*extra_np;
#if NDIMS==1
  int i,k,pz; // pz is momentum in units of grid spacing
  letype norm1=2.*pi*pw2(L)/pw2(pw2(dx)); // Normalization designed to match spectra to 3D results
  letype norm2=pw2(pw2((letype)N))/2./pi; // A different normalization is used for number and energy spectra
#elif NDIMS==2
  int i,j,k,py,pz; // py and pz are components of momentum in units of grid spacing
  letype norm1=pi*L/pw2(dx); // Normalization designed to match spectra to 3D results
  letype norm2=pw2((letype)N)/2.; // A different normalization is used for number and energy spectra
  letype fnyquist[2*N],fdnyquist[2*N]; // Used by FFT routine to store modes with k=nyquist
  int arraysize[]={N,N}; // Array of grid size in all dimensions - used by FFT routine
#elif NDIMS==3
  int i, j, k, px, py, pz; // px, py, and pz are components of momentum in units of grid spacing
  double norm1 = 1.0, norm2 = 1.0;

  int arraysize[] = {N, N, N}; // Array of grid size in all dimensions - used by FFT routine
#endif

  static int first=1;
  if(first) // Open output files
  {
    for(fld=0;fld<nfldsout;fld++)
    {
      sprintf(name_,"spectra%d%s",fld,ext_);
      spectra_[fld] = fopen(name_,mode_);
    }
    sprintf(name_,"spectratimes%s",ext_);
    spectratimes_=fopen(name_,mode_);
    first=0;
  }

  effective_mass(mass_sq, NULL); // Calculate effective masses of fields.
  // Include corrections involving scale factor derivatives (see documentation)
  for(fld=0;fld<nfldsout;fld++)
    mass_sq[fld] -= extra_omega;

  // Calculate magnitude of momentum in each bin
  for(i=0;i<numbins;i++)
    p[i]=dp*i;

  for(fld=0;fld<nfldsout;fld++) // Find power spectra for each field
  {
    for(i=0;i<numbins;i++) // Initialize all bins to 0
    {
      numpoints[i]=0; // Number of points in the bin
      f2[i]=0.; // |f_p|^2
      fd2[i]=0.; // |f_p'|^2
      ndensity[i]=0.; // omega |f_p|^2 + 1/omega |f_p'|^2
      rhodensity[i]=0.; // omega^2 |f_p|^2 + |f_p'|^2
    }


#if NDIMS==1
  fftr1((letype*) f[fld],  N, 1); // Transform field values to Fourier space
  fftr1((letype*) fd[fld], N, 1); // Transform field derivatives to Fourier space
#else
  fftrn((letype*) f[fld],  (letype*) fnyquist,  NDIMS, arraysize, 1); // Transform field values to Fourier space
  fftrn((letype*) fd[fld], (letype*) fdnyquist, NDIMS, arraysize, 1); // Transform field derivatives to Fourier space
#endif

#if NDIMS==1
  for(k=1;k<N/2;k++) 
  {
    pz=k; // z-component of momentum of modes at z=k
    pmagnitude=pz; // Magnitude of momentum of mode in units of momentum grid spacing
    fp2=pw2(f[fld][index(2*k)])+pw2(f[fld][index(2*k+1)]); // |Re(f_k)|^2 + |Im(f_k)|^2 for mode
    fpd2=pw2(fd[fld][index(2*k)])+pw2(fd[fld][index(2*k+1)]); // |Re(f_k')|^2 + |Im(f_k')|^2 for mode
    numpoints[(int)pmagnitude] += 2; // Iterate the count of points in this bin
    f2[(int)pmagnitude] += 2.*fp2; // Add the power of this mode to the bin
    fd2[(int)pmagnitude] += 2.*fpd2; // Add the power of the mode derivative to the bin
    fpd2=extra_fpd_2*(pw2(fd[fld][index(2*k)]+extra_fpd_1*f[fld][index(2*k)])+pw2(fd[fld][index(2*k+1)]+extra_fpd_1*f[fld][index(2*k+1)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
    omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
    if(omegasq > 0.) // Guard against letypeing point errors
    {
      omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
      ndensity[(int)pmagnitude] += 2.*extra_np*(fp2*omega + fpd2/omega)/pw2(pmagnitude); // Add this mode's contribution to number density
    }
    rhodensity[(int)pmagnitude] += 2.*extra_rhop*(fp2*omegasq + fpd2)/pw2(pmagnitude); // Add this mode's contribution to energy density
  }
  // Modes with k=0 or k=N/2 are only counted once
  for(k=0;k<=N/2;k+=N/2) // "Loop" over the two values k=0 and k=N/2
  {
    pz=k;
    pmagnitude=pz;
    if(k==0) // Amplitudes of k=0 modes
    {   
      fp2=pw2(f[fld][index(0)]);
      fpd2=pw2(fd[fld][index(0)]);
    }
    else // Modes with k=N/2 are stored in the imaginary component of the k=0 array entry
    {
      fp2=pw2(f[fld][index(1)]);
      fpd2=pw2(fd[fld][index(1)]);
    }
    numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin
    f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
    fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
    if(k==0)
      fpd2=extra_fpd_2*(pw2(fd[fld][index(0)]+extra_fpd_1*f[fld][index(0)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
    else
      fpd2=extra_fpd_2*(pw2(fd[fld][index(1)]+extra_fpd_1*f[fld][index(1)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
    omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
    if(pmagnitude==0.) // Avoid letypeing point errors
      continue;
    if(omegasq > 0.) // Guard against letypeing point errors
    {
      omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
      ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega)/pw2(pmagnitude); // Add this mode's contribution to number density
    }
    rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2)/pw2(pmagnitude); // Add this mode's contribution to energy density
  }
#elif NDIMS==2
  for(j=0;j<N;j++)
  {
    py=(j<=N/2 ? j : j-N); // y-component of momentum of modes at y=j
    // Modes with 0<k<N/2 are counted twice to account for the modes f(-p)=f(p)* that aren't explicitly included in the lattice
    for(k=1;k<N/2;k++) 
    {
      pz=k; // z-component of momentum of modes at z=k
      pmagnitude=sqrt(pw2(py)+pw2(pz)); // Magnitude of momentum of mode in units of momentum grid spacing
      fp2=pw2(f[fld][index(j, 2*k)]) + pw2(f[fld][index(j, 2*k+1)]); // |Re(f_k)|^2 + |Im(f_k)|^2 for mode
      fpd2=pw2(fd[fld][index(j, 2*k)]) + pw2(fd[fld][index(j, 2*k+1)]); // |Re(f_k')|^2 + |Im(f_k')|^2 for mode
      numpoints[(int)pmagnitude] += 2; // Iterate the count of points in this bin
      f2[(int)pmagnitude] += 2.*fp2; // Add the power of this mode to the bin
      fd2[(int)pmagnitude] += 2.*fpd2; // Add the power of the mode derivative to the bin
      fpd2=extra_fpd_2*(pw2(fd[fld][index(j, 2*k)] + extra_fpd_1*f[fld][index(j, 2*k)])+pw2(fd[fld][index(j, 2*k+1)] + extra_fpd_1*f[fld][index(j, 2*k+1)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
      omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
      if(omegasq > 0.) // Guard against letypeing point errors
      {
        omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
        ndensity[(int)pmagnitude] += 2.*extra_np*(fp2*omega + fpd2/omega)/pmagnitude; // Add this mode's contribution to number density
      }
      rhodensity[(int)pmagnitude] += 2.*extra_rhop*(fp2*omegasq + fpd2)/pmagnitude; // Add this mode's contribution to energy density
    }
    // Modes with k=0 or k=N/2 are only counted once
    for(k = 0; k <= N/2; k += N/2) // "Loop" over the two values k=0 and k=N/2
    {
      pz = k;
      pmagnitude = sqrt(pw2(py) + pw2(pz));
      if(k == 0) // Amplitudes of k=0 modes
      {   
        fp2 = pw2(f[fld][index(j, 0)]) + pw2(f[fld][index(j, 1)]);
        fpd2 = pw2(fd[fld][index(j, 0)]) + pw2(fd[fld][index(j, 1)]);
      }
      else // Modes with k=N/2 are stored in the array fnyquist
      {
        fp2 = pw2(fnyquist[2*j]) + pw2(fnyquist[2*j+1]);
        fpd2 = pw2(fdnyquist[2*j]) + pw2(fdnyquist[2*j+1]);
      }
      numpoints[(int) pmagnitude]++; // Iterate the count of points in this bin
      f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
      fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
      if(k == 0)
        fpd2 = extra_fpd_2*(pw2(fd[fld][index(j, 0)] + extra_fpd_1*f[fld][index(j, 0)]) + pw2(fd[fld][index(j, 1)] + extra_fpd_1*f[fld][index(j, 1)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
      else
        fpd2 = extra_fpd_2*(pw2(fdnyquist[2*j] + extra_fpd_1*fnyquist[2*j]) + pw2(fdnyquist[2*j+1] + extra_fpd_1*fnyquist[2*j + 1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
      omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
      if(pmagnitude == 0.) // Avoid letypeing point errors
        continue;
      if(omegasq > 0.) // Guard against letypeing point errors
      {
        omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
        ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega)/pmagnitude; // Add this mode's contribution to number density
      }
      rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2)/pmagnitude; // Add this mode's contribution to energy density
    }
  } // End of loop over j
#elif NDIMS==3

/*  printf("%d %d %d le vals: %e %e\n", posx, posy, posz, f[0][index(posz, posy, posx)], f[0][index(posz+1, posy, posx)]);
  printf("%d %d %d le vals: %e %e\n", posx, posy, posz, fnyquist[posx][2*posy], fnyquist[posx][2*posy+1]);

*/
    // Loop runs over all gridpoints. All points with k<N/2 are in the array f, while points with k=N/2 are in the array fnyquist.
    // px and py go over all mode values in wrap-around order, rising from 0 to N/2 and then from -N/2+1 to -1
    for(i = 0; i < N; i++)
    {
      px = (i <= N/2 ? i : i-N); // x-component of momentum of modes at x=i
      for(j = 0; j < N; j++)
      {
        py = (j <= N/2 ? j : j-N); // y-component of momentum of modes at y=j
        // Modes with 0<k<N/2 are counted twice to account for the modes f(-p)=f(p)* that aren't explicitly included in the lattice
        for(k = 1; k < N/2; k++) 
        {
          pz = k; // z-component of momentum of modes at z=k
          pmagnitude = sqrt(pw2(px) + pw2(py) + pw2(pz)); // Magnitude of momentum of mode in units of momentum grid spacing
          fp2 = pw2(f[fld][index(2*k, j, i)]) + pw2(f[fld][index(2*k + 1, j, i)]); // |Re(f_k)|^2 + |Im(f_k)|^2 for mode
          fpd2 = pw2(fd[fld][index(2*k, j, i)]) + pw2(fd[fld][index(2*k + 1, j, i)]); // |Re(f_k')|^2 + |Im(f_k')|^2 for mode
          numpoints[(int)pmagnitude] += 2; // Iterate the count of points in this bin
          f2[(int)pmagnitude] += 2.*fp2; // Add the power of this mode to the bin
          fd2[(int)pmagnitude] += 2.*fpd2; // Add the power of the mode derivative to the bin
          fpd2 = extra_fpd_2*(pw2(fd[fld][index(2*k, j, i)] + extra_fpd_1*f[fld][index(2*k, j, i)]) + pw2(fd[fld][index(2*k + 1, j, i)] + extra_fpd_1*f[fld][index(2*k + 1, j, i)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
          omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
          if(omegasq > 0.) // Guard against floating point errors
          {
            omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
            ndensity[(int)pmagnitude] += 2.*extra_np*(fp2*omega + fpd2/omega); // Add this mode's contribution to number density
          }
          rhodensity[(int)pmagnitude] += 2.*extra_rhop*(fp2*omegasq + fpd2); // Add this mode's contribution to energy density
        }
        // Modes with k=0 or k=N/2 are only counted once
        for(k = 0; k <= N/2; k += N/2) // "Loop" over the two values k=0 and k=N/2
        {
          pz = k;
          pmagnitude = sqrt(pw2(px) + pw2(py) + pw2(pz));
          if(k == 0) // Amplitudes of k=0 modes
          {   
            fp2 = pw2(f[fld][index(0, j, i)]) + pw2(f[fld][index(1, j, i)]);
            fpd2 = pw2(fd[fld][index(0, j, i)]) + pw2(fd[fld][index(1, j, i)]);
          }
          else // Modes with k=N/2 are stored in the array fnyquist
          {
            fp2 = pw2(fnyquist[i][2*j]) + pw2(fnyquist[i][2*j+1]);
            fpd2 = pw2(fdnyquist[i][2*j]) + pw2(fdnyquist[i][2*j+1]);
          }
          numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin
          f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
          fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
          if(k == 0)
            fpd2 = extra_fpd_2*(pw2(fd[fld][index(0, j, i)] + extra_fpd_1*f[fld][index(0, j, i)]) + pw2(fd[fld][index(1, j, i)] + extra_fpd_1*f[fld][index(1, j, i)])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
          else
            fpd2 = extra_fpd_2*(pw2(fdnyquist[i][2*j] + extra_fpd_1*fnyquist[i][2*j]) + pw2(fdnyquist[i][2*j+1] + extra_fpd_1*fnyquist[i][2*j+1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
          omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
          if(omegasq > 0.) // Guard against letypeing point errors
          {
            omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
            ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega); // Add this mode's contribution to number density
          }
          rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2); // Add this mode's contribution to energy density
        }
      } // End of loop over j
    } // End of loop over i
#endif

    for(i=0;i<numbins;i++)
    {
      if(numpoints[i]>0) // Convert sums to averages. (numpoints[i] should always be greater than zero.)
      {
        f2[i] = f2[i]/numpoints[i];
        fd2[i] = fd2[i]/numpoints[i];
        ndensity[i] = ndensity[i]/numpoints[i];
        rhodensity[i] = rhodensity[i]/numpoints[i];
      }
      omegasq = fabs(pw2(p[i]) + mass_sq[fld]); // Omega and p for each bin are recorded at the bottom of the bin (i.e. the lowest momentum)
      // Output the momentum, number of points, omega, and calculated spectra for each bin
      fprintf(spectra_[fld], "%e %d %e %e %e %e %e\n",
        p[i], numpoints[i], omegasq,norm1*f2[i], norm1*fd2[i], norm2*ndensity[i], norm2*rhodensity[i]);
    }
    fprintf(spectra_[fld],"\n");
    fflush(spectra_[fld]);

    // Transform field values back to position space.

#if NDIMS==1
    fftr1((letype*) f[fld],  N, -1);
    fftr1((letype*) fd[fld], N, -1);
#else
    fftrn((letype*) f[fld],  (letype*) fnyquist,  NDIMS, arraysize, -1);
    fftrn((letype*) fd[fld], (letype*) fdnyquist, NDIMS, arraysize, -1);
#endif
  } // End of loop over fields

  fprintf(spectratimes_, "%f\n", t); // Output time at which power spectra were recorded
  fflush(spectratimes_);

//printf("SPEC \t %.10f after %.20f\n", t, f[0][index(10, 10, 10)]);
  return;
}


inline letype gradsqbtest(int fld, INDEXLIST)
{
#if NDIMS==1
  letype resx = f[fld][index(mod(i + 1, N))] - f[fld][index(i)];
  return pw2(resx);
#elif NDIMS==2
  letype resx = f[fld][index(mod(i + 1, N), j)] - f[fld][index(i, j)];
  letype resy = f[fld][index(i, mod(j + 1, N))] - f[fld][index(i, j)];
  return pw2(resx) + pw2(resy);
#elif NDIMS==3
  letype resx = f[fld][index(mod(i + 1, N), j, k)] - f[fld][index(i, j, k)];
  letype resy = f[fld][index(i, mod(j + 1, N), k)] - f[fld][index(i, j, k)];
  letype resz = f[fld][index(i, j, mod(k + 1, N))] - f[fld][index(i, j, k)];
  return pw2(resx) + pw2(resy) + pw2(resz);
#endif
  return 0.0;
}

#if NDIMS == 1
  #define INDEX_ARG i
#elif NDIMS == 2
  #define INDEX_ARG i,j
#elif NDIMS == 3
  #define INDEX_ARG i,j,k
#endif

void testing_different_acc(letype grad)
{
	DECLARE_INDICES

	letype forward = 0.0;
	letype central2 = 0.0;
	letype central4 = 0.0;
	letype central6 = 0.0;
	letype central8 = 0.0;
	letype sta = 0.0;
	letype isoA = 0.0;
	letype isoB = 0.0;
	letype isoC = 0.0;

	letype norm = 1.0/pw2(dx);

	
    #pragma omp parallel for collapse(NDIMS) reduction(+:forward) reduction(+:central2) reduction(+:central4) reduction(+:central6) reduction(+:central8) reduction(+:sta) reduction(+:isoA) reduction(+:isoB) reduction(+:isoC)
    LOOP
    {
    	forward  += norm * pow(a,-2.*(rescale_s+1.)) * gradsqbtest(0,INDEX_ARG);
    	central2 += norm * pow(a,-2.*(rescale_s+1.)) * Stencil::fd_gradsq_test<letype, 2, NDIMS, N>(f[0], Position(i,j,k));
    	central4 += norm * pow(a,-2.*(rescale_s+1.)) * Stencil::fd_gradsq_test<letype, 4, NDIMS, N>(f[0], Position(i,j,k));
    	central6 += norm * pow(a,-2.*(rescale_s+1.)) * Stencil::fd_gradsq_test<letype, 6, NDIMS, N>(f[0], Position(i,j,k));
    	central8 += norm * pow(a,-2.*(rescale_s+1.)) * Stencil::fd_gradsq_test<letype, 8, NDIMS, N>(f[0], Position(i,j,k));
    	sta  +=     norm * pow(a,-2.*(rescale_s+1.)) * Stencil::isotropic_grad<letype, 0, NDIMS, N>(f[0], Position(i,j,k));
    	isoA +=     norm * pow(a,-2.*(rescale_s+1.)) * Stencil::isotropic_grad<letype, 1, NDIMS, N>(f[0], Position(i,j,k));
    	isoB +=     norm * pow(a,-2.*(rescale_s+1.)) * Stencil::isotropic_grad<letype, 2, NDIMS, N>(f[0], Position(i,j,k));
    	isoC +=     norm * pow(a,-2.*(rescale_s+1.)) * Stencil::isotropic_grad<letype, 3, NDIMS, N>(f[0], Position(i,j,k));
    }

    printf("\tcurr: %f\n", curr_gradientEnergy[0]);
    printf("\tLE:   %f   FW: %f\n\tCE2:  %f  CE4: %f  CE6: %f  CE8: %f\n\tsta: %f  ISOA: %f ISOB: %f ISOC: %f\n\n", grad, forward, central2, central4, central6, central8, sta, isoA, isoB, isoC);
}


void oscillon(letype totalEnergy)
{
  if(t == 0.0)
    return;

  static FILE* outfile[nflds] = {NULL};
  static char fileName[30];

  if(outfile[0] == NULL)
  {
    for(int fld = 0; fld < nflds; fld++)
    {
      snprintf(fileName, 29, "overdensities_%d.dat", fld);

      outfile[fld] = fopen(fileName, "w");
      if(outfile[fld] == NULL)
      {
        printf("ERROR: failed to open file %s for writing. Aborting.\n", fileName);
        abort();
      }
    }
  }

  DECLARE_INDICES

  const letype norm = 1/pw2(dx);

  //this is the total energy of one lattice point
  letype sTotalEnergy = 0.0;

  //these hold the gradient and potential energy of a single latticepoint
  letype sGradEnergy = 0.0;
  letype sPotEnergy = 0.0;

  //Variable for derivative energy
  letype var = 0.;
  letype vard = 0.;
  letype ffd = 0.;

  //this hold the derivative energy of a single lattice point
  letype sDerivEnergy = 0.0;

  for(int fld = 0; fld < nflds; fld++)
  {
    fprintf(outfile[fld], "%f %f %f ", t, a, ad); // Output time, scale factor and its derivative

    LOOP
    {
      sGradEnergy = 0.0;
      sPotEnergy = 0.0;
      sDerivEnergy = 0.0;

      //calculate derivative energy
      var = pw2(FIELD(fld)); //squared field value
      vard = pw2(FIELDD(fld)); //squared field derivative
      ffd = FIELD(fld)*FIELDD(fld); //product of field and derivative

      sDerivEnergy = .5*vard - rescale_r*(ad/a)*ffd + .5*pw2(rescale_r)*pw2(ad/a)*var;

      //calculate gradient energy
      sGradEnergy = 0.5 * pow(a,-2.*(rescale_s+1.)) * norm * Stencil::isotropic_grad<letype, 3, NDIMS, N>(f[fld], Position(i,j,k));
  
      //calculate potential energy
      for(int term = 0; term < num_potential_terms; term++)
        sPotEnergy = singlePotEnergy(term, f, i, j, k);

      sTotalEnergy = sGradEnergy + sPotEnergy + sDerivEnergy;

      //check if latticepoints exceed the overdensity defined in parameters.h
      if(sTotalEnergy / totalEnergy > overdensity)
      {
        //save lcation of this lattice point
        fprintf(outfile[fld], "%d-%d-%d_", i, j, k);
        //save overdensity, field value and its derivative
        fprintf(outfile[fld], "%f-%f-%f#", sTotalEnergy / totalEnergy, FIELD(fld), FIELDD(fld));
      }
    }

    fprintf(outfile[fld], "\n");
    fflush(outfile[fld]);
  }    
}

// Outputs the components of the total energy as a function of time
// The output file "energy" contains t, kinetic (time derivative) energy for each field, gradient energy for each field, and then all potential terms
// The file "conservation" contains rho/rho_initial for no expansion and the ratio of H^2 to 8 pi/3 rho for an expanding universe
// Note that energy output is not in physical units. See the documentation for conversions.
letype energy()
{
  static FILE *energy_,*conservation_;
  static letype totalinitial; // Initial value of energy density (used for checking conservation in Minkowski space)
  DECLARE_INDICES
  int fld;
  double var,vard,ffd; // Averaged field values over the grid (defined below)
  double deriv_energy,grad_energy,pot_energy,total=0.; // Total gives the total value of rho(t) for checking energy conservation

  static int first=1;
  if(first) // Open output files (first is set to zero at the bottom of the function)
  {
    sprintf(name_,"energy%s",ext_);
    energy_=fopen(name_,mode_);
    if(expansion!=1) // For power-law expansion don't calculate conservation
    {
      sprintf(name_,"conservation%s",ext_);
      conservation_=fopen(name_,mode_);
    }
  } // The variable first is used again at the end of the function, where it is then set to 0
  fprintf(energy_,"%f",t); // Output time

  // Calculate and output kinetic (time derivative) energy
  for(fld=0;fld<nflds;fld++) // Note that energy is output for all fields regardless of the value of nfldsout
  {
    var=0.;
    vard=0.;
    ffd=0.;
    #pragma omp parallel for collapse(NDIMS) reduction(+:var) reduction(+:vard) reduction(+:ffd)
    LOOP
    {
      var += pw2(FIELD(fld)); // Sum over squared field values
      vard += pw2(FIELDD(fld)); // Sum over squared field derivatives
      ffd += FIELD(fld)*FIELDD(fld); // Sum over product of field and derivative
    }
    // Convert sums to averages
    var = var/(letype)gridsize;
    vard = vard/(letype)gridsize;
    ffd = ffd/(letype)gridsize;
    deriv_energy = .5*vard - rescale_r*(ad/a)*ffd + .5*pw2(rescale_r)*pw2(ad/a)*var; // Extra terms account for model-dependent rescalings
    total += deriv_energy;
    fprintf(energy_," %e",deriv_energy);
  }

  startCalcMeanGradEnergy(f, curr_gradientEnergy);
  gpuErrchk(cudaPeekAtLastError()); 
  gpuErrchk(cudaDeviceSynchronize());

  // Calculate and output gradient energy
  for(fld = 0; fld < nflds; fld++)
  {
    grad_energy = pow(a,-2.*(rescale_s+1.)) * curr_gradientEnergy[fld] * 0.5 / pw2(dx) / (letype)gridsize;
    total += grad_energy;
    fprintf(energy_," %e",grad_energy);
  }

  letype pot_total = 0.0;
  // Calculate and output potential energy
  for(i=0;i<num_potential_terms;i++)
  {
    pot_energy = potential_energy(i,NULL); // Model dependent function for calculating potential energy terms
    total += pot_energy;
    pot_total += pot_energy;
    fprintf(energy_," %e",pot_energy);
  }

  fprintf(energy_,"\n");
  fflush(energy_);

  // Energy conservation
  if(first) // In Minkowski space record the initial value of the energy to be used for checking energy conservation
  {
    if(expansion==0)
      totalinitial=total;
    first=0; // Regardless of the expansion set first to 0 so file streams aren't opened again.
  }
  if(expansion!=1) // Conservation isn't checked for power law expansion
  {
    if(expansion==0) // In Minkowski space the file conservation_ records the ratio of rho(t) to rho_initial
      fprintf(conservation_,"%f %f\n",t,total/totalinitial);
    else // In an expanding universe the file conservation_ records the ratio of H^2(t) to 8 pi/3 rho(t)
      fprintf(conservation_,"%f %f\n",t,3.*pw2(rescale_A)/(8.*pi)*pow(a,2.*(rescale_r-1.))*pw2(ad)/total);
    fflush(conservation_);
  }

  return total;
	//testing_different_acc(grad_energy*2.0*N*N*N);//TODO: this is only for testing purposes
  //overdensities(total, 4, deriv_energy, grad_energy, pot_total);
}

// Outputs histograms for all output fields
// The histograms are stored in files "histogram<filenumber>" where <filenumber> runs from 0 to nfldsout.
// A separate file, "histogramtimes", records the times at which histograms are recorded, the minimum field value for each field, and the spacing (in field values) between successive bins for each time.
void histograms()
{
  static FILE *histogram_[nflds],*histogramtimes_;
  int i=0,j=0,k=0,fld;
  int binnum; // Index of bin for a given field value
  letype binfreq[nbins]; // The frequency of field values occurring within each bin
  letype bmin,bmax,df; // Minimum and maximum field values for each field and spacing (in field values) between bins
  int numpts; // Count the number of points in the histogram for each field. (Should be all lattice points unless explicit field limits are given.)

  static int first=1;
  if(first) // Open output files
  {
    for(fld=0;fld<nfldsout;fld++)
    {
      sprintf(name_,"histogram%d%s",fld,ext_);
      histogram_[fld]=fopen(name_,mode_);
    }
    sprintf(name_,"histogramtimes%s",ext_);
    histogramtimes_=fopen(name_,mode_);
    first=0;
  }

  fprintf(histogramtimes_,"%f",t); // Output time at which histograms were recorded
  
  for(fld=0;fld<nfldsout;fld++) // Main loop. Generate histogram for each field.
  {
    // Find the minimum and maximum values of the field
    if(histogram_max==histogram_min) // If no explicit limits are given use the current field values
    {
      i=0;j=0;k=0;
      bmin=FIELD(fld);
      bmax=bmin;
      LOOP
      {
        bmin = (FIELD(fld)<bmin ? FIELD(fld) : bmin);
        bmax = (FIELD(fld)>bmax ? FIELD(fld) : bmax);
      }
    }
    else
    {
      bmin=histogram_min;
      bmax=histogram_max;
    }
  
    // Find the difference (in field value) between successive bins
    df=(bmax-bmin)/(letype)(nbins); // bmin will be at the bottom of the first bin and bmax at the top of the last
  
    // Initialize all frequencies to zero
    for(i=0;i<nbins;i++)
      binfreq[i]=0.;

    // Iterate over grid to determine bin frequencies
    numpts=0;
    LOOP
    {
      binnum=(int)((FIELD(fld)-bmin)/df); // Find index of bin for each value
      if(FIELD(fld)==bmax) // The maximal field value is at the top of the highest bin
        binnum=nbins-1;
      if(binnum>=0 && binnum<nbins) // Increment frequency in the appropriate bin
      {
        binfreq[binnum]++;
        numpts++;
      }
    } // End of loop over grid

    // Output results
    for(i=0;i<nbins;i++)
      fprintf(histogram_[fld],"%e\n",binfreq[i]/(letype)numpts); // Output bin frequency normalized so the total equals 1
    fprintf(histogram_[fld],"\n"); // Stick a blank line between times to make the file more readable
    fflush(histogram_[fld]);
    fprintf(histogramtimes_," %e %e",bmin*rescaling,df*rescaling); // Output the starting point and stepsize for the bins at each time
  } // End loop over fields

  fprintf(histogramtimes_,"\n");
  fflush(histogramtimes_);
}

// Calculate two dimensional histograms of pairs of fields
// The histograms are stored in files "histogram2d<filenumber>_<filenumber>".
// A separate file, "histogramtimes", records the times at which histograms are recorded, the minimum field value for each field, and the spacing (in field values) between successive bins for each time.
void histograms_2d()
{
  static FILE **histogram_,*histogramtimes_; // histogram_ is an array of pointers to output files
  int i,j,k,pair; // pair iterates over pairs of fields for which histograms are being recorded
  int fld1,fld2; // Indices of the fields for a given histogram
  int binnum[2]; // Indices of bins for a given pair of field values
  letype binfreq[nbins0][nbins1]; // The frequency of field values occurring within each bin
  letype bmin[2],bmax[2],df[2]; // Minimum and maximum field values for each field and spacing (in field values) between bins

  static int first=1,numhists; // numhists is the numbers of 2d histograms being calculated
  if(first) // Open output files
  {
    numhists = (sizeof(hist2dflds)/sizeof(hist2dflds[0])/2); // Number of pairs in field list
    histogram_ = new FILE *[numhists]; // Allocate an array of FILE pointers
    for(pair=0;pair<numhists;pair++)
    {
      sprintf(name_,"histogram2d%d_%d%s",hist2dflds[2*pair],hist2dflds[2*pair+1],ext_);
      histogram_[pair]=fopen(name_,mode_);
    }
    sprintf(name_,"histogram2dtimes%s",ext_);
    histogramtimes_=fopen(name_,mode_);
    first=0;
  }

  fprintf(histogramtimes_,"%f",t); // Output time at which 2d histograms were recorded

  for(pair=0;pair<numhists;pair++) // Main loop. Generate histogram for each field pair
  {
    fld1=hist2dflds[2*pair]; // Find indices of fields for 2d histogram
    fld2=hist2dflds[2*pair+1];
    // Find the minimum and maximum values of the field
    if(histogram2d_min==histogram2d_max) // If no explicit limits are given use the current field values
    {
      i=0;j=0;k=0;
      bmin[0]=FIELD(fld1);
      bmax[0]=bmin[0];
      bmin[1]=FIELD(fld2);
      bmax[1]=bmin[1];
      LOOP
      {
        bmin[0] = (FIELD(fld1)<bmin[0] ? FIELD(fld1) : bmin[0]);
        bmax[0] = (FIELD(fld1)>bmax[0] ? FIELD(fld1) : bmax[0]);
        bmin[1] = (FIELD(fld2)<bmin[1] ? FIELD(fld2) : bmin[1]);
        bmax[1] = (FIELD(fld2)>bmax[1] ? FIELD(fld2) : bmax[1]);
      }
    }
    else
    {
      bmin[0]=histogram2d_min;
      bmax[0]=histogram2d_max;
      bmin[1]=histogram2d_min;
      bmax[1]=histogram2d_max;
    }

    // Find the difference (in field value) between successive bins
    df[0]=(bmax[0]-bmin[0])/(letype)(nbins0);
    df[1]=(bmax[1]-bmin[1])/(letype)(nbins1);
    
    // Initialize all frequencies to zero
    for(i=0;i<nbins0;i++)
      for(j=0;j<nbins1;j++)
        binfreq[i][j]=0.;
  
    // Iterate over grid to determine bin frequencies
    LOOP
    {
      binnum[0]=(int)((FIELD(fld1)-bmin[0])/df[0]); // Find index of bin for each value
      binnum[1]=(int)((FIELD(fld2)-bmin[1])/df[1]);
      if(FIELD(fld1)==bmax[0]) binnum[0]=nbins0-1; // The maximal field value is at the top of the highest bin
      if(FIELD(fld2)==bmax[1]) binnum[1]=nbins1-1; // The maximal field value is at the top of the highest bin
      if(binnum[0]>=0 && binnum[0]<nbins0 && binnum[1]>=0 && binnum[1]<nbins1)
	binfreq[binnum[0]][binnum[1]]++;
    } // End of loop over grid

    // Output results
    for(i=0;i<nbins0;i++)
      for(j=0;j<nbins1;j++)
        fprintf(histogram_[pair],"%e\n",binfreq[i][j]/(letype)gridsize);
    fprintf(histogram_[pair],"\n"); // Stick a blank line between times to make the file more readable
    fflush(histogram_[pair]);
    fprintf(histogramtimes_," %e %e %e %e",bmin[0]*rescaling,df[0]*rescaling,bmin[1]*rescaling,df[1]*rescaling); // Output the starting point and stepsize for the bins at each time
  } // End of loop over pairs

  fprintf(histogramtimes_,"\n");
  fflush(histogramtimes_);
}

// This function outputs the values of the fields    on a slice of the lattice
inline void slices()
{
  static FILE *slices_[nflds],*slicetimes_;
  static int adjusted_slicedim=slicedim; // Number of dimensions to include on slice
  static int final=slicelength*sliceskip;
  int i,j,k,fld;
  int x,y,z;
  int numpts; // Used for keeping track of how many points are being averaged in each output
  letype value; // Field value to be output

  static int first=1;
  if(first) // Open output files
  {
    for(fld=0;fld<nfldsout;fld++)
    {
      sprintf(name_,"slices%d%s",fld,ext_);
      slices_[fld]=fopen(name_,mode_);
    }
    sprintf(name_,"slicetimes%s",ext_);
    slicetimes_=fopen(name_,mode_);
    if(adjusted_slicedim>NDIMS)
      adjusted_slicedim=NDIMS;
    if(final>N)
      final=N;
    first=0;
  }

  for(fld=0;fld<nfldsout;fld++)
  {
    if(adjusted_slicedim==1)
    {
      for(k=0;k<final;k+=sliceskip)
      {
        if(sliceaverage==1) // Average over all "skipped" points
        {
          value=0.;
          numpts=0;
          for(z=k;z<k+sliceskip && z<N;z++)
          {
            value += FIELDPOINT(fld,0,0,z);
            numpts++;
          }
          value /= (letype)numpts;
        }
        else // ...or just output field values at the sampled points
          value = FIELDPOINT(fld,0,0,k);
        fprintf(slices_[fld],"%e\n",value*rescaling);
      }
    }
    else if(adjusted_slicedim==2)
    {
      for(j=0;j<final;j+=sliceskip)
        for(k=0;k<final;k+=sliceskip)
        {
          if(sliceaverage==1) // Average over all "skipped" points
          {
            value=0.;
            numpts=0;
            for(y=j;y<j+sliceskip && y<N;y++)
              for(z=k;z<k+sliceskip && z<N;z++)
              {
                value += FIELDPOINT(fld,0,y,z);
                numpts++;
              }
            value /= (letype)numpts;
          }
          else // ...or just output field values at the sampled points
            value = FIELDPOINT(fld,0,j,k);
          fprintf(slices_[fld],"%e\n",value*rescaling);
        }
    }
    else if(adjusted_slicedim==3)
    {
      for(i=0;i<final;i+=sliceskip)
        for(j=0;j<final;j+=sliceskip)
          for(k=0;k<final;k+=sliceskip)
          {
            if(sliceaverage==1) // Average over all "skipped" points
            {
              value=0.;
              numpts=0;
              for(x=i;x<i+sliceskip && x<N;x++)
                for(y=j;y<j+sliceskip && y<N;y++)
                  for(z=k;z<k+sliceskip && z<N;z++)
                  {
                    value += FIELDPOINT(fld,x,y,z);
                    numpts++;
                  }
              value *= rescaling/(letype)numpts;
            }
            else // Average over all "skipped" points
              value = FIELDPOINT(fld,i,j,k);
            fprintf(slices_[fld],"%e\n",value*rescaling);
          }
    }
    fprintf(slices_[fld],"\n");
    fflush(slices_[fld]);
  }
  fprintf(slicetimes_,"%f\n",t);
  fflush(slicetimes_);
}

// Outputs the gravitational wave spectrum
void gw()
{
  if(t == 0.0)
    return;


  static FILE* outfile_gw[gwnflds] = {NULL};
  static FILE* outfile_ast[gwnflds] = {NULL};

  static FILE* outfile_gw_nonTT[gwnflds] = {NULL};
  static FILE* outfile_ast_nonTT[gwnflds] = {NULL};


  
  static char fileName[30];


  if(outfile_gw[0] == NULL)
  {
    for(int fld = 0; fld < gwnflds; fld++)
    {
      snprintf(fileName, 29, "GW_spectrum_%d.dat", fld);

      outfile_gw[fld] = fopen(fileName, "w");
      if(outfile_gw[fld] == NULL)
      {
        printf("ERROR: failed to open file %s for writing. Aborting.\n", fileName);
        abort();
      }
    }
  }


  if constexpr(sast)
  {
    if(outfile_ast[0] == NULL)
    {
      for(int fld = 0; fld < gwnflds; fld++)
      {
        snprintf(fileName, 29, "AST_spectrum_%d.dat", fld);

        outfile_ast[fld] = fopen(fileName, "w");
        if(outfile_ast[fld] == NULL)
        {
          printf("ERROR: failed to open file %s for writing. Aborting.\n", fileName);
          abort();
        }
      }
    }
  }

  if constexpr(sastnonTT)
  {
    if(outfile_ast_nonTT[0] == NULL)
    {
      for(int fld = 0; fld < gwnflds; fld++)
      {
        snprintf(fileName, 29, "AST_spectrum_nonTT_%d.dat", fld);

        outfile_ast_nonTT[fld] = fopen(fileName, "w");
        if(outfile_ast_nonTT[fld] == NULL)
        {
          printf("ERROR: failed to open file %s for writing. Aborting.\n", fileName);
          abort();
        }
      }
    }
  }

  if constexpr(sgwnonTT)
  {
    if(outfile_gw_nonTT[0] == NULL)
    {
      for(int fld = 0; fld < gwnflds; fld++)
      {
        snprintf(fileName, 29, "GW_spectrum_nonTT_%d.dat", fld);

        outfile_gw_nonTT[fld] = fopen(fileName, "w");
        if(outfile_gw_nonTT[fld] == NULL)
        {
          printf("ERROR: failed to open file %s for writing. Aborting.\n", fileName);
          abort();
        }
      }
    }
  }


  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  
  //partially transform value from program units to physical units (up to a scaling)
  for(int gwfld = 0; gwfld < nflds; gwfld++)
    startAdjustDerivativeValues(h[gwfld], hd[gwfld], gwfld, 1);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  
/*  int posx = 20;
  int posy = -N/2+1;
  int posz = N/2;


  for(int i = 0; i < N; i++)
   for(int j = 0; j < N; j++)
     for(int k = 0; k < N; k++)
  {
    //f[0][index(i, j, k)] = N*N*N*sin(2.0 * M_PI * i * 20 / (double) N);
    
    hd[0][0][index(i, j, k)] = f[0][index(i, j, k)];
    //hd[0][0][index(i, j, k)] = 10.0 + sin(2.0 * M_PI * (posx * i + posy * j + posz * k) / (double) N) + sin(2.0 * M_PI * (8 * i - 23 * j + 41 * k) / (double) N);//f[0][index(i, j, k)];
    hd[0][1][index(i, j, k)] = 0.0;
    hd[0][2][index(i, j, k)] = 0.0;
    hd[0][3][index(i, j, k)] = 0.0;
    hd[0][4][index(i, j, k)] = 0.0;
    hd[0][5][index(i, j, k)] = 0.0;

    h[0][0][index(i, j, k)] = 0.0;
    h[0][1][index(i, j, k)] = 0.0;
    h[0][2][index(i, j, k)] = 0.0;
    h[0][3][index(i, j, k)] = 0.0;
    h[0][4][index(i, j, k)] = 0.0;
    h[0][5][index(i, j, k)] = 0.0;
  }*/

  for(int gwfld = 0; gwfld < nflds; gwfld++)
    dh_mom[gwfld].fourierTransform(hd[gwfld]);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());


  if constexpr(sast || sastnonTT)
  {
   for(int gwfld = 0; gwfld < nflds; gwfld++)
      EMT_mom[gwfld].fourierTransform(EMT[gwfld]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
/*  for(int i = 0; i < 3; i++) for(int j = 0; j <= i; j++)
    printf("%d %d FT: %.20f %.20f\n", i, j, dh_mom[0](i, j)(N/2, N/2, N/2).x, dh_mom[0](i, j)(N/2, N/2, N/2).y);

  printf("\n");*/
 /* printf("FT: %.20f %.20f\n", dh_mom[0](0, 0)(posx, posy, posz).x, dh_mom[0](0, 0)(posx, posy, posz).y);
  printf("FT: %.20f %.20f\n", dh_mom[0](0, 0)(posz, posy, posx).x, dh_mom[0](0, 0)(posz, posy, posx).y);
  */


/*  for(int i = -N/2 + 1; i <= N/2; i++)
   for(int j = -N/2 + 1; j <= N/2; j++)
     for(int k = -N/2 + 1; k <= N/2; k++)
  {
  // f[0][index(i, j, k)] = N*N*N*sin(2.0 * M_PI * i * 20 / (double) N);
    if(abs(dh_mom[0](0, 0)(i, j, k).x) > 1)
      printf("\there %d %d %d\n", i, j, k);

  }
  exit(1);*/
/*
  printf("%d %d %d my vals: %e %e\n", posx, posy, posz, dh_mom[0](0, 0)(posx, posy, posz).x, dh_mom[0](0, 0)(posx, posy, posz).y);
  printf("%d %d %d my vals: %e %e\n", posx, posy, posz, dh_mom[0](0, 0)(posz, posy, posx).x, dh_mom[0](0, 0)(posx, posy, posz).y);
  printf("%d %d %d my vals: %e %e\n", posx, posy, posz, dh_mom[0](0, 0)(-posx, -posy, -posz).x, dh_mom[0](0, 0)(-posx, -posy, -posz).y);
*//*  printf("my vals: %e %e\n", dh_mom[0](0, 0)(0, 20, 0).x, dh_mom[0](0, 0)(0, 20, 0).y);
  printf("my vals: %e %e\n", dh_mom[0](0, 0)(0, 0, 20).x, dh_mom[0](0, 0)(0, 0, 20).y);*/

  
  for(int gwfld = 0; gwfld < nflds; gwfld++)
  {
    for(int i = 0; i < 3; i++)
      for(int j = 0; j <= i; j++)
    {
      hdk[gwfld][indexTensor(i, j)]   = dh_mom[gwfld](i, j).getData();
      hdkTT[gwfld][indexTensor(i, j)] = dh_mom_TT[gwfld](i, j).getData(); 
    }
  }

  if constexpr(gwnflds > 1)
  {
    for(int i = 0; i < 3; i++)
        for(int j = 0; j <= i; j++)
    {
      hdkTT[gwnflds - 1][indexTensor(i, j)] = dh_mom_TT[gwnflds - 1](i, j).getData(); 
    }
  }


  if constexpr(sast || sastnonTT)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++)
    {
      for(int i = 0; i < 3; i++)
        for(int j = 0; j <= i; j++)
      {
        EMTk[gwfld][indexTensor(i, j)]   = EMT_mom[gwfld](i, j).getData();
        EMTkTT[gwfld][indexTensor(i, j)] = EMT_mom_TT[gwfld](i, j).getData(); 
      }
    }

    if constexpr(gwnflds > 1)
    {
      for(int i = 0; i < 3; i++)
          for(int j = 0; j <= i; j++)
      {
        EMTkTT[gwnflds - 1][indexTensor(i, j)] = EMT_mom_TT[gwnflds - 1](i, j).getData(); 
      }
    }
  }


/*  dh_mom[0](2, 0)(2, 20, 1).x = -1.0;//indexFT(2 + N/2, 20 + N/2, 1 + N/2)
  printf("%f\n", hdk[0][indexTensor(2, 1)][indexFT(2 + N/2, 20 + N/2, 1 + N/2)].x);
  exit(1);
*/

/*  for(int i = -N/2 + 1; i <= N/2; i++)
   for(int j = -N/2 + 1; j <= N/2; j++)
     for(int k = -N/2 + 1; k <= N/2; k++)
  {
  // f[0][index(i, j, k)] = N*N*N*sin(2.0 * M_PI * i * 20 / (double) N);
    if(abs(dh_mom[0](0, 0)(i, j, k).x) + abs(dh_mom[0](0, 0)(i, j, k).y) > 1)
      printf("\there %d %d %d %f %f\n", i, j, k, dh_mom[0](0, 0)(i, j, k).x, dh_mom[0](0, 0)(i, j, k).y);

  }

  printf("\n");

  for(int i = 1; i < N + 1; i++)
   for(int j = 1; j < N + 1; j++)
     for(int k = 1; k < N + 1; k++)
  {
  // f[0][index(i, j, k)] = N*N*N*sin(2.0 * M_PI * i * 20 / (double) N);
    if(abs(hdk[0][0][indexFT(i, j, k)].x) + abs(hdk[0][0][indexFT(i, j, k)].y) > 1)
      printf("\there %d %d %d %f %f\n", i - N/2, j - N/2, k - N/2, hdk[0][0][indexFT(i, j, k)].x, hdk[0][0][indexFT(i, j, k)].y);

  }
  printf("\n");*/

/*
  int testx = 5;
  int testy = N/2;
  int testz = 1;

  gpuErrchk(cudaDeviceSynchronize());
  hdk[0][indexTensor(0, 0)][indexFT(testx, testy, testz)].x = -1.234;
  printf("TEST %f\n", dh_mom[0](0, 0)(testz - N / 2, testy - N / 2, testx - N / 2).x);

  exit(1);*/
  /*
  printf("\t%.20f\n", hdk[0][0][indexFT(2, 1, 1)].x);
  printf("\t%.20f\n", dh_mom[0](0, 0)(2 - N/2, 1 - N/2, 1 - N/2).x);
  abort();*/
  
  for(int gwfld = 0; gwfld < nflds; gwfld++)
    startTTProj(hdk[gwfld], hdkTT[gwfld]);


  cudaDeviceSynchronize();

  if constexpr(sast)
  {
    for(int gwfld = 0; gwfld < nflds; gwfld++)
      startTTProj(EMTk[gwfld], EMTkTT[gwfld]);

      cudaDeviceSynchronize();
  }
/*  for(int i = 0; i < 3; i++) for(int j = 0; j <= i; j++)
  printf("%d %d FT: %.20f %.20f\n", i, j, dh_mom_TT[0](i, j)(N/2, N/2, N/2).x, dh_mom_TT[0](i, j)(N/2, N/2, N/2).y);
*/
// just copy data -> TT proj is effectively not used
/*  printf("skipping TT projection\n");
  for(int i = 0; i < 3; i++)
    for(int j = 0; j <= i; j++)
    {
  for(int k = 0; k < N + 1; k++)
    for(int l = 0; l < N + 1; l++)
      for(int m = 0; m < N + 1; m++)
      {
        hdkTT[0][indexTensor(i, j)][indexFT(k, l, m)].x = hdk[0][indexTensor(i, j)][indexFT(k, l, m)].x;
        hdkTT[0][indexTensor(i, j)][indexFT(k, l, m)].y = hdk[0][indexTensor(i, j)][indexFT(k, l, m)].y;
      }
    }*/


  //dh_mom[0].projTT(dh_mom_TT[0]);


  //startTestTTProj();
  gpuErrchk(cudaDeviceSynchronize());
/*
  printf("test1: %.20f %.20f\n", dh_mom_TT(1, 1)(1, 2, -N/2 + 1).x, dh_mom_TT(1, 1)(1, 1, 1).y);
  fflush(stdout);
  abort();
*/

/*
  cudaDeviceSynchronize();
  dh_mom.projTT(dh_mom_TT);
  cudaDeviceSynchronize();
  
  printf("test: %.20f %.20f\n", dh_mom_TT(1, 1)(1, 2, -N/2 + 1).x, dh_mom_TT(1, 1)(1, 1, 1).y);
  fflush(stdout);
  abort();
*/
  //set the last field in hdkTT to be the sum of all previous ones
  if constexpr(gwnflds > 1)
    startAddGWFields(gwnflds, hdkTT);

  if constexpr(gwnflds > 1 && sast)
    startAddGWFields(gwnflds, EMTkTT);

  gpuErrchk(cudaDeviceSynchronize());

  for(int gwfld = 0; gwfld < gwnflds; gwfld++)
    calc_GW_spectrum<fTT_type>(outfile_gw[gwfld], dh_mom_TT[gwfld]);

  if constexpr(sast)
  {
    for(int gwfld = 0; gwfld < gwnflds; gwfld++)
      calc_GW_spectrum<fTT_type>(outfile_ast[gwfld], EMT_mom_TT[gwfld]);
  }

  if constexpr(sastnonTT)
  {
    for(int gwfld = 0; gwfld < gwnflds; gwfld++)
      calc_GW_spectrum<cufft_type>(outfile_ast_nonTT[gwfld], EMT_mom[gwfld]);
  }

  if constexpr(sgwnonTT)
  {
    for(int gwfld = 0; gwfld < gwnflds; gwfld++)
      calc_GW_spectrum<cufft_type>(outfile_gw_nonTT[gwfld], dh_mom[gwfld]);
  }

  //transform back to program units
  for(int gwfld = 0; gwfld < nflds; gwfld++)
    startAdjustDerivativeValues(h[gwfld], hd[gwfld], gwfld, -1);
  gpuErrchk(cudaDeviceSynchronize());
}

// Output an image of all fields and derivatives on the lattice (and a few other variables) to a binary file
void checkpoint()
{
  static FILE *grid_; // File for storing lattice images
  // The following variables are all used for keeping track of when to close one field value file and begin writing to another.
  static int numtimes=sizeof(store_lattice_times)/sizeof(letype); // Number of times at which to switch grid image files
  static int current=0; // Index of time value at which to switch files
  static int open; // Indicates whether grid image file is open. (It should be unless the program failed to open the file.)
  int itime; // Integer value of the time up to which a file will be used. Used for generating file names.
  char filename[500];

  static int first=1;
  if(first)
  {
    if(numtimes>0 && store_lattice_times[0]==0.)
      numtimes=0;
    if(numtimes==0) // If no intermediate times are being output simply call the file grid.img
      grid_=fopen("grid.img","wb");
    else // Otherwise label file by the final time it will record data at.
    {
      sprintf(filename,"grid%d.img",(int)store_lattice_times[0]);
      grid_=fopen(filename,"wb");
    }
    first=0;
  }
  else if(current<numtimes && t>store_lattice_times[current]) // If one of the times listed above has been passed switch to another field value file.
  {
    if(open)
      fclose(grid_);
    current++;
    itime = (current < numtimes ? (int)store_lattice_times[current] : (int)tf); // After last time indicated name file with final time of run
    sprintf(filename,"grid%d.img",itime); // Files are labeled by the final time when they will be used, not by the time when they are opened
    grid_=fopen(filename,"wb");
  }

  if(grid_==NULL)
  {
    printf("Error: Grid checkpointing file not open\n");
    open=0;
  }
  else
    open=1;

  if(open) // Write a binary image of all current values to the file grid_.
  {
    rewind(grid_);
    fwrite(&run_number, sizeof(run_number), 1, grid_);
    fwrite(&t, sizeof(t), 1, grid_);
    fwrite(&a, sizeof(a), 1, grid_);
    fwrite(&ad, sizeof(ad), 1, grid_);
    fwrite(f, sizeof(letype), nflds*gridsize, grid_);
    fwrite(fd, sizeof(letype), nflds*gridsize, grid_);
    fflush(grid_);
  }
}

// Convert a time in seconds to a more readable form and print the results
void readable_time(int t, FILE *info_)
{
  int tminutes=60,thours=60*tminutes,tdays=24*thours;

  if(t==0)
  {
    fprintf(info_,"less than 1 second\n");
    return;
  }

  // Days
  if(t>tdays)
  {
    fprintf(info_,"%d days",t/tdays);
    t = t%tdays;
    if(t>0)
      fprintf(info_,", ");
  }  
  // Hours
  if(t>thours)
  {
    fprintf(info_,"%d hours",t/thours);
    t = t%thours;
    if(t>0)
      fprintf(info_,", ");
  }
  // Minutes
  if(t>tminutes)
  {
    fprintf(info_,"%d minutes",t/tminutes);
    t = t%tminutes;
    if(t>0)
      fprintf(info_,", ");
  }
  // Seconds
  if(t>0)
    fprintf(info_,"%d seconds",t);
  fprintf(info_,"\n");
  return;
}

/////////////////////////////////////////////////////
// Externally called function(s)
/////////////////////////////////////////////////////

// Output information about the run parameters.
// This need only be called at the beginning and end of the run.
void output_parameters()
{
  static FILE *info_;
  static time_t tStart,tFinish; // Keep track of elapsed clock time

  static int first=1;
  if(first) // At beginning of run output run parameters
  {
    sprintf(name_,"info%s",ext_);
    info_=fopen(name_,mode_);

    fprintf(info_,"--------------------------\n");
    fprintf(info_,"Model Specific Information\n");
    fprintf(info_,"--------------------------\n");
    modelinfo(info_);

    fprintf(info_,"\n--------------------------\n");
    fprintf(info_,"General Program Information\n");
    fprintf(info_,"-----------------------------\n");
    fprintf(info_,"Grid size = %d^%d\n",N,NDIMS);
    fprintf(info_,"Number of fields = %d\n",nflds);
    fprintf(info_,"L = %f\n",L);
    fprintf(info_,"dt = %f, dt/dx = %f\n",dt,dt/dx);
    fprintf(info_,"seed = %d\n", seed);
    if(expansion==0)
      fprintf(info_,"No expansion\n");
    else if(expansion==1)
      fprintf(info_,"Fixed background expansion\n");
    else if(expansion==2)
      fprintf(info_,"Expansion calculated self-consistently\n");
    time(&tStart);
    fprintf(info_,"\nRun began at %s",ctime(&tStart)); // Output date in readable form
    first=0;
  }
  else // If not at beginning record elapsed time for run
  {
    time(&tFinish);
    fprintf(info_,"Run ended at %s",ctime(&tFinish)); // Output ending date
    fprintf(info_,"\nRun from t=%f to t=%f took ",t0,t);
    readable_time((int)(tFinish-tStart),info_);
    fprintf(info_,"\n");
  }

  fflush(info_);
  return;
}

// Calculate and save quantities (means, variances, etc.). If force>0 all infrequent calculations will be performed
void save(int force)
{
  cudaDeviceSynchronize();
  //return;


  static letype tsave = t0; // Used to indicate when output files should be flushed.
  bool flush = false; // Indicates whether to flush files, checkpoint, and perform certain infrequent calculations

  if(checkpoint_interval == 0.) // In this case only checkpoint at the end of the run
    tsave = tf;

  if(t >= tsave || force > 0) // Check if the program should flush
  {
    tsave = t + checkpoint_interval;
    flush = true;
  }
  


  if(t > 0.0) // Synchronize field values and derivatives. There's a special case at t=0 when fields and derivatives should stay synchronized
    evolve_fields(-0.5*dt);


  //prefetchAsyncFields(f, fd, 0);
  //prefetchAsyncGWFields(h, hd, EMT, 0);




  if(expansion == 2) // Bring a'' up to date. Leapfrog correction isn't used since all quantities are known at the time t
    evolve_scale(0);

  prefetchAsyncFields(f, fd, cudaCpuDeviceId);
  //prefetchAsyncGWFields(h, hd, EMT, cudaCpuDeviceId);


  cudaDeviceSynchronize();

  // Model-specific output
  if(smodel) // Allows each model to define its own output function(s) and set rescaling for other outputs
    model_output(flush, ext_);
  if(smeansvars) // Calculate means and variances of fields
    meansvars(flush);
  if(sexpansion && expansion == 2) // Output scale factor and its derivatives
    scale(flush);
  // Infrequent calculations
  /*
  if(flush && t>=0.5)
  {
    dh_mom.projTT(dh_mom_TT);
    FILE* GW_spectrum = fopen("GW_spectrum.dat", "a");
    calc_GW_spectrum(GW_spectrum);
    fclose(GW_spectrum);
  }*/

  if((sgw || sgwnonTT || sast || sastnonTT) && flush)
    gw();


  letype totalEnergy = 0.0;
  if(senergy && flush && t>=tenergy) // Calculate all contributions to energy density
    totalEnergy = energy();
  
  if(soscillon && flush && senergy && t>= tenergy)
    oscillon(totalEnergy);

  if(shistograms && flush && t >= thistograms) // Calculate histograms of all output fields
    histograms();
  if(shistograms2d && flush && t >= thistograms2d) // Calculate two dimensional histograms of pairs of fields
    histograms_2d();
  if(sslices && flush && t >= tslices) // Calculate two dimensional histograms of pairs of fields
    slices();
  if(sspectra && flush && t >= tspectra) // Calculate and power spectra of all output fields
    spectra();

  if(t > 0.) // Desynchronize field values and derivatives (for leapfrog)
    evolve_fields(.5*dt);
  if(scheckpoint && flush && t >= tcheckpoint) // Save an image of the grid.
    checkpoint();
}