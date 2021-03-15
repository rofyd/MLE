#pragma once


/*
file for hilltop model (1803.08047)
*/

/*
General comments about the model.h file:
This file contains the following functions - all called externally
modelinfo(FILE *info_) outputs information about the model and model-specific parameters to a file.
modelinitialize() performs any model-specific initialization
potential_energy(int term, letype *field_values) calculates the average potential energy density term by term. The variable num_potential_terms (just above this function) specifies how many separate potential terms are used in this model.
dvdf(int fld, int i, int j, int k) calculates the potential term in the equation of motion, dV/dfield, for the field fld at the lattice point (i,j,k)
effective_mass(letype mass_sq[], letype *field_values) calculates the square masses of the fields and puts them in the array mass_sq. The parameter beginning tells the function to use initial field values - if this parameter is zero then the field quantities will be calculated dynamically.
model_output(int flush, char *ext_) allows each model to include its own specialized output function(s). The parameter flush is set to 1 when infrequent calculations are being performed and 0 otherwise. The string ext_ gives the extension for output filenames.
*/

/* - This section should be copied into parameters.h when using this model
// ---Adjustable parameters for TWOFLDLAMBDA model--- //
const double pmass = 5.35237e-8;//5.e-6; //mass of the inflaton
const double param_1st = 0.769129; //(phi_i / upsilon)^2

const int N = 128; // Number of points along each edge of the cubical lattice
const double L = 19.05;

const double initderivs[] = {0.158505 + res_r * 0.000678};

the following two values are the necessary changes to use this file for 1607.01314

f0 = 0.000159577
const double initderivs[] = {0.00218058};
const double L = 14.6969;
const double param_1st = 0.0064; //(phi_i / upsilon)^2
*/


// Rescaling parameters.
// The program variables ("_pr") are defined as
//   f_pr = rescale_A a^rescale_r f (f=field value)
//   x_pr = rescale_B x (x=distance)
//   dt_pr = rescale_B a^rescale_s dt (t=time)
// The constants beta, cpl, and f0 are used to set the variable rescalings rescale_A, rescale_B, rescale_r, and rescale_s.
// These rescaling constants may be reset independently; their settings in terms of beta, cpl, and f0 are suggestions. See the documentation for more details.
// These rescalings are intrinsic to the model and probably shouldn't be changed for individual runs. Adjustable parameters are stored in "parameters.h"


const double beta_exp = 2.0; // Exponent of the dominant term in the potential
const double cpl = pw2(pmass); // Coefficient of the dominant term in the potential (up to numerical factors - see documentation)
const double f0 = 0.001749361899560282;//0.00174936; // Initial value of phi in Planck units, typically the point at which phi'=0
// By default these are automatically set to A=1/f0, B=sqrt(cpl) f0^(-1+beta/2), R=6/(2+beta), S=3(2-beta)/(2+beta). They may be adjusted to different values, but the relationship S=2R-3 must be maintained for the program equations to remain correct.
const double rescale_A_inv = f0;
const double rescale_A = 1.0/rescale_A_inv;
const double rescale_B = pmass;
const double rescale_r = 1.0; //6.0 / (2.0 + beta_exp);
// The value of S in terms of R SHOULD NOT be changed.
const double rescale_s = 2.0 * rescale_r - 3.0;

// Other global variables
// The array model_vars is intended to hold any model-specific, non-constant global variables.
// These variables should be initialized in modelinitialize() below
// Even if you're not using any, num_model_vars should be at least 1 to keep some compilers happy.
const int num_model_vars = 1;
// Model specific variables: None are defined for this model.
extern letype model_vars[num_model_vars];

// Macros to make the equations more readable: The values of fld are 0=Phi,1=Chi
#define PHI FIELD(0)
//#define CHI FIELD(1)

// Model specific details about the run to be output to an information file
inline void modelinfo(FILE *info_)
{
  // Name and description of model
  fprintf(info_,"Hilltop model (1803.08047)\n");

  // Model specific parameter values
  fprintf(info_, "pmass = %f\n", pmass);
  fprintf(info_, "param 1 = %f\n", param_1st);
  //fprintf(info_, "g2m4 (g^2/m^4)= %f\n", param_2nd);
}

// Perform any model specific initialization
// This function is called twice, once before initializing the fields (which_call=1) and once after (which_call=2)
inline void modelinitialize(int which_call)
{
  if(which_call==1)
  {
    if(nflds!=1)
    {
      printf("Number of fields must be 1. Exiting.\n");
      exit(1);
    }
  }
}

// The constant num_potential_terms must be defined for use by outside functions
// Terms: term=0: 1/4 lambda phi^4 --- term=1: 1/2 g^2 phi^2 chi^2
const int num_potential_terms=1; // Number of terms that are calculated separately in the potential
// Potential energy terms
// See documentation for normalization of these terms.
// When setting initial conditions field values will be supplied in the array field_values. Otherwise this function will calculate them on the lattice.
inline letype singlePotEnergy(int term, letype** field, INDEXLIST)
{
  switch(term)
  {
    case 0:
      return 1.0 / (72.0 * param_1st) * pow(a, -2.0 * rescale_s + 2.0 * rescale_r) * pw2(1 - pow(a, -6.0 * rescale_r) * pow(param_1st, 3.0) * pow(field[0][index(i, j, k)], 6));
    default:
      break;
  }
    return 0.0;
}


const int num_pot_params = num_potential_terms + 1;

inline void prepareParamsForPotEnergy(letype* params)
{  
  params[0] = pow(param_1st, 3) * pow(a, -6.0 * rescale_r);
  params[1] = pow(a, -2.0 * rescale_s + 2.0 * rescale_r) / (72.0 * param_1st);
}

__device__ inline letype cu_singlePotEnergy(int term, const letype* __restrict__ params, letype** __restrict__ field, const size_t pos)
{
  switch(term)
  {
    case 0:
      return params[1] * cu_pow2(1 - params[0] * cu_pow(field[0][pos], 6.0));
    default:
      break;
  }

  return 0.0;
}

__device__ inline letype cu_singlePotEnergy(int term, const letype* __restrict__ params, letype** __restrict__ field, INDEXLIST)
{
  return cu_singlePotEnergy(term, params, field, index(i, j, k));
}


/*__device__ inline letype cu_singlePotEnergy(int term, const letype d_a, letype** field, INDEXLIST)
{
  letype rescaling_a = 1 / pow(d_a, 9);
  letype rescaling_val = pow(param_1st, 3);

  switch(term)
  {
    case 0:
      return pow(1 - rescaling_val * rescaling_a * pow(field[0][index(i, j, k)], 6), 2) * pow(d_a, 3) / (72.0 * param_1st);
    default:
      break;
  }

  return 0.0;
}*/

inline double potential_energy(int term, letype* field_values)
{
  DECLARE_INDICES
  double potential = 0.;

  if(field_values==NULL) // If no values are given calculate averages on the lattice
  {
    // Loop over grid to calculate potential term
    LOOP
    {
      potential += singlePotEnergy(term, f, i, j, k);    
    }

    // Convert sum to average
    potential /= (double)gridsize;
  }
  else
  {
    potential = singlePotEnergy(term, &field_values, 0, 0, 0);
  }

  return (potential);
}

// Potential terms in the equations of motion, dV/dfield, evaluated at point (i,j,k)
// See documentation for details on the normalization of these terms

const int num_dvdf_params = num_potential_terms + 1;

inline void prepareParamsForCalcLapl(letype* params)
{
  params[0] = pow(param_1st, 2) * pow(a, -2.0 * rescale_s - 4.0 * rescale_r) / 6.0;
  params[1] = pow(param_1st, 3) * pow(a, -6.0 * rescale_r);
}

__device__ inline letype d_dvdf(int fld, const letype* __restrict__ params, letype** __restrict__ field, letype d_a, INDEXLIST)
{
  const letype fValue = field[fld][index(i, j, k)];

  return( params[0] * cu_pow(fValue, 5.0) * ( -1.0 + params[1] * cu_pow(fValue, 6.0)) );
  //return( PHI - 4.0 * rescaling_a * param_1st * PHI * pw2(PHI) + param_2nd * 6.0 * pw2(rescaling_a) * PHI * pw2(pw2(PHI)));
}
/*
__device__ inline double d_dvdf(const letype* field, letype d_a, INDEXLIST)
{
  return( pow(param_1st, 2)/(6 * pow(d_a, 6)) * ( -pow(field[index(i, j, k)], 5) + pow(param_1st, 3)/pow(d_a, 9) * pow(field[index(i, j, k)], 11)) );
  //return( PHI - 4.0 * rescaling_a * param_1st * PHI * pw2(PHI) + param_2nd * 6.0 * pw2(rescaling_a) * PHI * pw2(pw2(PHI)));
}*/


inline letype dvdf(int fld, INDEXLIST)
{
  letype val1 = pow(param_1st, 2) * pow(a, -2.0 * rescale_s - 4.0 * rescale_r) / 6.0;
  letype val2 = pow(param_1st, 3) * pow(a, -6.0 * rescale_r);

  return( val1 * pow(PHI, 5) * ( -1.0 + val2 * pow(PHI, 6)) );
  //return( PHI - 4.0 * rescaling_a * param_1st * PHI * pw2(PHI) + param_2nd * 6.0 * pw2(rescaling_a) * PHI * pw2(pw2(PHI)));
}

// Calculate effective mass squared and put it into the array mass_sq[] (used for initial conditions and power spectra)
// See documentation for normalization of these terms
// When setting initial conditions field values will be supplied in the array field_values. Otherwise this function will calculate them on the lattice.
inline void effective_mass(double mass_sq[], letype* field_values)
{
  DECLARE_INDICES
  int fld;
  double fldpw4[nflds]; // Square value of field
  double fldpw10[nflds]; //quartic value of the field
  double correction; // Used to adjust masses by the appropriate power of the scale factor. (See documentation.)

  // Loop over fields to find mean-square value
  if(field_values==NULL) // If no values are given calculate averages on the lattice
  {
    for(fld=0;fld<nflds;fld++)
    {
      fldpw4[fld] = 0.;
      fldpw10[fld] = 0.;
      LOOP
      {
        fldpw4[fld] += pow(FIELD(fld), 4);
        fldpw10[fld] += pow(FIELD(fld), 10);
      }
      fldpw4[fld] /= (double)gridsize;
      fldpw10[fld] /= (double)gridsize;
    }
  }
  else // If field values are given then use them instead
    for(fld=0;fld<nflds;fld++)
    {
      fldpw4[fld] = pow(field_values[fld], 4);
      fldpw10[fld] = pow(field_values[fld], 10);
      //fldsqrd[fld] = pw2(field_values[fld]);
      //fldqrt[fld] = pw2(fldsqrd[fld]);
    }

  double val1 = pow(param_1st, 2) * pow(a, -2.0 * rescale_s - 4.0 * rescale_r) / 6.0;
  double val2 = pow(param_1st, 3) * pow(a, -6.0 * rescale_r);

  mass_sq[0] = val1 * (-5.0 * fldpw4[0] + 11.0 * val2 * fldpw10[0]);
  //mass_sq[0] = 1.0 - 12.0 * rescaling_a * param_1st * fldsqrd[0] + 30.0 * param_2nd * pw2(rescaling_a) * fldqrt[0];

  // Put in scale factor correction. This calculation should be the same for all models.
  if(expansion > 0) // If there's no expansion don't bother with this.
  {
    correction = pow(a, 2.*rescale_s + 2.);
    for(fld = 0; fld < nflds; fld++)
      mass_sq[fld] *= correction;
  }
}

// Model-specific output functions
inline void model_output(int flush, char *ext_){}

#undef PHI
#undef CHI
