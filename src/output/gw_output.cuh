#pragma once

#include "src/latticeeasy.cuh"
#include "src/utils/symmTensor.cuh"


//partially transform value from program units to physical units (up to a scaling)
//direction = 1 -> transform to physical units, direction = -1 -> transform to program units
void startAdjustDerivativeValues(letype** h, letype** hd, const int gwfld, const int direction);

//projects to TT space of f and saves the result to tf
void startTTProj(cufft_type** f, fTT_type** tf);

//test if the TT projected tensor actually is tracelessness and transverse
void startTestTTProj();

//set the last field in hdkTT to be the sum of all previous ones
void startAddGWFields(int gwnflds, fTT_type*** hdkTT);
