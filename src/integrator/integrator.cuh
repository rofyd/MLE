#pragma once

//#include "variables/parameters.h"
#include "src/latticeeasy.cuh"
#include "src/utils/symmTensor.cuh"
#include "variables/parameters.cuh"
#include "src/utils/utils.cuh"

void copyFieldsToConstantMemory();

void startCalcMeanPotEnergy(letype** f, letype* potEnergy);

void startCalcMeanGradEnergy(letype** f, letype curr_gradEnergy[nflds]);

void startResetEMT(letype** EMT);

void startEvolveGWd(letype** h, letype** hd, letype** EMT, letype dtime, const int gwfld);

void startCalcLaplacian(letype dtime, letype curr_gradEnergy[nflds]);

void startLeapFrog(int amt, letype** field, letype** fieldd, letype dtime);

void startLeapFrogMean(int gwfld, int amt, letype** field, letype** fieldd, letype dtime);

void meanPot2(letype** f);

void calchmean();

void calchdmean();

void startadjusthd();

void startadjusth();
//void runSimulation(int maxSteps, letype* f, letype* df, letype* mean);

template <unsigned int blockSize>
__global__ void cu_pot_reduce(const int term, const letype* __restrict__ params, letype** __restrict__ fields, letype* __restrict__ reduct, const size_t array_len);