# Modified LatticeEasy
Simulating the production of oscillons and the resulting GW spectrum for my M.Sc. thesis.

Main code (LATTICEEASY and CLUSTEREASY) is taken from http://www.felderbooks.com/latticeeasy/

Requirements:
* CUDA toolkit 11.2 (https://developer.nvidia.com/cuda-downloads)
* gcc 9.2.0

Program still has the same structure as the original LatticeEasy. See their official documentation for details http://www.felderbooks.com/latticeeasy/.
The parameters.cuh file contains five new switches that enable/disable the calculation of gravitational waves and the output of local energy densities. If you want to change the used stencil, then adjust the halo size in parameters.cuh and uncomment the respective stencil in src/utils/utils.cuh file. The models folder contains model files for three example models. You can copy them to variables/ and rename them to models.cuh in order to run the chosen model. The folder scripts/ contains a python file that creates a volume render of local overdensities.
The folder docs/ contains some information on design decision and the stencils.

Build via 'make' and run with './latticeeasy'.
