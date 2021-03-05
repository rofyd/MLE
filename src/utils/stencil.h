#pragma once

#include <iostream>
#include <vector>
#include <math.h>

#include "src/utils/utils.cuh"

namespace Stencil
{
	/*returns proper modulos e.g. (-4 % 5 = 1)*/
	inline int mod(int a, int b) { return (a % b + b) % b; }

	/*finite central difference with chosable accuracy. Also take the direction of differentiation
	note that: dir = 0 corresponds to x direction, 1 to y and 2 to z.*/
	template <typename T, int accuracy, int ndims, int N>
	inline T fd_central(T* field, const Position& pos, int dir)
	{
		std::vector<T> coeff(accuracy + 1); //TODO: check that accurace has appropriate values

		if constexpr(accuracy == 2)
			coeff = {-0.5, 0.0, 0.5};
		else if constexpr(accuracy == 4)
			coeff = {1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0};
		else if constexpr(accuracy == 6)
			coeff = {-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
		else if constexpr(accuracy == 8)
			coeff = {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0};

		const int length = coeff.size();
		T res = 0.0;


		if constexpr(ndims == 1)
		{
			for(int i = 0; i < length; i++)
				res += field[index(mod(pos.x - length/2 + i, N))] * coeff[i];
		}
		else if constexpr(ndims == 2)
		{
			switch(dir)
			{
				case 0:
					for(int i = 0; i < length; i++)
						res += field[index(mod(pos.x - length/2 + i, N), pos.y)] * coeff[i];
					break;
				case 1:
					for(int i = 0; i < length; i++)
						res += field[index(pos.x, mod(pos.y - length/2 + i, N))] * coeff[i];
					break;
			}
		}
		else if constexpr(ndims == 3)
		{
			/*
			for(int i = 0; i < length; i++)
				printf("%f ", field[(pos.x - length/2 + i) % N][pos.y][pos.z] * coeff[i]);
			printf("\n");*/
			switch(dir)
			{
				case 0:
					for(int i = 0; i < length; i++)
						res += field[index(mod(pos.x - length/2 + i, N), pos.y, pos.z)] * coeff[i];
					break;
				case 1:
					for(int i = 0; i < length; i++)
						res += field[index(pos.x, mod(pos.y - length/2 + i, N), pos.z)] * coeff[i];
					break;
				case 2:
					for(int i = 0; i < length; i++)
						res += field[index(pos.x, pos.y, mod(pos.z - length/2 + i, N))] * coeff[i];
					break;
			}
		}

		//printf("res %f, acc %d, dir %d\n", res, accuracy, dir);

		return res;
	}

	template <typename T, int accuracy, int ndims, int N>
	inline T isotropic_grad(T* field, const Position& pos)
	{
		std::vector<T> coeff(4);

		if constexpr(accuracy == 0)
			coeff = {-6.0, 1.0, 0.0, 0.0};
		else if constexpr(accuracy == 1)
			coeff = {-4.0, 1.0/3.0, 1.0/6.0, 0.0};
		else if constexpr(accuracy == 2)
			coeff = {-14.0/3.0, 2.0/3.0, 0.0, 1.0/12.0};
		else if constexpr(accuracy == 3)
			coeff = {-64.0/15.0, 7.0/15.0, 1.0/10.0, 1.0/30.0};

		T res = 0.0;

		for(int i = -1; i <= 1; i++) for(int j = -1; j <= 1; j++) for(int k = -1; k <= 1; k++)
			res += pow(field[index(mod(pos.x + i, N), mod(pos.y + j, N), mod(pos.z + k, N))] - field[index(pos.x, pos.y, pos.z)], 2) * coeff.at(abs(i) + abs(j) + abs(k));

		return 0.5 * res;
	}

	template <typename T, const int accuracy, const int ndims, const int N>
	inline T isotropic_lapl(const T* field, const Position& pos, const T norm)
	{
		std::vector<double> coeff(4); //TODO I would like to use letype instead of double but this messes up the accuracy for calcualting the gradient energy when using floats

		if constexpr(accuracy == 0)
			coeff = {-6.0, 1.0, 0.0, 0.0};
		else if constexpr(accuracy == 1)
			coeff = {-4.0, 1.0/3.0, 1.0/6.0, 0.0};
		else if constexpr(accuracy == 2)
			coeff = {-14.0/3.0, 2.0/3.0, 0.0, 1.0/12.0};
		else if constexpr(accuracy == 3)
			coeff = {-64.0/15.0, 7.0/15.0, 1.0/10.0, 1.0/30.0};

		for(int i = 0; i < 4; i++)
			coeff[i] *= norm;

		T res = 0.0;

		letype c = 0.0;
    	letype t = 0.0;
    	letype w = 0.0;

		for(int i = -1; i <= 1; i++) for(int j = -1; j <= 1; j++) for(int k = -1; k <= 1; k++)
		{
			w = coeff.at(abs(i) + abs(j) + abs(k)) * field[index(mod(pos.x + i, N), mod(pos.y + j, N), mod(pos.z + k, N))];
			t = res + w;
			if(abs(res) >= abs(w))
				c += (res - t) + w;
			else
				c += (w - t) + res;
			res = t;

			//res += coeff.at(abs(i) + abs(j) + abs(k)) * field[index(mod(pos.x + i, N), mod(pos.y + j, N), mod(pos.z + k, N))];
		}
		res += c;

		return res;
	}

	/*small test to calculate the gradient at a given position*/
	template <typename T, int accuracy, int ndims, int N>
	inline T fd_gradsq_test(T* field, const Position& pos)
	{
		T res = 0.0;
		for(int dir = 0; dir < ndims; dir++)
			res += pow(Stencil::fd_central<T, accuracy, ndims, N>(field, pos, dir), 2);
		return res;
	}
};
