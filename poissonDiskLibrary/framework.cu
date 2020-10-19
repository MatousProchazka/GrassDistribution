// Copyright 2020 Matous Prochazka, Bohemia Interactive, a.s.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   1. Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//   2. Redistributions in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//
//   3. Neither the name of the copyright holder nor the names of its contributors
//      may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "stdio.h"
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "thrust/system_error.h"
#include "thrust/device_vector.h"

#include <windows.h>

#include "cuda_poisson_lib.h"
/*
This main function serves only as a debugger for the actual library
*/


#define CUDA_CHECK(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

int main(int argc, char** argv) {
	std::vector<float> points;
	unsigned char* radVals;
	unsigned h, w;
	int radius = 800;
	int maxTries = 50;

	float bounds[4];

	bounds[0] = -3000.f;
	bounds[1] = 3000.f;
	bounds[2] = 3000.f;
	bounds[3] = -3000.f;


	try {
		//for the purpose of meassuring, maybe create the test of speed for the poisson disk I used on CPU?
		
		for(int i = 0; i < 1; i++){
			points.clear();

			CUDA_CHECK(cudaPoissonSampling::PoissonDiskDistribution(points, radius, maxTries, bounds));
		}
		for (int i = 0; i < 1; i++)
		{
			//points.clear();
			//CUDA_CHECK(cudaPoissonSampling::PoissonDiskDistribution(points, "terrainMap2.png", maxTries, bounds, 10, 80));
			cudaPoissonSampling::partitionAttributes part = {2,2,i};

			//CUDA_CHECK(cudaPoissonSampling::PoissonDiskDistribution(points, radVals,w,h, "snow_tigerG.png", maxTries, bounds, 10, 40, part));
			//CUDA_CHECK(cudaPoissonSampling::PoissonDiskDistribution(points, radVals,w,h, "snow_tigerG.png", maxTries, bounds, 10, 40));
			//CUDA_CHECK(cudaPoissonSampling::PoissonDiskDistribution(points, "terrainMap2.png", maxTries, bounds, 10, 80));

		}
		//CUDA_CHECK(PoissonDiskDistribution(points, radius, maxTries, bounds));
		
	}
	catch (thrust::system_error& c) {
		std::cout << c.what() <<std::endl;
	}
	catch (const std::exception& a) {
		std::cout << a.what() << std::endl;
	}
	
	
	DEBUG_TEXT("size %i\n", (int)points.size()/2);
	printf("size %i \n", (int) points.size()/2);

	for (int i = 0; i < points.size(); i += 2)
	{
		DEBUG_TEXT("x %f y %f \n", points[i], points[i+1]);
		if (abs(points[i]) > bounds[1] || abs(points[i + 1]) > bounds[1]) {
			printf("Error: the point is too big/small.\n");
		}
	}

	system("Pause");

	return 0;
}
