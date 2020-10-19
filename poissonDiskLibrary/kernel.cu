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

#include "cuda_poisson_lib.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "QuadTree.h"
#include "thrust/system_error.h"
#include "thrust/device_vector.h"

#include "external/lodepng.cpp"

#include <stdio.h>
#include "time.h"

#pragma("", off)

#define CUDA_CHECK(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort);
		throw thrust::system_error(code, thrust::cuda_category(), file + std::string(" ") + std::to_string(line));
	}
}


/****************************************************************************
*
*								****Header*****
*
**************************************************************************/

// function in parallel takes nodes in same tree level and for each creates new 4 children. Migrates down its own position based on position 
// and creates 3 more points for remaining children. 
// @tries determines how many times does the algorithmus attempt to throw random rart for new node position before giving up.
// Each position is upon creation checked whether distance is correct. If so, point is accepted into node and generation of further points stopped
// If no point is within correct distance, node remains empty.
__global__
void cuda_SampleThrow(int radius, int tries, quadNode** root, int treeLevelNodeFirstIndex, int numOfComputations, curandState* state, int maxCubesAmount);
//sector specific version of the funciton
__global__
void cuda_SampleThrow(int radius, int tries, quadNode** root, int treeLevelNodeFirstIndex, int numOfComputations, curandState* state, int maxCubesAmount, int sector, int upperRootCubesAmount);

__global__
void cuda_SampleThrow(unsigned char* radiusValues, int width, int height, int tries, quadNode** root, int treeLevelNodeFirstIndex, int numOfComputations, curandState* state, int maxCubesAmount, float4 bounds, int lowerThreshold, int upperThreshold, cudaPoissonSampling::partitionAttributes partition);
__global__
void cuda_SampleThrow(unsigned char* radiusValues, int width, int height, int tries, quadNode** root, int treeLevelNodeFirstIndex, int numOfComputations, curandState* state, int maxCubesAmount, float4 bounds, int lowerThreshold, int upperThreshold, cudaPoissonSampling::partitionAttributes partition, int sector, int upperRootCubesAmount);

//host version of cuda_SampleThrow (currently not used)
void SampleThrow(int radius, int tries, quadNode** root, int indexOfCurrentNode, int totalCubes);
void SampleThrow(int radius, int tries, quadNode* &root, int indexOfCurrentNode);

// unfolding the leaves positions from pointer array into vector
void UnfoldIntoVector(quadNode* root, std::vector<float> &result);
void UnfoldIntoVector(quadNode root, std::vector<float> &result); 
// unfolding the leaves positions from array into vector
void UnfoldArrayIntoVector(quadNode* nodeArray, std::vector<float> &result, int currLevel);


// computation of how many pontential points can the quadTree fit in
int ComputeNumOfCubes(int radius, const float* domain, int& resNumOfCubes, int& numOfLevels); 
int ComputeNumOfCubes(int radius, const float* domain, int& numOfLevels); 
int ComputeNumOfCubes(int numOfLevels);

//computation of width of currentLevel 
__host__ __device__
int ComputeWidthOfCurrentLevel(int numOfLevels);

// ---------QUAD TREE FUNCTIONS---------

//Add new node into subtree on pointerToNode position of subroot, afterwards pops node down until it is placed on leaf level
__host__ __device__
void AddNode(quadNode** root, float2 point, int pointerToNode, int maxCubeAmount); 
__host__ __device__
void AddNode(quadNode* &root, float2 point, int pointerToNode); 

//Find nearest node if there is any within maxDist
__host__ __device__
bool IsAnyNodeWithinDist(quadNode** root, float2 position, float maxDist); 
__host__ __device__
bool IsAnyNodeWithinDist(quadNode* root, float2 position, float maxDist); 
// Migrate points from current Level into lower level
// Deny parent point from being overriden and readd into it same node
__host__ __device__
bool MigratePointLower(quadNode** root, int& resSector, int currentNodeIndex, int maxCubeAmount);
__host__ __device__
bool MigratePointLower(quadNode* &root, int& resSector, int currentNodeIndex);

//helper function for detecting whether a node is a leaf
__host__ __device__
bool IsLeaf(quadNode* root);

//seems cudaFree does the trick?	
//Function to thoroughly clean the tree nodes
void ClearupTree(quadNode* &root);

// -----------RANDOM FUNCTIONS --------
// kernel to set up seed of the curand
__global__ void Setup_rand(curandState* state, int seed, int numOfNums);

////Generates rundom number within topLeft and botRight
__host__
float2 RandomThrow(float2 topLeft, float2 botRight);
__device__
float2 RandomThrow(float2 topLeft, float2 botRight, curandState* state);


// -----------HELPER FUNCTIONS --------
__host__ __device__
float CompDist(float firstPar, float secPar);
__host__ __device__
float Minim(float num1, float num2);
__host__ __device__
float Maxim(float num1, float num2);
__host__ __device__
inline float Dist(float2 a, float2 b);

//Creates new node on pointerToNode index
__host__ __device__
struct quadNode* NewNode(quadNode** root, float2 point, float2 topLeft, float2 botRight, int pointerToNode);
__host__ __device__
struct quadNode* NewNode(quadNode* &root, float2 point, float2 topLeft, float2 botRight, int pointerToNode);

//checks the least orthogonal distance to borders topLeft to botRight
__host__ __device__
int BoundaryMinDist(float2 position, float2 topLeft, float2 botRight);
//check if position is located within borders topLeft to botRight
__host__ __device__
bool InBoundary(float2 position, float2 topLeft, float2 botRight); 

//iteratively checkes whether the node is being inserted onto coresponding leaf place
__host__ __device__
bool InsertQuad(quadNode** root, float2 newPos, int indexOfCurrentNode, int maxCubesAmount);
__host__ __device__
bool InsertQuad(quadNode* &root, float2 newPos, int indexOfCurrentNode);

//if the children does not exist, then create new node, otherwise pushes into stack
//helper function for InsertQuad
__host__ __device__
void CheckAndCreateInChild(quadNode** root, quadNode* &node, int parentIdx, float2 newPos, float2 iTopLeft, float2 iBotRight, int sector, int stack[], int &stackIndex, int maxCubesAmount);

//iteratively looks for the closest neighbour of point
__host__ __device__
int FindNearest(quadNode** root, float2 point, float &minDist);
__host__ __device__
int FindNearest(quadNode* root, float2 point, float &minDist); 

//Checks distance of the leaf
//helper function for FindNearest
__host__ __device__
void CheckLeafDist(int currIndex, quadNode* leaf, float2 point, float &curDist, float &minDist, int stack[], int& index, int sector);

//Checks the childNode and puts it into corresponding stack
//helper function for FindNearest
__host__ __device__
void CheckChildNode(quadNode* &node, float2 point, int stack1[], int stack2[], int& index1, int& index2, int currIndex, int sector);

//currently potentially bias sectors [0, 1; 2, 3]
__device__ __host__
void ComputeBordersForSector(float2 topLeft, float2 botRight, int sector, float2& resTopLeft, float2& resBotRight); 

//determines in which child node does point belong to
__device__ __host__
int DetermineNodeSector(float2 position, float2 topLeft, float2 botRight);

//-----------TEST FUNCTIONS-------//
//test kernel to check integrity of root
__global__
void TestOfRootIntegrity(quadNode** root, int maxNumOfIndeces);
__global__
void TestOfRootIntegrity(quadNode* root, int maxNumOfIndeces, float2 position, float2 tl, float2 br);
//Testing random function
__global__
void testCRand(curandState* state);

//purely testing class to test if all points are indeed further than radius
// returns 1 if no node is within radius
//high complexity goes through whole tree and checks all nodes distance
__host__ __device__
int testDistOfDistance(quadNode** root, float2 point, int radius);

//------------------MEMORY FUNCTIONS-------------------//
//Kernel redirects the pointers of pointer array to array of structures
__global__
void SetpointerToData(quadNode** pointerArray, quadNode* dataMemory, int maxAmount);

//Kernel moves a pointer array into regular array
__global__
void MoveMemoryToArray(quadNode** d_root, quadNode* memory);
__global__
void MoveMemoryToArray(quadNode** d_root, quadNode* memory, int initialIdx, int nonLeafCubesNum, int upperRoots);

//Function copies the tree on device from tree on host
__host__
cudaError_t CopyTreeToDevice(quadNode** d_root, quadNode** c_root, int currentLevel, int numOfCubes);
//Function copies the tree from device to host array
__host__
cudaError_t CopyTreeFromDevice(quadNode**d_root, quadNode* &c_root, int currentLevel, int numOfCubes);

//Do not even check this ugly function 
//Used by conversion of tree from host to device
__global__
void SetArrayPartChildren(quadNode* memory, int index, bool topLeft, bool topRight, bool botLeft, bool botRight);

//Used by conversion of tree from host to device
cudaError CopyRecursivelyOnGPU(quadNode** root, quadNode* &gpuDataSpace, int index);

__host__
int LoadImage(std::vector<unsigned char> &image, std::string fileName, unsigned &width, unsigned &height);


__global__
void TestWrite(unsigned char* ar, int width, int height);


/****************************************************************************
*
*							*******CODE*******
*
**************************************************************************/

__global__
void InitNode(quadNode** root, float2 position, float2 topLeft, float2 botRight, curandState* state) {
	NewNode(root, RandomThrow(topLeft, botRight, state), topLeft, botRight, 0);
}

__global__
void TestWrite(unsigned char* ar, int width, int height)
{
	for (int i = 0; i < width * height; i++)
		printf("Values at %i %i is %i \n", i%width, i/width, ar[i]);
}
__global__
void TestIntWrite(unsigned char* arr, int size)
{
	for (int i = 0; i < size; i++)
		printf("Value is %i", arr[i]);
}

cudaError_t cudaPoissonSampling::PoissonDiskDistribution(std::vector<float>& positions,const int radius,const int maxTries, const float bounds[])
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	CUDA_CHECK(cudaSetDevice(0));
	
	int maxLevels = 0;
	float2 topL, botR;
	topL.x = bounds[0];
	topL.y = bounds[1];
	botR.x = bounds[2];
	botR.y = bounds[3];


	DEBUG_TEXT("topLeft x %f, topLeft y %f \n botRight x %f, botRight y %f \n", topL.x, topL.y, botR.x, botR.y);

	//need to compute the approximate size of quadtree for alocation
	const int numOfTotalCubes = ComputeNumOfCubes(radius, bounds, maxLevels);

	const int upperRootCubes = ComputeNumOfCubes(maxLevels - 2);

	//Initialization of random variables
	curandState* devStates;
	int seed = time(0);
	int randNumPerBlock = 64; //set these based on number of lists
	SOFT_DEBUG_TEXT("num of cubes is %i\n", numOfTotalCubes);
	int randNumPerGrid = Maxim(Minim((int) ceil(numOfTotalCubes / 64), 8192), 1);
	SOFT_DEBUG_TEXT("the block threads %i, the grid blocks %i\n", randNumPerBlock, randNumPerGrid);
	SOFT_DEBUG_TEXT("block num is %i\n", randNumPerGrid);
	CUDA_CHECK(cudaMalloc((void**)&devStates, numOfTotalCubes * sizeof(curandState)));
	
	SOFT_DEBUG_TEXT("Generating random number seeds\n");
	Setup_rand<<<randNumPerGrid, randNumPerBlock>>>(devStates, seed, numOfTotalCubes); //maybe make this variable??? That is how random should the numbers be
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	srand(time(0));
	// Allocate GPU buffers for tree
	
	quadNode** d_tree;
	CUDA_CHECK(cudaMalloc((void**)&d_tree, numOfTotalCubes * sizeof(quadNode*)));
	
	CUDA_CHECK(cudaMemset(d_tree, NULL, numOfTotalCubes * sizeof(quadNode*)));

	int cpuMaxLevel = 3;

	int cpuCubesLimit = 0;
	for (int i = 0; i <= cpuMaxLevel; i++) {
		cpuCubesLimit += pow(4.f, (float)i);
	}
	//while we only work with limited amount of cubes on CPU, we still allocate the whole array, since it will be receiving data from device
	int cpuCubesNum = numOfTotalCubes <= cpuCubesLimit ? numOfTotalCubes : cpuCubesLimit;

	//alocating heap for new operation
	size_t limit = size_t(numOfTotalCubes*7*8);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000);

	CUDA_CHECK(cudaGetLastError());
	SOFT_DEBUG_TEXT("num of cubes %i \n maxLevels %i\n", numOfTotalCubes, maxLevels);
	//initial throw to first space
	float2 randomPos = RandomThrow(topL, botR);
	InitNode<<<1,1>>>(d_tree, randomPos, topL, botR, devStates);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	//beginning the algorithm
	int curLevel = 0;
	int currMem = 0;

	SOFT_DEBUG_TEXT("\n==========================CUDA COMPUTATION BEGIN========================\n");
	int N = 1;
	int blockSize = 1;
	int gridSizeLimit = 2048;
	//first throw to make to get first partitions
	int kernelCycles = 1;
	int kernelComputationsLimit = ComputeNumOfCubes(9);
	if (curLevel < maxLevels)
	{
		curLevel++;
		cuda_SampleThrow << < N, blockSize >> > (radius, maxTries, d_tree, currMem++, N * blockSize, devStates, ComputeNumOfCubes(curLevel));
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	CUDA_CHECK(cudaGetLastError());
	//device cycles
	while (curLevel < maxLevels) {
		curLevel++;
		//potential increase if grid is 3D
		int cubes = ComputeNumOfCubes(curLevel);
		DEBUG_TEXT("Executing %i threads \n", blockSize * (int)Minim(gridSizeLimit, N));
		SOFT_DEBUG_TEXT("Executing %i threads, %i in block and %i in grid\n AllNodesNum %i \n", blockSize * (int)Minim(gridSizeLimit, N), blockSize, (int)Minim(gridSizeLimit, N), cubes);
		DEBUG_TEXT("curLevel %i,\n curMem is %i\n", curLevel, currMem);
		//throwing separatelly in 4 kernels to better sync
		if (kernelComputationsLimit > cubes)
		{
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, 0, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize())); 
			SOFT_DEBUG_TEXT("first kernel executed.\n");
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, 1, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("second kernel executed.\n");
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, 2, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("third kernel executed.\n");
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, 3, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("fouth kernel executed.\n");
		}
		else {
			SOFT_DEBUG_TEXT("Entering kernel fission part.\n");
			kernelCycles = (cubes / kernelComputationsLimit);
			int compNum = 4*(N * blockSize) / kernelCycles;
			for (int i = 0; i < kernelCycles; i++)
			{
				SOFT_DEBUG_TEXT("Executing %i iteration of kernels\n", i);
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem + i * compNum, compNum/4, devStates, cubes, 0, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("first kernel executed.\n");
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem + i * compNum, compNum/4, devStates, cubes, 1, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("second kernel executed.\n");
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem + i * compNum, compNum/4, devStates, cubes, 2, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("third kernel executed.\n");
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (radius, maxTries, d_tree, currMem + i * compNum, compNum/4, devStates, cubes, 3, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("fouth kernel executed.\n");
			}
		}

		//if the amount of positions is too high, we need to spread the workload into more kernels. 
		//It it because the queue is limited and the amount of computation grows faster then the amount of kernels we give kernels

		DEBUG_TEXT("%i run of kernel successfull.\n", curLevel);
		currMem += pow(4.f, (float)curLevel-1);
		if (curLevel-1 < 4)
		{
			blockSize *= 4;
			if (blockSize > 32)
			{
				blockSize *= 0.5;
				N *= 2;
			}
		}
		else {
				N *= 4;
		}
		CUDA_CHECK(cudaGetLastError());
		//number pointing to current level of memory of tree saved
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	SOFT_DEBUG_TEXT("numOfTotal cubes %i, numOfCubesForCurrLevel %i, numOfCubes for lowerLevel %i \n", numOfTotalCubes, ComputeNumOfCubes(curLevel), ComputeNumOfCubes(curLevel - 1));
	DEBUG_TEXT("currMem is %i", currMem);
	
	// If memory on GPU was used, copy output vector from GPU buffer to host memory.
	quadNode* cpuDataSpace = (quadNode*)malloc(numOfTotalCubes * sizeof(quadNode));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(CopyTreeFromDevice(d_tree, cpuDataSpace, curLevel, numOfTotalCubes));
	UnfoldArrayIntoVector(cpuDataSpace, positions, curLevel);
	
Error:
	if (d_tree)
		CUDA_CHECK(cudaFree(d_tree));
	if (devStates)
		CUDA_CHECK(cudaFree(devStates));
	if (cpuDataSpace)
		free(cpuDataSpace);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CUDA_CHECK(cudaDeviceReset());
	
	DEBUG_TEXT("total num of cubes %i", numOfTotalCubes);

	return cudaError::cudaSuccess;
}
cudaError_t cudaPoissonSampling::PoissonDiskDistribution(std::vector<float>& positions, unsigned char* &radiusValues, unsigned& width, unsigned& height, std::string textureLocation, const int maxTries, const float bounds[], int lowerThreshold, int upperThreshold, partitionAttributes partition)
{
	//unsigned width, height;
	std::vector<unsigned char> image;
	if (!LoadImage(image, textureLocation, width, height))
	{
		return cudaError::cudaErrorFileNotFound;
	}
	size_t arrSize = image.size();
	radiusValues = (unsigned char*)malloc(arrSize);
	std::copy(image.begin(), image.end(), radiusValues);


	printf("width is %f and height is %f\n", width, height);
	printf("random value is %i and %i\n", radiusValues[5], radiusValues[50]);

	float4 nBounds;
	nBounds.x = bounds[0];
	nBounds.y = bounds[1];
	nBounds.z = bounds[2];
	nBounds.w = bounds[3];

	unsigned char* d_radiusValues;

	CUDA_CHECK(cudaMalloc((void**)&d_radiusValues, arrSize * sizeof(unsigned char)));
	CUDA_CHECK(cudaMemcpy(d_radiusValues, radiusValues, arrSize * sizeof(unsigned char), cudaMemcpyHostToDevice));
	// Choose which GPU to run on, change this on a multi-GPU system.
	CUDA_CHECK(cudaSetDevice(0));

	if (radiusValues)
		free(radiusValues);

	TestWrite << <1, 1 >> > (d_radiusValues, 5, 1);

	CUDA_CHECK(cudaDeviceSynchronize());

	int maxLevels = 0;
	float2 topL, botR;
	topL.x = bounds[0];
	topL.y = bounds[1];
	botR.x = bounds[2];
	botR.y = bounds[3];


	DEBUG_TEXT("topLeft x %f, topLeft y %f \n botRight x %f, botRight y %f \n", topL.x, topL.y, botR.x, botR.y);

	//need to compute the approximate size of quadtree for alocation
	const int numOfTotalCubes = ComputeNumOfCubes(lowerThreshold, bounds, maxLevels);
	printf("numOfTotalCubes is %i\n", numOfTotalCubes);
	const int upperRootCubes = ComputeNumOfCubes(maxLevels - 2);

	//Initialization of random variables
	curandState* devStates;
	int seed = time(0);
	int randNumPerBlock = 64; //set these based on number of lists
	SOFT_DEBUG_TEXT("num of cubes is %i\n", numOfTotalCubes);
	int randNumPerGrid = Maxim(Minim((int)ceil(numOfTotalCubes / 64), 8192), 1);
	SOFT_DEBUG_TEXT("the block threads %i, the grid blocks %i\n", randNumPerBlock, randNumPerGrid);
	SOFT_DEBUG_TEXT("block num is %i\n", randNumPerGrid);
	CUDA_CHECK(cudaMalloc((void**)&devStates, numOfTotalCubes * sizeof(curandState)));

	SOFT_DEBUG_TEXT("Generating random number seeds\n");
	Setup_rand << <randNumPerGrid, randNumPerBlock >> > (devStates, seed, numOfTotalCubes); //maybe make this variable??? That is how random should the numbers be
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	srand(time(0));
	// Allocate GPU buffers for tree

	quadNode** d_tree;
	CUDA_CHECK(cudaMalloc((void**)&d_tree, numOfTotalCubes * sizeof(quadNode*)));

	CUDA_CHECK(cudaMemset(d_tree, NULL, numOfTotalCubes * sizeof(quadNode*)));


	int cpuMaxLevel = 3;

	int cpuCubesLimit = 0;
	for (int i = 0; i <= cpuMaxLevel; i++) {
		cpuCubesLimit += pow(4.f, (float)i);
	}
	//while we only work with limited amount of cubes on CPU, we still allocate the whole array, since it will be receiving data from device
	int cpuCubesNum = numOfTotalCubes <= cpuCubesLimit ? numOfTotalCubes : cpuCubesLimit;

	//alocating heap for new operation
	size_t limit = size_t(numOfTotalCubes * 7 * 8);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000);

	CUDA_CHECK(cudaGetLastError());
	SOFT_DEBUG_TEXT("num of cubes %i \n maxLevels %i\n", numOfTotalCubes, maxLevels);
	//initial throw to first space
	float2 randomPos = RandomThrow(topL, botR);
	InitNode << <1, 1 >> > (d_tree, randomPos, topL, botR, devStates);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	//beginning the algorithm
	int curLevel = 0;
	int currMem = 0;

	

	SOFT_DEBUG_TEXT("\n==========================CUDA COMPUTATION BEGIN========================\n");
	int N = 1;
	int blockSize = 1;
	int gridSizeLimit = 2048;
	//first throw to make to get first partitions
	int kernelCycles = 1;
	int kernelComputationsLimit = ComputeNumOfCubes(9);
	if (curLevel < maxLevels)
	{
		curLevel++;
		cuda_SampleThrow <<< N, blockSize >>> (d_radiusValues, width, height, maxTries, d_tree, currMem++, N * blockSize, devStates, ComputeNumOfCubes(curLevel), nBounds, lowerThreshold, upperThreshold, partition);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	CUDA_CHECK(cudaGetLastError());
	//device cycles
	while (curLevel < maxLevels) {
		curLevel++;
		//potential increase if grid is 3D
		int cubes = ComputeNumOfCubes(curLevel);
		DEBUG_TEXT("Executing %i threads \n", blockSize * (int)Minim(gridSizeLimit, N));
		SOFT_DEBUG_TEXT("Executing %i threads, %i in block and %i in grid\n AllNodesNum %i \n", blockSize * (int)Minim(gridSizeLimit, N), blockSize, (int)Minim(gridSizeLimit, N), cubes);
		DEBUG_TEXT("curLevel %i,\n curMem is %i\n", curLevel, currMem);
		//throwing separatelly in 4 kernels to better sync
		if (kernelComputationsLimit > cubes)
		{
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 0, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("first kernel executed.\n");
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 1, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("second kernel executed.\n");
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 2, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("third kernel executed.\n");
			cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem, N * blockSize, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 3, upperRootCubes);
			SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
			SOFT_DEBUG_TEXT("fouth kernel executed.\n");
		}
		else {
			SOFT_DEBUG_TEXT("Entering kernel fission part.\n");
			kernelCycles = (cubes / kernelComputationsLimit);
			int compNum = 4 * (N * blockSize) / kernelCycles;
			for (int i = 0; i < kernelCycles; i++)
			{
				SOFT_DEBUG_TEXT("Executing %i iteration of kernels\n", i);
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem + i * compNum, compNum / 4, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 0, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("first kernel executed.\n");
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem + i * compNum, compNum / 4, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 1, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("second kernel executed.\n");
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem + i * compNum, compNum / 4, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 2, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("third kernel executed.\n");
				cuda_SampleThrow << < Minim(gridSizeLimit, N), blockSize >> > (d_radiusValues, width, height, maxTries, d_tree, currMem + i * compNum, compNum / 4, devStates, cubes, nBounds, lowerThreshold, upperThreshold, partition, 3, upperRootCubes);
				SOFT_DEBUG_FUNCTION(CUDA_CHECK(cudaDeviceSynchronize()));
				SOFT_DEBUG_TEXT("fouth kernel executed.\n");
			}
		}

		//if the amount of positions is too high, we need to spread the workload into more kernels. 
		//It it because the queue is limited and the amount of computation grows faster then the amount of kernels we give kernels

		DEBUG_TEXT("%i run of kernel successfull.\n", curLevel);
		currMem += pow(4.f, (float)curLevel - 1);
		if (curLevel - 1 < 4)
		{
			blockSize *= 4;
			if (blockSize > 32)
			{
				blockSize *= 0.5;
				N *= 2;
			}
		}
		else {
			N *= 4;
		}
		CUDA_CHECK(cudaGetLastError());
		//number pointing to current level of memory of tree saved
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	SOFT_DEBUG_TEXT("numOfTotal cubes %i, numOfCubesForCurrLevel %i, numOfCubes for lowerLevel %i \n", numOfTotalCubes, ComputeNumOfCubes(curLevel), ComputeNumOfCubes(curLevel - 1));
	DEBUG_TEXT("currMem is %i", currMem);

	// If memory on GPU was used, copy output vector from GPU buffer to host memory.
	quadNode* cpuDataSpace = (quadNode*)malloc(numOfTotalCubes * sizeof(quadNode));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(CopyTreeFromDevice(d_tree, cpuDataSpace, curLevel, numOfTotalCubes));
	UnfoldArrayIntoVector(cpuDataSpace, positions, curLevel);

Error:
	if (d_tree)
		CUDA_CHECK(cudaFree(d_tree));
	if (devStates)
		CUDA_CHECK(cudaFree(devStates));
	if (cpuDataSpace)
		free(cpuDataSpace);

	if (d_radiusValues)
		CUDA_CHECK(cudaFree(d_radiusValues));


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CUDA_CHECK(cudaDeviceReset());

	DEBUG_TEXT("total num of cubes %i", numOfTotalCubes);

	return cudaError::cudaSuccess;
}
__global__
void SetpointerToData(quadNode** pointerArray, quadNode* dataMemory, int maxAmount) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < maxAmount; i += blockDim.x * gridDim.x) {
		quadNode* nd = &dataMemory[i];
		DEBUG_TEXT("node in dataMem is %f %f for %i thread\n", nd->position.x, nd->position.y, i);
		DEBUG_TEXT("in node %i is this on pointers to children %i %i %i %i\n", i, nd->topLeftTree, nd->topRightTree, nd->botLeftTree, nd->botRightTree);
		pointerArray[i] = &dataMemory[i];
		
		DEBUG_TEXT("node in pointerArray is %f %f for %i thread\n", pointerArray[i]->position.x, nd->position.y,i);
	}
}
__global__
void MoveMemoryToArray(quadNode** d_root, quadNode* memory)
{
	DEBUG_TEXT("value of root at %i is %f %f\n", threadIdx.x, d_root[threadIdx.x]->position.x, d_root[threadIdx.x]->position.y);
	memcpy(&memory[threadIdx.x], d_root[threadIdx.x], sizeof(quadNode));
}

__global__
void MoveMemoryToArray(quadNode** d_root, quadNode* memory, int initialIdx, int nonLeafCubesNum, int upperRoots)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x + initialIdx; i < nonLeafCubesNum; i += gridDim.x * blockDim.x)
	{
		DEBUG_TEXT("value of root on thread %i at %i is %f %f\n",i - initialIdx, i, d_root[i]->position.x, d_root[i]->position.y);
		if (d_root[i] == NULL)
		{
			memcpy(&memory[i], new quadNode(), sizeof(quadNode));
			continue;
		}else
			memcpy(&memory[i], d_root[i], sizeof(quadNode));
		DEBUG_TEXT("value of %i in memory after copy is %f %f \n", i, memory[i].position.x, memory[i].position.y);
		
		if (d_root[i]->topLeftTree)
		{
			DEBUG_TEXT("value of topLeftTree of %i is %f %f\n", i, d_root[i]->topLeftTree->position.x, d_root[i]->topLeftTree->position.y);
			if (i >= upperRoots) memcpy(&memory[4 * i + 1], d_root[i]->topLeftTree, sizeof(quadNode));
			memory[i].topLeftTree = &memory[4 * i + 1];
			DEBUG_TEXT("topLeft value of %i in memory after copy is %f %f \n", i, memory[i].topLeftTree->position.x, memory[i].topLeftTree->position.y);
		}
		if (d_root[i]->topRightTree)
		{
			DEBUG_TEXT("value of topRightTree of %i is %f %f\n", i, d_root[i]->topRightTree->position.x, d_root[i]->topRightTree->position.y);
			if (i >= upperRoots) memcpy(&memory[4 * i + 2], d_root[i]->topRightTree, sizeof(quadNode));
			memory[i].topRightTree = &memory[4 * i + 2];
			DEBUG_TEXT("topRight value of %i in memory after copy is %f %f \n", i, memory[i].topRightTree->position.x, memory[i].topRightTree->position.y);
		}
		if (d_root[i]->botLeftTree)
		{
			DEBUG_TEXT("value of botLeftTree of %i is %f %f\n", i, d_root[i]->botLeftTree->position.x, d_root[i]->botLeftTree->position.y);
			if (i >= upperRoots) memcpy(&memory[4 * i + 3], d_root[i]->botLeftTree, sizeof(quadNode));
			memory[i].botLeftTree = &memory[4 * i + 3];
			DEBUG_TEXT("botLeft value of %i in memory after copy is %f %f \n", i, memory[i].botLeftTree->position.x, memory[i].botLeftTree->position.y);
		}
		if (d_root[i]->botRightTree)
		{
			DEBUG_TEXT("value of botRightTree of %i is %f %f\n", i, d_root[i]->botRightTree->position.x, d_root[i]->botRightTree->position.y);
			if (i >= upperRoots) memcpy(&memory[4 * i + 4], d_root[i]->botRightTree, sizeof(quadNode));
			memory[i].botLeftTree = &memory[4 * i + 4];
			DEBUG_TEXT("botRight value of %i in memory after copy is %f %f \n", i, memory[i].botRightTree->position.x, memory[i].botRightTree->position.y);
		}
	}

}

cudaError_t CopyTreeToDevice(quadNode** d_root, quadNode** c_root, int currentLevel, int numOfCubes)
{
	cudaError_t cudaStatus;
	int numOfComputedValues = ComputeNumOfCubes(currentLevel); // nodes without leaves
	quadNode* gpuDataSpace;
	CUDA_CHECK(cudaMalloc((void**)&gpuDataSpace, numOfCubes * sizeof(quadNode)));
	CUDA_CHECK(cudaMemset(gpuDataSpace, NULL, sizeof(quadNode) * numOfCubes));
	DEBUG_TEXT("computed values copied: %i\n", numOfComputedValues);
	CUDA_CHECK(cudaMemcpy(d_root, c_root, numOfComputedValues * sizeof(quadNode*), cudaMemcpyHostToDevice));

	int lowerLvlNum = ComputeNumOfCubes(currentLevel - 1);
	
	if (c_root[0] != NULL)
	{
		CUDA_CHECK(cudaMemcpy(&gpuDataSpace[0], c_root[0], sizeof(quadNode), cudaMemcpyHostToDevice));
		CUDA_CHECK(CopyRecursivelyOnGPU(c_root, gpuDataSpace, 0));
	}
		
	SetpointerToData << < 1, 32 >> > (d_root, gpuDataSpace, 21);
	CUDA_CHECK(cudaFree(gpuDataSpace));
	return cudaError::cudaSuccess;
}

cudaError CopyRecursivelyOnGPU(quadNode** root, quadNode* &gpuDataSpace, int index) {
	bool topLeft = root[index]->topLeftTree != NULL, topRight = root[index]->topRightTree != NULL, 
		botLeft = root[index]->botLeftTree != NULL, botRight = root[index]->botRightTree != NULL;
	if (topLeft)
	{
		CUDA_CHECK(cudaMemcpy(&gpuDataSpace[4 * index + 1], root[index]->topLeftTree, sizeof(quadNode), cudaMemcpyHostToDevice));
		CUDA_CHECK(CopyRecursivelyOnGPU(root, gpuDataSpace, 4 * index + 1));
	}
	if (topRight)
	{
		CUDA_CHECK(cudaMemcpy(&gpuDataSpace[4 * index + 2], root[index]->topRightTree, sizeof(quadNode), cudaMemcpyHostToDevice));		
		CUDA_CHECK(CopyRecursivelyOnGPU(root, gpuDataSpace, 4 * index + 2));
	}
	if (botLeft)
	{
		CUDA_CHECK(cudaMemcpy(&gpuDataSpace[4 * index + 3], root[index]->botLeftTree, sizeof(quadNode), cudaMemcpyHostToDevice));		
		CUDA_CHECK(CopyRecursivelyOnGPU(root, gpuDataSpace, 4 * index + 3));
	}
	if (botRight)
	{
		CUDA_CHECK(cudaMemcpy(&gpuDataSpace[4 * index + 4], root[index]->botRightTree, sizeof(quadNode), cudaMemcpyHostToDevice));		
		CUDA_CHECK(CopyRecursivelyOnGPU(root, gpuDataSpace, 4 * index + 4));
	}
	SetArrayPartChildren << <1, 1 >> > (gpuDataSpace, index, topLeft, topRight, botLeft, botRight);
	return cudaError::cudaSuccess;
}
__host__
int LoadImage(std::vector<unsigned char> &image, std::string fileName, unsigned& width, unsigned &height)
{
	char* file = "D:/CUDA/poissonDiskLibrary/poissonDiskLibrary/gradientG.png";
	std::vector<unsigned char> output;

	lodepng::State state;

	if (unsigned error = lodepng::decode(image, width, height, fileName, LodePNGColorType::LCT_GREY))
	{
		printf("error is %u", error);
		return 0;
	}

	return 1;
}
__host__ __device__
int cudaPoissonSampling::GetPixelValueOnPosition(unsigned char* image, int imgWidth, int imgHeight, float2 position, int &resultValue, float4 bounds, int lowerThreshold, int upperThreshold, cudaPoissonSampling::partitionAttributes partition)
{
	int width, height;
	
	width = imgWidth / partition.widthPartitions;
	height = imgHeight / partition.heightPartitions;


	float normalizedWidth = (float)width / abs(bounds.x - bounds.z);
	float normalizedHeight = (float)height / abs(bounds.y - bounds.w);
	int outputX = normalizedWidth * (position.x - (Minim(bounds.x, bounds.z)));
	int outputY = normalizedHeight * (position.y - (Minim(bounds.y, bounds.w)));
	//get value on converted position within picture
	//printf("index tryed to reach is %i out of %i\n the index in width %i and height %i we reach is %i %i\n", outputY * imgWidth + outputX, imgWidth * imgHeight, imgWidth, imgHeight, outputX, outputY);
	
	int xCoord = outputX + (partition.partitionIdx % partition.widthPartitions) * width;
	int yCoord = outputY + (partition.partitionIdx / partition.widthPartitions) * height;
	
	if (yCoord * imgWidth + xCoord >= imgWidth * imgHeight)
	{
		printf("Trying to reach out of bounds within pixels of image.\n");
		return 0;
	}
	
	int result = image[yCoord * imgWidth + xCoord];
	int threshHold = upperThreshold - lowerThreshold;
	// converting output value to fit with threshhold values for radius
	float midRes = (float)result / 255.f;
	resultValue = (float)result / 255.f * (float)threshHold + lowerThreshold;
	if (resultValue == upperThreshold)
	{
		resultValue = 100000;
	}
		   
	return 1;
}
__global__
void testCRand(curandState * state)
{
	int tid = threadIdx.x + blockIdx.x * gridDim.x;
	printf("The thread %i produces rand %i\n",tid, curand(&state[tid]));
}

void SetPointersToChildren(quadNode* &arrayOfNodes, int numOfCubes, int index) {
	if (index < numOfCubes) {
		arrayOfNodes[index].topLeftTree = &arrayOfNodes[index * 4 + 1];
		arrayOfNodes[index].topRightTree = &arrayOfNodes[index * 4 + 2];
		arrayOfNodes[index].botLeftTree = &arrayOfNodes[index * 4 + 3];
		arrayOfNodes[index].botRightTree = &arrayOfNodes[index * 4 + 4];

		SetPointersToChildren(arrayOfNodes, numOfCubes, index * 4 + 1);
		SetPointersToChildren(arrayOfNodes, numOfCubes, index * 4 + 2);
		SetPointersToChildren(arrayOfNodes, numOfCubes, index * 4 + 3);
		SetPointersToChildren(arrayOfNodes, numOfCubes, index * 4 + 4);

		DEBUG_TEXT("setting up pointers to %i node to children\n", index);
		DEBUG_TEXT("topLeftTree has %f %f pointer value\n", arrayOfNodes[index].topLeftTree->position.x, arrayOfNodes[index].topLeftTree->position.y);
		DEBUG_TEXT("topRightTree has %f %f pointer value\n", arrayOfNodes[index].topRightTree->position.x, arrayOfNodes[index].topRightTree->position.y);
		DEBUG_TEXT("botLeftTree has %f %f pointer value\n", arrayOfNodes[index].botLeftTree->position.x, arrayOfNodes[index].botLeftTree->position.y);
		DEBUG_TEXT("botRightTree has %f %f pointer value\n", arrayOfNodes[index].botRightTree->position.x, arrayOfNodes[index].botRightTree->position.y);

		if (arrayOfNodes[index].topLeftTree == 0 || arrayOfNodes[index].topLeftTree == NULL)
		{
			arrayOfNodes[index].topLeftTree = NULL;
			printf("the value is NULL\n");
		}
		if (arrayOfNodes[index].topRightTree == 0 || arrayOfNodes[index].topRightTree == NULL)
		{
			arrayOfNodes[index].topRightTree = NULL;
			printf("the value is NULL\n");
		}
		if (arrayOfNodes[index].botLeftTree == 0 || arrayOfNodes[index].botLeftTree == NULL)
		{
			arrayOfNodes[index].botLeftTree = NULL;
			printf("the value is NULL\n");
		}
		if (arrayOfNodes[index].botRightTree == 0 || arrayOfNodes[index].botRightTree == NULL)
		{
			arrayOfNodes[index].botRightTree = NULL;
			printf("the value is NULL\n");
		}
	}
	else {
		arrayOfNodes[index].topLeftTree = NULL;
		arrayOfNodes[index].topRightTree = NULL;
		arrayOfNodes[index].botLeftTree = NULL;
		arrayOfNodes[index].botRightTree = NULL;
	}
	
}

__host__ cudaError_t CopyTreeFromDevice(quadNode ** d_root, quadNode * &cpuDataSpace, int currentLevel, int numOfCubes)
{
	SOFT_DEBUG_TEXT("Copying data from GPU.\n");
	quadNode* gpuDataSpace;
	CUDA_CHECK(cudaMalloc((void**)&gpuDataSpace, numOfCubes * sizeof(quadNode)));
	CUDA_CHECK(cudaGetLastError());

	DEBUG_TEXT("CopyTreeFromDevice before kernel for currLevel %i\n", currentLevel);
	int blockSize = 32;
	int gridSize = 2048;
	int nonLeafNodes = ComputeNumOfCubes(currentLevel - 1);
	int nodesLimitation = ComputeNumOfCubes(7);

	if (currentLevel == 0)
		MoveMemoryToArray <<< 1, 1 >>> (d_root, gpuDataSpace);
	else
	{
		if (nonLeafNodes < nodesLimitation)
			MoveMemoryToArray << <gridSize, blockSize >> > (d_root, gpuDataSpace, 0, nonLeafNodes, ComputeNumOfCubes(currentLevel - 2));
		else {
			int cyclesNum = ceil((float)(nonLeafNodes) / (float)(blockSize * gridSize));
			SOFT_DEBUG_TEXT("the cyclesNum is %i and is suposed to be %f\n", cyclesNum, ceil(((float)nonLeafNodes) / (float)(blockSize * gridSize)));
			float nonLeafPart = (float)nonLeafNodes / (float)cyclesNum;
			SOFT_DEBUG_TEXT("non Leaf nodes %i, nodes limitation is %i, therefore numOfCycles is %i, part is %f\n", nonLeafNodes, nodesLimitation, cyclesNum, nonLeafPart);
			for (int i = 0; i < cyclesNum; i++) {
				MoveMemoryToArray << <gridSize, blockSize >> > (d_root, gpuDataSpace,(int) i * nonLeafPart, (int)(i + 1) * nonLeafPart, ComputeNumOfCubes(currentLevel - 2));
				SOFT_DEBUG_TEXT("%i cycle with kernel executed.\n Borders for current level are %f to %f\n", i, i*nonLeafPart, (i+1)*nonLeafPart);
				CUDA_CHECK(cudaDeviceSynchronize());
			}	
		}
	}
		
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(cpuDataSpace, gpuDataSpace, numOfCubes * sizeof(quadNode), cudaMemcpyDeviceToHost));
	DEBUG_TEXT("The cpuDataSpace root is %f %f\n", cpuDataSpace[0].position.x, cpuDataSpace[0].position.y);
	int num = ComputeNumOfCubes(currentLevel);
	//no need to set up pointers
	//cpudataSpace can be used
	CUDA_CHECK(cudaFree(gpuDataSpace));

	return cudaError::cudaSuccess;
}
__global__
void SetArrayPartChildren(quadNode * memory, int index, bool topLeft, bool topRight, bool botLeft, bool botRight)
{
	DEBUG_TEXT("position of memory %f %f \n", memory[index].position.x, memory[index].position.y);
	if (topLeft)
		memory[index].topLeftTree = &memory[index * 4 + 1];
	else
		memory[index].topLeftTree = NULL;
	if (topRight)
		memory[index].topRightTree = &memory[index * 4 + 2];
	else
		memory[index].topRightTree = NULL;
	if (botLeft)
		memory[index].botLeftTree = &memory[index * 4 + 3];
	else
		memory[index].botLeftTree = NULL;
	if (botRight)
		memory[index].botRightTree = &memory[index * 4 + 4];
	else
		memory[index].botRightTree = NULL;
}

__global__
void cuda_SampleThrow(int radius, int tries, quadNode** root, int treeLevelNodeIndex, int numOfComputations, curandState* state, int maxCubesAmount)
{
	//make sure we wont spill out of grid
	for (int thrIdx = blockIdx.x * blockDim.x + threadIdx.x; thrIdx < numOfComputations; thrIdx += gridDim.x * blockDim.x) {
		int nodeId = treeLevelNodeIndex + thrIdx;
		DEBUG_TEXT("computedIndex %i for thread %i\n", nodeId, thrIdx);
		if (nodeId > numOfComputations + treeLevelNodeIndex)
		{
			printf("accessing places outside allocation\n");
			return;
		}
			
		quadNode* currNode = root[nodeId];
		DEBUG_TEXT("thread %i is beginning with the value %f %f\n", thrIdx, root[nodeId]->position.x, root[nodeId]->position.y);
		//use only for the remaining nodes of node (not for the one we migrated parent into)
		int skipSector = 5;
		if (!MigratePointLower(root, skipSector, nodeId, maxCubesAmount))
			printf("The given index to migrate does not point to valid node \n");
		if (skipSector > 3)
			printf("sector told to skip is higher than 4 \n");
		for (int i = 0; i < 4; i++) {
			// we wish to throw only to where there is no point yet
			DEBUG_TEXT("....cycle %i\n sector %i\n", i, skipSector);
			if (i == skipSector)
				continue;
			float2 point, sectorTopLeft, sectorBotRight;
			ComputeBordersForSector(currNode->topLeft, currNode->botRight, i, sectorTopLeft, sectorBotRight);
			bool found = false;
			//dart throwing
			for (int j = 0; j < tries; j++) {
				point = RandomThrow(sectorTopLeft, sectorBotRight, state);
				if (!IsAnyNodeWithinDist(root, point, radius)) {
					found = true;
					break;
				}
				DEBUG_TEXT("distance checked and is not nice for thread %i\n", thrIdx);
			}

			if (found)
			{
				DEBUG_FUNCTION(if (!testDistOfDistance(root, point, radius))
					DEBUG_TEXT("__The point %f %f is not far enough from other points\n", point.x, point.y));
				AddNode(root, point, nodeId, maxCubesAmount);
			}	
		}
	}
}
__global__
void cuda_SampleThrow(int radius, int tries, quadNode ** root, int treeLevelNodeIndex, int numOfComputations, curandState * state, int maxCubesAmount, int sector, int upperRootCubesAmount)
{
	//make sure we wont spill out of grid
	for (int thrIdx = blockIdx.x * blockDim.x + threadIdx.x; thrIdx < numOfComputations; thrIdx += gridDim.x * blockDim.x) {
		int nodeId = treeLevelNodeIndex + sector + thrIdx * 4;
		DEBUG_TEXT("computedIndex %i for thread %i\n", nodeId, thrIdx);
		if (nodeId > 4 * numOfComputations + treeLevelNodeIndex)
		{
			printf("accessing places outside allocation\n");
			return;
		}

		quadNode* currNode = root[nodeId];

		if (currNode == NULL || (currNode->position.x == 0.f && currNode->position.y == 0.0f))
		{
			DEBUG_TEXT("skipping point, since it is 0\n");
			continue;
		}
		DEBUG_TEXT("thread %i is beginning with the value %f %f\n", thrIdx, root[nodeId]->position.x, root[nodeId]->position.y);
		//use only for the remaining nodes of node (not for the one we migrated parent into)
		int skipSector = 5;
		if (!MigratePointLower(root, skipSector, nodeId, maxCubesAmount))
			printf("The given index to migrate does not point to valid node \n");
		if (skipSector > 3)
			printf("sector told to skip is higher than 4 \n");
		for (int i = 0; i < 4; i++) {
			// we wish to throw only to where there is no point yet
			DEBUG_TEXT("....cycle %i\n sector %i\n", i, skipSector);
			if (i == skipSector)
				continue;
			float2 point, sectorTopLeft, sectorBotRight;
			ComputeBordersForSector(currNode->topLeft, currNode->botRight, i, sectorTopLeft, sectorBotRight);
			bool found = false;
			//dart throwing
			for (int j = 0; j < tries; j++) {
				point = RandomThrow(sectorTopLeft, sectorBotRight, state);
				if (!IsAnyNodeWithinDist(root, point, radius)) {
					found = true;
					break;
				}
				DEBUG_TEXT("distance checked and is not nice for thread %i\n", thrIdx);
			}
			if (found)
			{
				DEBUG_FUNCTION(if (testDistOfDistance(root, point, radius) != !IsAnyNodeWithinDist(root, point, radius))
					DEBUG_TEXT("__The point %f %f is not far enough from other points\n", point.x, point.y))
				AddNode(root, point, nodeId, maxCubesAmount);
			}
		}
	}
}
__global__
void cuda_SampleThrow(unsigned char * radiusValues, int width, int height, int tries, quadNode ** root, int treeLevelNodeFirstIndex, int numOfComputations, curandState * state, int maxCubesAmount, float4 bounds, int lowerThreshold, int upperThreshold, cudaPoissonSampling::partitionAttributes partition)
{
	//make sure we wont spill out of grid
	for (int thrIdx = blockIdx.x * blockDim.x + threadIdx.x; thrIdx < numOfComputations; thrIdx += gridDim.x * blockDim.x) {
		int nodeId = treeLevelNodeFirstIndex + thrIdx;
		DEBUG_TEXT("computedIndex %i for thread %i\n", nodeId, thrIdx);
		if (nodeId > numOfComputations + treeLevelNodeFirstIndex)
		{
			printf("accessing places outside allocation\n");
			return;
		}

		quadNode* currNode = root[nodeId];
		DEBUG_TEXT("thread %i is beginning with the value %f %f\n", thrIdx, root[nodeId]->position.x, root[nodeId]->position.y);
		//use only for the remaining nodes of node (not for the one we migrated parent into)
		int skipSector = 5;
		if (!MigratePointLower(root, skipSector, nodeId, maxCubesAmount))
			printf("The given index to migrate does not point to valid node \n");
		if (skipSector > 3)
			printf("sector told to skip is higher than 4 \n");
		for (int i = 0; i < 4; i++) {
			// we wish to throw only to where there is no point yet
			DEBUG_TEXT("....cycle %i\n sector %i\n", i, skipSector);
			if (i == skipSector)
				continue;
			float2 point, sectorTopLeft, sectorBotRight;
			ComputeBordersForSector(currNode->topLeft, currNode->botRight, i, sectorTopLeft, sectorBotRight);
			bool found = false;
			int radius = 10000;
			for (int j = 0; j < tries; j++) {
				point = RandomThrow(sectorTopLeft, sectorBotRight, state);
				if (!cudaPoissonSampling::GetPixelValueOnPosition(radiusValues, width, height, point, radius, bounds, lowerThreshold, upperThreshold, partition))
					printf("Getting pixel was unsuccessfull.\n");
				if (!IsAnyNodeWithinDist(root, point, radius)) {
					found = true;
					break;
				}
				DEBUG_TEXT("distance checked and is not nice for thread %i\n", thrIdx);
			}

			if (found)
			{
				DEBUG_FUNCTION(if (!testDistOfDistance(root, point, radius))
					DEBUG_TEXT("__The point %f %f is not far enough from other points\n", point.x, point.y));
				AddNode(root, point, nodeId, maxCubesAmount);
			}
		}
	}
}
__global__
void cuda_SampleThrow(unsigned char* radiusValues, int width, int height, int tries, quadNode ** root, int treeLevelNodeFirstIndex, int numOfComputations, curandState * state, int maxCubesAmount, float4 bounds, int lowerThreshold, int upperThreshold, cudaPoissonSampling::partitionAttributes partition, int sector, int upperRootCubesAmount)
{
	//make sure we wont spill out of grid
	for (int thrIdx = blockIdx.x * blockDim.x + threadIdx.x; thrIdx < numOfComputations; thrIdx += gridDim.x * blockDim.x) {
		int nodeId = treeLevelNodeFirstIndex + sector + thrIdx * 4;
		DEBUG_TEXT("computedIndex %i for thread %i\n", nodeId, thrIdx);
		if (nodeId > 4 * numOfComputations + treeLevelNodeFirstIndex)
		{
			printf("accessing places outside allocation\n");
			return;
		}

		quadNode* currNode = root[nodeId];

		if (currNode == NULL || (currNode->position.x == 0.f && currNode->position.y == 0.0f))
		{
			DEBUG_TEXT("skipping point, since it is 0\n");
			continue;
		}
		DEBUG_TEXT("thread %i is beginning with the value %f %f\n", thrIdx, root[nodeId]->position.x, root[nodeId]->position.y);
		//use only for the remaining nodes of node (not for the one we migrated parent into)
		int skipSector = 5;
		if (!MigratePointLower(root, skipSector, nodeId, maxCubesAmount))
			printf("The given index to migrate does not point to valid node \n");
		if (skipSector > 3)
			printf("sector told to skip is higher than 4 \n");
		for (int i = 0; i < 4; i++) {
			// we wish to throw only to where there is no point yet
			DEBUG_TEXT("....cycle %i\n sector %i\n", i, skipSector);
			if (i == skipSector)
				continue;
			float2 point, sectorTopLeft, sectorBotRight;
			ComputeBordersForSector(currNode->topLeft, currNode->botRight, i, sectorTopLeft, sectorBotRight);
			bool found = false;
			int radius = 10000;

			//dart throwing
			for (int j = 0; j < tries; j++) {
				point = RandomThrow(sectorTopLeft, sectorBotRight, state);			
				if (!cudaPoissonSampling::GetPixelValueOnPosition(radiusValues, width, height, point, radius, bounds, lowerThreshold, upperThreshold, partition))
					printf("Getting pixel was unsuccessfull.\n");
				if (!IsAnyNodeWithinDist(root, point, radius)) {
					found = true;
					break;
				}
				DEBUG_TEXT("distance checked and is not nice for thread %i\n", thrIdx);
			}
			if (found)
			{
				DEBUG_FUNCTION(if (testDistOfDistance(root, point, radius) != !IsAnyNodeWithinDist(root, point, radius))
					DEBUG_TEXT("__The point %f %f is not far enough from other points\n", point.x, point.y))
					AddNode(root, point, nodeId, maxCubesAmount);
			}
		}
	}
}

void SampleThrow(int radius, int tries, quadNode ** root, int indexOfCurrentNode, int totalCubes)
{
	quadNode* currNode = root[indexOfCurrentNode];
	//use only for the remaining nodes of node (not for the one we migrated parent into)
	int skipSector;

	if (!MigratePointLower(root, skipSector, indexOfCurrentNode, totalCubes))
		printf("The given index to migrate does not point to valid point \n");
	if (skipSector > 3)
		printf("sector told to skip is higher than 4 \n");

	for (int i = 0; i < 4; i++) {
		// we wish to throw only to where there is no point yet
		DEBUG_TEXT("sample throw i %i \n", i);
		
		if (i == skipSector)
			continue;
		float2 point, sectorTopLeft, sectorBotRight;
		ComputeBordersForSector(currNode->topLeft, currNode->botRight, i, sectorTopLeft, sectorBotRight);
		bool found = false;
		//dart throwing
		for (int j = 0; j < tries; j++) {
			point = RandomThrow(sectorTopLeft, sectorBotRight);
			DEBUG_TEXT("randomly thrown\n");
			if (!IsAnyNodeWithinDist(root, point, radius)) {
				found = true;
				DEBUG_TEXT("found node not in distance \n");
				break;
			}
			DEBUG_TEXT("00000000000throwing again0000000000 \n");
		}
		if (found)
			AddNode(root, point, indexOfCurrentNode, totalCubes);
	}
}
void SampleThrow(int radius, int tries, quadNode * &root, int indexOfCurrentNode)
{
	quadNode* currNode = &root[indexOfCurrentNode];
	//use only for the remaining nodes of node (not for the one we migrated parent into)
	int skipSector;

	if (!MigratePointLower(root, skipSector, indexOfCurrentNode))
		printf("The given index to migrate does not point to valid point \n");
	if (skipSector > 3)
		printf("sector told to skip is higher than 4 \n");
	for (int i = 0; i < 4; i++) {
		// we wish to throw only to where there is no point yet
		if (i == skipSector)
			continue;
		float2 point, sectorTopLeft, sectorBotRight;
		ComputeBordersForSector(currNode->topLeft, currNode->botRight, i, sectorTopLeft, sectorBotRight);
		bool notFound = false;
		//dart throwing
		for (int j = 0; j < tries; j++) {
			point = RandomThrow(sectorTopLeft, sectorBotRight);
			if (!IsAnyNodeWithinDist(root, point, radius)) {
				notFound = true;
				break;
			}
		}
		if (notFound)
			AddNode(root, point, indexOfCurrentNode);
	}
}


//OPTIM Change reccursion for iteration
void UnfoldIntoVector(quadNode* root, std::vector<float> &result)
{
	printf("root val is %f %f", root->position);
	if (root != NULL && root != 0)
		if (IsLeaf(root)) {
			result.push_back(root->position.x);
			result.push_back(root->position.y);
		}
		else {
			UnfoldIntoVector(root->botLeftTree, result);
			UnfoldIntoVector(root->botRightTree, result);
			UnfoldIntoVector(root->topLeftTree, result);
			UnfoldIntoVector(root->topRightTree, result);
		}
}
void UnfoldIntoVector(quadNode root, std::vector<float> &result)
{
	quadNode* nd = &root;
	if (nd != NULL)
		if (IsLeaf(nd)) {
			result.push_back(nd->position.x);
			result.push_back(nd->position.y);
		}
		else {
			UnfoldIntoVector(nd->botLeftTree, result);
			UnfoldIntoVector(nd->botRightTree, result);
			UnfoldIntoVector(nd->topLeftTree, result);
			UnfoldIntoVector(nd->topRightTree, result);
		}
}

void UnfoldArrayIntoVector(quadNode* nodeArray, std::vector<float> &result, int currLevel)
{
	int numOfTotalCubes = ComputeNumOfCubes(currLevel);
	int numOfParentCubes = ComputeNumOfCubes(currLevel-1);
	for (int i = numOfParentCubes; i < numOfTotalCubes; i++)
	{
		if (nodeArray[i].position.x == 0.f && nodeArray[i].position.y == 0.f)
			continue;
		result.push_back(nodeArray[i].position.x);
		result.push_back(nodeArray[i].position.y);
	}
}

bool IsLeaf(quadNode* root) {
	return (!root->botLeftTree && !root->botRightTree && !root->topLeftTree && !root->topRightTree);
}

//OPTIM potential wasting memory if the domain is adaptive
int ComputeNumOfCubes(int radius, const float* domain, int& numOfLevels) {
	float domainW = CompDist(domain[0], domain[2]); 
	float domainH = CompDist(domain[1], domain[3]);
	float dom = Maxim(domainW, domainH);
	dom *= 0.5;
	const float radAdjusted = radius / sqrt(2);;
	int resNumOfCubes = 0;
	numOfLevels = 0;

	while (dom > radAdjusted) {
		dom *= 0.5; 
		resNumOfCubes += pow(4.f, (float)numOfLevels);
		numOfLevels++;
	}

	resNumOfCubes += pow(4.f,(float) numOfLevels); //OPTIM potentially takes precious memory in cache
	return resNumOfCubes;
}
int ComputeNumOfCubes(int numOfLevels)
{
	int result = 0;
	for (int i = 0; i <= numOfLevels; i++)
		result += pow(4.f, (float) i);
	return result;
}

__host__ __device__
int ComputeWidthOfCurrentLevel(int numOfLevels)
{
	int result = 0;
	for (int i = 0; i <= numOfLevels; i++)
		result = pow(2.f, (float)i);
	return result;
}
int ComputeNumOfCubes(int radius, const float* domain, int& resNumOfCubes, int& numOfLevels) {
	float domainW = CompDist(domain[0], domain[2]); 
	float domainH = CompDist(domain[1], domain[3]);
	float dom = Maxim(domainW, domainH);
	dom *= 0.5;
	const float radAdjusted = radius / sqrt(2);;
	resNumOfCubes = 0;
	numOfLevels = 0;

	while (dom > radAdjusted) {
		dom *= 0.5; 
		resNumOfCubes += pow(4.f, (float)numOfLevels);
		numOfLevels++;
	}

	resNumOfCubes += pow(4.f, (float) numOfLevels); //OPTIM potentially takes precious memory in cache
	return resNumOfCubes;
}

__host__
int random(float from, float to) {
	return rand() % (int)(to - from + 1) + from;
}

__device__
int random(float from, float to, curandState* state) {
	int tid = threadIdx.x + blockIdx.x * gridDim.x;
	curandState localState = state[tid];
	int x =(int) fabs((float)curand(&localState));
	state[tid] = localState;
	int mod = (float)(x % (int)(to - from + 1)) + from;
	DEBUG_TEXT("random num is %i for %f to %f\n", mod, from, to);
	return  mod;
}

float2 RandomThrow(float2 topLeft, float2 botRight) {
	float2 result;
	
	float sValx = Minim(topLeft.x, botRight.x);
	float bValx = Maxim(topLeft.x, botRight.x);
	float sValy = Minim(topLeft.y, botRight.y);
	float bValy = Maxim(topLeft.y, botRight.y);
	result.x = random(sValx+1, bValx-1);
	result.y = random(sValy+1, bValy-1);
	return result;
}
float2 RandomThrow(float2 topLeft, float2 botRight, curandState* state) {
	float2 result;
	float sValx = Minim(topLeft.x, botRight.x);
	float bValx = Maxim(topLeft.x, botRight.x);
	float sValy = Minim(topLeft.y, botRight.y);
	float bValy = Maxim(topLeft.y, botRight.y);
	result.x = random(sValx+1, bValx-1, state);
	result.y = random(sValy + 1, bValy - 1, state);
	DEBUG_TEXT("--random num is %f, %f\n", result.x, result.y);
	if (!InBoundary(result, topLeft, botRight))
		printf("The generated random point is not within bounds.\n");
	return result;
}
void ClearupTree(quadNode *& root)
{
	if (root->topLeftTree)
		if (!IsLeaf(root->topLeftTree))
			ClearupTree(root->topLeftTree);
	if(root->topRightTree)
		if (!IsLeaf(root->topRightTree))
			ClearupTree(root->topRightTree);
	if(root->botLeftTree)
		if (!IsLeaf(root->botLeftTree))
			ClearupTree(root->botLeftTree);
	if(root->botRightTree)
		if (!IsLeaf(root->botRightTree))
			ClearupTree(root->botRightTree);

	if (root->topLeftTree)
		delete root->topLeftTree;
	if (root->topRightTree)
		delete root->topRightTree;
	if (root->botLeftTree)
		delete root->botLeftTree;
	if (root->botRightTree)
		delete root->botRightTree;
}
__global__ void Setup_rand(curandState * state, int seed, int numOfNums)
{
	for (int tid = threadIdx.x + blockIdx.x * gridDim.x; tid < numOfNums; tid += gridDim.x * blockDim.x) {
		curand_init(seed, tid, 2, &state[tid]);
		__syncthreads();
	}
	
}
bool IsAnyNodeWithinDist(quadNode** root, float2 position, float maxDist)
{
	float distNearest = maxDist + 1;
	if (!FindNearest(root, position, distNearest))
		printf("Stack overflow!\n");
	if (maxDist < distNearest)
		return false;
	return true;
}
bool IsAnyNodeWithinDist(quadNode* root, float2 position, float maxDist)
{
	float distNearest = maxDist + 5;
	if (!FindNearest(root, position, distNearest))
		printf("Stack overflow!\n");
	if (maxDist < distNearest)
		return false;
	return true;
}

bool MigratePointLower(quadNode** root, int& resSector, int currentNodeIndex, int maxCubeAmount)
{
	quadNode* currentNode = root[currentNodeIndex];
	if (currentNode->position.x == 0.f && currentNode->position.y == 0.f)
		return false;
	AddNode(root, currentNode->position, currentNodeIndex, maxCubeAmount);
	//find out sector of added point
	resSector = DetermineNodeSector(currentNode->position, currentNode->topLeft, currentNode->botRight);
	if (resSector > 3)
		printf("Problem in migrate for node %i.\n", currentNodeIndex);
	return true;
}
bool MigratePointLower(quadNode* &root, int& resSector, int currentNodeIndex)
{
	quadNode* currentNode = &root[currentNodeIndex];
	if (currentNode->position.x == 0 && currentNode->position.y == 0)
		return false;
	AddNode(root, currentNode->position, currentNodeIndex);
	//find out sector of added point
	resSector = DetermineNodeSector(currentNode->position, currentNode->topLeft, currentNode->botRight);
	if (resSector > 3)
		printf("Problem in migrate.\n");
	return true;
}

int DetermineNodeSector(float2 position, float2 topLeft, float2 botRight) {
	int resSector;
	float distW = CompDist(topLeft.x, botRight.x) / 2;
	float distH = CompDist(topLeft.y, botRight.y) / 2;
	float2 midPoint, topRight, botLeft;
	topRight.y = topLeft.y;
	topRight.x = botRight.x;
	botLeft.y = botRight.y;
	botLeft.x = topLeft.x;
	midPoint.x = Minim(topLeft.x, botRight.x) + distW;
	midPoint.y = Minim(topLeft.y, botRight.y) + distH;
	if (InBoundary(position, topLeft, midPoint))
		resSector = 0;
	else if (InBoundary(position, midPoint, topRight))
		resSector = 1;
	else if (InBoundary(position, botLeft, midPoint))
		resSector = 2;
	else if (InBoundary(position, midPoint, botRight))
		resSector = 3;
	else {
		resSector = 4;
		printf("position %f %f not in boundary tL %f %f, bR %f %f Error.\n", position.x, position.y, topLeft.x, topLeft.y, botRight.x, botRight.y);
	}
		

	return resSector;
}

__global__
void TestOfRootIntegrity(quadNode ** root, int maxNumOfIndeces)
{
	for (int nodeidx = threadIdx.x + blockIdx.x * blockDim.x; nodeidx < maxNumOfIndeces; nodeidx += gridDim.x * blockDim.x)
	{
		printf("node on index %i has value %f %f\n", nodeidx, root[nodeidx]->position.x, root[nodeidx]->position.y);
	}
	if (threadIdx.x == 0)
	{
		quadNode* nd2 = root[3];
		if (nd2->botRightTree)
			printf("botRight son value is %f %f \n", nd2->botRightTree->position.x, nd2->botRightTree->position.y);
		nd2 = root[0];
		if (nd2->topLeftTree)
			printf("topLeft son value is %f %f \n", nd2->topLeftTree->position.x, nd2->topLeftTree->position.y);
	}
	
}
__global__
void TestOfRootIntegrity(quadNode* root, int maxNumOfIndeces, float2 position, float2 tl, float2 br)
{
	for (int nodeidx = threadIdx.x + blockIdx.x * blockDim.x; nodeidx < maxNumOfIndeces; nodeidx += gridDim.x * blockDim.x)
	{
		quadNode* nd = &root[nodeidx];
		printf("node on index %i has value %f %f\n", nodeidx, nd->position.x, nd->position.y);
	}
}


float CompDist(float firstPar, float secPar) {
	return fabsf(firstPar - secPar);
}
float Minim(float num1, float num2) {
	return (num1 <= num2) ? num1 : num2;
}
float Maxim(float num1, float num2) {
	return (num1 > num2) ? num1 : num2;
}
inline float Dist(float2 a, float2 b)
{
	float tx, ty, d;
	
	tx = a.x - b.x;
	ty = a.y - b.y;

	d = tx * tx + ty * ty;
	return sqrtf((float)d);
}
struct quadNode* NewNode(quadNode** root, float2 point, float2 topLeft, float2 botRight, int pointerToNode) {
	
	DEBUG_TEXT("index of new node %i, x %f y %f \n", pointerToNode, point.x, point.y);
	quadNode * tmp = new quadNode(point, topLeft, botRight);
	if (tmp == NULL)
		printf("Allocating new node failed.\n");
	root[pointerToNode] = tmp;
	root[pointerToNode]->topLeftTree = NULL;
	root[pointerToNode]->topRightTree = NULL;
	root[pointerToNode]->botLeftTree = NULL;
	root[pointerToNode]->botRightTree = NULL;
	return tmp;
}
struct quadNode* NewNode(quadNode* &root, float2 point, float2 topLeft, float2 botRight, int pointerToNode) {
	
	DEBUG_TEXT("index of new node %i, x %f y %f \n", pointerToNode, point.x, point.y);
	root[pointerToNode] = quadNode(point, topLeft, botRight);
	return &root[pointerToNode];
}

void AddNode(quadNode** root, float2 point, int pointerToNode, int maxCubeAmount)
{
	if (!InsertQuad(root, point, pointerToNode, maxCubeAmount))
	{
		printf("When adding new node, there was an error within InsertQuad - stack overflow or trying to make nodes smaller than 1 \n");
	}	
}
void AddNode(quadNode* &root, float2 point, int pointerToNode)
{
	if (!InsertQuad(root, point, 0))
	{
		printf("When adding new node, there was an error within InsertQuad - stack overflow or trying to make nodes smaller than 1 \n");
	}	
}
__device__ __host__
bool InBoundaryIn2D(float pos, float topLeft, float botRight)
{
	return pos >= Minim(topLeft, botRight) &&
		pos <= Maxim(botRight, topLeft);
}
__device__ __host__
bool InBoundary(float2 position, float2 topLeft, float2 botRight)
{
	return (InBoundaryIn2D(position.x, topLeft.x, botRight.x) &&
		InBoundaryIn2D(position.y, topLeft.y, botRight.y));
}

//if within boundaries returns 0;
int BoundaryMinDist(float2 position, float2 topLeft, float2 botRight) 
{
	if (InBoundary(position, topLeft, botRight))
		return 0;
	
	float minXdist; 
	float minYdist; 
	if (InBoundaryIn2D(position.x, topLeft.x, botRight.x))
	{
		minYdist = Minim(CompDist(topLeft.y, position.y), CompDist(botRight.y, position.y));
		return minYdist;
	}
	if (InBoundaryIn2D(position.y, topLeft.y, botRight.y))
	{
		minXdist = Minim(CompDist(topLeft.x, position.x), CompDist(botRight.x, position.x));
		return minXdist;
	}
	minXdist = Minim(CompDist(topLeft.x, position.x), CompDist(botRight.x, position.x));
	minYdist = Minim(CompDist(topLeft.y, position.y), CompDist(botRight.y, position.y));

	return Minim(minXdist, minYdist);
}

bool InsertQuad(quadNode **root, float2 newPos, int indexOfCurrentNode, int maxCubesAmount) {
	//simple stacks since the algorithm should not go into deep recursion
	int idxStack[4];
	int i = 0;
	idxStack[i++] = indexOfCurrentNode;
	bool noError = true;
	int currIndex;
	while (i > 0)
	{
		if (i > 3) {
			printf("stack overflow within the insert quad.\n");
			noError = false;
			break;
		}

		currIndex = idxStack[--i];
		//Index needs to be kept at most at the current leaf level (balancing)
		if (currIndex * 4 + 4 > maxCubesAmount)
		{
			printf("The plugin is trying to add quad on index %i out of bounds %i\n", currIndex*4 + 4, maxCubesAmount);
			break;
		}

		quadNode* currentNode = root[currIndex];

		// Current quad cannot contain it 
		if (!InBoundary(newPos, currentNode->topLeft, currentNode->botRight))
		{
			noError = noError & true;
			continue;
		}
		// We are at a quad of unit area 
		// We cannot subdivide this quad further 
		if (fabsf(currentNode->topLeft.x - currentNode->botRight.x) <= 1 &&
			fabsf(currentNode->topLeft.y - currentNode->botRight.y) <= 1)
		{
			//shouldnt happen, we are lower than allowed radius
			noError = noError & false;
			continue;
		}
		int sector = DetermineNodeSector(newPos, currentNode->topLeft, currentNode->botRight);
		if (sector > 3)
			printf("Problem in add.\n");
		switch (sector) {
		case 0:

			CheckAndCreateInChild(root, currentNode->topLeftTree, currIndex, newPos, currentNode->topLeft, currentNode->botRight, 1, idxStack, i, maxCubesAmount);
			break;
		case 1:
			CheckAndCreateInChild(root, currentNode->topRightTree, currIndex, newPos, currentNode->topLeft, currentNode->botRight, 2, idxStack, i, maxCubesAmount);
			break;
		case 2:
			CheckAndCreateInChild(root, currentNode->botLeftTree, currIndex, newPos, currentNode->topLeft, currentNode->botRight, 3, idxStack, i, maxCubesAmount);
			break;
		case 3:
			CheckAndCreateInChild(root, currentNode->botRightTree, currIndex, newPos, currentNode->topLeft, currentNode->botRight, 4, idxStack, i, maxCubesAmount);
			break;
		default:
			printf("Sector found within Adding new node invalid.\n");
		}
	}


	return noError;
}
bool InsertQuad(quadNode *&root, float2 newPos, int indexOfCurrentNode) {
	//simple stacks since the algorithm should not go into deep recursion
	int idxStack[40];
	int i = 0;
	idxStack[i++] = indexOfCurrentNode;

	bool noError = true;

	int currIndex;

	while (i > 0)
	{
		if (i > 39) {
			noError = false;
			break;
		}

		currIndex = idxStack[--i];

		quadNode* currentNode = &root[currIndex];

		// Current quad cannot contain it 
		if (!InBoundary(newPos, currentNode->topLeft, currentNode->botRight))
		{
			printf("point %f %f not in boundary.\n", newPos.x, newPos.y);
			noError = noError & true;
			continue;
		}
		// We are at a quad of unit area 
		// We cannot subdivide this quad further 
		if (fabsf(currentNode->topLeft.x - currentNode->botRight.x) <= 1 &&
			fabsf(currentNode->topLeft.y - currentNode->botRight.y) <= 1)
		{
			//shouldnt happen, we are lower than allowed radius
			noError = noError & false;
			continue;
		}
		float2 topL, botR;

		int sector = DetermineNodeSector(newPos, currentNode->topLeft, currentNode->botRight);
		if (sector > 3)
			printf("Problem in add.\n");
		switch (sector) {
		case 0:
			if (currentNode->topLeftTree == NULL) {
				ComputeBordersForSector(currentNode->topLeft, currentNode->botRight, 0, topL, botR);
				currentNode->topLeftTree = NewNode(root, newPos, topL, botR, currIndex * 4 + 1);
			}
			else
				idxStack[i++] = currIndex * 4 + 1;
			break;
		case 1:
			if (currentNode->topRightTree == NULL) {
				ComputeBordersForSector(currentNode->topLeft, currentNode->botRight, 1, topL, botR);
				currentNode->topRightTree = NewNode(root, newPos, topL, botR, currIndex * 4 + 2);
			}
			else
				idxStack[i++] = currIndex * 4 + 2;
			break;
		case 2:
			if (currentNode->botLeftTree == NULL) {
				ComputeBordersForSector(currentNode->topLeft, currentNode->botRight, 2, topL, botR);
				currentNode->botLeftTree = NewNode(root, newPos, topL, botR, currIndex * 4 + 3);
			}
			else
				idxStack[i++] = currIndex * 4 + 3;
			break;
		case 3:
			if (currentNode->botRightTree == NULL) {
				ComputeBordersForSector(currentNode->topLeft, currentNode->botRight, 3, topL, botR);
				currentNode->botRightTree = NewNode(root, newPos, topL, botR, currIndex * 4 + 4);
			}
			else
				idxStack[i++] = currIndex * 4 + 4;
			break;
		default:
			printf("Sector found within Adding new node invalid.\n");
		}
	}
	return noError;
}
__host__ __device__
void CheckAndCreateInChild(quadNode ** root, quadNode* &node, int parentIdx, float2 newPos, float2 iTopLeft, float2 iBotRight, int sector, int stack[], int & stackIndex, int maxCubesAmount)
{
	float2 topL, botR;
	int currIndex = parentIdx * 4 + sector;
	
	if (node == NULL || currIndex * 4 + 4 > maxCubesAmount) {
		ComputeBordersForSector(iTopLeft, iBotRight, sector-1, topL, botR);
		node = NewNode(root, newPos, topL, botR, currIndex);
	}
	else
		stack[stackIndex++] = currIndex;
}
__host__ __device__
void CheckLeafDist(int currIndex, quadNode* leaf, float2 point, float &curDist, float &minDist, int stack[], int& index, int sector) {
	if (leaf != NULL)
		if (minDist >= BoundaryMinDist(point, leaf->topLeft, leaf->botRight))
		{
			stack[index++] = currIndex * 4 + sector;
			DEBUG_TEXT("The added index is %i \n", stack[index-1]);
		}
}
__host__ __device__
void CheckChildNode(quadNode* &node, float2 point, int stack1[], int stack2[], int & index1, int & index2, int currIndex, int sector)
{
	if (node != NULL)
		if (InBoundary(point, node->topLeft, node->botRight))
			stack1[index1++] = currIndex * 4 + sector;
		else
			stack2[index2++] = currIndex * 4 + sector;
}

int testDistOfDistance(quadNode** root, float2 point, int radius)
{
	int stack[2048];
	int i = 0;
	int currIndex = 0;
	stack[i++] = currIndex;
	printf("\n ______Testing distance\n");
	quadNode* currNode;

	while (i > 0)
	{
		currIndex = stack[--i];
		currNode = root[currIndex];
		printf("testing node within %i node\n", currIndex);
		if (i > 2047)
		{
			printf("stack overflow within distTest\n");
			return 0;
		}
		if (currNode == NULL)
			continue; //ret 0

		float curDist = Dist(currNode->position, point);
		if ((float)radius > curDist)
			return 0;

		if (currNode->botRightTree != NULL)
				stack[i++] = currIndex * 4 + 4;
		if (currNode->botLeftTree != NULL)
				stack[i++] = currIndex * 4 + 3;
		if (currNode->topRightTree != NULL)
				stack[i++] = currIndex * 4 + 2;
		if (currNode->topLeftTree != NULL)
				stack[i++] = currIndex * 4 + 1;
	}
	
	return 1;
}

int FindNearest(quadNode** root, float2 point, float &minDist)
{
	DEBUG_TEXT("000000000000000Looking for nearest neighbour.0000000000000000000\n");
	int stack[40]; //serves to find the smallest node within which point is
	int stack2[2048]; // serves to compare distances to the point
	int i = 0, j = 0;

	int currIndex = 0;
	stack[i++] = currIndex;

	quadNode* currNode;

	//popping into the lowest node which contains point within its boundaries and stacking all children not containing point within boundaries
	while (i > 0)
	{
		currIndex = stack[--i];
		currNode = root[currIndex];
		if (i > 37 || j > 2040) //check to not overflow stack
		{
			printf("stack overflow in FindNearest.\n");
			return 0;
		}
			

		//recursion to pop into the smallest node that the point is in
		if (currNode == NULL)
			continue; 

		if (!InBoundary(point, currNode->topLeft, currNode->botRight))
			continue; 

		CheckChildNode(currNode->botRightTree, point, stack, stack2, i, j, currIndex, 4);
		CheckChildNode(currNode->botLeftTree, point, stack, stack2, i, j, currIndex, 3);
		CheckChildNode(currNode->topRightTree, point, stack, stack2, i, j, currIndex, 2);
		CheckChildNode(currNode->topLeftTree, point, stack, stack2, i, j, currIndex, 1);

		if (IsLeaf(currNode))
			stack2[j++] = currIndex;
	}
	
	//going through stacked nodes (which are the others than the ones contaning point) and testing distance
	while (j > 0)
	{
		currIndex = stack2[--j];
		currNode = root[currIndex];
		
		if (j > 2046) //check to not overflow stack
		{
			printf("stack overfloaw in FindNearest second part\n");
			return 0;
		}

		float curDist = Dist(currNode->position, point);
		if (minDist > curDist)
			minDist = curDist;

		CheckLeafDist(currIndex, currNode->botRightTree, point, curDist, minDist, stack2, j, 4);
		DEBUG_TEXT("Debug text outside leaf dist %i\n", stack2[j-1]);
		CheckLeafDist(currIndex, currNode->botLeftTree, point, curDist, minDist, stack2, j, 3);
		DEBUG_TEXT("Debug text outside leaf dist %i\n", stack2[j - 1]);
		CheckLeafDist(currIndex, currNode->topRightTree, point, curDist, minDist, stack2, j, 2);
		DEBUG_TEXT("Debug text outside leaf dist %i\n", stack2[j - 1]);
		CheckLeafDist(currIndex, currNode->topLeftTree, point, curDist, minDist, stack2, j, 1);
		DEBUG_TEXT("Debug text outside leaf dist %i\n", stack2[j - 1]);
	}
	return 1;
}


int FindNearest(quadNode* root, float2 point, float &minDist)
{
	int stack[40]; //serves to find the smallest node within which point is
	int stack2[2048]; // serves to compare distances to the point
	int i = 0, j = 0;

	int currIndex = 0;
	stack[i++] = currIndex;

	quadNode* currNode;

	//popping into the lowest node which contains point within its boundaries and stacking all children not containing point within boundaries
	while (i > 0)
	{
		currIndex = stack[--i];
		currNode = &root[currIndex];
		if (i > 37 || j > 2040) //check to not overflow stack
			return 0;

		//recursion to pop into the smallest node that the point is in
		if (currNode == NULL)
			continue; 

		if (!InBoundary(point, currNode->topLeft, currNode->botRight))
			continue;

		CheckChildNode(currNode->botRightTree, point, stack, stack2, i, j, currIndex, 4);
		CheckChildNode(currNode->botLeftTree, point, stack, stack2, i, j, currIndex, 3);
		CheckChildNode(currNode->topRightTree, point, stack, stack2, i, j, currIndex, 2);
		CheckChildNode(currNode->topLeftTree, point, stack, stack2, i, j, currIndex, 1);

		if (IsLeaf(currNode))
			stack2[j++] = currIndex;
	}
	
	//going through stacked nodes (which are the others than the ones contaning point) and testing distance
	while (j > 0)
	{
		currIndex = stack2[--j];
		currNode = &root[currIndex];
		
		if (j > 2046) //check to not overflow stack
			return 0;

		float curDist = Dist(currNode->position, point);
		if (minDist > curDist)
			minDist = curDist;

		CheckLeafDist(currIndex, currNode->botRightTree, point, curDist, minDist, stack2, j, 4);
		CheckLeafDist(currIndex, currNode->botLeftTree, point, curDist, minDist, stack2, j, 3);
		CheckLeafDist(currIndex, currNode->topRightTree, point, curDist, minDist, stack2, j, 2);
		CheckLeafDist(currIndex, currNode->topLeftTree, point, curDist, minDist, stack2, j, 1);
	
	}
	return 1;
}

void ComputeBordersForSector(float2 topLeft, float2 botRight, int sector, float2& resTopLeft, float2& resBotRight)
{
	float sValx = Minim(topLeft.x, botRight.x);
	float bValx = Maxim(topLeft.x, botRight.x);

	//just ensurance for cases when topLeft does not have higher y and botRight does not have higher x
	int tlx = (topLeft.x > botRight.x) ? -1 : 1;
	int tly = (topLeft.y < botRight.y) ? -1 : 1;
	
	//square hence the y val is same
	float distW = CompDist(topLeft.x, botRight.x) / 2;
	float distH = CompDist(topLeft.y, botRight.y) / 2;

	switch (sector) {
	case 0:
		resTopLeft = topLeft;
		resBotRight.x = topLeft.x + tlx* distW;
		resBotRight.y = topLeft.y - tly* distH;
		break;
	case 1:
		resTopLeft.x = topLeft.x + tlx* distW;
		resTopLeft.y = topLeft.y;
		resBotRight.x = botRight.x;
		resBotRight.y = topLeft.y - tly * distH;
		break;
	case 2:
		resTopLeft.x = topLeft.x;
		resTopLeft.y = topLeft.y - tly * distH;
		resBotRight.x = topLeft.x + tlx * distW;
		resBotRight.y = botRight.y;
		break;
	case 3:
		resTopLeft.x = topLeft.x + tlx * distW;
		resTopLeft.y = topLeft.y - tly * distH;
		resBotRight = botRight;
		break;
	default:
		printf("The given sector is higher than 3, hence invalid. \n");
	}

}
#pragma optimize("", on)