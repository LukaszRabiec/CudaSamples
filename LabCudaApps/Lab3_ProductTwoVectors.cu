// Lab3_ProductTwoVectors.cu : Defines the entry point for the console application.
// Author: £ukasz Pawe³ Rabiec (259049)

#include "handlers.h"
#include "stdafx.h"
#include <stdlib.h>

#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 1024
#define SIZE_OF_INT SIZE*sizeof(int)

void CudaInit()
{
	// Initialize device
	HANDLE_ERROR(cudaSetDevice(0));
}

void FillMatrixes(int* firstVector, int* secondVector)
{
	for (int i = 0; i < SIZE; i++)
	{
		firstVector[i] = 1;
		secondVector[i] = 1;
	}
}

//TODO: Przypomniec na pocz¹tku zajêæ
__global__ void ProductVectorsAtomic(const int* firstVector, const int* secondVector, int* result)
{
	register int sum = 0;

	for (int i = 0; i < SIZE; i++)
	{
		sum += firstVector[i] * secondVector[i];
	}

	*result = sum;
}

__global__ void ProductVectorsWithSumOnSingleThreadV1(const int* firstVector, const int* secondVector, int* result)
{
	__shared__ int cache[SIZE];
	register int localThreadId = threadIdx.x;

	cache[localThreadId] = firstVector[localThreadId] * secondVector[localThreadId];

	__syncthreads();

	// Sum on single thread (poor version)
	if (threadIdx.x == 0)
	{
		result[blockIdx.x] = 0;

		for (int i = 0; i < blockDim.x; i++)
		{
			result[blockIdx.x] += cache[i];
		}
	}
}

__global__ void ProductVectorsWithSumOnSingleThreadV2(const int* firstVector, const int* secondVector, int* result)
{
	__shared__ int product[SIZE];
	//register int prodGlob = blockDim.x * blockIdx.x + threadIdx.x;
	register int localThreadId = threadIdx.x;

	product[localThreadId] = firstVector[localThreadId] * secondVector[localThreadId];

	__syncthreads();

	// Sum on single thread (rich version)
	if (threadIdx.x == 0)
	{
		register int sum = 0;

		for (int i = 0; i < blockDim.x; i++)
		{
			sum += product[i];
		}

		result[blockIdx.x] = sum;
	}
}

int main()
{
	// Initialize
	CudaInit();

	int *firstVector, *secondVector, *multithreadsResult;
	int *devFirstVector, *devSecondVector, *devResult;
	int result;

	// Allocating memory
	firstVector = (int*)malloc(SIZE_OF_INT);
	secondVector = (int*)malloc(SIZE_OF_INT);
	HANDLE_ERROR(cudaMalloc((void**)&devFirstVector, SIZE_OF_INT));
	HANDLE_ERROR(cudaMalloc((void**)&devSecondVector, SIZE_OF_INT));
	HANDLE_ERROR(cudaMalloc((void**)&devResult, SIZE_OF_INT));

	// Operations
	FillMatrixes(firstVector, secondVector);

	HANDLE_ERROR(cudaMemcpy(devFirstVector, firstVector, SIZE_OF_INT, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(devSecondVector, secondVector, SIZE_OF_INT, cudaMemcpyHostToDevice));

	// Sum on single thread V1
	ProductVectorsWithSumOnSingleThreadV1<<<1, SIZE>>>(devFirstVector, devSecondVector, devResult);
	HANDLE_ERROR(cudaMemcpy(&result, devResult, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Product vectors with sum on single thread (v1): %d\n", result);

	// Sum on single thread V2
	ProductVectorsWithSumOnSingleThreadV2 << <1, SIZE >> >(devFirstVector, devSecondVector, devResult);
	HANDLE_ERROR(cudaMemcpy(&result, devResult, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Product vectors with sum on single thread (v2): %d\n", result);

	getchar();

	cudaFree(devFirstVector);
	cudaFree(devSecondVector);
	cudaFree(devResult);
	free(firstVector);
	free(secondVector);

	return 0;
}

