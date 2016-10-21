// Lab3_ProductTwoVectors.cu : Defines the entry point for the console application.
// Author: £ukasz Pawe³ Rabiec (259049)

#include "handlers.h"
#include "stdafx.h"
#include <stdlib.h>
#include <windows.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 1024
#define SIZE_OF_INT SIZE*SIZE*sizeof(int)

void CudaInit()
{
	// Initialize device
	HANDLE_ERROR(cudaSetDevice(0));
}

void FillMatrixes(int* firstVector, int* secondVector)
{
	for (int i = 0; i < SIZE * SIZE; i++)
	{
		firstVector[i] = 1;
		secondVector[i] = 1;
	}
}

int SumResults(int* result)
{
	int sum = 0;

	for (int i = 0; i < SIZE; i++)
	{
		sum += result[i];
	}

	return sum;
}

double CalculateTime(LARGE_INTEGER tim1, LARGE_INTEGER tim2, LARGE_INTEGER countPerSec)
{
	double time = 0;

	time = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;

	return time;
}

__global__ void ProductVectorsWithSumOnSingleThreadV1(const int* firstVector, const int* secondVector, int* result)
{
	__shared__ int cache[SIZE];
	register int prodGlob = blockDim.x * blockIdx.x + threadIdx.x;
	register int localThreadId = threadIdx.x;

	cache[localThreadId] = firstVector[prodGlob] * secondVector[prodGlob];

	__syncthreads();

	// Sum on single thread (poor version)
	if (threadIdx.x == 0)
	{
		result[blockIdx.x] = 0;		// Take this to registry (see next method)

		for (int i = 0; i < blockDim.x; i++)
		{
			result[blockIdx.x] += cache[i];
		}
	}
}

__global__ void ProductVectorsWithSumOnSingleThreadV2(const int* firstVector, const int* secondVector, int* result)
{
	__shared__ int product[SIZE];
	register int prodGlob = blockDim.x * blockIdx.x + threadIdx.x;
	register int localThreadId = threadIdx.x;

	product[localThreadId] = firstVector[prodGlob] * secondVector[prodGlob];

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

__global__ void ProductVectorsWithSumOnMultithreadsV1(const int* firstVector, const int* secondVector, int* result)
{
	__shared__ int product[SIZE];
	register int prodGlob = blockDim.x * blockIdx.x + threadIdx.x;
	register int localThreadId = threadIdx.x;

	product[localThreadId] = firstVector[prodGlob] * secondVector[prodGlob];

	// Sum with neighbor thread
	for (int i = 1; i < blockDim.x; i = i << 1)		// for (int i = 1; i < blockDim.x; i *= 2) // slower
	{
		__syncthreads();

		if (localThreadId % (i << 1) == 0)	// if (localThreadId % (2 * i) == 0) // slower
		{
			product[localThreadId] += product[localThreadId + i];
		}
	}

	if (localThreadId == 0)
	{
		result[blockIdx.x] = product[0];
	}
}

__global__ void ProductVectorsWithSumOnMultithreadsV2(const int* firstVector, const int* secondVector, int* result)
{
	__shared__ int product[SIZE];
	register int prodGlob = blockDim.x * blockIdx.x + threadIdx.x;
	register int localThreadId = threadIdx.x;

	product[localThreadId] = firstVector[prodGlob] * secondVector[prodGlob];

	// Sum with neighbor thread (half - fastest)
	for (int i = blockDim.x >> 1; i > 0; i >>= 1)		// for (int i = blockDim.x / 2; i > 0; i /= 2)
	{
		__syncthreads();

		if (localThreadId < i)		// if (localThreadId < blockDim.x / 2)
		{
			product[localThreadId] += product[localThreadId + i];	// product[localThreadId] += product[localThreadId + 1];
		}
	}

	if (localThreadId == 0)
	{
		result[blockIdx.x] = product[0];
	}
}

int main()
{
	// Initialize
	CudaInit();

	int *firstVector, *secondVector;
	int *devFirstVector, *devSecondVector, *devResult;
	int *result;
	int sum = 0;
	double time = 0;

	LARGE_INTEGER countPerSec, tim1, tim2;

	// Allocating memory
	firstVector = (int*)malloc(SIZE_OF_INT);
	secondVector = (int*)malloc(SIZE_OF_INT);
	result = (int*)malloc(SIZE_OF_INT);
	HANDLE_ERROR(cudaMalloc((void**)&devFirstVector, SIZE_OF_INT));
	HANDLE_ERROR(cudaMalloc((void**)&devSecondVector, SIZE_OF_INT));
	HANDLE_ERROR(cudaMalloc((void**)&devResult, SIZE_OF_INT));

	// Clock
	QueryPerformanceFrequency(&countPerSec);

	// Operations
	FillMatrixes(firstVector, secondVector);

	HANDLE_ERROR(cudaMemcpy(devFirstVector, firstVector, SIZE_OF_INT, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(devSecondVector, secondVector, SIZE_OF_INT, cudaMemcpyHostToDevice));

	// Sum on single thread V1
	QueryPerformanceCounter(&tim1);
	ProductVectorsWithSumOnSingleThreadV1<<<SIZE, SIZE>>>(devFirstVector, devSecondVector, devResult);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	HANDLE_ERROR(cudaMemcpy(result, devResult, SIZE_OF_INT, cudaMemcpyDeviceToHost));
	sum = SumResults(result);
	time = CalculateTime(tim1, tim2, countPerSec);
	printf("Product vectors with sum on single thread (v1): %d in time %lf ms\n", sum, time);

	// Sum on single thread V2
	QueryPerformanceCounter(&tim1);
	ProductVectorsWithSumOnSingleThreadV2<<<SIZE, SIZE>>>(devFirstVector, devSecondVector, devResult);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	HANDLE_ERROR(cudaMemcpy(result, devResult, SIZE_OF_INT, cudaMemcpyDeviceToHost));
	sum = SumResults(result);
	time = CalculateTime(tim1, tim2, countPerSec);
	printf("Product vectors with sum on single thread (v2): %d in time %lf ms\n", sum, time);

	// Sum on multithreads V1
	QueryPerformanceCounter(&tim1);
	ProductVectorsWithSumOnMultithreadsV1<<<SIZE, SIZE>>>(devFirstVector, devSecondVector, devResult);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	HANDLE_ERROR(cudaMemcpy(result, devResult, SIZE_OF_INT, cudaMemcpyDeviceToHost));
	sum = SumResults(result);
	time = CalculateTime(tim1, tim2, countPerSec);
	printf("Product vectors with sum on multithreads (v1): %d in time %lf ms\n", sum, time);

	// Sum on multithreads V2
	QueryPerformanceCounter(&tim1);
	ProductVectorsWithSumOnMultithreadsV2<<<SIZE, SIZE>>>(devFirstVector, devSecondVector, devResult);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	HANDLE_ERROR(cudaMemcpy(result, devResult, SIZE_OF_INT, cudaMemcpyDeviceToHost));
	sum = SumResults(result);
	time = CalculateTime(tim1, tim2, countPerSec);
	printf("Product vectors with sum on multithreads (v2): %d in time %lf ms\n", sum, time);

	getchar();

	cudaFree(devFirstVector);
	cudaFree(devSecondVector);
	cudaFree(devResult);
	free(firstVector);
	free(secondVector);
	free(result);

	return 0;
}

