// Lab5_MultiplyTwoMatrixes.cu : Defines the entry point for the console application.
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

#define SIZE 2
#define SIZE_OF_INT SIZE*SIZE*sizeof(int)

void CudaInit()
{
	// Initialize device
	HANDLE_ERROR(cudaSetDevice(0));
}

void FillMatrixes(int* firstMatrix, int* secondMatrix)
{
	for (int i = 0; i < SIZE * SIZE; i++)
	{
		firstMatrix[i] = i;
		secondMatrix[i] = 2*i;
	}
}

void PrintMatrix(int* matrix)
{
	for (int row = 0; row < SIZE; row++)
	{
		for (int col = 0; col < SIZE; col++)
		{
			printf("%d\t", matrix[row * SIZE + col]);
		}
		printf("\n");
	}
}

__global__ void MultiplyMatrixes(const int* firstMatrix, const int* secondMatrix, int* resultMatrix)
{

	int row = threadIdx.x;
	int col = threadIdx.y;
	int i;

	register int sum = 0;

	for (i = 0; i < blockDim.x; i++)
	{
		sum += firstMatrix[row * blockDim.x + i] * secondMatrix[i * blockDim.x + col];
	}

	resultMatrix[row * blockDim.x + col] += sum;
}

int main()
{
	// Initialize
	CudaInit();

	int *firstMatrix, *secondMatrix, *resultMatrix;
	int *devFirstMatrix, *devSecondMatrix, *devMatrix;

	// Allocating memory
	firstMatrix = (int*)malloc(SIZE_OF_INT);
	secondMatrix = (int*)malloc(SIZE_OF_INT);
	resultMatrix = (int*)malloc(SIZE_OF_INT);
	HANDLE_ERROR(cudaMalloc((void**)&devFirstMatrix, SIZE_OF_INT));
	HANDLE_ERROR(cudaMalloc((void**)&devSecondMatrix, SIZE_OF_INT));
	HANDLE_ERROR(cudaMalloc((void**)&devMatrix, SIZE_OF_INT));

	// Operations
	FillMatrixes(firstMatrix, secondMatrix);

	printf("First: \n");
	PrintMatrix(firstMatrix);
	printf("\nSecond: \n");
	PrintMatrix(secondMatrix);

	HANDLE_ERROR(cudaMemcpy(devFirstMatrix, firstMatrix, SIZE_OF_INT, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(devSecondMatrix, secondMatrix, SIZE_OF_INT, cudaMemcpyHostToDevice));

	dim3 block(SIZE, SIZE, 1);
	MultiplyMatrixes<<<1, block>>>(devFirstMatrix, devSecondMatrix, devMatrix);
	HANDLE_ERROR(cudaMemcpy(resultMatrix, devMatrix, SIZE_OF_INT, cudaMemcpyDeviceToHost));

	printf("\nResult: \n");
	PrintMatrix(resultMatrix);

	getchar();

	cudaFree(devFirstMatrix);
	cudaFree(devSecondMatrix);
	cudaFree(devMatrix);
	free(firstMatrix);
	free(secondMatrix);
	free(resultMatrix);

	return 0;
}

