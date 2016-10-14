// Lab2_AddingTwoVectors.cu : Defines the entry point for the console application.
// Author: £ukasz Pawe³ Rabiec (259049)

#include "stdafx.h"
#include "stdlib.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "handlers.h"

#define SIZE 32

void FillMatrixes(int* a, int* b)
{
	for (int i = 0; i < SIZE; i++)
	{
		a[i] = -2*i;
		b[i] = i*i;
	}
}

__global__ void AddVectors(int* a, int* b, int* c)
{
	int tid = blockIdx.x;

	if (tid < SIZE)
	{
		c[tid] = a[tid] + b[tid];
	}

}

int main()
{
	// Initialize device
	HANDLE_ERROR(cudaSetDevice(0));

	// Allocating memory on GPU
	int *a, *b, *c, *dev_a, *dev_b, *dev_c;

	size_t bytes = SIZE * sizeof(int);

	a = (int*)malloc(bytes);
	b = (int*)malloc(bytes);
	c = (int*)malloc(bytes);

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, bytes));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, bytes));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, bytes));

	FillMatrixes(a, b);

	HANDLE_ERROR(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));

	AddVectors<<<SIZE, 1>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost));

	for (int i = 0; i < SIZE; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	getchar();

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	free(a);
	free(b);
	free(c);

	return 0;
}

