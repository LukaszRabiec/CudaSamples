// Lab1_DeviceProps.cu : Defines the entry point for the console application.
// Author: £ukasz Pawe³ Rabiec (259049)

#include "stdafx.h"
#include "stdlib.h"
#include "cuda_runtime.h"


int main()
{
	// Devices
	cudaError_t err;
	int* countDevices = (int*)malloc(sizeof(int));

	err = cudaGetDeviceCount(countDevices);

	printf("CUDA Error: %s\n", cudaGetErrorString(err));
	printf("CUDA devices: %d\n\n", countDevices[0]);

	// Prop
	cudaDeviceProp prop;

	err = cudaGetDeviceProperties(&prop, countDevices[0] - 1);	// Devices counted from 0

	printf("CUDA Error: %s\n", cudaGetErrorString(err));
	printf("Name: %s\n", prop.name);
	printf("Device total memory: %d(MB)\n", prop.totalGlobalMem / 1024 / 1024);
	printf("Device shared memory: %d(KB)\n", prop.sharedMemPerBlock / 1024);
	printf("Maximum 32 bits registers per block: %d\n", prop.regsPerBlock);
	printf("Warp size: %d\n", prop.warpSize);
	printf("Compute capability (major, minor): (%d, %d)\n", prop.major, prop.minor);
	printf("Clock rate: %d(kHz)\n", prop.clockRate);
	printf("Maximum threads per multiprocessors: %d\n", prop.maxThreadsPerMultiProcessor);

	getchar();
	return 0;
}

