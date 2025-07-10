#include "gpu_data.h"
#include "allocator.h"

long GpuHelper::getTotalGlobalMem() {
	int nDevices;	
	cudaGetDeviceCount(&nDevices);
	
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		return prop.totalGlobalMem;
	}
	
	return 0;
}

/**
 * Assume one device for now.
 */
int GpuHelper::getMaxThreadsPerBlock() {
	int nDevices;	
	cudaGetDeviceCount(&nDevices);
	
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		return prop.maxThreadsPerBlock;
	}
	
	return 0;
}



