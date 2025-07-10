
#ifdef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <complex>
#include <memory>
#include <cfloat>

#include "gpu_func.h"

int MAX_THREADS = 1024;
int getMaxNoThread() {
	return MAX_THREADS;
}

#ifdef __CUDACC__
int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major){
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

std::string CNet::PrintInfo(bool& was_failure) {
	int nDevices;	
	std::ostringstream oss;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		oss	<< "Device Number: " << i << std::endl;
		oss	<< "  Device name: " << prop.name << std::endl;
		oss	<< "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
		oss	<< "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
		oss	<< "  Peak Memory Bandwidth (GB/s): " 
			<< 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
		oss	<< "  TotalGlobalMem: " << prop.totalGlobalMem / (1024 * 1024) << std::endl;
		oss	<< "  MaxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
		oss	<< "  SharedMemPerBlock: " << prop.sharedMemPerBlock / 1024 << std::endl;
		oss	<< "  WarpSize: " << prop.warpSize << std::endl;
		oss	<< "  MultiProcessorCount: " << prop.multiProcessorCount << std::endl;
		oss	<< "  CanMapHostMemory: " << prop.canMapHostMemory << std::endl;
		oss	<< "  CORES: " << getSPcores(prop) << std::endl;
	}
	
	return oss.str();
}
#endif


