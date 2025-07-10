#ifndef GPU_ALLOCATOR_H_
#define GPU_ALLOCATOR_H_

#include <cmath>
#include <map>
#include <utility>
#include <vector>
#include <stdio.h>

#include "kernels.h"
#include "compl.h"
#include "../impl/utils.h"

#ifdef __CUDACC__
    #include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include <cuda_runtime.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#else


// Implementing CUDA functions locally on CPU.

typedef enum {
	cudaMemcpyHostToHost,
	cudaMemcpyHostToDevice,
	cudaMemcpyDeviceToHost,
	cudaMemcpyDeviceToDevice
} cudaMemcpyKind;

inline std::string getCudaMemcpyKindName(cudaMemcpyKind kind) {
	if (kind == cudaMemcpyHostToDevice) {
		return "CPU->GPU";
	}
	if (kind == cudaMemcpyDeviceToHost) {
		return "GPU->CPU";
	}
	if (kind == cudaMemcpyHostToHost) {
		return "CPU->CPU";
	}
	return "GPU->GPU";
}

typedef enum {
	cudaSuccess, cudaErrorMemoryAllocation
} cudaError_t;

inline extern cudaError_t cudaMalloc(void** devPtr, size_t size) {
	std::cout << "WARNING: mocking the GPU, allocating on CPU." << std::endl;
	*devPtr = malloc(size);
	return *devPtr ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline extern cudaError_t cudaFree(void* devPtr) {
	std::cout << "WARNING: mocking the GPU, freeing on CPU." << std::endl;
	free(devPtr);
	return cudaSuccess;
}

inline extern cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
	std::cout << "WARNING: mocking the GPU, memset on CPU." << std::endl;
	memset(devPtr, (char)value, count);
	return cudaSuccess;
}

inline extern cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
		                      cudaMemcpyKind kind) {
	std::cout << "WARNING: mocking the GPU, copying "
			  << getCudaMemcpyKindName(kind)
			  << " from: " << src
			  << " to: " << dst << std::endl;
	memcpy(dst, src, count);
	return cudaSuccess;
}

#define CHECK(call)                                                            \
{                                                                              \
    if (call != cudaSuccess)                                                   \
    {                                                                          \
    	std::cout << "Error when allocating on CPU." << std::endl;             \
    }                                                                          \
}

#endif // __CUDACC__

class GpuHelper {
	friend class GpuHelperTest;

	std::vector<void*> gpu_data_;
	std::vector<long long> length_;
	long long total_length_ = 0;

public:

#ifdef __CUDACC__
	long getTotalGlobalMem();
	int getMaxThreadsPerBlock();
#else
	long getTotalGlobalMem() {
		return 1024 * 1024 * 1024;
	}

	int getMaxThreadsPerBlock() {
		return 1024;
	}
#endif

//============

	cmplx_* cmplx_allocate_on_gpu(int length) {
		cmplx_ *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(cmplx_));
		if (ret != cudaSuccess) {
			CHECK(ret);
#ifdef __CUDACC__
			fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(ret), __FILE__, __LINE__);
			std::cout.flush();
#else
#endif
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(cmplx_));
		total_length_ += length * sizeof(cmplx_);
		return data;
	}


	bool cmplx_copy_from_gpu(int length, cmplx_ *source, cmplx_ *dest) {
		auto ret = cudaMemcpy((void*) dest, (void*) source,
				sizeof(cmplx_) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool cmplx_copy_to_gpu(int length, const cmplx_ *source_cpu,
			cmplx_ *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(cmplx_) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	//=================

	GpuOutVar* out_var_allocate_on_gpu(int length) {
		GpuOutVar *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(GpuOutVar));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(GpuOutVar));
		total_length_ += length * sizeof(GpuOutVar);
		return data;
	}

	bool out_var_copy_to_gpu(int length, const GpuOutVar *source_cpu,
			GpuOutVar *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(GpuOutVar) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool out_var_copy_from_gpu(int length, GpuOutVar *source_gpu,
			GpuOutVar *dest_cpu) {
		auto ret = cudaMemcpy((void*) dest_cpu, (void*) source_gpu,
				sizeof(GpuOutVar) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	//=================
	GpuCloneVar* clone_var_allocate_on_gpu(int length) {
		GpuCloneVar *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(GpuCloneVar));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(GpuCloneVar));
		total_length_ += length * sizeof(GpuCloneVar);
		return data;
	}

	bool clone_var_copy_to_gpu(int length, const GpuCloneVar *source_cpu,
			GpuCloneVar *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(GpuCloneVar) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool clone_var_copy_from_gpu(int length, GpuCloneVar *source_gpu,
			GpuCloneVar *dest_cpu) {
		auto ret = cudaMemcpy((void*) dest_cpu, (void*) source_gpu,
				sizeof(GpuCloneVar) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	GpuInVar* in_var_allocate_on_gpu(int length) {
		GpuInVar *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(GpuInVar));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(GpuInVar));
		total_length_ += length * sizeof(GpuInVar);
		return data;
	}

	bool in_var_copy_to_gpu(int length, const GpuInVar *source_cpu,
			GpuInVar *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(GpuInVar) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool in_var_copy_from_gpu(int length, GpuInVar *source_gpu,
			GpuInVar *dest_cpu) {
		auto ret = cudaMemcpy((void*) dest_cpu, (void*) source_gpu,
				sizeof(GpuInVar) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	Grads* grad_allocate_on_gpu(int length) {
		Grads *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(Grads));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(Grads));
		total_length_ += length * sizeof(Grads);
		return data;
	}

	bool grad_copy_to_gpu(int length, const Grads *source_cpu,
			Grads *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(Grads) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool grad_copy_from_gpu(int length, Grads *source_gpu,
			Grads *dest_cpu) {
		auto ret = cudaMemcpy((void*) dest_cpu, (void*) source_gpu,
				sizeof(Grads) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	//=================
	float** ptr_float_allocate_on_gpu(int length) {
		float **data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(float*));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(float*));
		total_length_ += length * sizeof(float*);
		return data;
	}

	bool ptr_float_copy_to_gpu(int length, float **source_cpu, float **dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(float*) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	float* float_allocate_on_gpu(int length) {
		float *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(float));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(float));
		total_length_ += length * sizeof(float);
		return data;
	}

	bool float_copy_to_gpu(int length, const float *source_cpu,
			float *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(float) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool float_copy_from_gpu(int length, float *source_gpu, float *dest_cpu) {
		auto ret = cudaMemcpy((void*) dest_cpu, (void*) source_gpu,
				sizeof(float) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool float_var_copy_on_gpu(int length, float *source_gpu, float *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_gpu,
				sizeof(float) * length, cudaMemcpyDeviceToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool deallocate() {
		bool success = true;
		for (int d_indx = 0; d_indx < gpu_data_.size(); ++d_indx) {
			printf("Freeing %llu bytes at pointer %p...\n", length_[d_indx],
					gpu_data_[d_indx]);
			auto ret = cudaFree(gpu_data_[d_indx]);
			if (ret != cudaSuccess) {
				CHECK(ret);
				success = false;
			}
		}
		total_length_ = 0;
		gpu_data_.clear();
		length_.clear();

		return success;
	}

	virtual ~GpuHelper() {
		if (total_length_ == 0) {
			return;
		}
		for (int d_indx = 0; d_indx < gpu_data_.size(); ++d_indx) {
//			printf("Freeing %llu bytes at pointer %p...\n", length_[d_indx],
//					gpu_data_[d_indx]);
			auto ret = cudaFree(gpu_data_[d_indx]);
			if (ret != cudaSuccess) {
				CHECK(ret);
			}
		}
	}

	int* int_allocate_on_gpu(int length) {
		int *data;
		auto ret = cudaMalloc((void**) &data, length * sizeof(int));
		if (ret != cudaSuccess) {
			CHECK(ret);
			return NULL;
		}
		gpu_data_.push_back((void*) data);
		length_.push_back(length * sizeof(int));
		total_length_ += length * sizeof(int);
		return data;
	}

	bool int_copy_to_gpu(int length, const int *source_cpu, int *dest_gpu) {
		auto ret = cudaMemcpy((void*) dest_gpu, (void*) source_cpu,
				sizeof(int) * length, cudaMemcpyHostToDevice);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	bool int_copy_from_gpu(int length, int *source_gpu, int *dest_cpu) {
		auto ret = cudaMemcpy((void*) dest_cpu, (void*) source_gpu,
				sizeof(int) * length, cudaMemcpyDeviceToHost);
		CHECK(ret);
		return cudaSuccess == ret;
	}

	cmplx_* u_roots_allocate_and_copy_on_gpu(int u_root_len) {
		cmplx_ *data = NULL;
		cmplx_ uroot_[u_root_len];

		if ((data = cmplx_allocate_on_gpu(u_root_len)) == NULL) {
			return NULL;
		}

		for (int var = 0; var < u_root_len; ++var) {
			uroot_[var] = {std::cos(-2 * var * M_PI / u_root_len) / sqrt(u_root_len),
					       std::sin(-2 * var * M_PI / u_root_len) / sqrt(u_root_len)};
		}

		if (!cmplx_copy_to_gpu(u_root_len, uroot_, data)) {
			return NULL;
		}

		return data;
	}
};

#endif /* GPU_ALLOCATOR_H_ */
