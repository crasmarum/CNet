#ifndef GPU_GPURELU_H_
#define GPU_GPURELU_H_

#include "gpumapping.h"
#include "compl.h"
#include "../utils/stopwatch.h"

#ifdef __CUDACC__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA(x, y, z) <<< x, y, z >>>
#define CUDA2(x, y) <<< x, y >>>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   } else {
//	  fprintf(stderr,"All OK: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

#else

#include <stdio.h>

//Mocking CUDA for compiling it locally with g++ only.
#define CUDA(x, y, z)
#define CUDA2(x, y)
#define __global__

#define __device__
#define __shared__
#define __host__

struct dim3 {
	int x, y, z;
	dim3(int xx, int yy, int zz) {
		x = xx;
		y = yy;
		z = zz;
	}
};

typedef struct {
	int x;
} mockPoint;

const mockPoint blockDim = {0};
const mockPoint gridDim = {0};
const mockPoint blockIdx = {0};
const mockPoint threadIdx = {0};

inline void gpuErrchk(int);
inline int cudaPeekAtLastError() { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline void atomicAdd(float*, float) {};
inline void __syncthreads() {}
inline void __syncwarp() {}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}

#endif

__device__ const int MAX_BLOCK_SIZE = 1024;

void gpu_update_input(CFunc *func, float l_rate);

void gpu_update_adam_input(CFunc *func, float l_rate, float beta, int t);

void gpu_copy_to_clones(GpuCloneVar *in, int no_ancestors, int max_input_len);

void gpu_grad_from_clones(GpuCloneVar *in, int no_ancestors, int max_input_len);

__host__ __device__ inline float* Z_real_ (GpuInVar in, int pos) {
	return in.input_ptr_ + pos;
}

__host__ __device__ inline float* Z_imag_ (GpuInVar in, int pos) {
	return in.input_ptr_ + in.input_length_ + pos;
}

__host__ __device__ inline cmplx_ Z_(GpuInVar in, int pos) {
	return {*Z_real_(in, pos), *Z_imag_(in, pos)};
}

__host__ __device__ inline float* dZ_real_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 2 * in.input_length_ + pos;
}

__host__ __device__ inline float* dZ_imag_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 3 * in.input_length_ + pos;
}

__host__ __device__ inline cmplx_ dZ_(GpuInVar in, int pos) {
	return {*dZ_real_(in, pos), *dZ_imag_(in, pos)};
}

__host__ __device__ inline float* dZ_star_real_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 4 * in.input_length_ + pos;
}

__host__ __device__ inline float* dZ_star_imag_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 5 * in.input_length_ + pos;
}

__host__ __device__ inline cmplx_ dZ_star_(GpuInVar in, int pos) {
	return {*dZ_star_real_(in, pos), *dZ_star_imag_(in, pos)};
}

__host__ __device__ inline float* Z_real_ (GpuOutVar out, int pos) {
	return out.out_ptr_ + pos;
}

__host__ __device__ inline float* Z_imag_ (GpuOutVar out, int pos) {
	return out.out_ptr_ + out.out_length_ + pos;
}

__host__ __device__ inline cmplx_ Z_(GpuOutVar out, int pos) {
	return {*Z_real_(out, pos), *Z_imag_(out, pos)};
}

__host__ __device__ inline float* dZ_real_ (GpuOutVar out, int pos) {
	return out.out_ptr_ + 2 * out.out_length_ + pos;
}

__host__ __device__ inline float* dZ_imag_ (GpuOutVar out, int pos) {
	return out.out_ptr_ + 3 * out.out_length_ + pos;
}

__host__ __device__ inline cmplx_ dZ_(GpuOutVar out, int pos) {
	return {*dZ_real_(out, pos), *dZ_imag_(out, pos)};
}

__host__ __device__ inline float* dZ_star_real_ (GpuOutVar out, int pos) {
	return out.out_ptr_ + 4 * out.out_length_ + pos;
}

__host__ __device__ inline float* dZ_star_imag_ (GpuOutVar out, int pos) {
	return out.out_ptr_ + 5 * out.out_length_ + pos;
}

__host__ __device__ inline cmplx_ dZ_star_(GpuOutVar out, int pos) {
	return {*dZ_star_real_(out, pos), *dZ_star_imag_(out, pos)};
}

__host__ __device__ inline float* momentum_real_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 2 * in.input_length_ + pos;
}

__host__ __device__ inline float* momentum_imag_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 3 * in.input_length_ + pos;
}


class EmbeddingGpu : public GpuMapping {
	friend class CNet;

	int embedding_dim_;
	int no_embeddings_;
	int no_out_tokens_;

	int *gpu_tokens_;
	int no_allocated_tokens_;

public:
	EmbeddingGpu(int depth) : GpuMapping(depth), embedding_dim_(0), no_embeddings_(0),
		no_out_tokens_(0), gpu_tokens_(NULL) , no_allocated_tokens_(0) {
	}

	EmbeddingGpu(int depth, int embedding_dim, int no_embeddings, int no_out_tokens) : GpuMapping(depth), embedding_dim_(embedding_dim),
			no_embeddings_(no_embeddings), no_out_tokens_(no_out_tokens), gpu_tokens_(NULL), no_allocated_tokens_(0) {
	}

	void gpu_embedding_forward();

	virtual ~EmbeddingGpu() {
	}

	int noOutTokens() {
		return no_out_tokens_;
	}

	int outLength() {
		return no_out_tokens_ * no_embeddings_;
	}

	virtual void forward() {
		gpu_embedding_forward();
	}

	void gpu_embedding_backward(int label);

	virtual void backward(int label) {
		gpu_embedding_backward(label);
	}
};


class HadamardGpu : public GpuMapping {

public:
	HadamardGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~HadamardGpu() {
	}

	void gpu_hadamard_forward();

	virtual void forward() {
		gpu_hadamard_forward();
	}

	void hadamard_backward(int label);

	virtual void backward(int label) {
		hadamard_backward(label);
	}
};


class ResidualGpu : public GpuMapping {

public:
	ResidualGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~ResidualGpu() {
	}

	void gpu_residual_forward();

	virtual void forward() {
		gpu_residual_forward();
	}

	void gpu_residual_backward(int label);

	virtual void backward(int label) {
		gpu_residual_backward(label);
	}
};

class GeluGpu : public GpuMapping {

public:
	GeluGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~GeluGpu() {
	}

	void gpu_gelu_forward();

	virtual void forward() {
		gpu_gelu_forward();
	}

	void gpu_gelu_backward();

	virtual void backward(int label) {
		gpu_gelu_backward();
	}
};

class ReluGpu : public GpuMapping {

public:
	ReluGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~ReluGpu() {
	}

	void gpu_relu_forward();
	virtual void forward() {
		gpu_relu_forward();
	}

	void gpu_relu_backward();
	virtual void backward(int label) {
		gpu_relu_backward();
	}
};

class InputGpu : public GpuMapping {

public:
	InputGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~InputGpu() {
	}

	void gpu_input_forward();

	virtual void forward() {
		gpu_input_forward();
	}

	void gpu_input_backward();

	virtual void backward(int label) {
		gpu_input_backward();
	}
};

#endif /* GPU_GPURELU_H_ */
