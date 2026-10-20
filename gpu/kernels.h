#ifndef GPU_GPURELU_H_
#define GPU_GPURELU_H_

#ifndef CNET_LAYER_PROFILER_DEFINED
#define CNET_LAYER_PROFILER_DEFINED
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdio>
// ---- optional per-layer GPU profiler (off by default; enabled by the trainer's -profile_steps) ----
// Records CUDA events around every mapping's forward and backward and accumulates elapsed time by
// layer type (the CPU function name with its numeric uid suffix stripped). The events force a host
// wait per layer, so the absolute step time while profiling is a little higher than without.
struct CnetLayerProfiler {
	bool on = false;
	std::map<std::string, double> fwd_ms, bwd_ms;
	std::map<std::string, long> fwd_n, bwd_n;
	double grad_ms = 0;
	std::map<std::string, double> sub_ms;   // optional sub-sections recorded inside a layer (not in TOTAL)
	static std::string typeOf(const std::string &name) {
		size_t e = name.size();
		while (e > 0 && (isdigit((unsigned char) name[e - 1]) || name[e - 1] == '_' || name[e - 1] == ' ')) --e;
		return name.substr(0, e);
	}
	void reset() { fwd_ms.clear(); bwd_ms.clear(); fwd_n.clear(); bwd_n.clear(); grad_ms = 0; sub_ms.clear(); }
	void print(int steps) const {
		std::map<std::string, double> tot;
		double T = grad_ms;
		for (auto &kv : fwd_ms) { tot[kv.first] += kv.second; T += kv.second; }
		for (auto &kv : bwd_ms) { tot[kv.first] += kv.second; T += kv.second; }
		std::vector<std::pair<double, std::string>> order;
		for (auto &kv : tot) order.push_back({kv.second, kv.first});
		std::sort(order.rbegin(), order.rend());
		printf("PROFILE per step over %d steps (ms): layer type | forward | backward | total | share\n", steps);
		for (auto &o : order) {
			const std::string &k = o.second;
			double f = fwd_ms.count(k) ? fwd_ms.at(k) : 0, b = bwd_ms.count(k) ? bwd_ms.at(k) : 0;
			printf("PROFILE %-28s %9.2f %9.2f %9.2f %5.1f%%\n", k.c_str(), f / steps, b / steps, (f + b) / steps, 100.0 * (f + b) / T);
		}
		printf("PROFILE %-28s %9s %9s %9.2f %5.1f%%\n", "grad from clones", "", "", grad_ms / steps, 100.0 * grad_ms / T);
		printf("PROFILE %-28s %9s %9s %9.2f\n", "TOTAL (layers + grad)", "", "", T / steps);
		for (auto &kv : sub_ms) printf("SUBSECTION %-40s %9.2f\n", kv.first.c_str(), kv.second / steps);
	}
};
inline CnetLayerProfiler &cnetProfiler() { static CnetLayerProfiler p; return p; }
#endif


#include "gpumapping.h"
#include "compl.h"
#include "../utils/stopwatch.h"

#ifdef __CUDACC__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

// --- CUDA Graphs support ---------------------------------------------------
// All kernel launches go through g_cnet_stream (0 == per-thread default stream
// in eager mode; the capture stream while a graph is being recorded). While a
// graph is being captured, g_cnet_capturing is true and the per-kernel device
// syncs (CNET_KSYNC) become no-ops, since host-side syncs are illegal during
// stream capture. In eager mode CNET_KSYNC preserves the original behaviour
// (a device-wide barrier after every launch).
extern cudaStream_t g_cnet_stream;
extern bool         g_cnet_capturing;
// When true, eager-mode CNET_KSYNC only checks for launch errors and does not block the host:
// kernels on one stream already run in order, and every host read of device memory synchronizes
// on its own (cudaMemcpy). Off by default; dp_shakespeare -ksync false turns it on.
extern bool         g_cnet_nosync;
// When true, the shared cuBLAS handles use TF32 tensor cores (CUBLAS_TF32_TENSOR_OP_MATH) instead of
// strict FP32 (CUBLAS_PEDANTIC_MATH). Must be set before the first GEMM; dp_shakespeare -tf32 true.
extern bool         g_cnet_tf32;
#define CUDA(x, y, z) <<< x, y, z, g_cnet_stream >>>
#define CUDA2(x, y) <<< x, y, 0, g_cnet_stream >>>

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

// Post-launch sync: real stream sync in eager mode, no-op while capturing a graph.
#define CNET_KSYNC() do { \
    if (!g_cnet_capturing) { \
        gpuErrchk(cudaPeekAtLastError()); \
        if (!g_cnet_nosync) gpuErrchk(cudaStreamSynchronize(g_cnet_stream)); \
    } \
} while (0)

#else

#include <stdio.h>

//Mocking CUDA for compiling it locally with g++ only.
#define CUDA(x, y, z)
#define CUDA2(x, y)
#define CNET_KSYNC()
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

void gpu_update_true_adam_input(CFunc *func, float l_rate, float beta1,
		float beta2, float eps, int t, float grad_clip = 0.0f);

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

// Adam second moment v (segments 6,7). The first moment reuses the dz slots
// (2,3) via momentum_real_/imag_ above.
__host__ __device__ inline float* v_real_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 6 * in.input_length_ + pos;
}

__host__ __device__ inline float* v_imag_ (GpuInVar in, int pos) {
	return in.input_ptr_ + 7 * in.input_length_ + pos;
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


// GPU counterpart of the CPU Pad layer (impl/pad.h): scatters `length()` free
// kernel taps into an `getOutputLength()`-long zero vector at the fixed index
// set kernel_index_, then gathers gradients back.  gpu_kernel_index_ is a device
// copy of the CPU kernel_index_, uploaded once in GpuFunc::allocatePadKernelIndex().
class PadGpu : public GpuMapping {
public:
	int *gpu_kernel_index_ = 0;   // device int array, length == length()

	PadGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~PadGpu() {
	}

	void gpu_pad_forward();

	virtual void forward() {
		gpu_pad_forward();
	}

	void gpu_pad_backward(int label);

	virtual void backward(int label) {
		gpu_pad_backward(label);
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

// GPU complex dropout (see ComplexDropout in impl/dropout.h). Inverted dropout
// with a per-element mask = hash(seed, step, element); `step_` bumps each forward
// for a fresh mask, and backward reuses the same step_ so the masks match. seed_
// folds in the layer depth so different dropout layers get independent masks.
extern bool g_dropout_infer;   // set true around eval/inference -> DropoutGpu is identity

class DropoutGpu : public GpuMapping {
	float keep_;        // 1 - p
	unsigned seed_;
	int step_ = 0;

public:
	DropoutGpu(int depth, float p)
		: GpuMapping(depth), keep_(1.0f - p), seed_(0x9E3779B9u ^ (unsigned) depth) {}

	virtual ~DropoutGpu() {}

	void gpu_dropout_forward();

	virtual void forward() {
		++step_;
		gpu_dropout_forward();
	}

	void gpu_dropout_backward();

	virtual void backward(int label) {
		gpu_dropout_backward();
	}
};

// GPU conjugate lag product y_t = x_t * conj(x_{t-tau}) (see ConjLagProduct in
// impl/conjlag.h). The lag is within each clone's length-L vector.
class ConjLagGpu : public GpuMapping {
	int tau_;

public:
	ConjLagGpu(int depth, int tau) : GpuMapping(depth), tau_(tau) {}

	virtual ~ConjLagGpu() {}

	void gpu_conjlag_forward();

	virtual void forward() {
		gpu_conjlag_forward();
	}

	void gpu_conjlag_backward();

	virtual void backward(int label) {
		gpu_conjlag_backward();
	}
};

// GPU segmented (windowed) Fourier / STFT: block-diagonal DFT of each length-w
// window (see SegmentedFourier in impl/segfourier.h). Roots computed on the fly.
class SegFourierGpu : public GpuMapping {
	int window_;

public:
	SegFourierGpu(int depth, int window) : GpuMapping(depth), window_(window) {}

	virtual ~SegFourierGpu() {}

	void gpu_segfourier_forward();

	virtual void forward() {
		gpu_segfourier_forward();
	}

	void gpu_segfourier_backward();

	virtual void backward(int label) {
		gpu_segfourier_backward();
	}
};

// GPU multi-plane spectral conv (see HadamardConv in impl/hadamardconv.h):
// out[f,k] = sum_{p,j} taps[f,p,j]*X[p,k]*root(index[j]*k) (+ D*bias[f] at k=0),
// all operands read from the one concatenated input [X | taps | bias]. Dims
// (C,D,F) and the tap positions are read from the CPU func; gpu_kernel_index_ is
// a device copy of its kernel_index, uploaded once in
// GpuFunc::allocateHadamardConvKernelIndex() (mirrors PadGpu).
class HadamardConvGpu : public GpuMapping {
public:
	int *gpu_kernel_index_ = 0;   // device int array, length == K taps
	int C_ = 0, D_ = 0, F_ = 0, K_ = 0;   // set in allocateHadamardConvKernelIndex()

	HadamardConvGpu(int depth) : GpuMapping(depth) {}

	void setDims(int C, int D, int F, int K) { C_ = C; D_ = D; F_ = F; K_ = K; }

	virtual ~HadamardConvGpu() {}

	void gpu_hadamardconv_forward();

	virtual void forward() {
		gpu_hadamardconv_forward();
	}

	void gpu_hadamardconv_backward();

	virtual void backward(int label) {
		gpu_hadamardconv_backward();
	}
};

// GPU |z|^2 magnitude nonlinearity (see CModulus2 in impl/relu.h).
class CModulus2Gpu : public GpuMapping {

public:
	CModulus2Gpu(int depth) : GpuMapping(depth) {
	}

	virtual ~CModulus2Gpu() {
	}

	void gpu_modulus2_forward();

	virtual void forward() {
		gpu_modulus2_forward();
	}

	void gpu_modulus2_backward();

	virtual void backward(int label) {
		gpu_modulus2_backward();
	}
};

// GPU mean pooling over time (see MeanPool in impl/pool.h). width = L / P where
// L = length() (input length) and P = getOutputLength() (number of bins).
class MeanPoolGpu : public GpuMapping {

public:
	MeanPoolGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~MeanPoolGpu() {
	}

	void gpu_meanpool_forward();

	virtual void forward() {
		gpu_meanpool_forward();
	}

	void gpu_meanpool_backward();

	virtual void backward(int label) {
		gpu_meanpool_backward();
	}
};

// GPU holomorphic power w = z^M (see CPower in impl/relu.h). The exponent is a
// per-layer constant passed at construction.
class CPowerGpu : public GpuMapping {
	int power_;

public:
	CPowerGpu(int depth, int power) : GpuMapping(depth), power_(power) {
	}

	virtual ~CPowerGpu() {
	}

	void gpu_power_forward();

	virtual void forward() {
		gpu_power_forward();
	}

	void gpu_power_backward();

	virtual void backward(int label) {
		gpu_power_backward();
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
