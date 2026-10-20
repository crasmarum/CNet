#include <cstdlib>
#include "reduce.h"
#include "allocator.h"

#include "../utils/stopwatch.h"
#include "../utils/flags.h"

#ifdef __CUDACC__
#include <cufft.h>
#include <cublas_v2.h>
#endif

FLAG_INT(reduce_block_size, 512);

// When off, TokenwiseLinear uses the naive element-wise complex CUDA kernels; when on it
// uses batched cuBLAS GEMMs (~3x faster at LM-head width). Both are numerically equivalent
// (forward+backward match to ~1e-3, -tw_gpu; CUBLAS_PEDANTIC_MATH is set, so no TF32).
// HISTORICAL BUG (fixed): enabling cuBLAS used to collapse the Born-rule LM to the unigram
// loss under rope=true. The cause was NOT summation order -- it was a pointer-offset bug in
// tw_gemm_{forward,backward}: L and outlen were hoisted from entry 0, but Q, K, V are grouped
// into ONE batched TokenwiseLinearGpu mapping with DIFFERENT out_length_ per entry when
// rope=true (Q,K feed RotaryEmbed, out_length N*d; V feeds the BornAttention [Q;K;V] concat,
// out_length 3*N*d). Using entry 0's (Q's) outlen for V's writes/reads corrupted the [Q;K;V]
// buffer -- overwriting Im(Q) with Im(V) in the forward and misreading V's gradient segments
// in the backward -- which destroyed content-based routing while the lone output head learned
// the position-independent unigram marginal. (The element-wise kernels index per mapping entry,
// so they, rope=false, and single-layer unit tests were all fine.) Fixed by computing L/outlen
// per batch entry inside the loop (below). With the fix, -tw_cublas true trains identically to
// the element-wise path (char WikiText escapes the unigram floor and descends normally, 3.22 ->
// 2.19 by step 1200) at ~3x throughput. Default OFF only because the batched-GEMM pointer pool
// still issues a per-step cudaMemcpy (not CUDA-graph capturable); enable it for faster non-graph
// training.
FLAG_BOOL(tw_cublas, false)

// When set, the Linear layer uses the direct shared-memory GEMV kernels
// (forward and backward) instead of the generic reduce-then-sum path.
FLAG_BOOL(fast_linear, true)

// When set, the FourierTrans layer uses a cuFFT O(N log N) transform instead of
// the O(N^2) direct-DFT reduction. Off by default while it beds in.
FLAG_BOOL(fft_fast, false)

// When set, SoftMax uses the O(N) closed-form backward instead of the O(N^2)
// dense-Jacobian reduction.
FLAG_BOOL(fast_softmax, true)

#define Z(var, pos, len) cmplx((var)[(pos)], (var)[(len + pos)])

__device__ void warpReduce(cmplx_ *sdata, int tid) {
	cmplx_ tmp = {0.0, 0.0};
	tmp += sdata[tid + 16]; __syncwarp();
	sdata[tid] = tmp;       __syncwarp();
	tmp += sdata[tid + 8];  __syncwarp();
	sdata[tid] = tmp;       __syncwarp();
	tmp += sdata[tid + 4]; __syncwarp();
	sdata[tid] = tmp;      __syncwarp();
	tmp += sdata[tid + 2]; __syncwarp();
	sdata[tid] = tmp;      __syncwarp();
	tmp += sdata[tid + 1]; __syncwarp();
	sdata[tid] = tmp;
}

__device__ inline cmplx_ getDataForFft(int map_indx, int pos_in_segment, int init_seg_length,
		                 int segm_no, GpuInVar *in, unsigned int tid) {
	cmplx_ X = cmplx(in[map_indx].input_ptr_[pos_in_segment],
			in[map_indx].input_ptr_[init_seg_length + pos_in_segment]);
	assert(in->other_);
//	cmplx_ u_root = { std::cos(
//			-2 * pos_in_segment * segm_no * M_PI / init_seg_length), std::sin(
//			-2 * pos_in_segment * segm_no * M_PI / init_seg_length) };
//	return X * u_root;
	return X * in->other_[(pos_in_segment * segm_no) % init_seg_length];
}

__device__ inline cmplx_ getDataForT_Fft(int map_indx, int pos_in_segment, int init_seg_length,
        int segm_no, GpuInVar *in, unsigned int tid) {
	if (segm_no < pos_in_segment) {
		return {0.f, 0.f};
	}
	cmplx_ X = cmplx(in[map_indx].input_ptr_[pos_in_segment],
			in[map_indx].input_ptr_[init_seg_length + pos_in_segment]);
	assert(in->other_);
	return X * in->other_[(pos_in_segment * segm_no) % init_seg_length];
}

__device__ inline cmplx_ getDataForNorm(int map_indx, int pos_in_segment, int init_seg_length,
		                 int segm_no, GpuInVar *in, unsigned int tid) {
	cmplx_ X = cmplx(in[map_indx].input_ptr_[pos_in_segment],
			in[map_indx].input_ptr_[init_seg_length + pos_in_segment]);
	return X * conj_(X);
}

//__device__ cmplx_ getDataForLinear(int no_segments, int unpadded_length, int seg_out_len, GpuInVar *in, int segment_indx, int col) {
__device__ cmplx_ getDataForLinear(int fun_segment_indx, int col, int init_seg_length,
                                   int row, GpuInVar *in, unsigned int tid) {


	float *in_segment_start = in[fun_segment_indx].input_ptr_;
	int in_total_len = in[fun_segment_indx].input_length_;

	int offset = (row + 1) * init_seg_length;
	float *mat_row_starts = in_segment_start + offset;

/*
	printf("seg_start = %p \t row = %d \t col = %03d \t tid = %05d \t si = %d \t ul = %d \t tot_len = %d "
			"\t (%f, %f) * (%f, %f)\n",
			in_segment_start, row, col, tid, row, init_seg_length, in_total_len,
			mat_row_starts[col], mat_row_starts[in_total_len + col], in_segment_start[col], in_segment_start[in_total_len + col]);
//*/

	cmplx_ mat_val = Z(mat_row_starts, col, in_total_len);
	cmplx_ vec_val = Z(in_segment_start, col, in_total_len);

	return mat_val * vec_val;
}

__device__ inline cmplx_ getDataFor(int provider_id, int map_indx, int pos_in_segment, int init_seg_length,
									int segm_no, GpuInVar *in, unsigned int tid) {

	switch (provider_id) {
		case FFT_DATA_PROVIDER:
			return getDataForFft(map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
		case SOFTMAX_DATA_PROVIDER:
			return getDataForNorm(map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
		case NORM_DATA_PROVIDER:
			return getDataForNorm(map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
		case LINEAR_DATA_PROVIDER:
			return getDataForLinear(map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
			break;
		case T_FFT_DATA_PROVIDER:
			return getDataForT_Fft(map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
		default:
			return cmplx(0, 0);
			break;
	}
}

template<unsigned int blockSize>
__global__ void reducing_kernel__(int provider_id, GpuInVar *in, cmplx_ *out,
							int seg_length, int init_seg_length, int no_mappings, size_t max_no_threads) {
#ifdef __CUDACC__
	extern __shared__ cmplx_ sdata[];
#else
	cmplx_ sdata[1024];
#endif

    unsigned int tid = threadIdx.x;
	size_t thread_indx = blockIdx.x * blockDim.x + threadIdx.x;

	// max_threads = segment_len * no_segments * no_mappings;
	int pos_in_segment = thread_indx % seg_length;
    size_t out_indx = thread_indx / blockSize;

	// Every lane in the block must reach each __syncthreads() below, so
	// out-of-range and padding lanes contribute zero rather than returning
	// early: a divergent __syncthreads() is undefined behaviour (block hang
	// or corrupted partial sums).
	if (thread_indx >= max_no_threads || pos_in_segment >= init_seg_length) {
		sdata[tid] = {0, 0};
	} else {
		int no_segments = max_no_threads / no_mappings / seg_length;
		int map_indx = thread_indx / no_segments / seg_length;
		int segm_no = (thread_indx / seg_length) % no_segments;
		sdata[tid] = getDataFor(provider_id, map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
	}
    __syncthreads();

/*
	if (init_seg_length <= 32) {
		printf("map_indx = %d \t segm_no=%d  pos_in_segment = %05d \t out_indx = %d \t seg_length = %d \t init_seg_length = %d \t th_id = %05d \t"
				" X = (%f, %f) \t no_segments = %d \n",
				map_indx, segm_no, pos_in_segment, (int)out_indx, seg_length, init_seg_length, (int)thread_indx,
				sdata[tid].real, sdata[tid].imag, no_segments);
	}
//*/

	if (blockSize == 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512];} __syncthreads();}
	if (blockSize >= 512) { if (tid < 256) { sdata[tid]  += sdata[tid + 256];} __syncthreads();}
	if (blockSize >= 256) { if (tid < 128) { sdata[tid]  += sdata[tid + 128];} __syncthreads();}
	if (blockSize >= 128) { if (tid < 64) { sdata[tid]   += sdata[tid + 64];} __syncthreads();}
	if (blockSize >= 64) { if (tid < 32) { sdata[tid]    += sdata[tid + 32];} __syncthreads();}

//	if (reduce_use_warps) {
//		if (tid < 32) { warpReduce(sdata, blockIdx.x);}
//		__syncthreads();
//	} else {
	if (blockSize >= 32) { if (tid < 16) { sdata[tid]    += sdata[tid + 16];} __syncthreads();}
	if (blockSize >= 16) { if (tid < 8) { sdata[tid]    += sdata[tid + 8];} __syncthreads();}
	if (blockSize >= 8) { if (tid < 4) { sdata[tid]    += sdata[tid + 4];} __syncthreads();}
	if (blockSize >= 4) { if (tid < 2) { sdata[tid]    += sdata[tid + 2];} __syncthreads();}
	if (blockSize >= 2) { if (tid < 1) { sdata[tid]    += sdata[tid + 1];} __syncthreads();}

	if (tid == 0) {
		out[out_indx] = sdata[0];
	}
}

__global__ void fft_kernel_end__(int provider_id, int no_mappings, cmplx_ *tmp_in, GpuInVar *in, GpuOutVar *out,
		int in_stride, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	cmplx_ sum = cmplx(0.f, 0.f);
	int offset = thread_indx * in_stride;
	for (int var = 0; var < in_stride; ++var) {
		sum += (tmp_in + offset)[var];
	}

	int map_length = max_len / no_mappings;
	int map_indx = thread_indx / map_length;
	int pos = thread_indx - map_indx * map_length;
	int out_len = out[map_indx].out_length_;

//	if (thread_indx < 100) {
//		printf("map_length=%d \t map_indx=%d \t pos=%05d \t in_stride=%d \t no_mappings=%d \t thread_indx=%05d \t out_len=%d %f + %f \t %p\n",
//				map_length, map_indx, pos, in_stride, no_mappings, thread_indx, out_len, sum.real, sum.imag, out[map_indx].out_ptr_);
//	}

	if (pos < out_len) {
		out[map_indx].out_ptr_[pos] = sum.real;
		out[map_indx].out_ptr_[out_len + pos] = sum.imag;
	}
}

__global__ void softmax_kernel_end__(int no_mappings, cmplx_ *tmp_in, GpuInVar *in, GpuOutVar *out,
		int in_stride, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_length = max_len / no_mappings;
	int map_indx = thread_indx / map_length;
	int pos = thread_indx - map_indx * map_length;
	int out_len = out[map_indx].out_length_;

	cmplx_ sum = cmplx(0.f, 0.f);
	int offset = map_indx * in_stride;
	for (int var = 0; var < in_stride; ++var) {
		sum += (tmp_in + offset)[var];
	}
	if (thread_indx % map_length == 0) {
		// for computing gradient later
		out[map_indx].reduce_real_ = sum.real;
		sum.imag = pow(sum.real, 1.5);
		if(sum.imag <= 1.0e-15) {
			sum.imag = 1.0e-15;
		}
		out[map_indx].reduce_imag_ = sum.imag;
	}

	sum.real = sqrt(sum.real);
	if(sum.real <= 1.0e-15) {
		sum.real = 1.0e-15;
	}

//	if (thread_indx < 100) {
//		printf("map_length=%d \t map_indx=%d \t pos=%05d \t in_stride=%d \t no_mappings=%d \t thread_indx=%05d \t out_len=%d %f + %f \t %p\n",
//				map_length, map_indx, pos, in_stride, no_mappings, thread_indx, out_len, sum.real, sum.imag, out[map_indx].out_ptr_);
//	}

	if (pos < in[map_indx].input_length_) {   // per-entry length (defensive: grouped mappings may differ)
		out[map_indx].out_ptr_[pos] = in[map_indx].input_ptr_[pos] / sum.real;
		out[map_indx].out_ptr_[out_len + pos] = in[map_indx].input_ptr_[in[map_indx].input_length_ + pos] / sum.real;
	}
}

__global__ void norm_kernel_end__(int provider_id, int no_mappings, cmplx_ *tmp_in, GpuInVar *in, GpuOutVar *out,
		int in_stride, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	cmplx_ sum = cmplx(0.f, 0.f);
	int offset = thread_indx * in_stride;
	for (int var = 0; var < in_stride; ++var) {
		sum += (tmp_in + offset)[var];
	}

	int map_indx = thread_indx;

	if (thread_indx < no_mappings) {
		out[map_indx].reduce_real_ = sum.real;
		out[map_indx].reduce_imag_ = 0;
	}
}

void reducing_kernel(int provider_id, GpuInVar *in, cmplx_ *out,
				int no_segments, int seg_len, int no_mappings, int block_size) {
	int init_seg_length = seg_len;
	seg_len = seg_len % block_size == 0 ? seg_len
			: (seg_len + block_size - seg_len % block_size);

	size_t no_threads = (size_t) seg_len * no_segments * no_mappings;   // 64-bit: avoid int32 overflow at large channel/mapping counts
	unsigned grid = (unsigned) ((no_threads + block_size - 1) / block_size);

//	std::cout << "Launching reducing_kernel: "
//			 << " Grid size: " << grid << " Block Size: " << block_size
//			 << " segment_len: " << seg_len << " init_seg_length: " << init_seg_length
//			 << " no_segments: " << no_segments << " no_mappings: " << no_mappings
//			 << " no threads: " << no_threads << "\n";

	switch (block_size) {
		case 1024:
			reducing_kernel__ <1024> CUDA ( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 512:
			reducing_kernel__ <512> CUDA( grid , block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 256:
			reducing_kernel__ <256> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 128:
			reducing_kernel__ <128> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 64:
			reducing_kernel__ <64> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 32:
			reducing_kernel__ <32> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 16:
			reducing_kernel__ <16> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 8:
			reducing_kernel__ <8> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		case 4:
			reducing_kernel__ <4> CUDA( grid, block_size, block_size * sizeof(cmplx_) )
				(provider_id, in, out, seg_len, init_seg_length, no_mappings, no_threads);
			break;
		default:
			std::cerr << "Unsupported block size: " << block_size << ". Use: 1024, 512, 256, 128, 64, 32, 16, 8 or, 4." << std::endl;
			break;
	}


#ifdef __CUDACC__
	CNET_KSYNC();
#else
#endif

}

void fft_kernel_end(int provider_id, cmplx_ *temp_in, GpuInVar *in, GpuOutVar *out, int seg_len, int no_mappings, int block_size) {
	int in_stride = seg_len % block_size == 0 ? seg_len : (seg_len + block_size - seg_len % block_size);
	in_stride = in_stride / block_size;
	int no_threads = seg_len * no_mappings;

	unsigned grid = (no_threads + block_size - 1) / block_size;
//	std::cout << "Launching fft_kernel_end with Grid size: " << grid << " & Block Size:" << block_size
//			  << " in_stride: " << in_stride << " seg_len: " << seg_len
//			  << " no_mappings: " << no_mappings << "\n";

	fft_kernel_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
			(provider_id, no_mappings, temp_in, in, out, in_stride, no_threads);

#ifdef __CUDACC__
	CNET_KSYNC();
#else
#endif

}

void norm_kernel_end(int provider_id, cmplx_ *temp_in, GpuInVar *in, GpuOutVar *out, int seg_len, int no_mappings, int block_size) {
	int in_stride = seg_len % block_size == 0 ? seg_len : (seg_len + block_size - seg_len % block_size);
	in_stride = in_stride / block_size;
	int no_threads = no_mappings;

	unsigned grid = (no_threads + block_size - 1) / block_size;
//	std::cout << "Launching norm_kernel_end with Grid size: " << grid << " & Block Size:" << block_size
//			  << " in_stride: " << in_stride << " seg_len: " << seg_len
//			  << " no_mappings: " << no_mappings << "\n";

	norm_kernel_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
			(provider_id, no_mappings, temp_in, in, out, in_stride, no_threads);

#ifdef __CUDACC__
	CNET_KSYNC();
#else
#endif

}

void softmax_kernel_end(cmplx_ *temp_in, GpuInVar *in, GpuOutVar *out, int seg_len, int no_mappings, int block_size) {
	int in_stride = seg_len % block_size == 0 ? seg_len : (seg_len + block_size - seg_len % block_size);
	in_stride = in_stride / block_size;
	int no_threads = no_mappings * seg_len;

	unsigned grid = (no_threads + block_size - 1) / block_size;
//	std::cout << "Launching softmax_kernel_end with Grid size: " << grid << " & Block Size:" << block_size
//			  << " in_stride: " << in_stride << " seg_len: " << seg_len
//			  << " no_mappings: " << no_mappings << " no threads: " << no_threads << "\n";

	softmax_kernel_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
			(no_mappings, temp_in, in, out, in_stride, no_threads);

#ifdef __CUDACC__
	CNET_KSYNC();
#else
#endif

}

// ---- cuFFT fast path for FourierTrans (forward) ----
// CNet stores complex vectors planar (all reals then all imags) while cuFFT
// wants interleaved cufftComplex, so a gather/scatter pair brackets each
// transform. The batch of clones is laid out contiguously so one batched plan
// of size N (batch M) does the whole layer.

// planar input z (per clone) -> interleaved, contiguous per clone.
__global__ void gpu_fft_gather__(GpuInVar *in, cmplx_ *buf, int N, int no_mappings) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) {
		return;
	}
	int map = tid / N;
	int k = tid % N;
	buf[tid] = cmplx(*Z_real_(in[map], k), *Z_imag_(in[map], k));
}

// interleaved transform result -> planar output, scaled (overwrite).
__global__ void gpu_fft_scatter_out__(cmplx_ *buf, GpuOutVar *out, int N, int no_mappings, float scale) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) {
		return;
	}
	int map = tid / N;
	int k = tid % N;
	cmplx_ v = buf[tid];
	out[map].out_ptr_[k] = v.real * scale;
	out[map].out_ptr_[out[map].out_length_ + k] = v.imag * scale;
}

void FourierGpu::ensure_fft_plan(int N, int M) {
#ifdef __CUDACC__
	if (fft_plan_ != -1 && fft_N_ == N && fft_M_ == M) {
		return;
	}
	free_fft();
	cufftHandle plan;
	cufftResult r = cufftPlan1d(&plan, N, CUFFT_C2C, M);
	assert(r == CUFFT_SUCCESS);
	fft_plan_ = (int) plan;
	gpuErrchk(cudaMalloc((void**) &fft_buf_, sizeof(cmplx_) * (size_t) N * M));
	fft_N_ = N;
	fft_M_ = M;
#endif
}

void FourierGpu::free_fft() {
#ifdef __CUDACC__
	if (fft_plan_ != -1) {
		cufftDestroy((cufftHandle) fft_plan_);
		fft_plan_ = -1;
	}
	if (fft_buf_) {
		cudaFree(fft_buf_);
		fft_buf_ = NULL;
	}
	fft_N_ = 0;
	fft_M_ = 0;
#endif
}

void FourierGpu::gpu_fft_forward(int block_size) {
	if (fft_fast) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();
		ensure_fft_plan(N, M);

		int total = N * M;
		int tb = 256;
		unsigned grid = (total + tb - 1) / tb;
		float scale = 1.0f / sqrtf((float) N);

		gpu_fft_gather__ CUDA2(grid, tb) (gpu_in_ptr_, fft_buf_, N, M);
		cufftSetStream((cufftHandle) fft_plan_, g_cnet_stream);
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_FORWARD);
		gpu_fft_scatter_out__ CUDA2(grid, tb) (fft_buf_, gpu_out_ptr_, N, M, scale);

		CNET_KSYNC();
		return;
#endif
	}

	int b_out_length = getPaddedLength(block_size, this);
	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	auto gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	reducing_kernel(FFT_DATA_PROVIDER, gpu_in_ptr_, gpu_buffer, length(), length(), getNoMappings(), block_size);
	fft_kernel_end(FFT_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, length(), getNoMappings(), block_size);
}

// ===================== Born-rule attention (GPU) =====================
// Transcription of impl/bornattn.h (verified CPU reference). Input [Q;K;V] of
// length 3*N*d per clone (Q at pos t*d+dd, K at M+.., V at 2M+..), output N*d.
// One thread per (clone, query k); K[m]/V[m] receive from all queries k>=m, so
// their gradients use atomics.

#ifdef __CUDACC__
cublasHandle_t tw_blas_handle();
#endif
#ifdef __CUDACC__
static void born_memdbg(const char *tag) {
	static int on = -1;
	if (on < 0) on = getenv("CNET_MEMDBG") ? 1 : 0;
	if (!on) return;
	size_t fr = 0, to = 0; cudaDeviceSynchronize(); cudaMemGetInfo(&fr, &to);
	fprintf(stderr, "[memdbg] %s: used %.1f MB\n", tag, (double) (to - fr) / 1e6);
}
#endif

void BornAttentionGpu::ensure_scratch(int N, int d, int H, int M, int cols) {
#ifdef __CUDACC__
	if (ba_c_ && ba_N_ == N && ba_d_ == d && ba_H_ == H && ba_M_ == M && ba_cols_ == cols) return;
	free_scratch();
	ba_N_ = N; ba_d_ = d; ba_H_ = H; ba_M_ = M; ba_cols_ = cols;
	gpuErrchk(cudaMalloc((void**) &ba_c_, sizeof(cmplx_) * (size_t) M * H * N * cols));
	gpuErrchk(cudaMalloc((void**) &ba_A_, sizeof(float)  * (size_t) M * H * N * cols));
	gpuErrchk(cudaMalloc((void**) &ba_S_, sizeof(float)  * (size_t) M * H * N));
#endif
}
void BornAttentionGpu::free_scratch() {
#ifdef __CUDACC__
	if (ba_c_) { cudaFree(ba_c_); ba_c_ = NULL; }
	if (ba_A_) { cudaFree(ba_A_); ba_A_ = NULL; }
	if (ba_S_) { cudaFree(ba_S_); ba_S_ = NULL; }
	ba_N_ = ba_d_ = ba_H_ = ba_M_ = ba_cols_ = 0;
#endif
}

// One thread per (clone, head, query k). Head h owns dims [h*dh, (h+1)*dh) of
// the d-wide token; scratch row (clone*H + h)*N + k holds its scores.
__device__ inline float born_gamma__(int h, int decay) { return decay ? 1.f - exp2f(-(5.f + (float) h)) : 1.f; }

__global__ void gpu_born_forward__(GpuInVar *in, GpuOutVar *out,
		cmplx_ *C, float *A, float *S, int N, int d, int H, int B, int causal, float eps, int decay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * N) return;
	int clone = tid / (H * N), rem = tid - clone * H * N, h = rem / N, k = rem - h * N;
	const float gam = born_gamma__(h, decay);
	const int Mv = N * d, dh = d / H, d0 = h * dh;
	const int mhi = causal ? k : N - 1;
	GpuInVar inv = in[clone];
	size_t cb = ((size_t) clone * H + h) * N * N + (size_t) k * N;
	float Sk = 0.f;
	for (int m = 0; m <= mhi; ++m) {
		cmplx_ c = cmplx(0.f, 0.f);
		for (int j = 0; j < dh; ++j)
			c = c + Z_(inv, k * d + d0 + j) * conj_(Z_(inv, Mv + m * d + d0 + j));
		C[cb + m] = c;
		float s = c.real * c.real + c.imag * c.imag;
		if (decay) s *= powf(gam, fabsf((float) (k - m)));
		A[cb + m] = s;
		Sk += s;
	}
	S[((size_t) clone * H + h) * N + k] = Sk;
	float denomF = Sk + eps;
	float invS = denomF > 0.f ? 1.f / denomF : 0.f;
	for (int m = 0; m <= mhi; ++m) A[cb + m] *= invS;
	GpuOutVar ov = out[clone];
	for (int j = 0; j < dh; ++j) {
		const int dd = d0 + j;
		float orr = 0.f, oii = 0.f;
		for (int m = 0; m <= mhi; ++m) {
			float a = A[cb + m];
			cmplx_ v = Z_(inv, 2 * Mv + m * d + dd);
			orr += a * v.real; oii += a * v.imag;
		}
		ov.out_ptr_[k * d + dd] = orr;
		ov.out_ptr_[ov.out_length_ + k * d + dd] = oii;
	}
}

__global__ void gpu_born_backward__(GpuInVar *in, GpuOutVar *out,
		cmplx_ *C, float *A, float *S, int N, int d, int H, int B, int causal, float eps, int decay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * N) return;
	int clone = tid / (H * N), rem = tid - clone * H * N, h = rem / N, k = rem - h * N;
	const float gam = born_gamma__(h, decay);
	const int Mv = N * d, dh = d / H, d0 = h * dh;
	const int mhi = causal ? k : N - 1;
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	size_t cb = ((size_t) clone * H + h) * N * N + (size_t) k * N;
	float denomB = S[((size_t) clone * H + h) * N + k] + eps;
	float invS = denomB > 0.f ? 1.f / denomB : 0.f;
	float abar = 0.f;
	for (int m = 0; m <= mhi; ++m) {
		float am = 0.f;
		for (int j = 0; j < dh; ++j) {
			const int dd = d0 + j;
			cmplx_ gO = dZ_(ov, k * d + dd), gOb = dZ_star_(ov, k * d + dd);
			cmplx_ v = Z_(inv, 2 * Mv + m * d + dd);
			am += (gO * v).real + (gOb * conj_(v)).real;
		}
		abar += A[cb + m] * am;
	}
	for (int m = 0; m <= mhi; ++m) {
		float am = 0.f;
		for (int j = 0; j < dh; ++j) {
			const int dd = d0 + j;
			cmplx_ gO = dZ_(ov, k * d + dd), gOb = dZ_star_(ov, k * d + dd);
			cmplx_ v = Z_(inv, 2 * Mv + m * d + dd);
			am += (gO * v).real + (gOb * conj_(v)).real;
		}
		float b = (am - abar) * invS;
		if (decay) b *= powf(gam, fabsf((float) (k - m)));
		cmplx_ c = C[cb + m], cc = conj_(c);
		float Akm = A[cb + m];
		for (int j = 0; j < dh; ++j) {
			const int dd = d0 + j;
			cmplx_ gO = dZ_(ov, k * d + dd), gOb = dZ_star_(ov, k * d + dd);
			cmplx_ q = Z_(inv, k * d + dd), kk = Z_(inv, Mv + m * d + dd);
			atomicAdd(dZ_real_(inv, 2 * Mv + m * d + dd), gO.real * Akm);
			atomicAdd(dZ_imag_(inv, 2 * Mv + m * d + dd), gO.imag * Akm);
			atomicAdd(dZ_star_real_(inv, 2 * Mv + m * d + dd), gOb.real * Akm);
			atomicAdd(dZ_star_imag_(inv, 2 * Mv + m * d + dd), gOb.imag * Akm);
			cmplx_ dQ = cc * conj_(kk), dQb = c * kk;        // Q: b*conj(c)*conj(K), b*c*K
			atomicAdd(dZ_real_(inv, k * d + dd), b * dQ.real);
			atomicAdd(dZ_imag_(inv, k * d + dd), b * dQ.imag);
			atomicAdd(dZ_star_real_(inv, k * d + dd), b * dQb.real);
			atomicAdd(dZ_star_imag_(inv, k * d + dd), b * dQb.imag);
			cmplx_ dK = c * conj_(q), dKb = cc * q;          // K: b*c*conj(Q), b*conj(c)*Q
			atomicAdd(dZ_real_(inv, Mv + m * d + dd), b * dK.real);
			atomicAdd(dZ_imag_(inv, Mv + m * d + dd), b * dK.imag);
			atomicAdd(dZ_star_real_(inv, Mv + m * d + dd), b * dKb.real);
			atomicAdd(dZ_star_imag_(inv, Mv + m * d + dd), b * dKb.imag);
		}
	}
}

void BornAttentionGpu::gpu_born_forward() {
#ifdef __CUDACC__
	BornAttention *f = (BornAttention*) cpu_func_.front();
	if (f->linear()) { gpu_born_linear_forward(); return; }
	int N = f->nTokens(), d = f->dim(), H = f->heads(), B = getNoMappings();
	born_memdbg("quad fwd entry");
	ensure_scratch(N, d, H, B, N);
	born_memdbg("quad fwd after scratch");
	int total = B * H * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_born_forward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, ba_c_, ba_A_, ba_S_, N, d, H, B, f->causal() ? 1 : 0, f->eps(), f->decay() ? 1 : 0);
	CNET_KSYNC();
	born_memdbg("quad fwd exit");
#endif
}

void BornAttentionGpu::gpu_born_backward() {
#ifdef __CUDACC__
	BornAttention *f = (BornAttention*) cpu_func_.front();
	if (f->linear()) { gpu_born_linear_backward(); return; }
	int N = f->nTokens(), d = f->dim(), H = f->heads(), B = getNoMappings();
	int total = B * H * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_born_backward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, ba_c_, ba_A_, ba_S_, N, d, H, B, f->causal() ? 1 : 0, f->eps(), f->decay() ? 1 : 0);
	CNET_KSYNC();
#endif
}


// ======== BornAttention, exact chunkwise LINEAR form (density-matrix features) ========
//
// Per head (dh complex dims), lift x in C^dh to c(x) in R^{d2}, d2 = dh*dh, the packed
// Hermitian matrix x x^H: dh diagonal entries |x_m|^2, then sqrt2*Re(x_m conj x_n) and
// sqrt2*Im(x_m conj x_n) for m<n. Then <c(q), c(k)> = |<q,k>|^2 exactly, so with the
// prefix states Z_t = sum_{j<=t} c(K_j) (d2) and M_t = sum_{j<=t} c(K_j) [Re V_j | Im V_j]^T
// (d2 x 2dh): O_t = c(Q_t)^T M_t / (c(Q_t)^T Z_t + eps). Chunkwise: tokens in the current
// chunk are mixed with the quadratic kernel (C x C scores), earlier chunks through the
// state carried from the previous chunk; the state advances by one GEMM per chunk.
// Scratch is O(N*C) instead of O(N^2); time O(N (C + d2) dh) instead of O(N^2 dh).

// feature a -> (m, n): a < dh diagonal; then the dh(dh-1)/2 (m<n) pairs twice (sym, skew).
static void born_lift_index_table(int dh, std::vector<int> &idx) {
	const int d2 = dh * dh, T = dh * (dh - 1) / 2;
	idx.assign((size_t) 2 * d2, 0);
	for (int m = 0; m < dh; ++m) { idx[2 * m] = m; idx[2 * m + 1] = m; }
	int q = 0;
	for (int m = 0; m < dh; ++m)
		for (int n = m + 1; n < dh; ++n, ++q) {
			idx[2 * (dh + q)] = m;     idx[2 * (dh + q) + 1] = n;
			idx[2 * (dh + T + q)] = m; idx[2 * (dh + T + q) + 1] = n;
		}
}

// cX[((clone*H + h)*C + t)*d2 + a] = c(X[c0+t, head h])_a for X = Q (blk 0) or K (blk 1).
__global__ void gpu_born_lift__(GpuInVar *in, float *cX, const int *idx, int blk,
		int c0, int C, int N, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H, d2 = dh * dh, T = dh * (dh - 1) / 2;
	if (tid >= (size_t) B * H * C * d2) return;
	int a = (int) (tid % d2); size_t r = tid / d2;
	int t = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	GpuInVar inv = in[clone];
	const int base = blk * N * d + (c0 + t) * d + h * dh;
	const int m = idx[2 * a], n = idx[2 * a + 1];
	cmplx_ xm = Z_(inv, base + m);
	float val;
	if (m == n) val = xm.real * xm.real + xm.imag * xm.imag;
	else {
		cmplx_ p = xm * conj_(Z_(inv, base + n));                  // x_m conj(x_n)
		val = 1.41421356237f * (a < dh + T ? p.real : p.imag);
	}
	cX[tid] = val;
}

// U[(((clone*H + h)*2 + plane)*C + t)*dh + j] = plane(V[c0+t, head h, j]) (0 = Re, 1 = Im).
__global__ void gpu_born_packv__(GpuInVar *in, float *U, int c0, int C, int N, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * 2 * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int pl = (int) (r % 2); r /= 2;
	int h = (int) (r % H); int clone = (int) (r / H);
	GpuInVar inv = in[clone];
	const int p = 2 * N * d + (c0 + t) * d + h * dh + j;
	U[tid] = pl == 0 ? inv.input_ptr_[p] : inv.input_ptr_[inv.input_length_ + p];
}

// decay helpers (fixed gamma_h = 1 - 2^-(5+h)). dst[bh][t][a] = src * gamma_h^(C-1-t) (mode 0: keys) or
// gamma_h^(t+1) (mode 1: queries); one thread per element.
__global__ void gpu_born_rowscale__(float *dst, const float *src, int mode, int C, int d2, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (size_t) B * H * C * d2) return;
	size_t r = tid / d2; int t = (int) (r % C); r /= C; int h = (int) (r % H);
	const float gam = born_gamma__(h, 1);
	dst[tid] = src[tid] * powf(gam, mode == 0 ? (float) (C - 1 - t) : (float) (t + 1));
}
// p[bh][...per_bh...] *= gamma_h^C (state carried across a chunk boundary)
__global__ void gpu_born_scale_bh__(float *p, size_t per_bh, int BH, int H, int C) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (size_t) BH * per_bh) return;
	const int h = (int) ((tid / per_bh) % H);
	p[tid] *= powf(born_gamma__(h, 1), (float) C);
}

// One thread per (clone, head, query kc in the chunk): intra-chunk quadratic scores
// over keys c0..c0+kc, combined with the inter-chunk readout (num, den) from the state.
// Scratch rows are C wide: Cs/A[((clone*H + h)*N + k)*C + m], S[(clone*H + h)*N + k] =
// full score sum (intra + inter, without eps), A normalized by the full denominator.
__global__ void gpu_born_chunk_forward__(GpuInVar *in, GpuOutVar *out,
		cmplx_ *Cs, float *A, float *S, const float *num, const float *den,
		int c0, int C, int N, int d, int H, int B, float eps, int decay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * C) return;
	int kc = tid % C, r = tid / C, h = r % H, clone = r / H;
	const float gam = born_gamma__(h, decay);
	const int k = c0 + kc, dh = d / H, d0 = h * dh, Mv = N * d;
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	const size_t cb = (bh * N + k) * C;
	float Sk = 0.f;
	for (int m = 0; m <= kc; ++m) {
		cmplx_ c = cmplx(0.f, 0.f);
		for (int j = 0; j < dh; ++j)
			c = c + Z_(inv, k * d + d0 + j) * conj_(Z_(inv, Mv + (c0 + m) * d + d0 + j));
		Cs[cb + m] = c;
		float s = c.real * c.real + c.imag * c.imag;
		if (decay) s *= powf(gam, (float) (kc - m));
		A[cb + m] = s;
		Sk += s;
	}
	Sk += den[bh * C + kc];
	S[bh * N + k] = Sk;
	float denomF = Sk + eps;
	float invS = denomF > 0.f ? 1.f / denomF : 0.f;
	for (int m = 0; m <= kc; ++m) A[cb + m] *= invS;
	for (int m = kc + 1; m < C; ++m) { A[cb + m] = 0.f; Cs[cb + m] = cmplx(0.f, 0.f); }   // the backward GEMMs read full C x C blocks
	const float *nr = num + ((bh * 2 + 0) * C + kc) * dh, *ni = num + ((bh * 2 + 1) * C + kc) * dh;
	for (int j = 0; j < dh; ++j) {
		const int dd = d0 + j;
		float orr = nr[j] * invS, oii = ni[j] * invS;
		for (int m = 0; m <= kc; ++m) {
			float a = A[cb + m];
			cmplx_ v = Z_(inv, 2 * Mv + (c0 + m) * d + dd);
			orr += a * v.real; oii += a * v.imag;
		}
		ov.out_ptr_[k * d + dd] = orr;
		ov.out_ptr_[ov.out_length_ + k * d + dd] = oii;
	}
}


// ---- GEMM form of the linear-mode forward chunk step (same outputs as gpu_born_chunk_forward__) ----
// raw Q and K planes of the chunk: P[bh][2][C][dh]; offset 0 for Q, N*d for K
__global__ void gpu_born_packqk__(GpuInVar *in, float *Qp, float *Kp, int c0, int C, int N, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * 2 * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int pl = (int) (r % 2); r /= 2;
	int h = (int) (r % H); int clone = (int) (r / H);
	GpuInVar inv = in[clone];
	const int p = (c0 + t) * d + h * dh + j;
	const float *base = inv.input_ptr_ + (pl ? inv.input_length_ : 0);
	Qp[tid] = base[p];
	Kp[tid] = base[N * d + p];
}
// Cr holds <q_k, k_m> planar; write the complex score into Cs (zero above the diagonal) and the
// unnormalized weight |c|^2 gamma^(k-m) into A. One thread per (bh, k, m), m fastest.
__global__ void gpu_born_fwd_ew__(const float *Cr, cmplx_ *Cs, float *A, int c0, int C, int N, int H, int BH, int decay) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (size_t) BH * C * C) return;
	const int m = (int) (tid % C); size_t r = tid / C;
	const int kc = (int) (r % C); const size_t bh = r / C;
	const size_t cb = (bh * N + c0 + kc) * C + m;
	if (m > kc) { Cs[cb] = cmplx(0.f, 0.f); A[cb] = 0.f; return; }
	const size_t pr = (bh * 2 + 0) * C * C + (size_t) kc * C + m, pi = pr + (size_t) C * C;
	const cmplx_ c = cmplx(Cr[pr], Cr[pi]);
	Cs[cb] = c;
	float sc = c.real * c.real + c.imag * c.imag;
	if (decay) sc *= powf(born_gamma__((int) (bh % H), decay), (float) (kc - m));
	A[cb] = sc;
}
// one warp per (bh, k): S_k = sum_m A + den; normalize the row of A and the inter-chunk numerator
__global__ void gpu_born_fwd_rownorm__(float *A, float *S, float *num, const float *den, int c0, int C, int N, int dh, int BH, float eps) {
	const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
	if (row >= BH * C) return;
	const int kc = row % C; const size_t bh = row / C;
	float *Ar = A + (bh * N + c0 + kc) * C;
	float part = 0.f;
	for (int m = lane; m <= kc; m += 32) part += Ar[m];
	for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
	const float Sk = part + den[bh * C + kc];
	const float denomF = Sk + eps, invS = denomF > 0.f ? 1.f / denomF : 0.f;
	if (lane == 0) S[bh * N + c0 + kc] = Sk;
	for (int m = lane; m <= kc; m += 32) Ar[m] *= invS;
	float *nr = num + ((bh * 2 + 0) * C + kc) * dh, *ni = num + ((bh * 2 + 1) * C + kc) * dh;
	for (int j = lane; j < dh; j += 32) { nr[j] *= invS; ni[j] *= invS; }
}
// write the chunk output planes O[bh][2][C][dh] into the layer output
__global__ void gpu_born_out_scatter__(GpuOutVar *out, const float *O, int c0, int C, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	const size_t bh = (size_t) clone * H + h;
	GpuOutVar ov = out[clone];
	const int p = (c0 + t) * d + h * dh + j;
	ov.out_ptr_[p] = O[((bh * 2 + 0) * C + t) * dh + j];
	ov.out_ptr_[ov.out_length_ + p] = O[((bh * 2 + 1) * C + t) * dh + j];
}
// row-major Z[n x m] = alpha op(X)[n x k] op(Y)[k x m] + beta Z, batched (cuBLAS column-major via transposes)
static void born_rm_gemm(cublasHandle_t hb, bool tX, bool tY, int n, int m, int k, float alpha,
		const float *X, int ldX, long long sX, const float *Y, int ldY, long long sY, float beta, float *Z, int ldZ, long long sZ, int batch) {
	cublasSgemmStridedBatched(hb, tY ? CUBLAS_OP_T : CUBLAS_OP_N, tX ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
		Y, ldY, sY, X, ldX, sX, &beta, Z, ldZ, sZ, batch);
}

void BornAttentionGpu::ensure_linear_scratch(int N, int d, int H, int C, int B) {
#ifdef __CUDACC__
	const int dh = d / H, d2 = dh * dh, NC = N / C;
	if (lin_cQ_ && lin_C_ == C && lin_NC_ == NC && lin_d2_ == d2 && lin_B_ == B && lin_H_ == H && lin_dh_ == dh) return;
	free_linear_scratch();
	lin_C_ = C; lin_NC_ = NC; lin_d2_ = d2; lin_B_ = B; lin_H_ = H; lin_dh_ = dh;
	const size_t BH = (size_t) B * H;
	gpuErrchk(cudaMalloc((void**) &lin_cQ_,    sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &lin_cK_,    sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &lin_U_,     sizeof(float) * BH * 2 * C * dh));
	gpuErrchk(cudaMalloc((void**) &lin_num_,   sizeof(float) * BH * 2 * C * dh));
	gpuErrchk(cudaMalloc((void**) &lin_den_,   sizeof(float) * BH * C));
	gpuErrchk(cudaMalloc((void**) &lin_M_,     sizeof(float) * BH * 2 * d2 * dh));
	gpuErrchk(cudaMalloc((void**) &lin_Z_,     sizeof(float) * BH * d2));
	gpuErrchk(cudaMalloc((void**) &carry_M_,   sizeof(float) * BH * 2 * d2 * dh));
	gpuErrchk(cudaMalloc((void**) &carry_Z_,   sizeof(float) * BH * d2));
	gpuErrchk(cudaMemset(carry_M_, 0, sizeof(float) * BH * 2 * d2 * dh));
	gpuErrchk(cudaMemset(carry_Z_, 0, sizeof(float) * BH * d2));
	gpuErrchk(cudaMalloc((void**) &lin_Mhist_, sizeof(float) * (size_t) NC * BH * 2 * d2 * dh));
	gpuErrchk(cudaMalloc((void**) &lin_Zhist_, sizeof(float) * (size_t) NC * BH * d2));
	gpuErrchk(cudaMalloc((void**) &lin_ones_,  sizeof(float) * C));
	gpuErrchk(cudaMalloc((void**) &lin_Qp_,    sizeof(float) * BH * 2 * C * dh));
	gpuErrchk(cudaMalloc((void**) &lin_Kp_,    sizeof(float) * BH * 2 * C * dh));
	gpuErrchk(cudaMalloc((void**) &lin_Cr_,    sizeof(float) * BH * 2 * C * C));
	gpuErrchk(cudaMalloc((void**) &lin_cKw_,   sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &lin_cQd_,   sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &lin_idx_,   sizeof(int) * 2 * d2));
	gpuErrchk(cudaMalloc((void**) &bw_Gm_,    sizeof(float) * BH * 2 * C * dh));
	gpuErrchk(cudaMalloc((void**) &bw_Gt_,    sizeof(float) * BH * C * 2 * dh));
	gpuErrchk(cudaMalloc((void**) &bw_Ut_,    sizeof(float) * BH * C * 2 * dh));
	gpuErrchk(cudaMalloc((void**) &bw_Ug_,    sizeof(float) * BH * C * 4 * dh));
	gpuErrchk(cudaMalloc((void**) &bw_G4_,    sizeof(float) * BH * C * 4 * dh));
	gpuErrchk(cudaMalloc((void**) &bw_Am_,    sizeof(float) * BH * C * C));
	gpuErrchk(cudaMalloc((void**) &bw_Bm_,    sizeof(float) * BH * C * C));
	gpuErrchk(cudaMalloc((void**) &bw_W_,     sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &bw_Gq_,    sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &bw_Gk_,    sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &bw_cQs_,   sizeof(float) * BH * C * d2));
	gpuErrchk(cudaMalloc((void**) &bw_dV_,    sizeof(float) * BH * C * 4 * dh));
	gpuErrchk(cudaMalloc((void**) &bw_abar_,  sizeof(float) * BH * C));
	gpuErrchk(cudaMalloc((void**) &bw_abarD_, sizeof(float) * BH * C));
	gpuErrchk(cudaMalloc((void**) &bw_Ms_,    sizeof(float) * BH * d2 * 4 * dh));
	gpuErrchk(cudaMalloc((void**) &bw_Zs_,    sizeof(float) * BH * d2));
	std::vector<float> ones(C, 1.f);
	gpuErrchk(cudaMemcpy(lin_ones_, ones.data(), sizeof(float) * C, cudaMemcpyHostToDevice));
	std::vector<int> idx; born_lift_index_table(dh, idx);
	gpuErrchk(cudaMemcpy(lin_idx_, idx.data(), sizeof(int) * 2 * d2, cudaMemcpyHostToDevice));
#endif
}

void BornAttentionGpu::free_linear_scratch() {
#ifdef __CUDACC__
	float **ps[] = {&lin_cQ_, &lin_cK_, &lin_U_, &lin_num_, &lin_den_, &lin_M_, &lin_Z_, &lin_Mhist_, &lin_Zhist_, &lin_ones_, &lin_cKw_, &lin_cQd_, &lin_Qp_, &lin_Kp_, &lin_Cr_, &carry_M_, &carry_Z_,
		&bw_Gm_, &bw_Gt_, &bw_Ut_, &bw_Ug_, &bw_G4_, &bw_Am_, &bw_Bm_, &bw_W_, &bw_Gq_, &bw_Gk_, &bw_cQs_,
		&bw_dV_, &bw_abar_, &bw_abarD_, &bw_Ms_, &bw_Zs_};
	for (float **p : ps) if (*p) { cudaFree(*p); *p = NULL; }
	if (lin_idx_) { cudaFree(lin_idx_); lin_idx_ = NULL; }
	lin_C_ = lin_NC_ = lin_d2_ = lin_B_ = lin_H_ = lin_dh_ = 0;
#endif
}

void BornAttentionGpu::gpu_born_linear_forward() {
#ifdef __CUDACC__
	BornAttention *f = (BornAttention*) cpu_func_.front();
	const int N = f->nTokens(), d = f->dim(), H = f->heads(), B = getNoMappings(), C = f->chunk();
	if (!f->causal() || N % C != 0 || C < 1) {
		fprintf(stderr, "BornAttentionGpu linear mode: needs causal attention and N %% chunk == 0 (N=%d C=%d)\n", N, C);
		exit(1);
	}
	const int dh = d / H, d2 = dh * dh, NC = N / C;
	const size_t BH = (size_t) B * H;
	born_memdbg("lin fwd entry");
	ensure_scratch(N, d, H, B, C);
	ensure_linear_scratch(N, d, H, C, B);
	born_memdbg("lin fwd after scratch");
	cudaStream_t st = g_cnet_stream ? g_cnet_stream : cudaStreamPerThread;
	if (carry_) {
		gpuErrchk(cudaMemcpyAsync(lin_M_, carry_M_, sizeof(float) * BH * 2 * d2 * dh, cudaMemcpyDeviceToDevice, st));
		gpuErrchk(cudaMemcpyAsync(lin_Z_, carry_Z_, sizeof(float) * BH * d2, cudaMemcpyDeviceToDevice, st));
	} else {
		gpuErrchk(cudaMemsetAsync(lin_M_, 0, sizeof(float) * BH * 2 * d2 * dh, st));
		gpuErrchk(cudaMemsetAsync(lin_Z_, 0, sizeof(float) * BH * d2, st));
	}
	cublasHandle_t hb = tw_blas_handle();
	cublasSetStream(hb, st);
	born_memdbg("lin fwd after cublas handle");
	const float one = 1.f, zero = 0.f;
	const int tb = 128;
	const int decay = f->decay() ? 1 : 0;
	const size_t nLift = BH * C * d2, nPack = BH * 2 * C * dh, nChunk = BH * C;
	const float *cQr = decay ? lin_cQd_ : lin_cQ_, *cKw = decay ? lin_cKw_ : lin_cK_;   // readout queries / state keys
	for (int c = 0; c < NC; ++c) {
		const int c0 = c * C;
		gpu_born_lift__  CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (gpu_in_ptr_, lin_cQ_, lin_idx_, 0, c0, C, N, d, H, B);
		gpu_born_lift__  CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (gpu_in_ptr_, lin_cK_, lin_idx_, 1, c0, C, N, d, H, B);
		if (decay) {
			gpu_born_rowscale__ CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (lin_cKw_, lin_cK_, 0, C, d2, H, B);
			gpu_born_rowscale__ CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (lin_cQd_, lin_cQ_, 1, C, d2, H, B);
		}
		gpu_born_packv__ CUDA2((unsigned) ((nPack + tb - 1) / tb), tb) (gpu_in_ptr_, lin_U_, c0, C, N, d, H, B);
		// snapshot the exclusive prefix state at this chunk's start (for the backward)
		gpuErrchk(cudaMemcpyAsync(lin_Mhist_ + (size_t) c * BH * 2 * d2 * dh, lin_M_, sizeof(float) * BH * 2 * d2 * dh, cudaMemcpyDeviceToDevice, st));
		gpuErrchk(cudaMemcpyAsync(lin_Zhist_ + (size_t) c * BH * d2, lin_Z_, sizeof(float) * BH * d2, cudaMemcpyDeviceToDevice, st));
		// inter-chunk readout: num_pl[C,dh] = cQ[C,d2] * M_pl[d2,dh]; den[C] = cQ * Z
		// (row-major [r,c] with ld c == column-major [c,r]; the planes are strided batches)
		for (int pl = 0; pl < 2; ++pl)
			cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, dh, C, d2, &one,
				lin_M_ + (size_t) pl * d2 * dh, dh, (long long) 2 * d2 * dh,
				cQr, d2, (long long) C * d2, &zero,
				lin_num_ + (size_t) pl * C * dh, dh, (long long) 2 * C * dh, (int) BH);
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, 1, C, d2, &one,
			lin_Z_, 1, (long long) d2, cQr, d2, (long long) C * d2, &zero,
			lin_den_, 1, (long long) C, (int) BH);
		{	// intra-chunk step as GEMMs: scores, mask/decay, row normalization, weights x values
			const size_t nSq = BH * C * C, nQK = BH * 2 * C * dh;
			gpu_born_packqk__ CUDA2((unsigned) ((nQK + tb - 1) / tb), tb) (gpu_in_ptr_, lin_Qp_, lin_Kp_, c0, C, N, d, H, B);
			const float *Qr = lin_Qp_, *Qi = lin_Qp_ + (size_t) C * dh, *Kr = lin_Kp_, *Ki = lin_Kp_ + (size_t) C * dh;
			float *Cr = lin_Cr_, *Ci = lin_Cr_ + (size_t) C * C;
			const long long sP = 2LL * C * dh, sS = 2LL * C * C;
			// <q,k> = Q conj(K)^T:  Re = Qr Kr^T + Qi Ki^T,  Im = Qi Kr^T - Qr Ki^T
			born_rm_gemm(hb, false, true, C, C, dh,  1.f, Qr, dh, sP, Kr, dh, sP, 0.f, Cr, C, sS, (int) BH);
			born_rm_gemm(hb, false, true, C, C, dh,  1.f, Qi, dh, sP, Ki, dh, sP, 1.f, Cr, C, sS, (int) BH);
			born_rm_gemm(hb, false, true, C, C, dh,  1.f, Qi, dh, sP, Kr, dh, sP, 0.f, Ci, C, sS, (int) BH);
			born_rm_gemm(hb, false, true, C, C, dh, -1.f, Qr, dh, sP, Ki, dh, sP, 1.f, Ci, C, sS, (int) BH);
			gpu_born_fwd_ew__ CUDA2((unsigned) ((nSq + 255) / 256), 256) (lin_Cr_, ba_c_, ba_A_, c0, C, N, H, (int) BH, decay);
			gpu_born_fwd_rownorm__ CUDA2((unsigned) ((nChunk + 7) / 8), 256) (ba_A_, ba_S_, lin_num_, lin_den_, c0, C, N, dh, (int) BH, f->eps());
			// O = num * invS + A V  (A real, V complex planes)
			const float *Ab = ba_A_ + (size_t) c0 * C;
			const long long sA = (long long) N * C;
			for (int pl = 0; pl < 2; ++pl)
				born_rm_gemm(hb, false, false, C, dh, C, 1.f, Ab, C, sA, lin_U_ + (size_t) pl * C * dh, dh, sP, 1.f, lin_num_ + (size_t) pl * C * dh, dh, sP, (int) BH);
			gpu_born_out_scatter__ CUDA2((unsigned) ((BH * C * dh + tb - 1) / tb), tb) (gpu_out_ptr_, lin_num_, c0, C, d, H, B);
		}
		// advance the state with this chunk's keys/values: M <- gamma^C M + cKw^T U ; Z <- gamma^C Z + cKw^T 1
		if (decay) {
			gpu_born_scale_bh__ CUDA2((unsigned) ((BH * 2 * d2 * dh + tb - 1) / tb), tb) (lin_M_, (size_t) 2 * d2 * dh, (int) BH, H, C);
			gpu_born_scale_bh__ CUDA2((unsigned) ((BH * d2 + tb - 1) / tb), tb) (lin_Z_, (size_t) d2, (int) BH, H, C);
		}
		for (int pl = 0; pl < 2; ++pl)
			cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_T, dh, d2, C, &one,
				lin_U_ + (size_t) pl * C * dh, dh, (long long) 2 * C * dh,
				cKw, d2, (long long) C * d2, &one,
				lin_M_ + (size_t) pl * d2 * dh, dh, (long long) 2 * d2 * dh, (int) BH);
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, d2, 1, C, &one,
			cKw, d2, (long long) C * d2, lin_ones_, C, (long long) 0, &one,
			lin_Z_, d2, (long long) d2, (int) BH);
		if (c == 0) born_memdbg("lin fwd after first chunk");
	}
	if (carry_) {
		gpuErrchk(cudaMemcpyAsync(carry_M_, lin_M_, sizeof(float) * BH * 2 * d2 * dh, cudaMemcpyDeviceToDevice, st));
		gpuErrchk(cudaMemcpyAsync(carry_Z_, lin_Z_, sizeof(float) * BH * d2, cudaMemcpyDeviceToDevice, st));
	}
	cublasSetStream(hb, cudaStreamPerThread);
	CNET_KSYNC();
	born_memdbg("lin fwd exit");
#endif
}

void BornAttentionGpu::resetCarry() {
#ifdef __CUDACC__
	if (carry_M_) {
		const size_t BH = (size_t) lin_B_ * lin_H_;
		gpuErrchk(cudaMemset(carry_M_, 0, sizeof(float) * BH * 2 * lin_d2_ * lin_dh_));
		gpuErrchk(cudaMemset(carry_Z_, 0, sizeof(float) * BH * lin_d2_));
	}
#endif
}


// ---------------- linear form: backward ----------------
//
// With a_kj = <g_k, u_j> (g_k = packed output gradient, u_j = packed V_j), abar_k =
// sum_j A_kj a_kj and b_kj = (a_kj - abar_k)/D_k (exactly the quadratic backward), the
// Q/K gradients are dz*(Q_k) = Gamma_k Q_k with Gamma_k = sum_j b_kj K_j K_j^H, dz*(K_j) =
// Gamma'_j K_j with Gamma'_j = sum_k b_kj Q_k Q_k^H (dz = conj of dz*), and V gets
// dz(V_j) = sum_k A_kj gO_k, dz*(V_j) = sum_k A_kj gOb_k. In lifted space pack(Gamma_k) =
// sum_j b_kj c(K_j): the within-chunk part is a C x C GEMM, the earlier-chunk part reads
// the FORWARD prefix state: (M_c0 g_k - abar_k Z_c0)/D_k. pack(Gamma'_j) and dz(V_j)
// collect from LATER chunks through a suffix state built from queries, swept in reverse:
// Ms = sum_{k later} (c(Q_k)/D_k) [gO_r|gO_i|gOb_r|gOb_i]_k^T (d2 x 4dh), Zs = sum (abar_k/D_k) c(Q_k).

#define BORN_MAX_DH 128

// per (clone, head, t in chunk, j): packs output gradients and values for the chunk
__global__ void gpu_born_bw_pack__(GpuInVar *in, GpuOutVar *out,
		float *Gm, float *Gt, float *Ut, float *Ug, float *G4, int c0, int C, int N, int d, int H, int B, int decay) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone]; GpuOutVar ov = out[clone];
	const int p = (c0 + t) * d + h * dh + j;
	cmplx_ gO = dZ_(ov, p), gOb = dZ_star_(ov, p), v = Z_(inv, 2 * N * d + p);
	const float gr = gO.real + gOb.real, gi = -gO.imag + gOb.imag;   // <g, u> = Re(gO v + gOb conj v)
	Gm[((bh * 2 + 0) * C + t) * dh + j] = gr;
	Gm[((bh * 2 + 1) * C + t) * dh + j] = gi;
	float *gt = Gt + (bh * C + t) * 2 * dh; gt[j] = gr; gt[dh + j] = gi;
	float *ut = Ut + (bh * C + t) * 2 * dh; ut[j] = v.real; ut[dh + j] = v.imag;
	const float wj = decay ? powf(born_gamma__(h, 1), (float) (C - 1 - t)) : 1.f;   // key weight for later-chunk reads
	float *ug = Ug + (bh * C + t) * 4 * dh; ug[j] = wj * v.real; ug[dh + j] = -wj * v.imag; ug[2 * dh + j] = wj * v.real; ug[3 * dh + j] = wj * v.imag;
	float *g4 = G4 + (bh * C + t) * 4 * dh; g4[j] = gO.real; g4[dh + j] = gO.imag; g4[2 * dh + j] = gOb.real; g4[3 * dh + j] = gOb.imag;
}

// per (clone, head, k in chunk): abar_k, b_kj (C x C), and c(Q_k)/D_k
__global__ void gpu_born_bw_scalars__(const float *A, const float *S, const float *Am, const float *W, const float *cQ,
		float *Bm, float *cQs, float *abar_out, float *abarD_out,
		int c0, int C, int N, int d2, int BH, float eps, int H, int decay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= BH * C) return;
	const int kc = tid % C, bh = tid / C, k = c0 + kc;
	const float gam = born_gamma__(bh % H, decay);   // cQ here is the gamma^(k+1)-weighted lift when decay is on
	const float D = S[(size_t) bh * N + k] + eps;
	const float invD = D > 0.f ? 1.f / D : 0.f;
	const float *Arow = A + ((size_t) bh * N + k) * C, *Amrow = Am + ((size_t) bh * C + kc) * C;
	const float *Wrow = W + ((size_t) bh * C + kc) * d2, *cQrow = cQ + ((size_t) bh * C + kc) * d2;
	float abar = 0.f, cw = 0.f;
	for (int m = 0; m <= kc; ++m) abar += Arow[m] * Amrow[m];
	for (int a = 0; a < d2; ++a) cw += cQrow[a] * Wrow[a];
	abar += cw * invD;
	abar_out[tid] = abar; abarD_out[tid] = abar * invD;
	float *Brow = Bm + ((size_t) bh * C + kc) * C;
	for (int m = 0; m < C; ++m) Brow[m] = m <= kc ? (Amrow[m] - abar) * invD * (decay ? powf(gam, (float) (kc - m)) : 1.f) : 0.f;
	float *cs = cQs + ((size_t) bh * C + kc) * d2;
	for (int a = 0; a < d2; ++a) cs[a] = cQrow[a] * invD;
}

// y = Gamma x for the packed Hermitian Gamma = unpack(G); accumulates dz* = y, dz = conj(y)
__device__ inline void born_unpack_apply(const float *G, const int *idx, int dh, GpuInVar inv, int base) {
	cmplx_ x[BORN_MAX_DH], y[BORN_MAX_DH];
	for (int m = 0; m < dh; ++m) { x[m] = Z_(inv, base + m); y[m] = cmplx(G[m] * x[m].real, G[m] * x[m].imag); }
	const int T = dh * (dh - 1) / 2;
	const float r2 = 0.70710678118f;
	for (int q = 0; q < T; ++q) {
		const int m = idx[2 * (dh + q)], n = idx[2 * (dh + q) + 1];
		cmplx_ g = cmplx(G[dh + q] * r2, G[dh + T + q] * r2);        // Gamma_mn
		y[m] = y[m] + g * x[n];
		y[n] = y[n] + conj_(g) * x[m];
	}
	for (int m = 0; m < dh; ++m) {
		atomicAdd(dZ_star_real_(inv, base + m), y[m].real);
		atomicAdd(dZ_star_imag_(inv, base + m), y[m].imag);
		atomicAdd(dZ_real_(inv, base + m), y[m].real);
		atomicAdd(dZ_imag_(inv, base + m), -y[m].imag);
	}
}

// per (clone, head, k in chunk): Q gradient from Gq (intra) + (W - abar Z_c0)/D (inter)
__global__ void gpu_born_bw_q__(GpuInVar *in, const float *Gq, const float *W, const float *Zc0,
		const float *S, const float *abar, const int *idx, int c0, int C, int N, int d, int H, int B, float eps, int decay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * C) return;
	const int kc = tid % C, r = tid / C, h = r % H, clone = r / H, k = c0 + kc, dh = d / H, d2 = dh * dh;
	const size_t bh = (size_t) clone * H + h;
	const float D = S[bh * N + k] + eps, ab = abar[bh * C + kc];
	const float invD = (D > 0.f ? 1.f / D : 0.f) * (decay ? powf(born_gamma__(h, 1), (float) (kc + 1)) : 1.f);   // inter term carries gamma^(k+1)
	float G[4096];   // dh <= 64 -> d2 <= 4096 (checked in the launcher)
	const float *gq = Gq + (bh * C + kc) * d2, *w = W + (bh * C + kc) * d2, *z = Zc0 + bh * d2;
	for (int a = 0; a < d2; ++a) G[a] = gq[a] + (w[a] - ab * z[a]) * invD;
	born_unpack_apply(G, idx, dh, in[clone], k * d + h * dh);
}

// per (clone, head, j in chunk): K gradient from Gk (intra + inter GEMMs) - Zs
__global__ void gpu_born_bw_k__(GpuInVar *in, const float *Gk, const float *Zs, const int *idx,
		int c0, int C, int N, int d, int H, int B, int decay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * C) return;
	const int jc = tid % C, r = tid / C, h = r % H, clone = r / H, j = c0 + jc, dh = d / H, d2 = dh * dh;
	const size_t bh = (size_t) clone * H + h;
	const float wj = decay ? powf(born_gamma__(h, 1), (float) (C - 1 - jc)) : 1.f;   // Zs term carries the key weight
	float G[4096];
	const float *gk = Gk + (bh * C + jc) * d2, *z = Zs + bh * d2;
	for (int a = 0; a < d2; ++a) G[a] = gk[a] - wj * z[a];
	born_unpack_apply(G, idx, dh, in[clone], N * d + j * d + h * dh);
}

// per (clone, head, j in chunk, m): V gradient planes [dz_r | dz_i | dz*_r | dz*_i]
__global__ void gpu_born_bw_v__(GpuInVar *in, const float *dV, int c0, int C, int N, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * C * dh) return;
	int m = (int) (tid % dh); size_t r = tid / dh;
	int jc = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone];
	const float *dv = dV + (bh * C + jc) * 4 * dh;
	const int p = 2 * N * d + (c0 + jc) * d + h * dh + m;
	atomicAdd(dZ_real_(inv, p), dv[m]);
	atomicAdd(dZ_imag_(inv, p), dv[dh + m]);
	atomicAdd(dZ_star_real_(inv, p), dv[2 * dh + m]);
	atomicAdd(dZ_star_imag_(inv, p), dv[3 * dh + m]);
}

// Warp-per-token form of gpu_born_bw_q__ (mode 0) and gpu_born_bw_k__ (mode 1): the packed Hermitian
// Gamma is never materialized per thread. Each warp holds x (the token's Q or K) and y = Gamma x in
// shared memory; lanes walk consecutive packed entries a (coalesced reads of the source rows) and
// compute G[a] on the fly:  mode 0: G = Gq + (W - abar Zc0) invD;  mode 1: G = Gk - wj Zs.
// 8 warps (tokens) per block; dh <= 64.
__global__ void gpu_born_bw_qk_warp__(GpuInVar *in, const float *Gx, const float *W, const float *Z,
		const float *S, const float *abar, const int *idx, int c0, int C, int N, int d, int H, int B, float eps, int decay, int mode) {
	__shared__ float xs[8][2][64], ys[8][2][64];
	const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
	const int tok = blockIdx.x * 8 + w;
	const bool live = tok < B * H * C;
	const int kc = live ? tok % C : 0, r = live ? tok / C : 0, h = r % H, clone = r / H;
	const int dh = d / H, d2 = dh * dh, T = dh * (dh - 1) / 2;
	const size_t bh = (size_t) clone * H + h;
	float cst = 0.f, scale = 1.f;   // G[a] = gx[a] + scale * (w[a] - cst * z[a])
	if (live) {
		if (mode == 0) {
			const float D = S[bh * N + c0 + kc] + eps;
			scale = (D > 0.f ? 1.f / D : 0.f) * (decay ? powf(born_gamma__(h, 1), (float) (kc + 1)) : 1.f);
			cst = abar[bh * C + kc];
		} else {
			scale = -(decay ? powf(born_gamma__(h, 1), (float) (C - 1 - kc)) : 1.f);   // - wj * Zs
			cst = 0.f;
		}
	}
	const int base = mode == 0 ? (c0 + kc) * d + h * dh : N * d + (c0 + kc) * d + h * dh;
	GpuInVar inv = in[live ? clone : 0];
	const float *gx = Gx + (bh * C + kc) * d2;
	const float *wr = mode == 0 ? W + (bh * C + kc) * d2 : nullptr;
	const float *zr = Z + bh * d2;
	#define GVAL(a) (gx[a] + scale * ((mode == 0 ? wr[a] : 0.f) - (mode == 0 ? cst * zr[a] : -zr[a])))
	if (live) for (int m = lane; m < dh; m += 32) {
		cmplx_ x = Z_(inv, base + m);
		xs[w][0][m] = x.real; xs[w][1][m] = x.imag;
		const float g = GVAL(m);
		ys[w][0][m] = g * x.real; ys[w][1][m] = g * x.imag;
	}
	__syncwarp();
	const float r2 = 0.70710678118f;
	if (live) for (int q = lane; q < T; q += 32) {
		const int m = idx[2 * (dh + q)], n = idx[2 * (dh + q) + 1];
		const float gr = GVAL(dh + q) * r2, gi = GVAL(dh + T + q) * r2;   // Gamma_mn = gr + i gi
		const float xnr = xs[w][0][n], xni = xs[w][1][n], xmr = xs[w][0][m], xmi = xs[w][1][m];
		atomicAdd(&ys[w][0][m], gr * xnr - gi * xni); atomicAdd(&ys[w][1][m], gr * xni + gi * xnr);   // y_m += g x_n
		atomicAdd(&ys[w][0][n], gr * xmr + gi * xmi); atomicAdd(&ys[w][1][n], gr * xmi - gi * xmr);   // y_n += conj(g) x_m
	}
	#undef GVAL
	__syncwarp();
	if (live) for (int m = lane; m < dh; m += 32) {
		const float yr = ys[w][0][m], yi = ys[w][1][m];
		atomicAdd(dZ_star_real_(inv, base + m), yr);
		atomicAdd(dZ_star_imag_(inv, base + m), yi);
		atomicAdd(dZ_real_(inv, base + m), yr);
		atomicAdd(dZ_imag_(inv, base + m), -yi);
	}
}

void BornAttentionGpu::gpu_born_linear_backward() {
#ifdef __CUDACC__
	BornAttention *f = (BornAttention*) cpu_func_.front();
	const int N = f->nTokens(), d = f->dim(), H = f->heads(), B = getNoMappings(), C = f->chunk();
	const int dh = d / H, d2 = dh * dh, NC = N / C;
	const size_t BH = (size_t) B * H;
	if (dh > 64) { fprintf(stderr, "BornAttentionGpu linear backward: head dim %d > 64 not supported\n", dh); exit(1); }
	born_memdbg("lin bwd entry");
	cudaStream_t st = g_cnet_stream ? g_cnet_stream : cudaStreamPerThread;
	cublasHandle_t hb = tw_blas_handle();
	cublasSetStream(hb, st);
	gpuErrchk(cudaMemsetAsync(bw_Ms_, 0, sizeof(float) * BH * d2 * 4 * dh, st));
	gpuErrchk(cudaMemsetAsync(bw_Zs_, 0, sizeof(float) * BH * d2, st));
	const float one = 1.f, zero = 0.f;
	const int tb = 128;
	const int decay = f->decay() ? 1 : 0;
	const size_t nLift = BH * C * d2, nTok = BH * C, nTokDh = BH * C * dh;
	const float eps = f->eps();
	const float *cQd = decay ? lin_cQd_ : lin_cQ_, *cKw = decay ? lin_cKw_ : lin_cK_;
	for (int c = NC - 1; c >= 0; --c) {
		const int c0 = c * C;
		const float *Mc0 = lin_Mhist_ + (size_t) c * BH * 2 * d2 * dh, *Zc0 = lin_Zhist_ + (size_t) c * BH * d2;
		const float *Ablk = ba_A_ + (size_t) c0 * C;                       // rows of chunk c, per-bh stride N*C
		gpu_born_lift__ CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (gpu_in_ptr_, lin_cQ_, lin_idx_, 0, c0, C, N, d, H, B);
		gpu_born_lift__ CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (gpu_in_ptr_, lin_cK_, lin_idx_, 1, c0, C, N, d, H, B);
		if (decay) {
			gpu_born_rowscale__ CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (lin_cKw_, lin_cK_, 0, C, d2, H, B);
			gpu_born_rowscale__ CUDA2((unsigned) ((nLift + tb - 1) / tb), tb) (lin_cQd_, lin_cQ_, 1, C, d2, H, B);
		}
		gpu_born_bw_pack__ CUDA2((unsigned) ((nTokDh + tb - 1) / tb), tb) (gpu_in_ptr_, gpu_out_ptr_,
			bw_Gm_, bw_Gt_, bw_Ut_, bw_Ug_, bw_G4_, c0, C, N, d, H, B, decay);
		// Am[k][j] = <g_k, u_j>  (row-major [k][j] == column-major [j][k])
		cublasSgemmStridedBatched(hb, CUBLAS_OP_T, CUBLAS_OP_N, C, C, 2 * dh, &one,
			bw_Ut_, 2 * dh, (long long) C * 2 * dh, bw_Gt_, 2 * dh, (long long) C * 2 * dh, &zero,
			bw_Am_, C, (long long) C * C, (int) BH);
		// W[k][a] = sum_pl M_c0,pl[a][j] g_k,pl[j]
		for (int pl = 0; pl < 2; ++pl)
			cublasSgemmStridedBatched(hb, CUBLAS_OP_T, CUBLAS_OP_N, d2, C, dh, &one,
				Mc0 + (size_t) pl * d2 * dh, dh, (long long) 2 * d2 * dh,
				bw_Gm_ + (size_t) pl * C * dh, dh, (long long) 2 * C * dh, pl == 0 ? &zero : &one,
				bw_W_, d2, (long long) C * d2, (int) BH);
		gpu_born_bw_scalars__ CUDA2((unsigned) ((nTok + tb - 1) / tb), tb) (ba_A_, ba_S_, bw_Am_, bw_W_, cQd,
			bw_Bm_, bw_cQs_, bw_abar_, bw_abarD_, c0, C, N, d2, (int) BH, eps, H, decay);
		// Gq[k][a] = sum_j Bm[k][j] cK[j][a]
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, d2, C, C, &one,
			lin_cK_, d2, (long long) C * d2, bw_Bm_, C, (long long) C * C, &zero,
			bw_Gq_, d2, (long long) C * d2, (int) BH);
		gpu_born_bw_qk_warp__ CUDA2((unsigned) ((nTok + 7) / 8), 256) (gpu_in_ptr_, bw_Gq_, bw_W_, Zc0, ba_S_, bw_abar_,
			lin_idx_, c0, C, N, d, H, B, eps, decay, 0);
		// Gk[j][a] = sum_k Bm[k][j] cQ[k][a]  +  sum_p Ug[j][p] Ms[a][p]
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_T, d2, C, C, &one,
			lin_cQ_, d2, (long long) C * d2, bw_Bm_, C, (long long) C * C, &zero,
			bw_Gk_, d2, (long long) C * d2, (int) BH);
		cublasSgemmStridedBatched(hb, CUBLAS_OP_T, CUBLAS_OP_N, d2, C, 4 * dh, &one,
			bw_Ms_, 4 * dh, (long long) d2 * 4 * dh, bw_Ug_, 4 * dh, (long long) C * 4 * dh, &one,
			bw_Gk_, d2, (long long) C * d2, (int) BH);
		gpu_born_bw_qk_warp__ CUDA2((unsigned) ((nTok + 7) / 8), 256) (gpu_in_ptr_, bw_Gk_, nullptr, bw_Zs_, ba_S_, bw_abar_,
			lin_idx_, c0, C, N, d, H, B, eps, decay, 1);
		// dV[j][p] = sum_k A[k][j] G4[k][p]  +  sum_a cK[j][a] Ms[a][p]
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_T, 4 * dh, C, C, &one,
			bw_G4_, 4 * dh, (long long) C * 4 * dh, Ablk, C, (long long) N * C, &zero,
			bw_dV_, 4 * dh, (long long) C * 4 * dh, (int) BH);
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, 4 * dh, C, d2, &one,
			bw_Ms_, 4 * dh, (long long) d2 * 4 * dh, cKw, d2, (long long) C * d2, &one,
			bw_dV_, 4 * dh, (long long) C * 4 * dh, (int) BH);
		gpu_born_bw_v__ CUDA2((unsigned) ((nTokDh + tb - 1) / tb), tb) (gpu_in_ptr_, bw_dV_, c0, C, N, d, H, B);
		// suffix state: Ms <- gamma^C Ms + sum_k cQs[k] G4[k]; Zs <- gamma^C Zs + sum_k cQd[k] abarD[k]
		if (decay) {
			gpu_born_scale_bh__ CUDA2((unsigned) ((BH * d2 * 4 * dh + tb - 1) / tb), tb) (bw_Ms_, (size_t) d2 * 4 * dh, (int) BH, H, C);
			gpu_born_scale_bh__ CUDA2((unsigned) ((BH * d2 + tb - 1) / tb), tb) (bw_Zs_, (size_t) d2, (int) BH, H, C);
		}
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_T, 4 * dh, d2, C, &one,
			bw_G4_, 4 * dh, (long long) C * 4 * dh, bw_cQs_, d2, (long long) C * d2, &one,
			bw_Ms_, 4 * dh, (long long) d2 * 4 * dh, (int) BH);
		cublasSgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, d2, 1, C, &one,
			cQd, d2, (long long) C * d2, bw_abarD_, C, (long long) C, &one,
			bw_Zs_, d2, (long long) d2, (int) BH);
	}
	cublasSetStream(hb, cudaStreamPerThread);
	CNET_KSYNC();
	born_memdbg("lin bwd exit");
#endif
}

// ======== TokenNorm (per-token complex RMS normalization) ===================
void TokenNormGpu::ensure_scratch(int N, int M) {
#ifdef __CUDACC__
	if (tn_r_ && tn_N_ == N && tn_M_ == M) return;
	free_scratch();
	tn_N_ = N; tn_M_ = M;
	gpuErrchk(cudaMalloc((void**) &tn_r_, sizeof(float) * (size_t) M * N));
#endif
}
void TokenNormGpu::free_scratch() {
#ifdef __CUDACC__
	if (tn_r_) { cudaFree(tn_r_); tn_r_ = NULL; }
	tn_N_ = tn_M_ = 0;
#endif
}

// r_t = sqrt(mean_dd |x[t,dd]|^2 + eps); y = x / r_t. One thread per (clone,token).
__global__ void gpu_tn_forward__(GpuInVar *in, GpuOutVar *out,
		float *R, int N, int d, float eps, int B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * N) return;
	int clone = tid / N, t = tid % N;
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	float s = 0.f;
	for (int dd = 0; dd < d; ++dd) {
		cmplx_ x = Z_(inv, t * d + dd);
		s += x.real * x.real + x.imag * x.imag;
	}
	float r = sqrtf(s / (float) d + eps);
	R[(size_t) clone * N + t] = r;
	float invr = 1.f / r;
	for (int dd = 0; dd < d; ++dd) {
		cmplx_ x = Z_(inv, t * d + dd);
		ov.out_ptr_[t * d + dd] = x.real * invr;
		ov.out_ptr_[ov.out_length_ + t * d + dd] = x.imag * invr;
	}
}

// dL/dx[dd]  = gO[dd]/r  - conj(x[dd]) * T/(2 d r^3)
// dL/dx*[dd] = gOb[dd]/r - x[dd]       * T/(2 d r^3),  T = sum(gO x + gOb conj x)
// Written with explicit real/imag to avoid relying on complex*scalar overloads.
__global__ void gpu_tn_backward__(GpuInVar *in, GpuOutVar *out,
		float *R, int N, int d, int B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * N) return;
	int clone = tid / N, t = tid % N;
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	float r = R[(size_t) clone * N + t];
	float invr = 1.f / r;
	float coup = 1.f / (2.f * (float) d * r * r * r);
	float Tre = 0.f, Tim = 0.f;
	for (int dd = 0; dd < d; ++dd) {
		cmplx_ gO = dZ_(ov, t * d + dd), gOb = dZ_star_(ov, t * d + dd);
		cmplx_ x = Z_(inv, t * d + dd);
		// gO*x
		Tre += gO.real * x.real - gO.imag * x.imag;
		Tim += gO.real * x.imag + gO.imag * x.real;
		// gOb*conj(x)
		Tre += gOb.real * x.real + gOb.imag * x.imag;
		Tim += -gOb.real * x.imag + gOb.imag * x.real;
	}
	float Tcr = Tre * coup, Tci = Tim * coup;
	for (int dd = 0; dd < d; ++dd) {
		cmplx_ gO = dZ_(ov, t * d + dd), gOb = dZ_star_(ov, t * d + dd);
		cmplx_ x = Z_(inv, t * d + dd);
		// dz = gO/r - conj(x)*Tc ; conj(x)*Tc = (xr*Tcr + xi*Tci, xr*Tci - xi*Tcr)
		float dzr = gO.real * invr - (x.real * Tcr + x.imag * Tci);
		float dzi = gO.imag * invr - (x.real * Tci - x.imag * Tcr);
		// dz* = gOb/r - x*Tc ; x*Tc = (xr*Tcr - xi*Tci, xr*Tci + xi*Tcr)
		float dsr = gOb.real * invr - (x.real * Tcr - x.imag * Tci);
		float dsi = gOb.imag * invr - (x.real * Tci + x.imag * Tcr);
		atomicAdd(dZ_real_(inv, t * d + dd), dzr);
		atomicAdd(dZ_imag_(inv, t * d + dd), dzi);
		atomicAdd(dZ_star_real_(inv, t * d + dd), dsr);
		atomicAdd(dZ_star_imag_(inv, t * d + dd), dsi);
	}
}

void TokenNormGpu::gpu_tn_forward() {
#ifdef __CUDACC__
	TokenNorm *f = (TokenNorm*) cpu_func_.front();
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	ensure_scratch(N, B);
	int total = B * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_tn_forward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, tn_r_, N, d, f->eps(), B);
	CNET_KSYNC();
#endif
}

void TokenNormGpu::gpu_tn_backward() {
#ifdef __CUDACC__
	TokenNorm *f = (TokenNorm*) cpu_func_.front();
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	int total = B * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_tn_backward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, tn_r_, N, d, B);
	CNET_KSYNC();
#endif
}

// ======== RotaryEmbed (RoPE position rotation for complex tokens) ============
// y[t,dd] = x[t,dd] * exp(i * base^{-dd/d} * t). One thread per (clone, element).
__global__ void gpu_rope_forward__(GpuInVar *in, GpuOutVar *out,
		int N, int d, int dh, float base, int B, int pos) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int M = N * d;
	if (tid >= B * M) return;
	int clone = tid / M, p = tid - clone * M;
	int t = p / d, dd = p - t * d;
	// absolute position pos + t (pos = window offset in streaming); double like the CPU layer, since
	// at positions in the millions a float product loses the phase
	double ang = pow((double) base, -(double) (dd % dh) / (double) dh) * (double) (pos + t);   // ladder restarts per head
	float cr = (float) cos(ang), ci = (float) sin(ang);
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	cmplx_ x = Z_(inv, p);
	ov.out_ptr_[p]                   = x.real * cr - x.imag * ci;   // Re(x*r)
	ov.out_ptr_[ov.out_length_ + p]  = x.real * ci + x.imag * cr;   // Im(x*r)
}

__global__ void gpu_rope_backward__(GpuInVar *in, GpuOutVar *out,
		int N, int d, int dh, float base, int B, int pos) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int M = N * d;
	if (tid >= B * M) return;
	int clone = tid / M, p = tid - clone * M;
	int t = p / d, dd = p - t * d;
	// absolute position pos + t (pos = window offset in streaming); double like the CPU layer, since
	// at positions in the millions a float product loses the phase
	double ang = pow((double) base, -(double) (dd % dh) / (double) dh) * (double) (pos + t);   // ladder restarts per head
	float cr = (float) cos(ang), ci = (float) sin(ang);
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	cmplx_ gO = dZ_(ov, p), gOb = dZ_star_(ov, p);
	// dz = gO * r  (r = cr + i ci)
	float dzr = gO.real * cr - gO.imag * ci;
	float dzi = gO.real * ci + gO.imag * cr;
	// dz* = gOb * conj(r)  (conj r = cr - i ci)
	float dsr =  gOb.real * cr + gOb.imag * ci;
	float dsi = -gOb.real * ci + gOb.imag * cr;
	atomicAdd(dZ_real_(inv, p), dzr);
	atomicAdd(dZ_imag_(inv, p), dzi);
	atomicAdd(dZ_star_real_(inv, p), dsr);
	atomicAdd(dZ_star_imag_(inv, p), dsi);
}

void RotaryEmbedGpu::gpu_rope_forward() {
#ifdef __CUDACC__
	RotaryEmbed *f = (RotaryEmbed*) cpu_func_.front();
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	int total = B * N * d, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_rope_forward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, N, d, d / f->heads(), (float) f->base(), B, f->pos());
	CNET_KSYNC();
#endif
}

void RotaryEmbedGpu::gpu_rope_backward() {
#ifdef __CUDACC__
	RotaryEmbed *f = (RotaryEmbed*) cpu_func_.front();
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	int total = B * N * d, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_rope_backward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, N, d, d / f->heads(), (float) f->base(), B, f->pos());
	CNET_KSYNC();
#endif
}

// Inverse DFT forward: identical to FourierGpu::gpu_fft_forward but with
// CUFFT_INVERSE. Reuses the same gather/scatter kernels, plan and 1/sqrt(N)
// scaling (the unitary inverse). cuFFT is required for the inverse layer.
void InverseFourierGpu::gpu_ifft_forward(int block_size) {
	if (fft_fast) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();
		ensure_fft_plan(N, M);

		int total = N * M;
		int tb = 256;
		unsigned grid = (total + tb - 1) / tb;
		float scale = 1.0f / sqrtf((float) N);

		gpu_fft_gather__ CUDA2(grid, tb) (gpu_in_ptr_, fft_buf_, N, M);
		cufftSetStream((cufftHandle) fft_plan_, g_cnet_stream);
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_INVERSE);
		gpu_fft_scatter_out__ CUDA2(grid, tb) (fft_buf_, gpu_out_ptr_, N, M, scale);

		CNET_KSYNC();
		return;
#endif
	}
	// The slow (unity-root) path implements only the forward DFT; the inverse
	// layer therefore requires the cuFFT fast path.
	assert(fft_fast && "InverseFourierGpu requires -fft_fast true");
}

// ---- Bluestein fast path for TriangFourier ----
// Causal linear convolution a*h via a zero-padded batched FFT of size L. All
// chirp/kernel phases are precomputed on the host (k^2 reduced mod 2N) so float
// never has to hold k^2 directly. The lower-triangular mask is free: it is
// exactly the one-sided range of a linear convolution.

// x_q chirp_q, zero-padded to L, interleaved per clone (forward input gather).
__global__ void gpu_bl_gather_chirp__(GpuInVar *in, cmplx_ *chirp, cmplx_ *buf,
		int N, int L, int no_mappings) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * L) return;
	int map = tid / L;
	int k = tid % L;
	buf[tid] = (k < N) ? (Z_(in[map], k) * chirp[k]) : cmplx(0.f, 0.f);
}

// pointwise multiply by a length-L frequency-domain kernel (broadcast over clones).
__global__ void gpu_bl_kmul__(cmplx_ *buf, cmplx_ *K, int L, int no_mappings) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * L) return;
	int k = tid % L;
	buf[tid] = buf[tid] * K[k];
}

// T_p = chirp_p * conv[p] * scale  -> planar output (overwrite).
__global__ void gpu_bl_scatter_out__(cmplx_ *buf, cmplx_ *chirp, GpuOutVar *out,
		int N, int L, int no_mappings, float scale) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) return;
	int map = tid / N;
	int p = tid % N;
	cmplx_ v = (chirp[p] * buf[map * L + p]) * scale;
	out[map].out_ptr_[p] = v.real;
	out[map].out_ptr_[out[map].out_length_ + p] = v.imag;
}

void TrianFourierGpu::ensure_bluestein(int N, int M) {
#ifdef __CUDACC__
	if (bl_plan_ != -1 && bl_N_ == N && bl_M_ == M) return;
	free_bluestein();

	int L = 1;
	while (L < 2 * N) L *= 2;                  // L >= 2N-1, power of two
	bl_N_ = N; bl_M_ = M; bl_L_ = L;

	cufftHandle plan;
	cufftResult r = cufftPlan1d(&plan, L, CUFFT_C2C, M);
	assert(r == CUFFT_SUCCESS);
	bl_plan_ = (int) plan;

	gpuErrchk(cudaMalloc((void**) &bl_buf_,   sizeof(cmplx_) * (size_t) L * M));
	gpuErrchk(cudaMalloc((void**) &bl_chirp_, sizeof(cmplx_) * N));
	gpuErrchk(cudaMalloc((void**) &bl_Hf_,    sizeof(cmplx_) * L));
	gpuErrchk(cudaMalloc((void**) &bl_KrA_,   sizeof(cmplx_) * L));
	gpuErrchk(cudaMalloc((void**) &bl_KrB_,   sizeof(cmplx_) * L));

	// Host-built chirp and kernels. chirp_k = e^{i pi (k^2 mod 2N) / N}.
	const double PI = acos(-1.0);
	const long twoN = 2L * (long) N;
	std::vector<cmplx_> chirp(N);
	std::vector<cmplx_> hpad(L, cmplx(0.f, 0.f));   // conj(chirp), normal order  (forward conv)
	std::vector<cmplx_> krA(L,  cmplx(0.f, 0.f));   // reverse(conj(chirp))        (backward dz)
	std::vector<cmplx_> krB(L,  cmplx(0.f, 0.f));   // reverse(chirp)              (backward dz*)
	// chirp_k = w^{k^2/2} with w = e^{-i 2pi / N} (UnityRoots uses the negative
	// convention), i.e. e^{-i pi k^2 / N}; k^2 is reduced mod 2N first.
	for (int k = 0; k < N; ++k) {
		long kk = ((long) k * (long) k) % twoN;
		double ph = PI * (double) kk / (double) N;
		chirp[k] = cmplx(cos(ph), -sin(ph));
	}
	for (int k = 0; k < N; ++k) hpad[k] = cmplx(chirp[k].real, -chirp[k].imag);
	krA[0] = cmplx(chirp[0].real, -chirp[0].imag);
	krB[0] = chirp[0];
	for (int m = 1; m < N; ++m) {
		krA[L - m] = cmplx(chirp[m].real, -chirp[m].imag);
		krB[L - m] = chirp[m];
	}
	gpuErrchk(cudaMemcpy(bl_chirp_, chirp.data(), sizeof(cmplx_) * N, cudaMemcpyHostToDevice));

	// FFT the three fixed kernels once with a batch-1 size-L plan.
	cufftHandle kplan;
	r = cufftPlan1d(&kplan, L, CUFFT_C2C, 1);
	assert(r == CUFFT_SUCCESS);
	cmplx_ *dsts[3] = { bl_Hf_, bl_KrA_, bl_KrB_ };
	std::vector<cmplx_> *srcs[3] = { &hpad, &krA, &krB };
	for (int i = 0; i < 3; ++i) {
		gpuErrchk(cudaMemcpy(dsts[i], srcs[i]->data(), sizeof(cmplx_) * L, cudaMemcpyHostToDevice));
		cufftSetStream(kplan, g_cnet_stream);
		cufftExecC2C(kplan, (cufftComplex*) dsts[i], (cufftComplex*) dsts[i], CUFFT_FORWARD);
	}
	cufftDestroy(kplan);
	CNET_KSYNC();
#endif
}

void TrianFourierGpu::free_bluestein() {
#ifdef __CUDACC__
	if (bl_plan_ != -1) { cufftDestroy((cufftHandle) bl_plan_); bl_plan_ = -1; }
	if (bl_buf_)   { cudaFree(bl_buf_);   bl_buf_ = NULL; }
	if (bl_chirp_) { cudaFree(bl_chirp_); bl_chirp_ = NULL; }
	if (bl_Hf_)    { cudaFree(bl_Hf_);    bl_Hf_ = NULL; }
	if (bl_KrA_)   { cudaFree(bl_KrA_);   bl_KrA_ = NULL; }
	if (bl_KrB_)   { cudaFree(bl_KrB_);   bl_KrB_ = NULL; }
	bl_N_ = bl_L_ = bl_M_ = 0;
#endif
}

void TrianFourierGpu::gpu_T_fft_forward(int block_size) {
	if (fft_fast) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();
		ensure_bluestein(N, M);
		int L = bl_L_;

		int tb = 256;
		unsigned gridL = ((size_t) L * M + tb - 1) / tb;
		unsigned gridN = ((size_t) N * M + tb - 1) / tb;
		float scale = 1.0f / ((float) L * sqrtf((float) N));   // cuFFT IFFT is unnormalised (x L)

		gpu_bl_gather_chirp__ CUDA2(gridL, tb) (gpu_in_ptr_, bl_chirp_, bl_buf_, N, L, M);
		cufftSetStream((cufftHandle) bl_plan_, g_cnet_stream);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_FORWARD);
		gpu_bl_kmul__ CUDA2(gridL, tb) (bl_buf_, bl_Hf_, L, M);
		cufftSetStream((cufftHandle) bl_plan_, g_cnet_stream);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_INVERSE);
		gpu_bl_scatter_out__ CUDA2(gridN, tb) (bl_buf_, bl_chirp_, gpu_out_ptr_, N, L, M, scale);

		CNET_KSYNC();
		return;
#endif
	}

	int b_out_length = getPaddedLength(block_size, this);
	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	auto gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	reducing_kernel(T_FFT_DATA_PROVIDER, gpu_in_ptr_, gpu_buffer, length(), length(), getNoMappings(), block_size);
	fft_kernel_end(T_FFT_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, length(), getNoMappings(), block_size);
}

void L2Gpu::gpu_norm_forward(int block_size) {
	int b_out_length = getPaddedLength(block_size, this);

	cmplx_* gpu_buffer;
#ifdef __CUDACC__
	// Persistent scratch: allocate once (first/warmup call), reuse thereafter, so
	// the steady-state forward issues no cudaMalloc and is CUDA-graph capturable.
	if (!l2_scratch_ || l2_scratch_len_ < b_out_length) {
		if (l2_scratch_) cudaFree(l2_scratch_);
		gpuErrchk(cudaMalloc((void**) &l2_scratch_, sizeof(cmplx_) * (size_t) b_out_length));
		l2_scratch_len_ = b_out_length;
	}
	gpu_buffer = l2_scratch_;
#else
	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	assert(gpu_buffer);
#endif

	reducing_kernel(NORM_DATA_PROVIDER, gpu_in_ptr_, gpu_buffer, 1, length(), getNoMappings(), block_size);
	norm_kernel_end(NORM_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, length(), getNoMappings(), block_size);
}


__global__ void gpu_norm_backward__ (GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	auto z = Z_(in[map_indx], pos);

	atomicAdd(dZ_real_(in[map_indx],      pos), conj_(z).real);
	atomicAdd(dZ_imag_(in[map_indx],      pos), conj_(z).imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), z.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), z.imag);
}

void L2Gpu::gpu_norm_backward() {
	int total_threads = length() * getNoMappings();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_norm_backward__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads);
	CNET_KSYNC();
}


void SoftMaxGpu::gpu_soft_max_forward(int block_size) {
	int b_out_length = getPaddedLength(block_size, this);

	// TODO: make this global
	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	auto gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}
	//	helper.cmplx_copy_from_gpu(b_out_length, gpu_buffer, &output[0]);
	//	for (int var = 0; var < output.size(); ++var) {
	//		std::cout << std::setfill('0') << std::setw(5) << var << "\t" << output[var] << std::endl;
	//	}

	reducing_kernel(SOFTMAX_DATA_PROVIDER, gpu_in_ptr_, gpu_buffer, 1, length(), getNoMappings(), block_size);
	softmax_kernel_end(gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, length(), getNoMappings(), block_size);
}


__global__ void cross_ent_end__(int no_mappings, cmplx_ *tmp_in, GpuInVar *in, GpuOutVar *out,
		int* labels, int in_stride, int max_len) {

	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	cmplx_ sum = cmplx(0.f, 0.f);
	int offset = thread_indx * in_stride;
	for (int var = 0; var < in_stride; ++var) {
		sum += (tmp_in + offset)[var];
	}

	int map_indx = thread_indx;

	if (thread_indx < no_mappings) {
		out[map_indx].reduce_real_ = sum.real;
		if (labels) {
			int label = labels[map_indx];
			float ret = pow(in[map_indx].input_ptr_[label], 2)
					   + pow(in[map_indx].input_ptr_[in[map_indx].input_length_ + label], 2);
			sum.real = (sum.real < 1e-15 ? 1e-15 : sum.real);
			ret = ret / sum.real;
			ret = (ret < 1e-15 ? 1e-15 : ret);
			out[map_indx].reduce_imag_ = -std::log(ret);

//			printf("ce %d %d %d %f %f\n", thread_indx, map_indx, label, out[map_indx].reduce_imag_, out[map_indx].reduce_real_);
		}
	}
}

void CrossEntropyGpu::gpu_cross_ent_forward(int block_size) {
	int b_out_length = getPaddedLength(block_size, this);

	// TODO: make this global
	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	auto gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	reducing_kernel(SOFTMAX_DATA_PROVIDER, gpu_in_ptr_, gpu_buffer, 1, length(), getNoMappings(), block_size);
	//norm_kernel_end(NORM_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, length(), getNoMappings(), block_size);


	int in_stride = length() % block_size == 0 ? length() : (length() + block_size - length() % block_size);
	in_stride = in_stride / block_size;
	int no_threads = getNoMappings();

	unsigned grid = (no_threads + block_size - 1) / block_size;

//	std::cout << "Launching cross_ent_end__ grid: " << grid << " block: "
//			  << block_size << " no_threads: " << no_threads << std::endl;

	cross_ent_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
		(getNoMappings(), gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, gpu_labels_, in_stride, no_threads);

	CNET_KSYNC();
}

__global__ void new_l_kernel_end__(int no_mappings, cmplx_ *tmp_in, GpuInVar *in, GpuOutVar *out,
		int in_stride, int out_len, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	cmplx_ sum = cmplx(0.f, 0.f);
	int offset = thread_indx * in_stride;
	for (int var = 0; var < in_stride; ++var) {
		sum += (tmp_in + offset)[var];
	}

	int map_indx = thread_indx / out_len;
	int pos = thread_indx % out_len;

//	if (thread_indx < 100) {
//		int map_length = max_len / no_mappings;
//		printf("Map_length=%d \t map_indx=%d \t pos=%05d \t in_stride=%d \t no_mappings=%d \t thread_indx=%05d \t out_len=%d %.3f + %.3f \t %p\n",
//				map_length, map_indx, pos, in_stride, no_mappings, thread_indx, out_len, sum.real, sum.imag, out[0].out_ptr_);
//	}

	if (pos < out_len && map_indx < no_mappings) {
		out[map_indx].out_ptr_[pos] = sum.real;
		out[map_indx].out_ptr_[out[map_indx].out_length_ + pos] = sum.imag;   // per-entry imag-plane offset (defensive)
	}
}

void new_l_kernel_end(cmplx_ *temp_in, GpuInVar *in, GpuOutVar *out, int seg_len, int no_mappings, int out_length, int block_size) {
	int in_stride = seg_len % block_size == 0 ? seg_len : (seg_len + block_size - seg_len % block_size);
	in_stride = in_stride / block_size;
	int no_threads = seg_len * no_mappings;

	unsigned grid = (no_threads + block_size - 1) / block_size;
//	std::cout << "Launching new_l_kernel_end with Grid size: " << grid << " & Block Size:" << block_size
//			  << " in_stride: " << in_stride << " seg_len: " << seg_len
//			  << " no_mappings: " << no_mappings << "\n";

	new_l_kernel_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
			(no_mappings, temp_in, in, out, in_stride, out_length, no_threads);

#ifdef __CUDACC__
	CNET_KSYNC();
#else
#endif

}

// Optimized Linear (matrix-vector) forward.
//
// Each block handles a tile of output rows for a single batch element (map).
// The input vector (N complex numbers) is loaded once into shared memory and
// reused across every row the block computes, so the O(N*M) global reads of the
// vector collapse to O(N) per block. This replaces the generic reduce-then-sum
// path (two kernel launches plus a per-call scratch cudaMalloc) with a single
// direct GEMV kernel.
__global__ void gpu_linear_forward__(GpuInVar *in, GpuOutVar *out,
		int N, int M, int no_mappings, int blocks_per_map) {
#ifdef __CUDACC__
	extern __shared__ cmplx_ svec[];
#else
	cmplx_ svec[1];   // CPU build never executes device kernels
#endif
	// blockIdx.x is uniform across a block, so this early-out is uniform and
	// can never cause a divergent __syncthreads().
	int map_indx = blockIdx.x / blocks_per_map;
	if (map_indx >= no_mappings) {
		return;
	}

	float *in_ptr = in[map_indx].input_ptr_;
	int in_len = in[map_indx].input_length_;

	// Cooperatively cache the input vector in shared memory.
	for (int c = threadIdx.x; c < N; c += blockDim.x) {
		svec[c] = Z(in_ptr, c, in_len);
	}
	__syncthreads();

	int row = (blockIdx.x % blocks_per_map) * blockDim.x + threadIdx.x;
	if (row < M) {
		float *mat_row = in_ptr + (row + 1) * N;
		cmplx_ sum = cmplx(0.f, 0.f);
		for (int c = 0; c < N; ++c) {
			sum += svec[c] * Z(mat_row, c, in_len);
		}
		out[map_indx].out_ptr_[row] = sum.real;
		out[map_indx].out_ptr_[out[map_indx].out_length_ + row] = sum.imag;
	}
}

void LinearGpu::gpu_linear_forward(int block_size) {
	int seg_in_len = ((Linear*)getCpuFun()[0])->firstInputLength();   // E_in (contraction)
	int no_segments = ((Linear*)getCpuFun()[0])->outSize();           // E_out (output rows)

#ifdef __CUDACC__
	// Dense Linear is the position-wise linear with a single token (N = 1).
	if (tw_cublas && getNoMappings() > 0) {
		tw_gemm_forward(1, seg_in_len, no_segments, in_, out_, d_pool_, tw_cap_);
		return;
	}
#endif

	// Fast path: cache the input vector in shared memory and compute each
	// output row directly. Used whenever the vector fits in shared memory.
	// The direct GEMV uses one thread per output row, so it only pays off when
	// there are enough rows to fill the GPU. For few output rows the generic
	// reduction (which parallelises over the contraction dimension) is faster,
	// so fall back to it below the threshold.
	size_t shmem = (size_t) seg_in_len * sizeof(cmplx_);
	if (fast_linear && no_segments >= 512 && shmem <= 48u * 1024u) {
		// Keep blocks small (one row per thread, 128 rows per block) so that a
		// single layer spreads over many SMs instead of one giant block.
		int threads = 128;
		if (threads > no_segments) {
			threads = no_segments;
		}
		if (threads < 1) {
			threads = 1;
		}
		int blocks_per_map = (no_segments + threads - 1) / threads;
		unsigned grid = (unsigned) getNoMappings() * blocks_per_map;

		gpu_linear_forward__ CUDA(grid, threads, shmem)
				(gpu_in_ptr_, gpu_out_ptr_, seg_in_len, no_segments, getNoMappings(), blocks_per_map);

#ifdef __CUDACC__
		CNET_KSYNC();
#endif
		return;
	}

	// Fallback for very wide inputs that do not fit in shared memory: the
	// original reduce-then-sum path.
	int padded_segment_len = seg_in_len % block_size == 0 ? seg_in_len
						   : (seg_in_len + block_size - seg_in_len % block_size);
	int b_out_length = (padded_segment_len / block_size) * getNoMappings() * seg_in_len;

	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	auto gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	reducing_kernel(LINEAR_DATA_PROVIDER, gpu_in_ptr_, gpu_buffer, no_segments, seg_in_len, getNoMappings(),
			        MAX_BLOCK_SIZE);
	new_l_kernel_end(gpu_buffer, gpu_in_ptr_, gpu_out_ptr_, seg_in_len, getNoMappings(), no_segments,
			        MAX_BLOCK_SIZE);
}

// ======== TokenwiseLinear: shared e_in x e_out weight applied per token slice.
// Input layout per map: [ data: n_tokens*e_in ][ weight: e_out*e_in ].
// out(t,r) = sum_c data(t,c) * weight(r,c), index t*e_out + r.

__global__ void gpu_tokenwise_forward__(GpuInVar *in, GpuOutVar *out,
		int n_tokens, int e_in, int e_out, int no_mappings) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	size_t per_map = (size_t) n_tokens * e_out;
	if (tid >= (size_t) no_mappings * per_map) {
		return;
	}
	int map_indx = (int) (tid / per_map);
	int o = (int) (tid % per_map);     // output index within this map
	int t = o / e_out;
	int r = o % e_out;
	int w_base = n_tokens * e_in;

	cmplx_ sum = cmplx(0.f, 0.f);
	for (int c = 0; c < e_in; ++c) {
		sum += Z_(in[map_indx], t * e_in + c) * Z_(in[map_indx], w_base + r * e_in + c);
	}
	out[map_indx].out_ptr_[o] = sum.real;
	out[map_indx].out_ptr_[out[map_indx].out_length_ + o] = sum.imag;
}

#ifdef __CUDACC__
// Shared cuBLAS handle bound to the per-thread stream (one per host thread; the
// multi-process data-parallel runs are single-threaded per GPU).
cublasHandle_t tw_blas_handle() {
	static thread_local cublasHandle_t h = 0;
	if (!h) {
		cublasCreate(&h);
		cublasSetStream(h, cudaStreamPerThread);
		cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
		// Force true IEEE FP32 (no TF32): on Ampere+ cuBLAS defaults to TF32 for
		// SGEMM, whose ~10-bit mantissa is too coarse near the Born-loss unigram
		// saddle and leaves training stuck at chance. The element-wise fallback uses
		// full FP32; match it so the cuBLAS speedup does not cost correctness.
		cublasSetMathMode(h, g_cnet_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH);
	}
	return h;
}
#endif

void tw_free_pool(float **&pool, int &cap) {
#ifdef __CUDACC__
	if (pool) cudaFree(pool);
	pool = 0; cap = 0;
#endif
}

#ifdef __CUDACC__
// Complex Out[N,E_out] = Data[N,E_in] * W[E_out,E_in]^T via 4 real GEMMs on the
// planar real/imag segments (the row-major planes already ARE the column-major
// matrices cuBLAS expects): Out_cm[E_out,N] = W_cm[E_in,E_out]^T * D_cm[E_in,N].
// slots: 0 Dr, 1 Di, 2 Wr, 3 Wi, 4 Or, 5 Oi. Linear calls this with N = 1.
void tw_gemm_forward(int N, int e_in, int e_out,
		const std::vector<GpuInVar> &in_, const std::vector<GpuOutVar> &out_, float **&pool, int &cap) {
	const int B = (int) in_.size();
	// Clone pointers/lengths are static after allocateOnGpu(), so build+upload the
	// batched-pointer pool ONCE (on first call / batch change). Steady-state issues no
	// host->device copy, so the GEMMs are CUDA-graph-capturable.
	if (cap != B) {
		tw_free_pool(pool, cap);
		cap = B;
		gpuErrchk(cudaMalloc((void**) &pool, (size_t) 6 * B * sizeof(float*)));
		const int wbase = N * e_in;
		std::vector<float*> hp((size_t) 6 * B, nullptr);
		for (int b = 0; b < B; ++b) {
			const int L = in_[b].input_length_;         // per-entry: Q/K/V share input len,
			const int outlen = out_[b].out_length_;     // but out_length_ differs (V -> 3*N*d concat vs Q/K -> N*d under rope)
			float *base = in_[b].input_ptr_, *ob = out_[b].out_ptr_;
			hp[0*B+b]=base;         hp[1*B+b]=base+L;
			hp[2*B+b]=base+wbase;   hp[3*B+b]=base+L+wbase;
			hp[4*B+b]=ob;           hp[5*B+b]=ob+outlen;
		}
		gpuErrchk(cudaMemcpy(pool, hp.data(), (size_t) 6 * B * sizeof(float*), cudaMemcpyHostToDevice));
	}
	float **Dr=pool,**Di=pool+B,**Wr=pool+2*B,**Wi=pool+3*B,**Or=pool+4*B,**Oi=pool+5*B;
	cublasHandle_t h = tw_blas_handle();
	const float one=1.f, neg=-1.f, zero=0.f;
	cublasSgemmBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, e_out, N, e_in, &one, (const float* const*)Wr, e_in, (const float* const*)Dr, e_in, &zero, Or, e_out, B);
	cublasSgemmBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, e_out, N, e_in, &neg, (const float* const*)Wi, e_in, (const float* const*)Di, e_in, &one,  Or, e_out, B);
	cublasSgemmBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, e_out, N, e_in, &one, (const float* const*)Wi, e_in, (const float* const*)Dr, e_in, &zero, Oi, e_out, B);
	cublasSgemmBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, e_out, N, e_in, &one, (const float* const*)Wr, e_in, (const float* const*)Di, e_in, &one,  Oi, e_out, B);
	CNET_KSYNC();
}
#endif

void TokenwiseLinearGpu::gpu_tokenwise_forward() {
	TokenwiseLinear *cpu = (TokenwiseLinear*) getCpuFun()[0];
	int n_tokens = cpu->nTokens(), e_in = cpu->inDim(), e_out = cpu->outDim();
	int B = getNoMappings();
#ifdef __CUDACC__
	if (tw_cublas && B > 0) {
		tw_gemm_forward(n_tokens, e_in, e_out, in_, out_, d_pool_, tw_cap_);
		return;
	}
#endif
	size_t total = (size_t) B * n_tokens * e_out;
	int tb = 256;
	unsigned grid = (unsigned) ((total + tb - 1) / tb);
	gpu_tokenwise_forward__ CUDA2(grid, tb)
			(gpu_in_ptr_, gpu_out_ptr_, n_tokens, e_in, e_out, B);
#ifdef __CUDACC__
	CNET_KSYNC();
#endif
}

// ======== Gradients


__global__ void gpu_cross_ent_backward__(GpuInVar *in, GpuOutVar *out, int map_length, int label,
		int *labels, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / map_length;
	int pos = thread_indx - map_indx * map_length;
	float square_norm_ = out[map_indx].reduce_real_;

	float square_mod = (Z_(in[map_indx], pos) * conj_(Z_(in[map_indx], pos))).real;
	square_mod = (square_mod < 1e-15 ? 1e-15 : square_mod);

	float grad_real = 0;
	float grad_imag = 0;

	label = labels ? labels[map_indx] : label;

	if (label == pos) {
		grad_real = - *Z_real_(in[map_indx], pos) * (square_norm_ - square_mod) / (square_mod * square_norm_);
		grad_imag = - *Z_imag_(in[map_indx], pos) * (square_norm_ - square_mod) / (square_mod * square_norm_);
	} else {
		grad_real = *Z_real_(in[map_indx], pos) / square_norm_;
		grad_imag = *Z_imag_(in[map_indx], pos) / square_norm_;
	}

	atomicAdd(dZ_star_real_(in[map_indx], pos), grad_real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), grad_imag);
	atomicAdd(dZ_real_(in[map_indx], pos), grad_real);
	atomicAdd(dZ_imag_(in[map_indx], pos), -grad_imag);
}


void CrossEntropyGpu::gpu_cross_ent_backward(int label) {
	int no_threads = getNoMappings() * length();

	unsigned grid = (no_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
//	std::cout << "Launching gpu_cross_ent_backward with Grid size: " << grid << " & Block Size:" << MAX_BLOCK_SIZE
//			  << " segment len: " << length() << " no_mappings: " << getNoMappings() << "\n";

	gpu_cross_ent_backward__ CUDA2( grid, MAX_BLOCK_SIZE )
			(gpu_in_ptr_, gpu_out_ptr_, length(), label, gpu_labels_, no_threads);

	CNET_KSYNC();
}

// ======== SequenceCrossEntropy (autoregressive per-position loss) ============
// Batched over B clones. Targets are laid out per clone: targets[map*n_pos + p].
// one thread per (clone, position): ||z_p||^2 and -log p_{target}. total = B*n_pos.
__global__ void seq_ce_forward__(GpuInVar *in, int vocab, int n_pos,
		int *targets, float *sqnorm, float *poss_loss, float smooth, int total) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= total) {
		return;
	}
	int map_indx = t / n_pos;
	int p = t - map_indx * n_pos;
	int base = p * vocab;
	float sum = 0.f;
	for (int k = 0; k < vocab; ++k) {
		cmplx_ z = Z_(in[map_indx], base + k);
		sum += z.real * z.real + z.imag * z.imag;
	}
	if (sum < 1e-15f) sum = 1e-15f;
	sqnorm[t] = sum;   // raw Born norm; backward recomputes the smoothing from it
	cmplx_ zt = Z_(in[map_indx], base + targets[t]);
	// Laplace-smoothed Born, RELATIVE (scale-invariant like Born itself):
	//   s = smooth * sum / V,   p_t = (|z_t|^2 + s) / (sum + V*s) = (|z_t|^2+s)/(sum(1+smooth)).
	// Uniform input -> exactly 1/V; |z_t|^2->0 -> floor smooth/(V(1+smooth)) > 0 (no cliff).
	// smooth==0 -> exact Born.
	float s = smooth * sum / vocab;
	float num = zt.real * zt.real + zt.imag * zt.imag + s;
	float den = sum * (1.0f + smooth);
	float prob = num / den;
	if (prob < 1e-15f) prob = 1e-15f;
	poss_loss[t] = -logf(prob);
}

// one thread per (clone, position, vocab index): the per-position Born gradient,
// scaled by 1/n_pos to match the per-clone mean loss (the clone sum + l_rate/B
// then yields the batch mean). max_len = B*n_pos*vocab.
__global__ void seq_ce_backward__(GpuInVar *in, int vocab, int n_pos,
		int *targets, float *sqnorm, float inv_n, float eps, float smooth, int max_len) {
	int gi = blockIdx.x * blockDim.x + threadIdx.x;
	if (gi >= max_len) {
		return;
	}
	int per_clone = n_pos * vocab;
	int map_indx = gi / per_clone;
	int within = gi - map_indx * per_clone;   // index inside this clone's logits
	int p = within / vocab;
	int k = within - p * vocab;
	int tpos = map_indx * n_pos + p;
	float sqn = sqnorm[tpos];
	float zr = *Z_real_(in[map_indx], within);
	float zi = *Z_imag_(in[map_indx], within);
	float gr, gi_;
	if (smooth > 0.0f) {
		// Gradient of the relative Laplace-smoothed Born loss (matches forward):
		//   L = -log(|z_t|^2 + a*sum/V) + log(sum) + const,  a = smooth, N = |z_t|^2 + a*sum/V.
		//   dL/d|z_k|^2 = 1/sum - (a/V)/N - [k==target]/N.
		int tk = targets[tpos];
		float a = smooth;
		float s = a * sqn / vocab;
		int base_p = (within - k);                       // = p*vocab (target lives here + tk)
		float ztr = *Z_real_(in[map_indx], base_p + tk);
		float zti = *Z_imag_(in[map_indx], base_p + tk);
		float N = ztr * ztr + zti * zti + s;
		if (N < 1e-15f) N = 1e-15f;
		float gf = 1.0f / sqn - (a / vocab) / N;
		if (k == tk) gf -= 1.0f / N;
		gr  = gf * zr * inv_n;
		gi_ = gf * zi * inv_n;
	} else if (k == targets[tpos]) {
		float sqmod = zr * zr + zi * zi;
		if (sqmod < 1e-15f) sqmod = 1e-15f;
		// target probability floor (matches CPU SequenceCrossEntropy): bound the
		// 1/|z_t| blow-up of the exact gradient. eps == 0 -> exact Born gradient.
		if (eps > 0.0f) {
			float floor = eps * sqn;
			if (sqmod < floor) sqmod = floor;
		}
		float f = (sqn - sqmod) / (sqmod * sqn);
		gr  = -zr * f * inv_n;
		gi_ = -zi * f * inv_n;
	} else {
		gr  = zr / sqn * inv_n;
		gi_ = zi / sqn * inv_n;
	}
	atomicAdd(dZ_star_real_(in[map_indx], within), gr);
	atomicAdd(dZ_star_imag_(in[map_indx], within), gi_);
	atomicAdd(dZ_real_(in[map_indx], within), gr);
	atomicAdd(dZ_imag_(in[map_indx], within), -gi_);
}

SequenceCrossEntropyGpu::~SequenceCrossEntropyGpu() {
#ifdef __CUDACC__
	if (gpu_targets_)   cudaFree(gpu_targets_);
	if (gpu_sqnorm_)    cudaFree(gpu_sqnorm_);
	if (gpu_poss_loss_) cudaFree(gpu_poss_loss_);
#endif
}

void SequenceCrossEntropyGpu::gpu_seq_ce_forward() {
	SequenceCrossEntropy *cpu = (SequenceCrossEntropy*) getCpuFun()[0];
	vocab_ = cpu->vocab();
	n_pos_ = cpu->nPos();
	int B = getNoMappings();

	// Raw cudaMalloc (freed in the destructor): a scoped GpuHelper would free
	// these the moment it went out of scope, leaving dangling members. The
	// sqnorm / loss buffers are B*n_pos; gpu_targets_ (self-owned) is only used
	// in the batch=1 path -- batched runs read the CNet-owned gpu_batch_targets_.
	if (!gpu_sqnorm_) {
		gpuErrchk(cudaMalloc((void**) &gpu_sqnorm_, B * n_pos_ * sizeof(float)));
		gpuErrchk(cudaMalloc((void**) &gpu_poss_loss_, B * n_pos_ * sizeof(float)));
		if (!gpu_batch_targets_) {
			gpuErrchk(cudaMalloc((void**) &gpu_targets_, n_pos_ * sizeof(int)));
		}
	}
	int *targets;
	if (gpu_batch_targets_) {
		targets = gpu_batch_targets_;   // already uploaded by CNet::batchToGpu
	} else {
		gpuErrchk(cudaMemcpy(gpu_targets_, cpu->targetsData(), n_pos_ * sizeof(int),
				cudaMemcpyHostToDevice));
		targets = gpu_targets_;
	}

	int total = B * n_pos_;
	int tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	seq_ce_forward__ CUDA2(grid, tb)
			(gpu_in_ptr_, vocab_, n_pos_, targets, gpu_sqnorm_, gpu_poss_loss_, cpu->smooth(), total);
	CNET_KSYNC();
}

void SequenceCrossEntropyGpu::gpu_seq_ce_backward() {
	int B = getNoMappings();
	int total = B * n_pos_ * vocab_;
	float inv_n = 1.0f / (float) n_pos_;
	SequenceCrossEntropy *cf = (SequenceCrossEntropy*) getCpuFun()[0];
	float eps = cf->eps(), smooth = cf->smooth();
	int *targets = gpu_batch_targets_ ? gpu_batch_targets_ : gpu_targets_;
	int tb = 256;
	unsigned grid = (total + tb - 1) / tb;
	seq_ce_backward__ CUDA2(grid, tb)
			(gpu_in_ptr_, vocab_, n_pos_, targets, gpu_sqnorm_, inv_n, eps, smooth, total);
	CNET_KSYNC();
}

void SequenceCrossEntropyGpu::readLosses(std::vector<float> &out) {
	int n = getNoMappings() * n_pos_;
	out.resize(n);
	gpuErrchk(cudaMemcpy(&out[0], gpu_poss_loss_, n * sizeof(float), cudaMemcpyDeviceToHost));
}

float SequenceCrossEntropyGpu::readLoss() {
	int n = getNoMappings() * n_pos_;
	std::vector<float> loss(n);
	gpuErrchk(cudaMemcpy(&loss[0], gpu_poss_loss_, n * sizeof(float),
			cudaMemcpyDeviceToHost));
	double s = 0.0;
	for (float l : loss) s += l;
	return (float) (s / n);
}

// One thread per batch element: argmax_k |z_k|^2 (the prediction) compared to
// the label, writing 1/0 into out_correct[map]. |z_k|^2/||z||^2 has the same
// argmax as |z_k|^2, so no normalisation is needed. Cheap: no_mappings threads,
// N work each, reusing the activations already on the device.
__global__ void gpu_argmax_correct__(GpuInVar *in, int *labels, int *out_correct,
		int N, int no_mappings) {
	int map_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (map_indx >= no_mappings) {
		return;
	}
	int best = 0;
	float best_val = -1.f;
	for (int k = 0; k < N; ++k) {
		float re = *Z_real_(in[map_indx], k);
		float im = *Z_imag_(in[map_indx], k);
		float m = re * re + im * im;
		if (m > best_val) {
			best_val = m;
			best = k;
		}
	}
	out_correct[map_indx] = (best == labels[map_indx]) ? 1 : 0;
}

void gpu_argmax_correct(GpuInVar *in, int *labels, int *out_correct, int N, int no_mappings) {
	int block = 256;
	unsigned grid = (no_mappings + block - 1) / block;
	gpu_argmax_correct__ CUDA2(grid, block) (in, labels, out_correct, N, no_mappings);
	CNET_KSYNC();
}

// Like gpu_argmax_correct but writes the predicted class (Born argmax of |z_k|^2)
// per mapping, for building a confusion matrix on the host.
__global__ void gpu_argmax_predict__(GpuInVar *in, int *out_pred, int N, int no_mappings) {
	int map_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (map_indx >= no_mappings) {
		return;
	}
	int best = 0;
	float best_val = -1.f;
	for (int k = 0; k < N; ++k) {
		float re = *Z_real_(in[map_indx], k);
		float im = *Z_imag_(in[map_indx], k);
		float m = re * re + im * im;
		if (m > best_val) {
			best_val = m;
			best = k;
		}
	}
	out_pred[map_indx] = best;
}

void gpu_argmax_predict(GpuInVar *in, int *out_pred, int N, int no_mappings) {
	int block = 256;
	unsigned grid = (no_mappings + block - 1) / block;
	gpu_argmax_predict__ CUDA2(grid, block) (in, out_pred, N, no_mappings);
	CNET_KSYNC();
}

