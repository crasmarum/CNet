#include "reduce.h"
#include "allocator.h"

#include "../utils/stopwatch.h"
#include "../utils/flags.h"

#ifdef __CUDACC__
#include <cufft.h>
#endif

FLAG_INT(reduce_block_size, 512);

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

	if (pos < in->input_length_) {
		out[map_indx].out_ptr_[pos] = in[map_indx].input_ptr_[pos] / sum.real;
		out[map_indx].out_ptr_[out_len + pos] = in[map_indx].input_ptr_[in->input_length_ + pos] / sum.real;
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

	size_t no_threads = seg_len * no_segments * no_mappings;
	unsigned grid = (no_threads + block_size - 1) / block_size;

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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_FORWARD);
		gpu_fft_scatter_out__ CUDA2(grid, tb) (fft_buf_, gpu_out_ptr_, N, M, scale);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
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

void BornAttentionGpu::ensure_scratch(int N, int d, int M) {
#ifdef __CUDACC__
	if (ba_c_ && ba_N_ == N && ba_d_ == d && ba_M_ == M) return;
	free_scratch();
	ba_N_ = N; ba_d_ = d; ba_M_ = M;
	gpuErrchk(cudaMalloc((void**) &ba_c_, sizeof(cmplx_) * (size_t) M * N * N));
	gpuErrchk(cudaMalloc((void**) &ba_A_, sizeof(float)  * (size_t) M * N * N));
	gpuErrchk(cudaMalloc((void**) &ba_S_, sizeof(float)  * (size_t) M * N));
#endif
}
void BornAttentionGpu::free_scratch() {
#ifdef __CUDACC__
	if (ba_c_) { cudaFree(ba_c_); ba_c_ = NULL; }
	if (ba_A_) { cudaFree(ba_A_); ba_A_ = NULL; }
	if (ba_S_) { cudaFree(ba_S_); ba_S_ = NULL; }
	ba_N_ = ba_d_ = ba_M_ = 0;
#endif
}

__global__ void gpu_born_forward__(GpuInVar *in, GpuOutVar *out,
		cmplx_ *C, float *A, float *S, int N, int d, int B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * N) return;
	int clone = tid / N, k = tid % N, Mv = N * d;
	GpuInVar inv = in[clone];
	size_t cb = (size_t) clone * N * N + (size_t) k * N;
	float Sk = 0.f;
	for (int m = 0; m <= k; ++m) {
		cmplx_ c = cmplx(0.f, 0.f);
		for (int dd = 0; dd < d; ++dd)
			c = c + Z_(inv, k * d + dd) * conj_(Z_(inv, Mv + m * d + dd));
		C[cb + m] = c;
		float s = c.real * c.real + c.imag * c.imag;
		A[cb + m] = s;
		Sk += s;
	}
	S[(size_t) clone * N + k] = Sk;
	float invS = Sk > 0.f ? 1.f / Sk : 0.f;
	for (int m = 0; m <= k; ++m) A[cb + m] *= invS;
	GpuOutVar ov = out[clone];
	for (int dd = 0; dd < d; ++dd) {
		float orr = 0.f, oii = 0.f;
		for (int m = 0; m <= k; ++m) {
			float a = A[cb + m];
			cmplx_ v = Z_(inv, 2 * Mv + m * d + dd);
			orr += a * v.real; oii += a * v.imag;
		}
		ov.out_ptr_[k * d + dd] = orr;
		ov.out_ptr_[ov.out_length_ + k * d + dd] = oii;
	}
}

__global__ void gpu_born_backward__(GpuInVar *in, GpuOutVar *out,
		cmplx_ *C, float *A, float *S, int N, int d, int B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * N) return;
	int clone = tid / N, k = tid % N, Mv = N * d;
	GpuInVar inv = in[clone];
	GpuOutVar ov = out[clone];
	size_t cb = (size_t) clone * N * N + (size_t) k * N;
	float invS = S[(size_t) clone * N + k] > 0.f ? 1.f / S[(size_t) clone * N + k] : 0.f;
	float abar = 0.f;
	for (int m = 0; m <= k; ++m) {
		float am = 0.f;
		for (int dd = 0; dd < d; ++dd) {
			cmplx_ gO = dZ_(ov, k * d + dd), gOb = dZ_star_(ov, k * d + dd);
			cmplx_ v = Z_(inv, 2 * Mv + m * d + dd);
			am += (gO * v).real + (gOb * conj_(v)).real;
		}
		abar += A[cb + m] * am;
	}
	for (int m = 0; m <= k; ++m) {
		float am = 0.f;
		for (int dd = 0; dd < d; ++dd) {
			cmplx_ gO = dZ_(ov, k * d + dd), gOb = dZ_star_(ov, k * d + dd);
			cmplx_ v = Z_(inv, 2 * Mv + m * d + dd);
			am += (gO * v).real + (gOb * conj_(v)).real;
		}
		float b = (am - abar) * invS;
		cmplx_ c = C[cb + m], cc = conj_(c);
		float Akm = A[cb + m];
		for (int dd = 0; dd < d; ++dd) {
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
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	ensure_scratch(N, d, B);
	int total = B * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_born_forward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, ba_c_, ba_A_, ba_S_, N, d, B);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void BornAttentionGpu::gpu_born_backward() {
#ifdef __CUDACC__
	BornAttention *f = (BornAttention*) cpu_func_.front();
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	int total = B * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_born_backward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, ba_c_, ba_A_, ba_S_, N, d, B);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void TokenNormGpu::gpu_tn_backward() {
#ifdef __CUDACC__
	TokenNorm *f = (TokenNorm*) cpu_func_.front();
	int N = f->nTokens(), d = f->dim(), B = getNoMappings();
	int total = B * N, tb = 128;
	unsigned grid = (total + tb - 1) / tb;
	gpu_tn_backward__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, tn_r_, N, d, B);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_INVERSE);
		gpu_fft_scatter_out__ CUDA2(grid, tb) (fft_buf_, gpu_out_ptr_, N, M, scale);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
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
		cufftExecC2C(kplan, (cufftComplex*) dsts[i], (cufftComplex*) dsts[i], CUFFT_FORWARD);
	}
	cufftDestroy(kplan);
	gpuErrchk(cudaDeviceSynchronize());
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
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_FORWARD);
		gpu_bl_kmul__ CUDA2(gridL, tb) (bl_buf_, bl_Hf_, L, M);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_INVERSE);
		gpu_bl_scatter_out__ CUDA2(gridN, tb) (bl_buf_, bl_chirp_, gpu_out_ptr_, N, L, M, scale);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
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

	// TODO: make this global
	std::vector<cmplx_> output(b_out_length);
	GpuHelper helper;
	auto gpu_buffer = helper.cmplx_allocate_on_gpu(output.size());
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
		out[map_indx].out_ptr_[out->out_length_ + pos] = sum.imag;
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	int seg_in_len = ((Linear*)getCpuFun()[0])->firstInputLength();   // N
	int no_segments = ((Linear*)getCpuFun()[0])->outSize();           // M

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
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
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

void TokenwiseLinearGpu::gpu_tokenwise_forward() {
	TokenwiseLinear *cpu = (TokenwiseLinear*) getCpuFun()[0];
	int n_tokens = cpu->nTokens();
	int e_in = cpu->inDim();
	int e_out = cpu->outDim();

	size_t total = (size_t) getNoMappings() * n_tokens * e_out;
	int tb = 256;
	unsigned grid = (unsigned) ((total + tb - 1) / tb);
	gpu_tokenwise_forward__ CUDA2(grid, tb)
			(gpu_in_ptr_, gpu_out_ptr_, n_tokens, e_in, e_out, getNoMappings());
#ifdef __CUDACC__
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

// ======== SequenceCrossEntropy (autoregressive per-position loss) ============
// Batched over B clones. Targets are laid out per clone: targets[map*n_pos + p].
// one thread per (clone, position): ||z_p||^2 and -log p_{target}. total = B*n_pos.
__global__ void seq_ce_forward__(GpuInVar *in, int vocab, int n_pos,
		int *targets, float *sqnorm, float *poss_loss, int total) {
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
	sqnorm[t] = sum;
	cmplx_ zt = Z_(in[map_indx], base + targets[t]);
	float prob = (zt.real * zt.real + zt.imag * zt.imag) / sum;
	if (prob < 1e-15f) prob = 1e-15f;
	poss_loss[t] = -logf(prob);
}

// one thread per (clone, position, vocab index): the per-position Born gradient,
// scaled by 1/n_pos to match the per-clone mean loss (the clone sum + l_rate/B
// then yields the batch mean). max_len = B*n_pos*vocab.
__global__ void seq_ce_backward__(GpuInVar *in, int vocab, int n_pos,
		int *targets, float *sqnorm, float inv_n, float eps, int max_len) {
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
	if (k == targets[tpos]) {
		float sqmod = zr * zr + zi * zi;
		if (sqmod < 1e-15f) sqmod = 1e-15f;
		// target probability floor (matches CPU SequenceCrossEntropy): bound the
		// 1/|z_t| gradient blow-up. eps == 0 -> exact Born gradient.
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
			(gpu_in_ptr_, vocab_, n_pos_, targets, gpu_sqnorm_, gpu_poss_loss_, total);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void SequenceCrossEntropyGpu::gpu_seq_ce_backward() {
	int B = getNoMappings();
	int total = B * n_pos_ * vocab_;
	float inv_n = 1.0f / (float) n_pos_;
	float eps = ((SequenceCrossEntropy*) getCpuFun()[0])->eps();
	int *targets = gpu_batch_targets_ ? gpu_batch_targets_ : gpu_targets_;
	int tb = 256;
	unsigned grid = (total + tb - 1) / tb;
	seq_ce_backward__ CUDA2(grid, tb)
			(gpu_in_ptr_, vocab_, n_pos_, targets, gpu_sqnorm_, inv_n, eps, total);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

