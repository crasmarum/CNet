
#include "reduce.h"
#include "allocator.h"

#include "../utils/stopwatch.h"
#include "../utils/flags.h"

#ifdef __CUDACC__
#include <cufft.h>
#include <cublas_v2.h>
#endif

extern bool fast_linear;    // defined in reduce.cpp
extern bool fft_fast;       // defined in reduce.cpp
extern bool fast_softmax;   // defined in reduce.cpp
extern bool tw_cublas;      // defined in reduce.cpp

#ifdef __CUDACC__
// This TU's cuBLAS handle (bound to the per-thread stream); the forward TU keeps
// its own -- both are thread_local and cheap.
static cublasHandle_t twg_blas_handle() {
	static thread_local cublasHandle_t h = 0;
	if (!h) {
		cublasCreate(&h);
		cublasSetStream(h, cudaStreamPerThread);
		cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
	}
	return h;
}
#endif
#define TWG_SLOTS 16

__device__ cmplx_ Fdz(int provider_id, GpuInVar in, int in_indx, GpuOutVar out, int out_indx) {
	switch (provider_id) {
		case SOFTMAX_DATA_PROVIDER:
		{
//			printf("ii=%d \t oi=%d \t (rr=%f ri=%f) \t (zr=%f zi=%f) \t \n", in_indx, in_indx,
//					out.reduce_real_, out.reduce_imag_, Z_(in, in_indx).real, Z_(in, in_indx).imag);
			if (in_indx == out_indx) {
				return 0.5f * (cmplx(2.f * out.reduce_real_, 0.f) - Z_(in, in_indx) * conj_(Z_(in, in_indx)))
						/ out.reduce_imag_;
			}
			return -0.5f * Z_(in, out_indx) * conj_(Z_(in, in_indx)) / out.reduce_imag_;
		}
		case LINEAR_DATA_PROVIDER:
		{
			int in1_len = in.input_length_ / (1 + in.output_length_);
			auto o_index = (out_indx + 1) * in1_len + in_indx;

			// we update the other gradients now for efficiency
			auto dLdz      = dZ_(out, out_indx);
			auto dLdz_star = dZ_star_(out, out_indx);

			auto other = Z_(in, in_indx);
			auto dz = dLdz * other;
			auto dz_star = dLdz_star * conj_(other);

			atomicAdd(dZ_real_(in, o_index), dz.real);
			atomicAdd(dZ_imag_(in, o_index), dz.imag);
			atomicAdd(dZ_star_real_(in, o_index), dz_star.real);
			atomicAdd(dZ_star_imag_(in, o_index), dz_star.imag);

//			printf("X oi=%03d \t ii=%03d \t (z=(%f %f)) \t MI=%03d \t (MT=(%f %f) \t iL1=%d \t oL=%d\n",
//					out_indx, in_indx,
//					Z_(in, o_index).real, Z_(in, o_index).imag, o_index, dz.real, dz.imag, in1_len, in.output_length_);

			// and return what is needed for reduction
			return Z_(in, o_index);
		}
		case FFT_DATA_PROVIDER:
		{
			return in.other_[(in_indx * out_indx) % in.input_length_];
		}
		case T_FFT_DATA_PROVIDER: {
			return in_indx > out_indx ? cmplx_(0.f, 0.f) : in.other_[(in_indx * out_indx) % in.input_length_];
		}
		default:
			break;
	}
	return {0, 0};
}

__device__ cmplx_ Fdz_star(int provider_id, GpuInVar in, int in_indx, GpuOutVar out, int out_indx) {
	switch (provider_id) {
		case SOFTMAX_DATA_PROVIDER:
		{
			if (out_indx == in_indx) {
				return -0.5f * Z_(in, in_indx) * Z_(in, in_indx) / out.reduce_imag_;
			}
			return -0.5f * Z_(in, in_indx) * Z_(in, out_indx) / out.reduce_imag_;
		}
		case LINEAR_DATA_PROVIDER:
			return {0, 0};
		case FFT_DATA_PROVIDER:
			return {0, 0};
		default:
			break;
	}
	return {0, 0};
}

template<unsigned int blockSize>
__global__ void grad_reducing_kernel__(int provider_id, GpuInVar *in, int in_seg_len, GpuOutVar *out, Grads *buff,
							           int out_seg_length, int init_out_seg_length, size_t max_no_threads) {
#ifdef __CUDACC__
	extern __shared__ Grads sdata[];
#else
	Grads sdata[1024];
#endif

    unsigned int tid = threadIdx.x;
	size_t thread_indx = blockIdx.x * blockDim.x + threadIdx.x;

	int out_indx = thread_indx % out_seg_length;
    size_t buff_indx = thread_indx / blockSize;

	// we have max_no_threads = out_seg_length * in_seg_len * no_mappings;
	// out_seg_length is the padded initial out segment.
	// Every lane in the block must reach each __syncthreads() below, so
	// out-of-range and padding lanes contribute a zero element rather than
	// returning early: a divergent __syncthreads() is undefined behaviour.
	if (thread_indx >= max_no_threads || out_indx >= init_out_seg_length) {
		sdata[tid] = Grads();
	} else {
		int map_indx = thread_indx / in_seg_len / out_seg_length;
		int in_indx = (thread_indx / out_seg_length) % in_seg_len;

		auto dLdz = dZ_(out[map_indx], out_indx);
		auto dLdz_star = dZ_star_(out[map_indx], out_indx);

		auto dz      =      Fdz(provider_id, in[map_indx], in_indx, out[map_indx], out_indx);
		auto dz_star = Fdz_star(provider_id, in[map_indx], in_indx, out[map_indx], out_indx);

		sdata[tid].grad_ = dLdz * dz + dLdz_star * conj_(dz_star);
		sdata[tid].grad_star_ = dLdz * dz_star + dLdz_star * conj_(dz);
	}

//    if (!blockIdx.x) {
//    	printf("out_l=%03d \t mid=%d \t tid=%03d \t iidx=%03d \t oidx=%03d \t bindx=%03d \t Tid=%03d\t  %f %f \n",
//    			init_out_seg_length, map_indx,   tid,       in_indx,  out_indx, (int)buff_indx,  (int)thread_indx,
//    			sdata[tid].grad_.real, sdata[tid].grad_.imag);
//    }

    __syncthreads();

/*
    printf("DATA \t bi=%03d \t ti=%03d \t oi=%03d \t ii=%03d \t mi=%03d \t dz=(%.3f %.3f) \t dz*=(%.3f %.3f) \t fdz=(%.3f %.3f) \t TI=%d \n",
    		blockIdx.x, tid, out_indx, in_indx, map_indx,
    		(dLdz * dz + dLdz_star * conj_(dz_star)).real, (dLdz * dz + dLdz_star * conj_(dz_star)).imag,
			(dLdz * dz_star + dLdz_star * conj_(dz)).real, (dLdz * dz_star + dLdz_star * conj_(dz)).imag,
			dz.real, dz.imag, (int)thread_indx);
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
		buff[buff_indx] = sdata[0];
//		printf("dz_star \t %d \t %d %f %f \t %f %f \t %d \n", in_indx, out_indx,
//				sdata[0].grad_.real, sdata[0].grad_.imag, sdata[0].grad_star_.real, sdata[0].grad_star_.imag, (int)buff_indx);
	}
}

void grad_reducing_kernel(int provider_id, GpuInVar *in, int input_length, GpuOutVar *out, int out_length, Grads *buff,
		int no_mappings, int block_size) {
	int out_seg_len = out_length % block_size == 0 ? out_length
			: (out_length + block_size - out_length % block_size);

	size_t no_threads = out_seg_len * input_length * no_mappings;
	unsigned grid = (no_threads + block_size - 1) / block_size;

	switch (block_size) {
		case 1024:
			grad_reducing_kernel__ <1024> CUDA ( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 512:
			grad_reducing_kernel__ <512> CUDA( grid , block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 256:
			grad_reducing_kernel__ <256> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 128:
			grad_reducing_kernel__ <128> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 64:
			grad_reducing_kernel__ <64> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 32:
			grad_reducing_kernel__ <32> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 16:
			grad_reducing_kernel__ <16> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
			break;
		case 8:
			grad_reducing_kernel__ <8> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, no_mappings, no_threads);
			break;
		case 4:
			grad_reducing_kernel__ <4> CUDA( grid, block_size, block_size * sizeof(Grads) )
				(provider_id, in, input_length, out, buff, out_seg_len, out_length, no_threads);
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

__global__ void grad_kernel_end__(int provider_id, int no_mappings, Grads *buff, GpuInVar *in, int in_stride, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	Grads sum;
	int offset = thread_indx * in_stride;
	for (int var = 0; var < in_stride; ++var) {
		sum += (buff + offset)[var];
	}

	int map_length = max_len / no_mappings;
	int map_indx = thread_indx / map_length;
	int pos = thread_indx - map_indx * map_length;
	int out_len = in->input_length_;

//	if (thread_indx < 100) {
//		printf("map_length=%d \t map_indx=%d \t pos=%05d \t in_stride=%d \t no_mappings=%d \t thread_indx=%05d \t out_len=%d %f + %f \t %p\n",
//				map_length, map_indx, pos, in_stride, no_mappings, thread_indx, out_len,
//				sum.grad_.real, sum.grad_.imag, in[map_indx].input_ptr_);
//	}

	if (pos < out_len) {
		atomicAdd(dZ_real_(in[map_indx], pos), sum.grad_.real);
		atomicAdd(dZ_imag_(in[map_indx], pos), sum.grad_.imag);
		atomicAdd(dZ_star_real_(in[map_indx], pos), sum.grad_star_.real);
		atomicAdd(dZ_star_imag_(in[map_indx], pos), sum.grad_star_.imag);
	}
}

void grad_kernel_end(int provider_id, Grads *buff, GpuInVar *in, int seg_len, int no_mappings, int block_size) {
	int in_stride = seg_len % block_size == 0 ? seg_len : (seg_len + block_size - seg_len % block_size);
	in_stride = in_stride / block_size;
	int no_threads = seg_len * no_mappings;

	unsigned grid = (no_threads + block_size - 1) / block_size;
//	std::cout << "Launching grad_kernel_end with Grid size: " << grid << " & Block Size:" << block_size
//			  << " in_stride: " << in_stride << " seg_len: " << seg_len
//			  << " no_mappings: " << no_mappings << "\n";

	grad_kernel_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
			(provider_id, no_mappings, buff, in, in_stride, no_threads);

#ifdef __CUDACC__
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#else
#endif

}

// ---- O(N) SoftMax backward (fast_softmax) ----
// A(z)=z/||z||. With P = sum_j dLdz_j z_j and Q = sum_j dLdz*_j conj(z_j),
// R = (P+Q)/||z||^3, the input gradient is
//   dz[i]      += dLdz_i/||z||  - 0.5 conj(z_i) R
//   dz_star[i] += dLdz*_i/||z|| - 0.5 z_i      R
// replacing the O(N^2) dense Jacobian with two O(N) passes.

// (1) One block per map: reduce P and Q over the N elements.
__global__ void gpu_softmax_pq__(GpuInVar *in, GpuOutVar *out, cmplx_ *pq, int N, int no_mappings) {
#ifdef __CUDACC__
	extern __shared__ cmplx_ sh[];   // [0,bd) = P partials, [bd,2bd) = Q partials
#else
	cmplx_ sh[2];
#endif
	int map = blockIdx.x;            // uniform across the block
	if (map >= no_mappings) {
		return;
	}
	int tid = threadIdx.x;
	cmplx_ p = cmplx(0.f, 0.f), q = cmplx(0.f, 0.f);
	for (int j = tid; j < N; j += blockDim.x) {
		cmplx_ zj = Z_(in[map], j);
		p += dZ_(out[map], j) * zj;
		q += dZ_star_(out[map], j) * conj_(zj);
	}
	sh[tid] = p;
	sh[blockDim.x + tid] = q;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sh[tid] += sh[tid + stride];
			sh[blockDim.x + tid] += sh[blockDim.x + tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0) {
		pq[2 * map] = sh[0];
		pq[2 * map + 1] = sh[blockDim.x];
	}
}

// (2) One thread per (map, i): apply the closed-form gradient, accumulated.
__global__ void gpu_softmax_apply_grad__(GpuInVar *in, GpuOutVar *out, cmplx_ *pq, int N, int no_mappings) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) {
		return;
	}
	int map = tid / N;
	int i = tid % N;
	float s = sqrtf(out[map].reduce_real_);    // ||z||
	if (s < 1e-15f) {
		s = 1e-15f;
	}
	float s3 = out[map].reduce_imag_;           // ||z||^3 (floored in forward)
	cmplx_ R = (pq[2 * map] + pq[2 * map + 1]) / s3;
	cmplx_ zi = Z_(in[map], i);
	cmplx_ sum_dz = dZ_(out[map], i) / s - 0.5f * conj_(zi) * R;
	cmplx_ sum_dz_star = dZ_star_(out[map], i) / s - 0.5f * zi * R;
	atomicAdd(dZ_real_(in[map], i), sum_dz.real);
	atomicAdd(dZ_imag_(in[map], i), sum_dz.imag);
	atomicAdd(dZ_star_real_(in[map], i), sum_dz_star.real);
	atomicAdd(dZ_star_imag_(in[map], i), sum_dz_star.imag);
}

void SoftMaxGpu::gpu_soft_max_backward(int label, int block_size) {
	if (fast_softmax) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();

		GpuHelper helper;
		cmplx_ *pq = helper.cmplx_allocate_on_gpu(2 * M);   // P,Q per map
		assert(pq);

		int pqt = 256;   // power of two for the tree reduction
		gpu_softmax_pq__ CUDA(M, pqt, 2 * pqt * sizeof(cmplx_))
				(gpu_in_ptr_, gpu_out_ptr_, pq, N, M);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		int total = N * M;
		int tb = 256;
		unsigned grid = (total + tb - 1) / tb;
		gpu_softmax_apply_grad__ CUDA2(grid, tb) (gpu_in_ptr_, gpu_out_ptr_, pq, N, M);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		return;
#endif
	}

	int b_out_length = getPaddedOutLength(block_size, this);

	GpuHelper helper;
	auto gpu_buffer = helper.grad_allocate_on_gpu(b_out_length);
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	grad_reducing_kernel(SOFTMAX_DATA_PROVIDER, gpu_in_ptr_, length(), gpu_out_ptr_, getOutputLength(),
			gpu_buffer, getNoMappings(), block_size);
	grad_kernel_end(SOFTMAX_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, length(), getNoMappings(), block_size);
}

void linear_kernel_end(int provider_id, Grads *buff, GpuInVar *in, int seg_len, int out_len, int no_mappings, int block_size) {
	int in_stride = out_len % block_size == 0 ? out_len : (out_len + block_size - out_len % block_size);
	in_stride = in_stride / block_size;
	int no_threads = seg_len * no_mappings;

	unsigned grid = (no_threads + block_size - 1) / block_size;
//	std::cout << "Launching linear_kernel_end with Grid size: " << grid << " & Block Size:" << block_size
//			  << " in_stride: " << in_stride << " seg_len: " << seg_len << " out_len: " << out_len
// 			  << " no_mappings: " << no_mappings << "\n";

	grad_kernel_end__ CUDA( grid, block_size, block_size * sizeof(cmplx_) )
			(provider_id, no_mappings, buff, in, in_stride, no_threads);

#ifdef __CUDACC__
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#else
#endif

}

// Optimized Linear backward (used when fast_linear is set).
//
// Two direct kernels replace the generic reduce-then-sum path (a tree reduction
// over rows plus a per-call scratch cudaMalloc):
//   (1) matrix gradient  grad_W[row][col] = dLdz[row] * vec[col]   (outer product)
//   (2) vector gradient  grad_v[col] = sum_row dLdz[row] * mat[row][col]  (reduction)
// They write to disjoint regions of the input gradient buffer and both
// accumulate with atomicAdd, matching the original semantics exactly.

// (1) Matrix gradient: one thread per matrix entry (map, row, col).
__global__ void gpu_linear_mat_grad__(GpuInVar *in, GpuOutVar *out,
		int N, int M, int no_mappings) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t) no_mappings * M * N;
	if (tid >= total) {
		return;
	}
	int col = (int) (tid % N);
	int row = (int) ((tid / N) % M);
	int map_indx = (int) (tid / ((size_t) M * N));

	cmplx_ dLdz      = dZ_(out[map_indx], row);
	cmplx_ dLdz_star = dZ_star_(out[map_indx], row);
	cmplx_ vec       = Z_(in[map_indx], col);

	int o_index = (row + 1) * N + col;
	cmplx_ dz      = dLdz * vec;
	cmplx_ dz_star = dLdz_star * conj_(vec);

	atomicAdd(dZ_real_(in[map_indx], o_index), dz.real);
	atomicAdd(dZ_imag_(in[map_indx], o_index), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], o_index), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], o_index), dz_star.imag);
}

// (2) Vector gradient: one thread per vector position (map, col); the incoming
// output gradient (M complex, both dz and dz_star) is cached in shared memory
// and reused across every column the block computes.
__global__ void gpu_linear_vec_grad__(GpuInVar *in, GpuOutVar *out,
		int N, int M, int no_mappings, int blocks_per_map) {
#ifdef __CUDACC__
	extern __shared__ cmplx_ sh[];     // [0,M) = dLdz, [M,2M) = dLdz_star
#else
	cmplx_ sh[1];   // CPU build never executes device kernels
#endif
	// blockIdx.x is uniform across a block, so this early-out is uniform and
	// can never cause a divergent __syncthreads().
	int map_indx = blockIdx.x / blocks_per_map;
	if (map_indx >= no_mappings) {
		return;
	}

	for (int r = threadIdx.x; r < M; r += blockDim.x) {
		sh[r]     = dZ_(out[map_indx], r);
		sh[M + r] = dZ_star_(out[map_indx], r);
	}
	__syncthreads();

	int col = (blockIdx.x % blocks_per_map) * blockDim.x + threadIdx.x;
	if (col < N) {
		cmplx_ sum_dz      = cmplx(0.f, 0.f);
		cmplx_ sum_dz_star = cmplx(0.f, 0.f);
		for (int r = 0; r < M; ++r) {
			cmplx_ m = Z_(in[map_indx], (r + 1) * N + col);
			sum_dz      += sh[r]     * m;
			sum_dz_star += sh[M + r] * conj_(m);
		}
		atomicAdd(dZ_real_(in[map_indx], col), sum_dz.real);
		atomicAdd(dZ_imag_(in[map_indx], col), sum_dz.imag);
		atomicAdd(dZ_star_real_(in[map_indx], col), sum_dz_star.real);
		atomicAdd(dZ_star_imag_(in[map_indx], col), sum_dz_star.imag);
	}
}

void LinearGpu::gpu_linear_backward(int label, int block_size) {
	int N = ((Linear*)getCpuFun()[0])->firstInputLength();   // E_in (vector length)
	int M = ((Linear*)getCpuFun()[0])->outSize();            // E_out (output rows)
	int no_mappings = getNoMappings();

#ifdef __CUDACC__
	// Dense Linear is the position-wise linear with a single token (N = 1).
	if (tw_cublas && no_mappings > 0) {
		tw_ensure_pool(d_pool_, tw_cap_, no_mappings);
		tw_gemm_backward(1, N, M, in_, out_, d_pool_);
		return;
	}
#endif

	// The direct kernels beat the generic reduction when there are few output
	// rows M (the reduction then wastes most of each 512-wide block on padding).
	// For large M the reduction's row-parallelism wins, so fall back to it.
	size_t shmem = (size_t) 2 * M * sizeof(cmplx_);          // dLdz + dLdz_star
	if (fast_linear && M <= 256 && shmem <= 48u * 1024u) {
		// (1) matrix gradient: one thread per matrix entry.
		size_t total = (size_t) no_mappings * M * N;
		int mt = 256;
		unsigned mgrid = (unsigned) ((total + mt - 1) / mt);
		gpu_linear_mat_grad__ CUDA2(mgrid, mt)
				(gpu_in_ptr_, gpu_out_ptr_, N, M, no_mappings);
#ifdef __CUDACC__
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		// (2) vector gradient: one thread per column, output grad cached in shared.
		int vt = 128;
		if (vt > N) {
			vt = N;
		}
		if (vt < 1) {
			vt = 1;
		}
		int blocks_per_map = (N + vt - 1) / vt;
		unsigned vgrid = (unsigned) no_mappings * blocks_per_map;
		gpu_linear_vec_grad__ CUDA(vgrid, vt, shmem)
				(gpu_in_ptr_, gpu_out_ptr_, N, M, no_mappings, blocks_per_map);
#ifdef __CUDACC__
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		return;
	}

	// Fallback: the original reduce-then-sum path.
	int b_out_length = getPaddedOutLength(block_size, this);

	GpuHelper helper;
	auto gpu_buffer = helper.grad_allocate_on_gpu(b_out_length);
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}
	int seg_len = length() / (1 + getOutputLength());

	grad_reducing_kernel(LINEAR_DATA_PROVIDER, gpu_in_ptr_, seg_len, gpu_out_ptr_, getOutputLength(),
			gpu_buffer, getNoMappings(), block_size);
	linear_kernel_end(LINEAR_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, seg_len, getOutputLength(), getNoMappings(), block_size);
}

// ======== TokenwiseLinear backward (direct kernels).
// weight gradient: one thread per (token, out, in) entry, accumulated across
// tokens into the shared weight (hence atomicAdd). data gradient: one thread per
// (token, in) element, summing over the out dimension.

__global__ void gpu_tokenwise_mat_grad__(GpuInVar *in, GpuOutVar *out,
		int n_tokens, int e_in, int e_out, int no_mappings) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	size_t per_map = (size_t) n_tokens * e_out * e_in;
	if (tid >= (size_t) no_mappings * per_map) {
		return;
	}
	int map_indx = (int) (tid / per_map);
	size_t rem = tid % per_map;
	int t = (int) (rem / ((size_t) e_out * e_in));
	int rc = (int) (rem % ((size_t) e_out * e_in));
	int r = rc / e_in;
	int c = rc % e_in;
	int w_base = n_tokens * e_in;

	cmplx_ dLdz      = dZ_(out[map_indx], t * e_out + r);
	cmplx_ dLdz_star = dZ_star_(out[map_indx], t * e_out + r);
	cmplx_ d = Z_(in[map_indx], t * e_in + c);

	int w_index = w_base + r * e_in + c;
	cmplx_ dz      = dLdz * d;
	cmplx_ dz_star = dLdz_star * conj_(d);
	atomicAdd(dZ_real_(in[map_indx], w_index), dz.real);
	atomicAdd(dZ_imag_(in[map_indx], w_index), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], w_index), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], w_index), dz_star.imag);
}

__global__ void gpu_tokenwise_vec_grad__(GpuInVar *in, GpuOutVar *out,
		int n_tokens, int e_in, int e_out, int no_mappings) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	size_t per_map = (size_t) n_tokens * e_in;
	if (tid >= (size_t) no_mappings * per_map) {
		return;
	}
	int map_indx = (int) (tid / per_map);
	int dc = (int) (tid % per_map);
	int t = dc / e_in;
	int c = dc % e_in;
	int w_base = n_tokens * e_in;

	cmplx_ sum_dz      = cmplx(0.f, 0.f);
	cmplx_ sum_dz_star = cmplx(0.f, 0.f);
	for (int r = 0; r < e_out; ++r) {
		cmplx_ dLdz      = dZ_(out[map_indx], t * e_out + r);
		cmplx_ dLdz_star = dZ_star_(out[map_indx], t * e_out + r);
		cmplx_ w = Z_(in[map_indx], w_base + r * e_in + c);
		sum_dz      += dLdz * w;
		sum_dz_star += dLdz_star * conj_(w);
	}
	int d_index = t * e_in + c;
	atomicAdd(dZ_real_(in[map_indx], d_index), sum_dz.real);
	atomicAdd(dZ_imag_(in[map_indx], d_index), sum_dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], d_index), sum_dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], d_index), sum_dz_star.imag);
}

#ifdef __CUDACC__
// Backward for the position-wise linear (the dense Linear calls with N = 1). Both
// grads are complex GEMMs accumulated (beta=1) onto the pre-zeroed gradient planes:
//   data grad:   gD[E_in,N]    += (conj)W[E_in,E_out] * dOut[E_out,N]
//   weight grad: gW[E_in,E_out] += (conj)D[E_in,N]    * dOut[E_out,N]^T  (sum over tokens)
// split into two real GEMMs per Wirtinger component (dz and dz*).
void tw_gemm_backward(int N, int e_in, int e_out,
		const std::vector<GpuInVar> &in_, const std::vector<GpuOutVar> &out_, float **pool) {
	const int B = (int) in_.size();
	const int L = in_[0].input_length_, wbase = N * e_in, outlen = out_[0].out_length_;
	std::vector<float*> hp((size_t) 16 * B, nullptr);
	for (int b = 0; b < B; ++b) {
		float *base = in_[b].input_ptr_, *ob = out_[b].out_ptr_;
		hp[0*B+b]=base;              hp[1*B+b]=base+L;                 // Dr, Di
		hp[2*B+b]=base+wbase;        hp[3*B+b]=base+L+wbase;           // Wr, Wi
		hp[4*B+b]=ob+2*outlen;       hp[5*B+b]=ob+3*outlen;            // dOut r/i  (dz)
		hp[6*B+b]=ob+4*outlen;       hp[7*B+b]=ob+5*outlen;            // dOut r/i  (dz*)
		hp[8*B+b]=base+2*L;          hp[9*B+b]=base+3*L;               // gData dz r/i
		hp[10*B+b]=base+4*L;         hp[11*B+b]=base+5*L;              // gData dz* r/i
		hp[12*B+b]=base+2*L+wbase;   hp[13*B+b]=base+3*L+wbase;        // gWeight dz r/i
		hp[14*B+b]=base+4*L+wbase;   hp[15*B+b]=base+5*L+wbase;        // gWeight dz* r/i
	}
	gpuErrchk(cudaMemcpy(pool, hp.data(), (size_t) 16 * B * sizeof(float*), cudaMemcpyHostToDevice));
	#define P(i) (pool + (size_t)(i) * B)
	float **Dr=P(0),**Di=P(1),**Wr=P(2),**Wi=P(3),**GOr=P(4),**GOi=P(5),**GSr=P(6),**GSi=P(7),
		  **gDr=P(8),**gDi=P(9),**gSDr=P(10),**gSDi=P(11),**gWr=P(12),**gWi=P(13),**gWSr=P(14),**gWSi=P(15);
	cublasHandle_t h = twg_blas_handle();
	const float one = 1.f;
	// data grad: C[E_in,N] = A[E_in,E_out] * Bm[E_out,N], accumulate
	auto GD = [&](float a, float **A, float **Bm, float **C) {
		cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, e_in, N, e_out,
				&a, (const float* const*) A, e_in, (const float* const*) Bm, e_out, &one, C, e_in, B);
	};
	// weight grad: C[E_in,E_out] = A[E_in,N] * Bm[E_out,N]^T, accumulate (sum over N)
	auto GW = [&](float a, float **A, float **Bm, float **C) {
		cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, e_in, e_out, N,
				&a, (const float* const*) A, e_in, (const float* const*) Bm, e_out, &one, C, e_in, B);
	};
	GD( 1.f, Wr, GOr, gDr);  GD(-1.f, Wi, GOi, gDr);
	GD( 1.f, Wr, GOi, gDi);  GD( 1.f, Wi, GOr, gDi);
	GD( 1.f, Wr, GSr, gSDr); GD( 1.f, Wi, GSi, gSDr);
	GD( 1.f, Wr, GSi, gSDi); GD(-1.f, Wi, GSr, gSDi);
	GW( 1.f, Dr, GOr, gWr);  GW(-1.f, Di, GOi, gWr);
	GW( 1.f, Dr, GOi, gWi);  GW( 1.f, Di, GOr, gWi);
	GW( 1.f, Dr, GSr, gWSr); GW( 1.f, Di, GSi, gWSr);
	GW( 1.f, Dr, GSi, gWSi); GW(-1.f, Di, GSr, gWSi);
	#undef P
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
#endif

void TokenwiseLinearGpu::gpu_tokenwise_backward() {
	TokenwiseLinear *cpu = (TokenwiseLinear*) getCpuFun()[0];
	int n_tokens = cpu->nTokens(), e_in = cpu->inDim(), e_out = cpu->outDim();
	int no_mappings = getNoMappings();
#ifdef __CUDACC__
	if (tw_cublas && no_mappings > 0) {
		tw_ensure_pool(d_pool_, tw_cap_, no_mappings);
		tw_gemm_backward(n_tokens, e_in, e_out, in_, out_, d_pool_);
		return;
	}
#endif

	size_t mtot = (size_t) no_mappings * n_tokens * e_out * e_in;
	int mt = 256;
	unsigned mgrid = (unsigned) ((mtot + mt - 1) / mt);
	gpu_tokenwise_mat_grad__ CUDA2(mgrid, mt)
			(gpu_in_ptr_, gpu_out_ptr_, n_tokens, e_in, e_out, no_mappings);
#ifdef __CUDACC__
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	size_t vtot = (size_t) no_mappings * n_tokens * e_in;
	int vt = 256;
	unsigned vgrid = (unsigned) ((vtot + vt - 1) / vt);
	gpu_tokenwise_vec_grad__ CUDA2(vgrid, vt)
			(gpu_in_ptr_, gpu_out_ptr_, n_tokens, e_in, e_out, no_mappings);
#ifdef __CUDACC__
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

// ---- cuFFT fast path for FourierTrans (backward) ----
// The FFT is a unitary linear map, so its Wirtinger adjoint is itself an FFT:
//   dz[q]      += (1/sqrt N) sum_p dLdz[p]      e^{-i 2pi pq/N} = FFT_forward(dLdz)/sqrt N
//   dz_star[q] += (1/sqrt N) sum_p dLdz_star[p] e^{+i 2pi pq/N} = FFT_inverse(dLdz_star)/sqrt N
// so the backward is a forward transform of the incoming dz and an inverse
// transform of the incoming dz_star, each scaled and accumulated.

// gather output gradient (dz, or dz_star when use_star) -> interleaved buffer.
__global__ void gpu_fft_gather_grad__(GpuOutVar *out, cmplx_ *buf, int N, int no_mappings, int use_star) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) {
		return;
	}
	int map = tid / N;
	int k = tid % N;
	buf[tid] = use_star ? dZ_star_(out[map], k) : dZ_(out[map], k);
}

// scatter transformed gradient -> accumulate into the input dz / dz_star, scaled.
__global__ void gpu_fft_scatter_grad__(cmplx_ *buf, GpuInVar *in, int N, int no_mappings,
		float scale, int use_star) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) {
		return;
	}
	int map = tid / N;
	int k = tid % N;
	cmplx_ v = buf[tid];
	if (use_star) {
		atomicAdd(dZ_star_real_(in[map], k), v.real * scale);
		atomicAdd(dZ_star_imag_(in[map], k), v.imag * scale);
	} else {
		atomicAdd(dZ_real_(in[map], k), v.real * scale);
		atomicAdd(dZ_imag_(in[map], k), v.imag * scale);
	}
}

void FourierGpu::gpu_fft_backward(int label, int block_size) {
	if (fft_fast) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();
		ensure_fft_plan(N, M);

		int total = N * M;
		int tb = 256;
		unsigned grid = (total + tb - 1) / tb;
		float scale = 1.0f / sqrtf((float) N);

		// dz += FFT_forward(dLdz) / sqrt(N)
		gpu_fft_gather_grad__ CUDA2(grid, tb) (gpu_out_ptr_, fft_buf_, N, M, 0);
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_FORWARD);
		gpu_fft_scatter_grad__ CUDA2(grid, tb) (fft_buf_, gpu_in_ptr_, N, M, scale, 0);

		// dz_star += FFT_inverse(dLdz_star) / sqrt(N)
		gpu_fft_gather_grad__ CUDA2(grid, tb) (gpu_out_ptr_, fft_buf_, N, M, 1);
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_INVERSE);
		gpu_fft_scatter_grad__ CUDA2(grid, tb) (fft_buf_, gpu_in_ptr_, N, M, scale, 1);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		return;
#endif
	}

	int b_out_length = getPaddedOutLength(block_size, this);

	GpuHelper helper;
	auto gpu_buffer = helper.grad_allocate_on_gpu(b_out_length);
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	grad_reducing_kernel(FFT_DATA_PROVIDER, gpu_in_ptr_, length(), gpu_out_ptr_, getOutputLength(),
			gpu_buffer, getNoMappings(), block_size);
	grad_kernel_end(SOFTMAX_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, length(), getNoMappings(), block_size);
}

// Inverse DFT backward: the adjoint of the inverse map is the forward unitary
// DFT, i.e. FourierGpu::gpu_fft_backward with the two cuFFT directions swapped:
//   dz      += FFT_inverse(dLdz)      / sqrt(N)
//   dz_star += FFT_forward(dLdz_star) / sqrt(N)
void InverseFourierGpu::gpu_ifft_backward(int label, int block_size) {
	if (fft_fast) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();
		ensure_fft_plan(N, M);

		int total = N * M;
		int tb = 256;
		unsigned grid = (total + tb - 1) / tb;
		float scale = 1.0f / sqrtf((float) N);

		// dz += FFT_inverse(dLdz) / sqrt(N)
		gpu_fft_gather_grad__ CUDA2(grid, tb) (gpu_out_ptr_, fft_buf_, N, M, 0);
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_INVERSE);
		gpu_fft_scatter_grad__ CUDA2(grid, tb) (fft_buf_, gpu_in_ptr_, N, M, scale, 0);

		// dz_star += FFT_forward(dLdz_star) / sqrt(N)
		gpu_fft_gather_grad__ CUDA2(grid, tb) (gpu_out_ptr_, fft_buf_, N, M, 1);
		cufftExecC2C((cufftHandle) fft_plan_, (cufftComplex*) fft_buf_,
				(cufftComplex*) fft_buf_, CUFFT_FORWARD);
		gpu_fft_scatter_grad__ CUDA2(grid, tb) (fft_buf_, gpu_in_ptr_, N, M, scale, 1);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		return;
#endif
	}
	assert(fft_fast && "InverseFourierGpu requires -fft_fast true");
}

// ---- Bluestein fast path for TriangFourier (backward / adjoint) ----
// The adjoint of the lower-triangular DFT is the upper-triangular sum
//   dz[q]      += (1/sqrt N) sum_{p>=q} dLdz[p]      w^{pq}
//   dz_star[q] += (1/sqrt N) sum_{p>=q} dLdz_star[p] conj(w^{pq})
// which chirps into a correlation (reversed kernel): with b = g*chirp (dz) or
// g*conj(chirp) (dz*), grad = chirp (resp. conj(chirp)) * corr(b, k) / sqrt N,
// corr computed as b convolved with the reversed kernel KrA/KrB precomputed at
// setup. Same O(N log N) batched FFT as the forward.

// b = g * (chirp | conj chirp), zero-padded to L, read from output gradients.
__global__ void gpu_blb_gather_grad__(GpuOutVar *out, cmplx_ *chirp, cmplx_ *buf,
		int N, int L, int no_mappings, int use_star) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * L) return;
	int map = tid / L;
	int k = tid % L;
	if (k < N) {
		cmplx_ g  = use_star ? dZ_star_(out[map], k) : dZ_(out[map], k);
		cmplx_ ch = use_star ? conj_(chirp[k]) : chirp[k];
		buf[tid] = g * ch;
	} else {
		buf[tid] = cmplx(0.f, 0.f);
	}
}

__global__ void gpu_blb_kmul__(cmplx_ *buf, cmplx_ *K, int L, int no_mappings) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * L) return;
	int k = tid % L;
	buf[tid] = buf[tid] * K[k];
}

// grad_in[q] += (chirp | conj chirp)[q] * corr[q] * scale  (accumulate).
__global__ void gpu_blb_scatter_grad__(cmplx_ *buf, cmplx_ *chirp, GpuInVar *in,
		int N, int L, int no_mappings, float scale, int use_star) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= no_mappings * N) return;
	int map = tid / N;
	int q = tid % N;
	cmplx_ ch = use_star ? conj_(chirp[q]) : chirp[q];
	cmplx_ v = (ch * buf[map * L + q]) * scale;
	if (use_star) {
		atomicAdd(dZ_star_real_(in[map], q), v.real);
		atomicAdd(dZ_star_imag_(in[map], q), v.imag);
	} else {
		atomicAdd(dZ_real_(in[map], q), v.real);
		atomicAdd(dZ_imag_(in[map], q), v.imag);
	}
}

void TrianFourierGpu::gpu_T_fft_backward(int label, int block_size) {
	if (fft_fast) {
#ifdef __CUDACC__
		int N = length();
		int M = getNoMappings();
		ensure_bluestein(N, M);
		int L = bl_L_;

		int tb = 256;
		unsigned gridL = ((size_t) L * M + tb - 1) / tb;
		unsigned gridN = ((size_t) N * M + tb - 1) / tb;
		float scale = 1.0f / ((float) L * sqrtf((float) N));

		// dz += L^T dLdz  (kernel KrA = FFT(reverse(conj chirp)))
		gpu_blb_gather_grad__ CUDA2(gridL, tb) (gpu_out_ptr_, bl_chirp_, bl_buf_, N, L, M, 0);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_FORWARD);
		gpu_blb_kmul__ CUDA2(gridL, tb) (bl_buf_, bl_KrA_, L, M);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_INVERSE);
		gpu_blb_scatter_grad__ CUDA2(gridN, tb) (bl_buf_, bl_chirp_, gpu_in_ptr_, N, L, M, scale, 0);

		// dz_star += L^H dLdz_star  (kernel KrB = FFT(reverse(chirp)))
		gpu_blb_gather_grad__ CUDA2(gridL, tb) (gpu_out_ptr_, bl_chirp_, bl_buf_, N, L, M, 1);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_FORWARD);
		gpu_blb_kmul__ CUDA2(gridL, tb) (bl_buf_, bl_KrB_, L, M);
		cufftExecC2C((cufftHandle) bl_plan_, (cufftComplex*) bl_buf_, (cufftComplex*) bl_buf_, CUFFT_INVERSE);
		gpu_blb_scatter_grad__ CUDA2(gridN, tb) (bl_buf_, bl_chirp_, gpu_in_ptr_, N, L, M, scale, 1);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		return;
#endif
	}

	int b_out_length = getPaddedOutLength(block_size, this);

	GpuHelper helper;
	auto gpu_buffer = helper.grad_allocate_on_gpu(b_out_length);
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	grad_reducing_kernel(T_FFT_DATA_PROVIDER, gpu_in_ptr_, length(), gpu_out_ptr_, getOutputLength(),
			gpu_buffer, getNoMappings(), block_size);
	grad_kernel_end(SOFTMAX_DATA_PROVIDER, gpu_buffer, gpu_in_ptr_, length(), getNoMappings(), block_size);
}

