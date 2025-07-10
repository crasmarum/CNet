
#include "reduce.h"
#include "allocator.h"

#include "../utils/stopwatch.h"
#include "../utils/flags.h"

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
	if (thread_indx >= max_no_threads) {
		return;
	}

	int out_indx = thread_indx % out_seg_length;

	int map_indx = thread_indx / in_seg_len / out_seg_length;
	//int init_out_seg_length = 8;//out[map_indx].out_length_;

	// we have max_no_threads = out_seg_length * in_seg_len * no_mappings;
	// we need to do this as out_seg is the padded initial out segment.
	if (out_indx >= init_out_seg_length) {
		sdata[tid] = Grads();
		return;
	}
	__syncthreads();

	int in_indx = (thread_indx / out_seg_length) % in_seg_len;

    size_t buff_indx = thread_indx / blockSize;

    auto dLdz = dZ_(out[map_indx], out_indx);
    auto dLdz_star = dZ_star_(out[map_indx], out_indx);

    auto dz      =      Fdz(provider_id, in[map_indx], in_indx, out[map_indx], out_indx);
    auto dz_star = Fdz_star(provider_id, in[map_indx], in_indx, out[map_indx], out_indx);

    sdata[tid].grad_ = dLdz * dz + dLdz_star * conj_(dz_star);
    sdata[tid].grad_star_ = dLdz * dz_star + dLdz_star * conj_(dz);

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

void SoftMaxGpu::gpu_soft_max_backward(int label, int block_size) {
	int b_out_length = getPaddedOutLength(block_size, this);

	GpuHelper helper;
	auto gpu_buffer = helper.grad_allocate_on_gpu(b_out_length);
	if (!gpu_buffer) {
		assert(gpu_buffer);
	}

	//std::vector<Grads> output(b_out_length);
	//	helper.cmplx_copy_from_gpu(b_out_length, gpu_buffer, &output[0]);
	//	for (int var = 0; var < output.size(); ++var) {
	//		std::cout << std::setfill('0') << std::setw(5) << var << "\t" << output[var] << std::endl;
	//	}

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

void LinearGpu::gpu_linear_backward(int label, int block_size) {
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

void FourierGpu::gpu_fft_backward(int label, int block_size) {
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

void TrianFourierGpu::gpu_T_fft_backward(int label, int block_size) {
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

