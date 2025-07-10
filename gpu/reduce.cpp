#include "reduce.h"
#include "allocator.h"

#include "../utils/stopwatch.h"
#include "../utils/flags.h"

FLAG_INT(reduce_block_size, 512);

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
	if (thread_indx >= max_no_threads) {
		return;
	}

	// max_threads = segment_len * no_segments * no_mappings;
	int no_segments = max_no_threads / no_mappings / seg_length;
	int pos_in_segment = thread_indx % seg_length;
	if (pos_in_segment >= init_seg_length) {
		sdata[tid] = {0, 0};	// we need to do this!
		return;
	}

	int map_indx = thread_indx / no_segments / seg_length;
	int segm_no = (thread_indx / seg_length) % no_segments;
    size_t out_indx = thread_indx / blockSize;

	sdata[tid] = getDataFor(provider_id, map_indx, pos_in_segment, init_seg_length, segm_no, in, tid);
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

void FourierGpu::gpu_fft_forward(int block_size) {
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

void TrianFourierGpu::gpu_T_fft_forward(int block_size) {
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

void LinearGpu::gpu_linear_forward(int block_size) {
	int seg_in_len = ((Linear*)getCpuFun()[0])->firstInputLength();
	int no_segments = ((Linear*)getCpuFun()[0])->outSize();

//	int seg_out_len = mp->getOuts().front().out_length_;
//	int seg_in_len = mp->length() / (seg_out_len + 1); // here the assumption is that is only one output! not true

	int padded_segment_len = seg_in_len % block_size == 0 ? seg_in_len
						   : (seg_in_len + block_size - seg_in_len % block_size);
	int b_out_length = (padded_segment_len / block_size) * getNoMappings() * seg_in_len;

	// TODO: make this global
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

