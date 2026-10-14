#include "kernels.h"
#include "gpumapping.h"

#include "../utils/flags.h"

const int maxNoThreads = 1024; // TODO


/*
 * [I1] ...[I_nf]           [O1]...[O_nf]  nf = number of functions
 *  Ij = [i_0, ..., i_len]   Oj = [o_0, ..., o_len]
 *  Inmem:
 *  [i_00, ..., i_0_len][i_10, ..., i_1_length] ... [i_nf_0, ... , i_nf_len]
 *  in_ptr[1]->input_ptr_ = i_10, out_ptr[1]->out_ptr_ = o_10
 *
 *  ex: [1 2 3] [4, 5, 6] len=3 nf=2
 */
__global__ void gpu_relu_forward__(GpuInVar *in, GpuOutVar *out, int no_func,
		int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;


	bool is_pos = (*Z_real_(in[map_indx], pos)) > 0
			&& (*Z_imag_(in[map_indx], pos)) > 0;

//	printf("thread_indx=%05d \t func_indx=%05d \t pos=%05d \t re=%f im=%f\n",
//			thread_indx, func_indx, pos, (*in_pos_real), (*in_pos_imag));

	*Z_real_(out[map_indx], pos) = is_pos ? *Z_real_(in[map_indx], pos) : 0;
	*Z_imag_(out[map_indx], pos) = is_pos ? *Z_imag_(in[map_indx], pos) : 0;
}

void ReluGpu::gpu_relu_forward() {
	int total_threads = getNoMappings() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	// gpu_in_ptr_, gpu_out_ptr_, in_.size(), length()
	gpu_relu_forward__ CUDA2(no_blocks, maxNoThreads)
		(gpu_in_ptr_, gpu_out_ptr_, in_.size(), length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_relu_backward__(GpuInVar *in, GpuOutVar *out, int no_func,
		int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	bool is_pos = (*Z_real_(in[map_indx], pos)) > 0
			&& (*Z_imag_(in[map_indx], pos)) > 0;

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);

	atomicAdd(dZ_real_(in[map_indx],      pos), is_pos ? dLdz.real : 0);
	atomicAdd(dZ_imag_(in[map_indx],      pos), is_pos ? dLdz.imag : 0);
	atomicAdd(dZ_star_real_(in[map_indx], pos), is_pos ? dLdz_star.real : 0);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), is_pos ? dLdz_star.imag : 0);
}

void ReluGpu::gpu_relu_backward() {
	int total_threads = getNoMappings() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	// gpu_in_ptr_, gpu_out_ptr_, in_.size(), length()
	gpu_relu_backward__ CUDA2(no_blocks, maxNoThreads)
		(gpu_in_ptr_, gpu_out_ptr_, in_.size(), length(), total_threads);
	CNET_KSYNC();
}

/*
 * [I1] ...[I_nf]           [O1]...[O_nf]  nf = number of functions
 *  Ij = [i_0, ..., i_len]   Oj = [o_0, ..., o_len]
 *  Inmem:
 *  [i_00, ..., i_0_len][i_10, ..., i_1_length] ... [i_nf_0, ... , i_nf_len]
 *  in_ptr[1]->input_ptr_ = i_10, out_ptr[1]->out_ptr_ = o_10
 *
 *  ex: [1 2 3] [4, 5, 6] len=3 nf=2
 */
__global__ void gpu_input_forward__(GpuInVar *in, GpuOutVar *out, int no_func,
		int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int func_indx = thread_indx / length;
	int pos = thread_indx % length;

	float *in_fc_start = (in + func_indx)->input_ptr_;
	float *in_pos_real = in_fc_start + pos;
	float *in_pos_imag = in_fc_start + length + pos;

	float *out_fc_start = (out + func_indx)->out_ptr_;
	float *out_pos_real = out_fc_start + pos;
	float *out_pos_imag = out_fc_start + (out + func_indx)->out_length_ + pos;

	*out_pos_real = *in_pos_real;
	*out_pos_imag = *in_pos_imag;
}

// 	void gpu_input_forward(GpuInVar *in, GpuOutVar *out, int no_func, int length);
void InputGpu::gpu_input_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_input_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, in_.size(), length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_input_backward__(GpuInVar *in, GpuOutVar *out, int no_maps, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int current_map = thread_indx / length;
	int pos = thread_indx - length * current_map;

//	printf("m=%d\t pim=%d\t T=%d\t piT=%d\t tid=%d\t op=%p\n", current_map, pos_in_map, current_token, pos_in_token,
//			thread_indx, out[current_map].out_ptr_);

	atomicAdd(dZ_star_real_(in[current_map], pos), *dZ_star_real_(out[current_map], pos));
	atomicAdd(dZ_star_imag_(in[current_map], pos), *dZ_star_imag_(out[current_map], pos));
}

// 	void gpu_input_forward(GpuInVar *in, GpuOutVar *out, int no_func, int length);
void InputGpu::gpu_input_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_input_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, in_.size(), length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_residual_forward__ (GpuInVar *in, GpuOutVar *out, int no_maps,
		int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	auto Z = Z_(in[map_indx], pos) + Z_(in[map_indx], length + pos);
	*Z_real_(out[map_indx], pos) = Z.real;
	*Z_imag_(out[map_indx], pos) = Z.imag;
}

void ResidualGpu::gpu_residual_forward() {
	int total_threads = in_.size() * length() / 2;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_residual_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, in_.size(), length() / 2, total_threads);
	CNET_KSYNC();
}

__global__ void gpu_residual_backward__ (GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);

	atomicAdd(dZ_real_(in[map_indx], pos), dLdz.real);
	atomicAdd(dZ_imag_(in[map_indx], pos), dLdz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), dLdz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), dLdz_star.imag);


	atomicAdd(dZ_real_(in[map_indx], length + pos), dLdz.real);
	atomicAdd(dZ_imag_(in[map_indx], length + pos), dLdz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], length + pos), dLdz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], length + pos), dLdz_star.imag);

//	printf("thread_indx=%05d \t func_indx=%05d \t pos=%05d \t re=%f im=%f\n",
//			thread_indx, map_indx, pos, dLdz.real, dLdz.imag);

}

void ResidualGpu::gpu_residual_backward(int label) {
	int total_threads = in_.size() * length() / 2;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_residual_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length() / 2, total_threads);
	CNET_KSYNC();
}

__global__ void gpu_hadamard_forward__ (GpuInVar *in, GpuOutVar *out, int no_func,
		int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int func_indx = thread_indx / length;
	int pos = thread_indx % length;
	float *in_fc_start = (in + func_indx)->input_ptr_;

	float *in_pos_real_1 = in_fc_start + pos;
	float *in_pos_imag_1 = in_fc_start + 2 * length + pos;

	float *in_pos_real_2 = in_fc_start + length + pos;
	float *in_pos_imag_2 = in_fc_start + 3 * length + pos;

	float *out_fc_start = (out + func_indx)->out_ptr_;
	float *out_pos_real = out_fc_start + pos;
	float *out_pos_imag = out_fc_start + (out + func_indx)->out_length_ + pos;

//	printf("thread_indx=%05d \t func_indx=%05d \t pos=%05d \t re=%f im=%f\n",
//			thread_indx, func_indx, pos, (*in_pos_real), (*in_pos_imag));

	auto z = cmplx(*in_pos_real_1, *in_pos_imag_1) * cmplx(*in_pos_real_2, *in_pos_imag_2);

	*out_pos_real = z.real;
	*out_pos_imag = z.imag;
}

//void gpu_hadamard_forward(GpuInVar *in, GpuOutVar *out, int no_func, int length);
void HadamardGpu::gpu_hadamard_forward() {
	int total_threads = in_.size() * length() / 2;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_hadamard_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, in_.size(), length() / 2, total_threads);
	CNET_KSYNC();
}

__global__ void gpu_hadamard_backward__ (GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);

	auto z = Z_(in[map_indx], length + pos);
	atomicAdd(dZ_real_(in[map_indx],      pos), (dLdz * z).real);
	atomicAdd(dZ_imag_(in[map_indx],      pos), (dLdz * z).imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), (dLdz_star * conj_(z)).real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), (dLdz_star * conj_(z)).imag);

	z = Z_(in[map_indx], pos);
	atomicAdd(dZ_real_(in[map_indx],      length + pos), (dLdz * z).real);
	atomicAdd(dZ_imag_(in[map_indx],      length + pos), (dLdz * z).imag);
	atomicAdd(dZ_star_real_(in[map_indx], length + pos), (dLdz_star * conj_(z)).real);
	atomicAdd(dZ_star_imag_(in[map_indx], length + pos), (dLdz_star * conj_(z)).imag);
}

void HadamardGpu::hadamard_backward(int label) {
	int total_threads = in_.size() * length() / 2;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_hadamard_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length() / 2, total_threads);
	CNET_KSYNC();
}

// ---- Pad: scatter `length` taps into an `out_length` zero vector ----
// Non-kernel output positions must be re-zeroed every forward (the downstream
// FFT reads the full padded vector, and the arena slot may hold stale data).
__global__ void gpu_pad_zero__(GpuOutVar *out, int out_length, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map_indx = t / out_length;
	int pos = t % out_length;
	*Z_real_(out[map_indx], pos) = 0.0f;
	*Z_imag_(out[map_indx], pos) = 0.0f;
}

__global__ void gpu_pad_forward__(GpuInVar *in, GpuOutVar *out, int *kidx,
		int length, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map_indx = t / length;
	int k = t % length;
	int opos = kidx[k];
	*Z_real_(out[map_indx], opos) = *Z_real_(in[map_indx], k);
	*Z_imag_(out[map_indx], opos) = *Z_imag_(in[map_indx], k);
}

void PadGpu::gpu_pad_forward() {
	int out_len = getOutputLength();          // padded length (e.g. 784)

	int zero_threads = in_.size() * out_len;
	int zero_blocks = (zero_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_pad_zero__ CUDA2(zero_blocks, maxNoThreads) (gpu_out_ptr_, out_len, zero_threads);
	CNET_KSYNC();

	int total_threads = in_.size() * length();  // one thread per tap (length() == 49)
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_pad_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_,
			gpu_kernel_index_, length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_pad_backward__(GpuInVar *in, GpuOutVar *out, int *kidx,
		int length, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map_indx = t / length;
	int k = t % length;
	int opos = kidx[k];
	auto dz      = dZ_(out[map_indx], opos);
	auto dz_star = dZ_star_(out[map_indx], opos);
	atomicAdd(dZ_real_(in[map_indx],      k), dz.real);
	atomicAdd(dZ_imag_(in[map_indx],      k), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], k), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], k), dz_star.imag);
}

void PadGpu::gpu_pad_backward(int label) {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_pad_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_,
			gpu_kernel_index_, length(), total_threads);
	CNET_KSYNC();
}

__device__ const float GPU_GELU_SCALING_FACTOR = 0.7978845608; // sqrtf(2.0f / M_PI);

__device__ inline float gpu_gelu(float xi) {
	float cube = 0.044715f * xi * xi * xi;
	return 0.5f * xi * (1.0f + tanhf(GPU_GELU_SCALING_FACTOR * (xi + cube)));
}

__device__ inline float gpu_dx_gelu(float x) {
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    return 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
}

__global__ void gpu_gelu_forward__(GpuInVar *in, GpuOutVar *out, int no_func,
		int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int func_indx = thread_indx / length;
	int pos = thread_indx % length;

	float *in_fc_start = (in + func_indx)->input_ptr_;
	float *in_pos_real = in_fc_start + pos;
	float *in_pos_imag = in_fc_start + length + pos;

	float *out_fc_start = (out + func_indx)->out_ptr_;
	float *out_pos_real = out_fc_start + pos;
	float *out_pos_imag = out_fc_start + (out + func_indx)->out_length_ + pos;

//	printf("thread_indx=%05d \t func_indx=%05d \t pos=%05d \t re=%f im=%f\n",
//			thread_indx, func_indx, pos, (*in_pos_real), (*in_pos_imag));

	*out_pos_real = gpu_gelu(*in_pos_real);
	*out_pos_imag = gpu_gelu(*in_pos_imag);
}

// gpu_gelu_forward(gpu_in_ptr_, gpu_out_ptr_, in_.size(), length());
void GeluGpu::gpu_gelu_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;

	gpu_gelu_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, in_.size(), length(), total_threads);

	CNET_KSYNC();
}

__global__ void gpu_gelu_backward__ (GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);

	float dx = 0.5 * gpu_dx_gelu(*Z_real_(in[map_indx], pos));
	float dy = 0.5 * gpu_dx_gelu(*Z_imag_(in[map_indx], pos));
	auto dz = dLdz * (dx + dy) + dLdz_star * (dx - dy);
	auto dz_star = dLdz * (dx - dy) +  dLdz_star * (dx + dy);

	atomicAdd(dZ_real_(in[map_indx], pos), dz.real);
	atomicAdd(dZ_imag_(in[map_indx], pos), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), dz_star.imag);

//	printf("thread_indx=%05d \t func_indx=%05d \t pos=%05d \t re=%f im=%f\n",
//			thread_indx, map_indx, pos, dx + dy, dLdz.real);
}

void GeluGpu::gpu_gelu_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_gelu_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads);
	CNET_KSYNC();
}

// ---- complex dropout (inverted; mask = hash(seed, step, element)) ----
// Returns the kept/scaled factor c: 0 if dropped, else 1/keep. Deterministic in
// (seed, step, idx) so forward and backward recompute the same mask.
__device__ inline float gpu_drop_factor(unsigned seed, int step, int idx, float keep) {
	unsigned h = seed ^ 0x9E3779B9u;
	h ^= (unsigned) step * 0x85EBCA6Bu; h = (h ^ (h >> 15)) * 0x2545F491u;
	h ^= (unsigned) idx  * 0xC2B2AE35u; h = (h ^ (h >> 13)) * 0x27D4EB2Fu;
	h ^= h >> 16;
	float u = (h & 0x00FFFFFFu) / 16777216.0f;     // uniform [0,1)
	return (u < keep) ? (1.0f / keep) : 0.0f;
}

__global__ void gpu_dropout_forward__(GpuInVar *in, GpuOutVar *out, int no_func,
		int length, int max_len, unsigned seed, int step, float keep) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) return;
	int func_indx = thread_indx / length;
	int pos = thread_indx % length;
	float c = gpu_drop_factor(seed, step, thread_indx, keep);

	float *in_fc_start  = (in  + func_indx)->input_ptr_;
	float *out_fc_start = (out + func_indx)->out_ptr_;
	int out_len = (out + func_indx)->out_length_;
	out_fc_start[pos]           = in_fc_start[pos]          * c;   // real
	out_fc_start[out_len + pos] = in_fc_start[length + pos] * c;   // imag
}

// When true, DropoutGpu is the identity (keep=1): set around held-out eval /
// inference so dropout is a no-op there, matching the CPU (eval) path. The GPU
// forward otherwise always applies the training mask.
bool g_dropout_infer = false;

void DropoutGpu::gpu_dropout_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	float eff_keep = g_dropout_infer ? 1.0f : keep_;    // keep=1 -> factor 1 -> identity
	gpu_dropout_forward__ CUDA2(no_blocks, maxNoThreads)
		(gpu_in_ptr_, gpu_out_ptr_, in_.size(), length(), total_threads, seed_, step_, eff_keep);
	CNET_KSYNC();
}

__global__ void gpu_dropout_backward__(GpuInVar *in, GpuOutVar *out, int length,
		int max_len, unsigned seed, int step, float keep) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) return;
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	float c = gpu_drop_factor(seed, step, thread_indx, keep);   // same mask as forward

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);
	auto dz      = dLdz      * c;    // out = c * in (c real scalar)
	auto dz_star = dLdz_star * c;

	atomicAdd(dZ_real_(in[map_indx], pos), dz.real);
	atomicAdd(dZ_imag_(in[map_indx], pos), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), dz_star.imag);
}

void DropoutGpu::gpu_dropout_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_dropout_backward__ CUDA2(no_blocks, maxNoThreads)
		(gpu_in_ptr_, gpu_out_ptr_, length(), total_threads, seed_, step_, keep_);
	CNET_KSYNC();
}

// ---- conjugate lag product y_t = x_t * conj(x_{t-tau}) ----
__global__ void gpu_conjlag_forward__(GpuInVar *in, GpuOutVar *out, int length, int max_len, int tau) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) return;
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	if (pos >= tau) {
		cmplx_ a = cmplx(*Z_real_(in[map_indx], pos),       *Z_imag_(in[map_indx], pos));
		cmplx_ b = cmplx(*Z_real_(in[map_indx], pos - tau), *Z_imag_(in[map_indx], pos - tau));
		cmplx_ y = a * conj_(b);
		*Z_real_(out[map_indx], pos) = y.real;
		*Z_imag_(out[map_indx], pos) = y.imag;
	} else {
		*Z_real_(out[map_indx], pos) = 0.0f;
		*Z_imag_(out[map_indx], pos) = 0.0f;
	}
}

void ConjLagGpu::gpu_conjlag_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_conjlag_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads, tau_);
	CNET_KSYNC();
}

__global__ void gpu_conjlag_backward__(GpuInVar *in, GpuOutVar *out, int length, int max_len, int tau) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) return;
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	cmplx_ dz_acc = cmplx(0.0f, 0.0f), dzs_acc = cmplx(0.0f, 0.0f);
	if (pos >= tau) {                                     // role a: output t = pos
		cmplx_ dy  = dZ_(out[map_indx], pos);
		cmplx_ dys = dZ_star_(out[map_indx], pos);
		cmplx_ xlag = cmplx(*Z_real_(in[map_indx], pos - tau), *Z_imag_(in[map_indx], pos - tau));
		dz_acc  = dz_acc  + dy  * conj_(xlag);
		dzs_acc = dzs_acc + dys * xlag;
	}
	if (pos + tau < length) {                             // role b: output t = pos + tau
		cmplx_ dy  = dZ_(out[map_indx], pos + tau);
		cmplx_ dys = dZ_star_(out[map_indx], pos + tau);
		cmplx_ xfwd = cmplx(*Z_real_(in[map_indx], pos + tau), *Z_imag_(in[map_indx], pos + tau));
		dz_acc  = dz_acc  + dys * conj_(xfwd);
		dzs_acc = dzs_acc + dy  * xfwd;
	}
	atomicAdd(dZ_real_(in[map_indx], pos), dz_acc.real);
	atomicAdd(dZ_imag_(in[map_indx], pos), dz_acc.imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), dzs_acc.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), dzs_acc.imag);
}

void ConjLagGpu::gpu_conjlag_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_conjlag_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads, tau_);
	CNET_KSYNC();
}

// ---- segmented (windowed) Fourier / STFT: block-diagonal DFT, window w ----
__device__ inline cmplx_ gpu_root(int k, int w) {   // e^{i 2pi k / w}
	float a = 6.2831853071795864f * ((float) (k % w)) / ((float) w);
	return cmplx(cosf(a), sinf(a));
}

__global__ void gpu_segfourier_forward__(GpuInVar *in, GpuOutVar *out, int length, int max_len, int w) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) return;
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	int b = pos / w, p = pos % w;
	float scale = sqrtf((float) w);
	cmplx_ acc = cmplx(0.0f, 0.0f);
	for (int q = 0; q < w; ++q) {
		cmplx_ x = cmplx(*Z_real_(in[map_indx], b * w + q), *Z_imag_(in[map_indx], b * w + q));
		acc = acc + x * conj_(gpu_root(p * q, w));   // CPU UnityRoots uses e^{-i...}
	}
	*Z_real_(out[map_indx], pos) = acc.real / scale;
	*Z_imag_(out[map_indx], pos) = acc.imag / scale;
}

void SegFourierGpu::gpu_segfourier_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_segfourier_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads, window_);
	CNET_KSYNC();
}

__global__ void gpu_segfourier_backward__(GpuInVar *in, GpuOutVar *out, int length, int max_len, int w) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) return;
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	int b = pos / w, q = pos % w;
	float scale = sqrtf((float) w);
	cmplx_ sum_dz = cmplx(0.0f, 0.0f), sum_star_dz = cmplx(0.0f, 0.0f);
	for (int p = 0; p < w; ++p) {
		cmplx_ dLdz      = dZ_(out[map_indx], b * w + p);
		cmplx_ dLdz_star = dZ_star_(out[map_indx], b * w + p);
		cmplx_ r = conj_(gpu_root(p * q, w));   // CPU UnityRoots uses e^{-i...}
		sum_dz      = sum_dz + dLdz * r;
		sum_star_dz = sum_star_dz + dLdz_star * conj_(r);
	}
	atomicAdd(dZ_real_(in[map_indx], pos), sum_dz.real / scale);
	atomicAdd(dZ_imag_(in[map_indx], pos), sum_dz.imag / scale);
	atomicAdd(dZ_star_real_(in[map_indx], pos), sum_star_dz.real / scale);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), sum_star_dz.imag / scale);
}

void SegFourierGpu::gpu_segfourier_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_segfourier_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads, window_);
	CNET_KSYNC();
}

// ---- HadamardConv: multi-plane spectral conv (see impl/hadamardconv.h) ----
// Input (per clone) is the concatenation [ X : C*D | taps : F*C*K | bias : F ];
// output is F*D. root(m) = e^{i2pi m/D}. All maps are holomorphic so dOut/d(.)* = 0.
__global__ void gpu_hadamardconv_forward__(GpuInVar *in, GpuOutVar *out, int *kidx,
		int C, int D, int F, int K, size_t max_threads) {
	size_t t = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_threads) return;
	int outlen = F * D;
	int m = t / outlen, o = t % outlen, f = o / D, k = o % D;
	int tb = C * D, bb = C * D + F * C * K;
	cmplx_ acc = cmplx(0.0f, 0.0f);
	for (int p = 0; p < C; ++p) {
		cmplx_ W = cmplx(0.0f, 0.0f);
		for (int j = 0; j < K; ++j) {
			int ti = tb + f * C * K + p * K + j;
			cmplx_ tap = cmplx(*Z_real_(in[m], ti), *Z_imag_(in[m], ti));
			W = W + tap * conj_(gpu_root((kidx[j] * k) % D, D));
		}
		cmplx_ x = cmplx(*Z_real_(in[m], p * D + k), *Z_imag_(in[m], p * D + k));
		acc = acc + x * W;
	}
	if (k == 0) {
		cmplx_ b = cmplx(*Z_real_(in[m], bb + f), *Z_imag_(in[m], bb + f));
		acc = acc + cmplx((float) D, 0.0f) * b;
	}
	*Z_real_(out[m], o) = acc.real;
	*Z_imag_(out[m], o) = acc.imag;
}

__global__ void gpu_hadamardconv_backward_x__(GpuInVar *in, GpuOutVar *out, int *kidx,
		int C, int D, int F, int K, size_t max_threads) {
	size_t t = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_threads) return;
	int xlen = C * D;
	int m = t / xlen, idx2 = t % xlen, p = idx2 / D, k = idx2 % D;
	int tb = C * D;
	cmplx_ sdz = cmplx(0.0f, 0.0f), sdzs = cmplx(0.0f, 0.0f);
	for (int f = 0; f < F; ++f) {
		cmplx_ W = cmplx(0.0f, 0.0f);
		for (int j = 0; j < K; ++j) {
			int ti = tb + f * C * K + p * K + j;
			cmplx_ tap = cmplx(*Z_real_(in[m], ti), *Z_imag_(in[m], ti));
			W = W + tap * conj_(gpu_root((kidx[j] * k) % D, D));
		}
		cmplx_ dz  = dZ_(out[m], f * D + k);
		cmplx_ dzs = dZ_star_(out[m], f * D + k);
		sdz  = sdz  + dz  * W;
		sdzs = sdzs + dzs * conj_(W);
	}
	atomicAdd(dZ_real_(in[m], p * D + k), sdz.real);
	atomicAdd(dZ_imag_(in[m], p * D + k), sdz.imag);
	atomicAdd(dZ_star_real_(in[m], p * D + k), sdzs.real);
	atomicAdd(dZ_star_imag_(in[m], p * D + k), sdzs.imag);
}

__global__ void gpu_hadamardconv_backward_taps__(GpuInVar *in, GpuOutVar *out, int *kidx,
		int C, int D, int F, int K, size_t max_threads) {
	size_t t = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_threads) return;
	int plen = F * C * K;
	int m = t / plen, idx2 = t % plen;
	int f = idx2 / (C * K), r = idx2 % (C * K), p = r / K, j = r % K;
	int tb = C * D;
	cmplx_ sdz = cmplx(0.0f, 0.0f), sdzs = cmplx(0.0f, 0.0f);
	for (int k = 0; k < D; ++k) {
		cmplx_ x = cmplx(*Z_real_(in[m], p * D + k), *Z_imag_(in[m], p * D + k));
		cmplx_ xr = x * conj_(gpu_root((kidx[j] * k) % D, D));
		cmplx_ dz  = dZ_(out[m], f * D + k);
		cmplx_ dzs = dZ_star_(out[m], f * D + k);
		sdz  = sdz  + dz  * xr;
		sdzs = sdzs + dzs * conj_(xr);
	}
	int ti = tb + f * C * K + p * K + j;
	atomicAdd(dZ_real_(in[m], ti), sdz.real);
	atomicAdd(dZ_imag_(in[m], ti), sdz.imag);
	atomicAdd(dZ_star_real_(in[m], ti), sdzs.real);
	atomicAdd(dZ_star_imag_(in[m], ti), sdzs.imag);
}

__global__ void gpu_hadamardconv_backward_bias__(GpuInVar *in, GpuOutVar *out,
		int C, int D, int F, int K, size_t max_threads) {
	size_t t = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_threads) return;
	int m = t / F, f = t % F;
	int bb = C * D + F * C * K;
	cmplx_ dz  = dZ_(out[m], f * D + 0);
	cmplx_ dzs = dZ_star_(out[m], f * D + 0);
	atomicAdd(dZ_real_(in[m], bb + f), (float) D * dz.real);
	atomicAdd(dZ_imag_(in[m], bb + f), (float) D * dz.imag);
	atomicAdd(dZ_star_real_(in[m], bb + f), (float) D * dzs.real);
	atomicAdd(dZ_star_imag_(in[m], bb + f), (float) D * dzs.imag);
}

void HadamardConvGpu::gpu_hadamardconv_forward() {
	size_t nmap = in_.size();
	size_t total = nmap * (size_t) F_ * D_;
	unsigned no_blocks = (unsigned) ((total + maxNoThreads - 1) / maxNoThreads);
	gpu_hadamardconv_forward__ CUDA2(no_blocks, maxNoThreads)
		(gpu_in_ptr_, gpu_out_ptr_, gpu_kernel_index_, C_, D_, F_, K_, total);
	CNET_KSYNC();
}

void HadamardConvGpu::gpu_hadamardconv_backward() {
	size_t nmap = in_.size();
	// dL/dX
	{
		size_t total = nmap * (size_t) C_ * D_;
		unsigned nb = (unsigned) ((total + maxNoThreads - 1) / maxNoThreads);
		gpu_hadamardconv_backward_x__ CUDA2(nb, maxNoThreads)
			(gpu_in_ptr_, gpu_out_ptr_, gpu_kernel_index_, C_, D_, F_, K_, total);
	}
	// dL/dtaps
	{
		size_t total = nmap * (size_t) F_ * C_ * K_;
		unsigned nb = (unsigned) ((total + maxNoThreads - 1) / maxNoThreads);
		gpu_hadamardconv_backward_taps__ CUDA2(nb, maxNoThreads)
			(gpu_in_ptr_, gpu_out_ptr_, gpu_kernel_index_, C_, D_, F_, K_, total);
	}
	// dL/dbias
	{
		size_t total = nmap * (size_t) F_;
		unsigned nb = (unsigned) ((total + maxNoThreads - 1) / maxNoThreads);
		gpu_hadamardconv_backward_bias__ CUDA2(nb, maxNoThreads)
			(gpu_in_ptr_, gpu_out_ptr_, C_, D_, F_, K_, total);
	}
	CNET_KSYNC();
}

// ---- |z|^2 magnitude nonlinearity ----
__global__ void gpu_modulus2_forward__(GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	float re = *Z_real_(in[map_indx], pos);
	float im = *Z_imag_(in[map_indx], pos);
	*Z_real_(out[map_indx], pos) = re * re + im * im;
	*Z_imag_(out[map_indx], pos) = 0.0f;
}

void CModulus2Gpu::gpu_modulus2_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_modulus2_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_modulus2_backward__(GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);
	cmplx_ z = cmplx(*Z_real_(in[map_indx], pos), *Z_imag_(in[map_indx], pos));
	cmplx_ g = dLdz + dLdz_star;
	cmplx_ dz      = g * conj_(z);   // dL/dz
	cmplx_ dz_star = g * z;          // dL/dz~

	atomicAdd(dZ_real_(in[map_indx], pos), dz.real);
	atomicAdd(dZ_imag_(in[map_indx], pos), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), dz_star.imag);
}

void CModulus2Gpu::gpu_modulus2_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_modulus2_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads);
	CNET_KSYNC();
}

// ---- mean pooling over time ----
__global__ void gpu_meanpool_forward__(GpuInVar *in, GpuOutVar *out, int P, int width, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / P;
	int p = t % P;
	float sre = 0, sim = 0;
	for (int j = 0; j < width; ++j) {
		sre += *Z_real_(in[map], p * width + j);
		sim += *Z_imag_(in[map], p * width + j);
	}
	*Z_real_(out[map], p) = sre / width;
	*Z_imag_(out[map], p) = sim / width;
}

void MeanPoolGpu::gpu_meanpool_forward() {
	int P = getOutputLength();
	int width = length() / P;
	int total_threads = in_.size() * P;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_meanpool_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, P, width, total_threads);
	CNET_KSYNC();
}

__global__ void gpu_meanpool_backward__(GpuInVar *in, GpuOutVar *out, int L, int width, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / L;
	int i = t % L;
	int p = i / width;
	atomicAdd(dZ_real_(in[map], i),      *dZ_real_(out[map], p) / width);
	atomicAdd(dZ_imag_(in[map], i),      *dZ_imag_(out[map], p) / width);
	atomicAdd(dZ_star_real_(in[map], i), *dZ_star_real_(out[map], p) / width);
	atomicAdd(dZ_star_imag_(in[map], i), *dZ_star_imag_(out[map], p) / width);
}

void MeanPoolGpu::gpu_meanpool_backward() {
	int L = length();
	int width = L / getOutputLength();
	int total_threads = in_.size() * L;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_meanpool_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, L, width, total_threads);
	CNET_KSYNC();
}

// ---- holomorphic power w = z^M ----
__device__ inline cmplx_ gpu_ipow(cmplx_ z, int n) {
	cmplx_ r = cmplx(1.0f, 0.0f);
	for (int i = 0; i < n; ++i) {
		r = r * z;
	}
	return r;
}

__global__ void gpu_power_forward__(GpuInVar *in, GpuOutVar *out, int power, int length, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / length;
	int pos = t % length;
	cmplx_ z = cmplx(*Z_real_(in[map], pos), *Z_imag_(in[map], pos));
	cmplx_ w = gpu_ipow(z, power);
	*Z_real_(out[map], pos) = w.real;
	*Z_imag_(out[map], pos) = w.imag;
}

void CPowerGpu::gpu_power_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_power_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, power_, length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_power_backward__(GpuInVar *in, GpuOutVar *out, int power, int length, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / length;
	int pos = t % length;
	auto dLdz      = dZ_(out[map], pos);
	auto dLdz_star = dZ_star_(out[map], pos);
	cmplx_ z = cmplx(*Z_real_(in[map], pos), *Z_imag_(in[map], pos));
	cmplx_ zpow = gpu_ipow(z, power - 1);          // z^(M-1)
	cmplx_ dz      = dLdz * ((float) power * zpow);
	cmplx_ dz_star = dLdz_star * ((float) power * conj_(zpow));

	atomicAdd(dZ_real_(in[map], pos), dz.real);
	atomicAdd(dZ_imag_(in[map], pos), dz.imag);
	atomicAdd(dZ_star_real_(in[map], pos), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map], pos), dz_star.imag);
}

void CPowerGpu::gpu_power_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_power_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, power_, length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_embedding_forward__(GpuInVar *in, int *tokens, int no_out_tokens, GpuOutVar *out,
		                                int embedding_dim, int map_size, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int current_map = thread_indx / map_size;
	int tok_indx = (thread_indx / embedding_dim) % no_out_tokens;
	int current_token = tokens[in[current_map].tok_offset_ + tok_indx];
	int pos_in_map = thread_indx - map_size * current_map;
	int pos_in_token = pos_in_map % embedding_dim;

//	printf("m=%03d\t pim=%03d\t Tx=%03d \t T=%03d\t piT=%03d\t tid=%03d\t op=%p \t t_off=%d \t %f %f \t %d \t %d \t %d \n",
//			current_map, pos_in_map, tok_indx, current_token, pos_in_token,
//			thread_indx, out[current_map].out_ptr_, in[current_map].t_offset_,
//			current_token < 0 ? -1 : *Z_real_(in[current_map], embedding_dim * current_token + pos_in_token),
//			current_token < 0 ?  0 : *Z_imag_(in[current_map], embedding_dim * current_token + pos_in_token),
//			out[current_map].out_length_, no_out_tokens, embedding_dim);

	if (current_token < 0) {
		*Z_real_(out[current_map], pos_in_map) = 0;
		*Z_imag_(out[current_map], pos_in_map) = 0;
		return;
	}
	// (no __syncthreads here: each thread writes its own output element, there is
	// no shared data, and a barrier after the divergent return above would be
	// undefined behaviour since the token<0 lanes have already exited.)

	*Z_real_(out[current_map], pos_in_map) = *Z_real_(in[current_map], embedding_dim * current_token + pos_in_token);
	*Z_imag_(out[current_map], pos_in_map) = *Z_imag_(in[current_map], embedding_dim * current_token + pos_in_token);
}

void EmbeddingGpu::gpu_embedding_forward() {
	int total_threads = no_out_tokens_ * embedding_dim_ * in_.size();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;

	gpu_embedding_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_tokens_, no_out_tokens_, gpu_out_ptr_,
			embedding_dim_, no_out_tokens_ * embedding_dim_, total_threads);

	CNET_KSYNC();
}

__global__ void gpu_embedding_backward__(GpuInVar *in, int *tokens, int no_out_tokens, GpuOutVar *out,
		                                int embedding_dim, int map_size, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int current_map = thread_indx / map_size;
	int tok_indx = (thread_indx / embedding_dim) % no_out_tokens;
	int current_token = tokens[in[current_map].tok_offset_ + tok_indx];
	int pos_in_map = thread_indx - map_size * current_map;
	int pos_in_token = pos_in_map % embedding_dim;

//	printf("m=%d\t pim=%d\t T=%d\t piT=%d\t tid=%d\t op=%p\n", current_map, pos_in_map, current_token, pos_in_token,
//			thread_indx, out[current_map].out_ptr_);

	if (current_token < 0) {
		return;   // padding token contributes no gradient
	}

	atomicAdd(dZ_star_real_(in[current_map], embedding_dim * current_token + pos_in_token),
			*dZ_star_real_(out[current_map], pos_in_map));
	atomicAdd(dZ_star_imag_(in[current_map], embedding_dim * current_token + pos_in_token),
			*dZ_star_imag_(out[current_map], pos_in_map));
}

void EmbeddingGpu::gpu_embedding_backward(int label) {
	int total_threads = no_out_tokens_ * embedding_dim_ * in_.size();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;

	gpu_embedding_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_tokens_, no_out_tokens_, gpu_out_ptr_,
			embedding_dim_, no_out_tokens_ * embedding_dim_, total_threads);

	CNET_KSYNC();
}

__global__ void gpu_zero_gradients__(GpuInVar *in, int map_length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	int map_indx = thread_indx / map_length;
	int pos = thread_indx - map_indx * map_length;

//	printf("map=%d \t pos=%d \t ptr=%p \n", map_indx, pos, in[map_indx].input_ptr_);

	*dZ_real_(in[map_indx], pos) = 0;
	*dZ_imag_(in[map_indx], pos) = 0;
	*dZ_star_real_(in[map_indx], pos) = 0;
	*dZ_star_imag_(in[map_indx], pos) = 0;
}

void GpuMapping::zeroGradients() {
	int total_threads = getNoMappings() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_zero_gradients__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (gpu_in_ptr_, length(), total_threads);

	CNET_KSYNC();
}

__global__ void gpu_update_input__(GpuInVar in, float l_rate, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

	*Z_real_(in, thread_indx) -= l_rate * (*dZ_star_real_(in, thread_indx));
	*Z_imag_(in, thread_indx) -= l_rate * (*dZ_star_imag_(in, thread_indx));

	*dZ_star_real_(in, thread_indx) = 0;
	*dZ_star_imag_(in, thread_indx) = 0;
}

void gpu_update_input(CFunc *func, float l_rate) {
	int total_threads = func->input().length_;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_update_input__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (func->gpu_var_, l_rate, total_threads);

	CNET_KSYNC();
}

__global__ void gpu_adam_update_input__(GpuInVar in, float l_rate, float beta, int t, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

//	printf("adam %03d \t lr=%f \t b=%f \t %f \t %f \n",
//	      thread_indx, l_rate, beta, *momentum_real_(in, thread_indx), *dZ_star_real_(in, thread_indx));

	// The GPU arena is not zero-initialised, so the momentum buffer is garbage
	// until the first step seeds it. Start the EMA from 0 at t==0.
	float m_real = (t == 0) ? 0.0f : *momentum_real_(in, thread_indx);
	float m_imag = (t == 0) ? 0.0f : *momentum_imag_(in, thread_indx);

	*momentum_real_(in, thread_indx) = m_real * beta
			+ (1 - beta) * (*dZ_star_real_(in, thread_indx));
	*momentum_imag_(in, thread_indx) = m_imag * beta
			+ (1 - beta) * (*dZ_star_imag_(in, thread_indx));

	*Z_real_(in, thread_indx) -= l_rate * (*momentum_real_(in, thread_indx))
								/ (1 - pow(beta, t + 1));
	*Z_imag_(in, thread_indx) -= l_rate * (*momentum_imag_(in, thread_indx))
								/ (1 - pow(beta, t + 1));

	*dZ_star_real_(in, thread_indx) = 0;
	*dZ_star_imag_(in, thread_indx) = 0;
}

void gpu_update_adam_input(CFunc *func, float l_rate, float beta, int t) {
	int total_threads = func->input().length_;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_adam_update_input__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (func->gpu_var_, l_rate, beta, t, total_threads);

	CNET_KSYNC();
}

// True Adam: first moment m (dz slots 2,3) AND second moment v (slots 6,7),
// each with bias correction, and the per-parameter sqrt(v)+eps step. The arena
// is not zero-initialised, so m and v start from 0 at t==0. Treats the real and
// imaginary components as independent real parameters (matches CPU trueAdam).
__global__ void gpu_true_adam_update_input__(GpuInVar in, float l_rate,
		float beta1, float beta2, float eps, int t, float grad_clip, int max_len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= max_len) {
		return;
	}

	float gr = *dZ_star_real_(in, i);
	float gi = *dZ_star_imag_(in, i);

	// Optional clip-by-value: bound each gradient component to +/- grad_clip
	// before the moment update, taming the rare large spikes that otherwise
	// drive training to NaN (e.g. from z^M power features).
	if (grad_clip > 0.0f) {
		gr = fmaxf(-grad_clip, fminf(grad_clip, gr));
		gi = fmaxf(-grad_clip, fminf(grad_clip, gi));
	}

	float m_r = (t == 0) ? 0.0f : *momentum_real_(in, i);
	float m_i = (t == 0) ? 0.0f : *momentum_imag_(in, i);
	float v_r = (t == 0) ? 0.0f : *v_real_(in, i);
	float v_i = (t == 0) ? 0.0f : *v_imag_(in, i);

	m_r = beta1 * m_r + (1 - beta1) * gr;
	m_i = beta1 * m_i + (1 - beta1) * gi;
	v_r = beta2 * v_r + (1 - beta2) * gr * gr;
	v_i = beta2 * v_i + (1 - beta2) * gi * gi;

	*momentum_real_(in, i) = m_r;
	*momentum_imag_(in, i) = m_i;
	*v_real_(in, i) = v_r;
	*v_imag_(in, i) = v_i;

	float bc1 = 1 - pow(beta1, t + 1);
	float bc2 = 1 - pow(beta2, t + 1);
	float mhat_r = m_r / bc1, mhat_i = m_i / bc1;
	float vhat_r = v_r / bc2, vhat_i = v_i / bc2;

	*Z_real_(in, i) -= l_rate * mhat_r / (sqrtf(vhat_r) + eps);
	*Z_imag_(in, i) -= l_rate * mhat_i / (sqrtf(vhat_i) + eps);

	*dZ_star_real_(in, i) = 0;
	*dZ_star_imag_(in, i) = 0;
}

void gpu_update_true_adam_input(CFunc *func, float l_rate, float beta1,
		float beta2, float eps, int t, float grad_clip) {
	int total_threads = func->input().length_;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_true_adam_update_input__ CUDA2(no_blocks, MAX_BLOCK_SIZE)
			(func->gpu_var_, l_rate, beta1, beta2, eps, t, grad_clip, total_threads);

	CNET_KSYNC();
}


__global__ void gpu_copy_to_clones__(GpuCloneVar *in, int max_input_len, size_t total_threads) {
	size_t thread_indx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= total_threads) {
		return;
	}

	int current_input = thread_indx / max_input_len;
	int pos = thread_indx % max_input_len;
	if (pos >= in[current_input].data_len_) {
		return;
	}

//	if (current_input == 0) {
//		printf("ci=%d\t %04d\t maxl=%d\t cdl=%d\t aptr=%p\t c0=%p\t nc=%d\n",
//			current_input, pos, max_input_len, in[current_input].data_len_,
//			in[current_input].ancestor_ptr_, in[current_input].clone_array_ptr_[0],
//			in[current_input].no_clones_);
//	}

	float val = in[current_input].ancestor_ptr_[pos];
	for (int clone_indx = 0; clone_indx < in[current_input].no_clones_; ++clone_indx) {
		*(in[current_input].clone_array_ptr_[clone_indx] + pos) = val;
	}
}

void gpu_copy_to_clones(GpuCloneVar *in, int no_ancestors, int max_input_len) {
	if (no_ancestors == 0 || max_input_len == 0) {
		return;   // batch_size == 1: no clones to broadcast to
	}
	int len = max_input_len
		+ (MAX_BLOCK_SIZE - max_input_len % MAX_BLOCK_SIZE) % MAX_BLOCK_SIZE;

	size_t total_threads = (size_t) len * no_ancestors;   // 64-bit: avoid int32 overflow at large channel counts
	unsigned no_blocks = (unsigned) ((total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE);

	gpu_copy_to_clones__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (in, len, total_threads);

	CNET_KSYNC();
}

__global__ void gpu_grad_from_clones__(GpuCloneVar *in, int max_input_len, size_t total_threads) {
	size_t thread_indx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= total_threads) {
		return;
	}

	int current_input = thread_indx / max_input_len;
	int pos = thread_indx % max_input_len;

	// in[current_input].data_len_ is set to 4 * VarIn.length, for dZ_star we need half.
	if (pos >= in[current_input].data_len_ / 2) {
		return;
	}

	int start_pos = in[current_input].data_len_; // grad start pos
	for (int clone_indx = 0; clone_indx < in[current_input].no_clones_; ++clone_indx) {
		float *clone_grad = in[current_input].clone_array_ptr_[clone_indx] + start_pos + pos;
		atomicAdd(in[current_input].ancestor_ptr_ + start_pos + pos, *clone_grad);
		// Reset the clone gradient after folding it into the ancestor. Clones are
		// never touched by the optimizer (only the ancestor is updated+zeroed), and
		// backward accumulates with atomicAdd, so without this the clone gradients
		// would pile up across steps and the ancestor would receive a running sum.
		*clone_grad = 0.0f;
	}
}

void gpu_grad_from_clones(GpuCloneVar *in, int no_ancestors, int max_input_len) {
	int len = max_input_len
		+ (MAX_BLOCK_SIZE - max_input_len % MAX_BLOCK_SIZE) % MAX_BLOCK_SIZE;

	size_t total_threads = (size_t) len * no_ancestors;   // 64-bit: avoid int32 overflow at large channel counts
	unsigned no_blocks = (unsigned) ((total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE);

	gpu_grad_from_clones__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (in, len, total_threads);

	CNET_KSYNC();
}

