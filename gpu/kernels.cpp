#include <cmath>
#include <cstdint>
#include <vector>
#include "kernels.h"
#include "gpumapping.h"
#include "../impl/slice.h"
#include "../impl/melpower.h"
#include "../impl/tfconv.h"
#include "../impl/interleave.h"

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

// ---- slice out[i] = in[start + i] (see Slice in impl/slice.h); one start per map ----
__global__ void gpu_slice_forward__(GpuInVar *in, GpuOutVar *out, int P, const int *starts, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / P;
	int i = t % P;
	int start = starts[map];
	*Z_real_(out[map], i) = *Z_real_(in[map], start + i);
	*Z_imag_(out[map], i) = *Z_imag_(in[map], start + i);
}

SliceGpu::~SliceGpu() {
#ifdef __CUDACC__
	if (gpu_starts_) { cudaFree(gpu_starts_); gpu_starts_ = NULL; }
#endif
}

// The mapped CPU layers (ancestor and clones of every Slice in the block) are known only after the
// mapping is built, so the per-map starts are uploaded on first use.
void SliceGpu::ensureStarts() {
	const int n = (int) in_.size();
	if (gpu_starts_ && n_starts_ == n) return;
	std::vector<int> st(n);
	auto &funs = getCpuFun();
	for (int i = 0; i < n; ++i) st[i] = ((Slice*) funs[i])->start();
#ifdef __CUDACC__
	// Owned by the mapping (a GpuHelper frees its allocations when it goes out of scope).
	if (gpu_starts_) { cudaFree(gpu_starts_); gpu_starts_ = NULL; }
	gpuErrchk(cudaMalloc((void**) &gpu_starts_, sizeof(int) * (size_t) n));
	gpuErrchk(cudaMemcpy(gpu_starts_, st.data(), sizeof(int) * (size_t) n, cudaMemcpyHostToDevice));
#endif
	n_starts_ = n;
}

void SliceGpu::gpu_slice_forward() {
	ensureStarts();
	int P = getOutputLength();
	int total_threads = in_.size() * P;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_slice_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, P, gpu_starts_, total_threads);
	CNET_KSYNC();
}

__global__ void gpu_slice_backward__(GpuInVar *in, GpuOutVar *out, int P, const int *starts, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / P;
	int i = t % P;
	int start = starts[map];
	atomicAdd(dZ_real_(in[map], start + i),      *dZ_real_(out[map], i));
	atomicAdd(dZ_imag_(in[map], start + i),      *dZ_imag_(out[map], i));
	atomicAdd(dZ_star_real_(in[map], start + i), *dZ_star_real_(out[map], i));
	atomicAdd(dZ_star_imag_(in[map], start + i), *dZ_star_imag_(out[map], i));
}

void SliceGpu::gpu_slice_backward() {
	ensureStarts();
	int P = getOutputLength();
	int total_threads = in_.size() * P;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_slice_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, P, gpu_starts_, total_threads);
	CNET_KSYNC();
}

// ---- compressed spectral loss (see CompressedL2 in impl/compressed_l2.h) ----
// Input [Y (M) | T (M)], target S = -T. One block of kCl2Threads threads reduces one chunk of one map.
static const int kCl2Threads = 256;
static const float kCl2Eps = 1e-8f;

__global__ void gpu_cl2_zero__(GpuOutVar *out, int n) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t < n) { out[t].reduce_real_ = 0.f; out[t].reduce_imag_ = 0.f; }
}

__global__ void gpu_cl2_forward__(GpuInVar *in, GpuOutVar *out, int M, int chunks, float c, float beta, float scale) {
	__shared__ float sh[256];
	int map = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
	int k = chunk * blockDim.x + threadIdx.x;
	float v = 0.f;
	if (k < M) {
		float yr = *Z_real_(in[map], k), yi = *Z_imag_(in[map], k);
		float sr = -*Z_real_(in[map], M + k), si = -*Z_imag_(in[map], M + k);
		float a = sqrtf(yr * yr + yi * yi + kCl2Eps), b = sqrtf(sr * sr + si * si + kCl2Eps);
		float pa = powf(a, c - 1.f), pb = powf(b, c - 1.f);
		float dr = pa * yr - pb * sr, di = pa * yi - pb * si, dm = pa * a - pb * b;
		v = scale * (beta * (dr * dr + di * di) + (1.f - beta) * dm * dm);
	}
	sh[threadIdx.x] = v;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(&out[map].reduce_real_, sh[0]);
}

void CompressedL2Gpu::gpu_cl2_forward() {
	const int n = (int) in_.size(), M = length() / 2, chunks = (M + kCl2Threads - 1) / kCl2Threads;
	gpu_cl2_zero__ CUDA2((n + kCl2Threads - 1) / kCl2Threads, kCl2Threads) (gpu_out_ptr_, n);
	CNET_KSYNC();
	gpu_cl2_forward__ CUDA2(n * chunks, kCl2Threads) (gpu_in_ptr_, gpu_out_ptr_, M, chunks, c_, beta_, scale_);
	CNET_KSYNC();
}

// dL/dz~ with respect to the first element of the pair (y; s is the target), see the CPU header.
__device__ inline cmplx_ gpu_cl2_grad(cmplx_ y, cmplx_ s, float c, float beta) {
	float y2 = y.real * y.real + y.imag * y.imag, s2 = s.real * s.real + s.imag * s.imag;
	float a = sqrtf(y2 + kCl2Eps), b = sqrtf(s2 + kCl2Eps);
	float pa = powf(a, c - 1.f), pb = powf(b, c - 1.f), pa3 = pa / (a * a);
	cmplx_ d = cmplx(pa * y.real - pb * s.real, pa * y.imag - pb * s.imag);
	float k = 0.5f * (c - 1.f);
	float r1 = pa + k * y2 * pa3;
	cmplx_ yy = y * y;
	cmplx_ g1 = cmplx(d.real * r1, d.imag * r1) + conj_(d) * yy * (k * pa3);
	float m2 = c * (pa * a - pb * b) * pa / a;
	return cmplx(beta * g1.real + (1.f - beta) * m2 * y.real, beta * g1.imag + (1.f - beta) * m2 * y.imag);
}

__global__ void gpu_cl2_backward__(GpuInVar *in, GpuOutVar *out, int M, float c, float beta, float scale, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) return;
	int map = t / M, k = t % M;
	cmplx_ y = cmplx(*Z_real_(in[map], k), *Z_imag_(in[map], k));
	cmplx_ s = cmplx(-*Z_real_(in[map], M + k), -*Z_imag_(in[map], M + k));
	cmplx_ gy = gpu_cl2_grad(y, s, c, beta) * scale;
	cmplx_ gs = gpu_cl2_grad(s, y, c, beta) * scale;  // w.r.t. s; T = -s flips the sign
	atomicAdd(dZ_star_real_(in[map], k), gy.real);
	atomicAdd(dZ_star_imag_(in[map], k), gy.imag);
	atomicAdd(dZ_real_(in[map], k), gy.real);
	atomicAdd(dZ_imag_(in[map], k), -gy.imag);
	atomicAdd(dZ_star_real_(in[map], M + k), -gs.real);
	atomicAdd(dZ_star_imag_(in[map], M + k), -gs.imag);
	atomicAdd(dZ_real_(in[map], M + k), -gs.real);
	atomicAdd(dZ_imag_(in[map], M + k), gs.imag);
}

void CompressedL2Gpu::gpu_cl2_backward() {
	const int M = length() / 2;
	int total_threads = in_.size() * M;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_cl2_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, M, c_, beta_, scale_, total_threads);
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

// ---- log power log(eps + |z|^2) ----
__global__ void gpu_logpower_forward__(GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;
	float re = *Z_real_(in[map_indx], pos);
	float im = *Z_imag_(in[map_indx], pos);
	*Z_real_(out[map_indx], pos) = logf(kLogPowerEps + re * re + im * im);
	*Z_imag_(out[map_indx], pos) = 0.0f;
}

void CLogPowerGpu::gpu_logpower_forward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_logpower_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads);
	CNET_KSYNC();
}

__global__ void gpu_logpower_backward__(GpuInVar *in, GpuOutVar *out, int length, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}
	int map_indx = thread_indx / length;
	int pos = thread_indx % length;

	auto dLdz      = dZ_(out[map_indx], pos);
	auto dLdz_star = dZ_star_(out[map_indx], pos);
	float re = *Z_real_(in[map_indx], pos), im = *Z_imag_(in[map_indx], pos);
	cmplx_ z = cmplx(re, im);
	float s = 1.0f / (kLogPowerEps + re * re + im * im);
	cmplx_ g = dLdz + dLdz_star;
	g = cmplx(g.real * s, g.imag * s);
	cmplx_ dz      = g * conj_(z);   // dL/dz
	cmplx_ dz_star = g * z;          // dL/dz~

	atomicAdd(dZ_real_(in[map_indx], pos), dz.real);
	atomicAdd(dZ_imag_(in[map_indx], pos), dz.imag);
	atomicAdd(dZ_star_real_(in[map_indx], pos), dz_star.real);
	atomicAdd(dZ_star_imag_(in[map_indx], pos), dz_star.imag);
}

void CLogPowerGpu::gpu_logpower_backward() {
	int total_threads = in_.size() * length();
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_logpower_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, length(), total_threads);
	CNET_KSYNC();
}

// ---- mean over tokens, per dimension ----
__global__ void gpu_tokenmeanpool_forward__(GpuInVar *in, GpuOutVar *out, int P, int N, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / P;
	int p = t % P;
	float sre = 0, sim = 0;
	for (int k = 0; k < N; ++k) {
		sre += *Z_real_(in[map], k * P + p);
		sim += *Z_imag_(in[map], k * P + p);
	}
	*Z_real_(out[map], p) = sre / N;
	*Z_imag_(out[map], p) = sim / N;
}

void TokenMeanPoolGpu::gpu_tokenmeanpool_forward() {
	int P = getOutputLength();
	int N = length() / P;
	int total_threads = in_.size() * P;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_tokenmeanpool_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, P, N, total_threads);
	CNET_KSYNC();
}

__global__ void gpu_tokenmeanpool_backward__(GpuInVar *in, GpuOutVar *out, int L, int P, int N, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / L;
	int i = t % L;
	int p = i % P;
	atomicAdd(dZ_real_(in[map], i),      *dZ_real_(out[map], p) / N);
	atomicAdd(dZ_imag_(in[map], i),      *dZ_imag_(out[map], p) / N);
	atomicAdd(dZ_star_real_(in[map], i), *dZ_star_real_(out[map], p) / N);
	atomicAdd(dZ_star_imag_(in[map], i), *dZ_star_imag_(out[map], p) / N);
}

void TokenMeanPoolGpu::gpu_tokenmeanpool_backward() {
	int L = length();
	int P = getOutputLength();
	int total_threads = in_.size() * L;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	gpu_tokenmeanpool_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, L, P, L / P, total_threads);
	CNET_KSYNC();
}

// ---- fixed log-mel front end ----
__global__ void gpu_melpower_forward__(GpuInVar *in, GpuOutVar *out, const float *H, const int *lo, const int *hi,
		int N, int W, int M, int K, float mean, float scale, int max_len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= max_len) {
		return;
	}
	int map = idx / (N * M);
	int o = idx % (N * M);
	int t = o / M, m = o % M;
	float P = 0.f;
	for (int k = lo[m]; k <= hi[m]; ++k) {
		float re = *Z_real_(in[map], t * W + k), im = *Z_imag_(in[map], t * W + k);
		P += H[m * K + k] * (re * re + im * im);
	}
	*Z_real_(out[map], o) = (logf(kLogPowerEps + P) - mean) / scale;
	*Z_imag_(out[map], o) = 0.f;
}

__global__ void gpu_melpower_backward__(GpuInVar *in, GpuOutVar *out, const float *H, const int *mlo, const int *mhi,
		int N, int W, int M, int K, float mean, float scale, int max_len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= max_len) {
		return;
	}
	int map = idx / (N * K);
	int r = idx % (N * K);
	int t = r / K, k = r % K;
	if (mhi[k] < 0) {
		return;
	}
	cmplx_ s = cmplx(0.f, 0.f);
	for (int m = mlo[k]; m <= mhi[k]; ++m) {
		float h = H[m * K + k];
		if (h == 0.f) continue;
		int o = t * M + m;
		float inv = expf(-(*Z_real_(out[map], o) * scale + mean));   // 1 / (eps + P_m)
		cmplx_ g = dZ_(out[map], o) + dZ_star_(out[map], o);
		float w = h * inv / scale;
		s = s + cmplx(g.real * w, g.imag * w);
	}
	int pos = t * W + k;
	cmplx_ z = cmplx(*Z_real_(in[map], pos), *Z_imag_(in[map], pos));
	cmplx_ dz = s * conj_(z);
	cmplx_ dzs = s * z;
	atomicAdd(dZ_real_(in[map], pos), dz.real);
	atomicAdd(dZ_imag_(in[map], pos), dz.imag);
	atomicAdd(dZ_star_real_(in[map], pos), dzs.real);
	atomicAdd(dZ_star_imag_(in[map], pos), dzs.imag);
}

void MelPowerGpu::ensureFilters() {
	if (gpu_H_) {
		return;
	}
#ifdef __CUDACC__
	const MelPower *mel = (const MelPower*) getCpuFun().front();
	const int M = mel->bands(), K = mel->bins();
	gpuErrchk(cudaMalloc((void**) &gpu_H_, sizeof(float) * (size_t) M * K));
	gpuErrchk(cudaMemcpy(gpu_H_, mel->filters().data(), sizeof(float) * (size_t) M * K, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &gpu_lo_, sizeof(int) * (size_t) M));
	gpuErrchk(cudaMemcpy(gpu_lo_, mel->bandLo().data(), sizeof(int) * (size_t) M, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &gpu_hi_, sizeof(int) * (size_t) M));
	gpuErrchk(cudaMemcpy(gpu_hi_, mel->bandHi().data(), sizeof(int) * (size_t) M, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &gpu_mlo_, sizeof(int) * (size_t) K));
	gpuErrchk(cudaMemcpy(gpu_mlo_, mel->binLo().data(), sizeof(int) * (size_t) K, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &gpu_mhi_, sizeof(int) * (size_t) K));
	gpuErrchk(cudaMemcpy(gpu_mhi_, mel->binHi().data(), sizeof(int) * (size_t) K, cudaMemcpyHostToDevice));
#endif
}

MelPowerGpu::~MelPowerGpu() {
#ifdef __CUDACC__
	if (gpu_H_) cudaFree(gpu_H_);
	if (gpu_lo_) cudaFree(gpu_lo_);
	if (gpu_hi_) cudaFree(gpu_hi_);
	if (gpu_mlo_) cudaFree(gpu_mlo_);
	if (gpu_mhi_) cudaFree(gpu_mhi_);
#endif
	gpu_H_ = NULL;
	gpu_lo_ = gpu_hi_ = gpu_mlo_ = gpu_mhi_ = NULL;
}

void MelPowerGpu::gpu_melpower_forward() {
	ensureFilters();
	const MelPower *mel = (const MelPower*) getCpuFun().front();
	const int N = mel->tokens(), W = mel->window(), M = mel->bands(), K = mel->bins();
	int total_threads = in_.size() * N * M;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_melpower_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, gpu_H_, gpu_lo_, gpu_hi_,
			N, W, M, K, mel->mean(), mel->scale(), total_threads);
	CNET_KSYNC();
}

void MelPowerGpu::gpu_melpower_backward() {
	ensureFilters();
	const MelPower *mel = (const MelPower*) getCpuFun().front();
	const int N = mel->tokens(), W = mel->window(), M = mel->bands(), K = mel->bins();
	int total_threads = in_.size() * N * K;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_melpower_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, gpu_H_, gpu_mlo_, gpu_mhi_,
			N, W, M, K, mel->mean(), mel->scale(), total_threads);
	CNET_KSYNC();
}

// ---- max-magnitude token, per dimension ----
__global__ void gpu_tokenmaxpool_forward__(GpuInVar *in, GpuOutVar *out, int *arg, int P, int N, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / P;
	int p = t % P;
	float best = -1.f;
	int bt = 0;
	for (int k = 0; k < N; ++k) {
		float re = *Z_real_(in[map], k * P + p), im = *Z_imag_(in[map], k * P + p);
		float m = re * re + im * im;
		if (m > best) {   // first maximum on ties, as on the CPU
			best = m;
			bt = k;
		}
	}
	arg[t] = bt;
	*Z_real_(out[map], p) = *Z_real_(in[map], bt * P + p);
	*Z_imag_(out[map], p) = *Z_imag_(in[map], bt * P + p);
}

__global__ void gpu_tokenmaxpool_backward__(GpuInVar *in, GpuOutVar *out, const int *arg, int P, int max_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= max_len) {
		return;
	}
	int map = t / P;
	int p = t % P;
	int i = arg[t] * P + p;
	atomicAdd(dZ_real_(in[map], i),      *dZ_real_(out[map], p));
	atomicAdd(dZ_imag_(in[map], i),      *dZ_imag_(out[map], p));
	atomicAdd(dZ_star_real_(in[map], i), *dZ_star_real_(out[map], p));
	atomicAdd(dZ_star_imag_(in[map], i), *dZ_star_imag_(out[map], p));
}

void TokenMaxPoolGpu::ensureArg() {
	const int n = (int) in_.size() * getOutputLength();
	if (gpu_arg_ && n_arg_ == n) {
		return;
	}
#ifdef __CUDACC__
	if (gpu_arg_) {
		cudaFree(gpu_arg_);
		gpu_arg_ = NULL;
	}
	gpuErrchk(cudaMalloc((void**) &gpu_arg_, sizeof(int) * (size_t) n));
	gpuErrchk(cudaMemset(gpu_arg_, 0, sizeof(int) * (size_t) n));
#endif
	n_arg_ = n;
}

TokenMaxPoolGpu::~TokenMaxPoolGpu() {
#ifdef __CUDACC__
	if (gpu_arg_) {
		cudaFree(gpu_arg_);
	}
#endif
	gpu_arg_ = NULL;
}

void TokenMaxPoolGpu::gpu_tokenmaxpool_forward() {
	ensureArg();
	int P = getOutputLength();
	int N = length() / P;
	int total_threads = in_.size() * P;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_tokenmaxpool_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, gpu_arg_, P, N, total_threads);
	CNET_KSYNC();
}

void TokenMaxPoolGpu::gpu_tokenmaxpool_backward() {
	ensureArg();
	int P = getOutputLength();
	int total_threads = in_.size() * P;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_tokenmaxpool_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, gpu_arg_, P, total_threads);
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
		float beta1, float beta2, float eps, int t, float grad_clip, int max_len, int *nonfinite) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= max_len) {
		return;
	}

	float gr = *dZ_star_real_(in, i);
	float gi = *dZ_star_imag_(in, i);

	// A non-finite gradient component (or one whose square overflows, which would make v infinite for the rest of
	// the run) is dropped (treated as 0) and counted, instead of being turned into +grad_clip by fminf/fmaxf below
	// (which silently locked a run into a constant drift).
	const bool bad_r = !cnet_finite(gr * gr), bad_i = !cnet_finite(gi * gi);
	if (bad_r || bad_i) {
		if (nonfinite) atomicAdd(nonfinite, (int) bad_r + (int) bad_i);   // components dropped
		gr = bad_r ? 0.0f : gr;
		gi = bad_i ? 0.0f : gi;
	}

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

// One device counter per GPU, allocated on first use on the then-current device (the data-parallel replicas set
// their device before every call, so a counter is never dereferenced from another GPU): non-finite gradient
// components dropped by the Adam update.
static int *nonfiniteCounter(bool create) {
#ifdef __CUDACC__
	static int *counters[64] = {NULL};
	int dev = 0;
	if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return NULL;
	if (!counters[dev] && create) {
		gpuErrchk(cudaMalloc((void**) &counters[dev], sizeof(int)));
		gpuErrchk(cudaMemset(counters[dev], 0, sizeof(int)));
	}
	return counters[dev];
#else
	return NULL;
#endif
}

void gpu_update_true_adam_input(CFunc *func, float l_rate, float beta1,
		float beta2, float eps, int t, float grad_clip) {
	int *counter = nonfiniteCounter(true);
	int total_threads = func->input().length_;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_true_adam_update_input__ CUDA2(no_blocks, MAX_BLOCK_SIZE)
			(func->gpu_var_, l_rate, beta1, beta2, eps, t, grad_clip, total_threads, counter);

	CNET_KSYNC();
}

// Number of non-finite gradient components dropped by the Adam update on the current device since the last call
// (then zeroed).
int gpu_nonfinite_gradients_reset() {
	int n = 0;
#ifdef __CUDACC__
	int *counter = nonfiniteCounter(false);
	if (counter) {
		gpuErrchk(cudaMemcpy(&n, counter, sizeof(int), cudaMemcpyDeviceToHost));
		if (n) gpuErrchk(cudaMemset(counter, 0, sizeof(int)));
	}
#endif
	return n;
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


// ---- causal time-frequency convolution (see TimeFreqConv in impl/tfconv.h) ----
// Layout as on the CPU: data (t, b, c) at (t B + b) Ci + c; tap (o, c, tau, j) at base + ((o Ci + c) T + tau) F + j;
// output (t, bo, o) at (t Bo + bo) Co + o, reading data(t - tau, bo S + j - F/2, c) (zero outside the plane).
__global__ void gpu_tfconv_forward__(GpuInVar *in, GpuOutVar *out, int N, int B, int Ci, int Co, int T, int F, int S,
		int Bo, int base, int max_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= max_len) {
		return;
	}
	const int per_map = N * Bo * Co;
	const int map = tid / per_map, oi = tid % per_map;
	const int o = oi % Co, bo = (oi / Co) % Bo, t = oi / (Co * Bo);
	float sr = 0.f, si = 0.f;
	for (int tau = 0; tau < T; ++tau) {
		const int ts = t - tau;
		if (ts < 0) continue;
		for (int j = 0; j < F; ++j) {
			const int b = bo * S + j - F / 2;
			if (b < 0 || b >= B) continue;
			for (int c = 0; c < Ci; ++c) {
				const int di = (ts * B + b) * Ci + c, wi = base + ((o * Ci + c) * T + tau) * F + j;
				const float wr = *Z_real_(in[map], wi), wm = *Z_imag_(in[map], wi);
				const float dr = *Z_real_(in[map], di), dm = *Z_imag_(in[map], di);
				sr += wr * dr - wm * dm;
				si += wr * dm + wm * dr;
			}
		}
	}
	*Z_real_(out[map], oi) = sr;
	*Z_imag_(out[map], oi) = si;
}

// dL/d data(ts, b, c): gathers every output that read it: dz += dLo_z * tap, dz~ += dLo_z~ * conj(tap).
__global__ void gpu_tfconv_backward_data__(GpuInVar *in, GpuOutVar *out, int N, int B, int Ci, int Co, int T, int F,
		int S, int Bo, int base, int max_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= max_len) {
		return;
	}
	const int per_map = N * B * Ci;
	const int map = tid / per_map, di = tid % per_map;
	const int c = di % Ci, b = (di / Ci) % B, ts = di / (Ci * B);
	float gr = 0.f, gi = 0.f, sr = 0.f, si = 0.f;
	for (int tau = 0; tau < T; ++tau) {
		const int t = ts + tau;
		if (t >= N) break;
		for (int j = 0; j < F; ++j) {
			const int num = b - j + F / 2;                    // = bo S for the output bin that reads b through tap j
			if (num < 0 || num % S != 0) continue;
			const int bo = num / S;
			if (bo >= Bo) continue;
			for (int o = 0; o < Co; ++o) {
				const int oi = (t * Bo + bo) * Co + o, wi = base + ((o * Ci + c) * T + tau) * F + j;
				const float wr = *Z_real_(in[map], wi), wm = *Z_imag_(in[map], wi);
				const float ar = *dZ_real_(out[map], oi), ai = *dZ_imag_(out[map], oi);
				const float br = *dZ_star_real_(out[map], oi), bi = *dZ_star_imag_(out[map], oi);
				gr += ar * wr - ai * wm;
				gi += ar * wm + ai * wr;
				sr += br * wr + bi * wm;
				si += bi * wr - br * wm;
			}
		}
	}
	atomicAdd(dZ_real_(in[map], di), gr);        // a layer with several consumers is mapped twice on the same input
	atomicAdd(dZ_imag_(in[map], di), gi);        // (gpu_data.h addGpuOuts), so the accumulation must be atomic
	atomicAdd(dZ_star_real_(in[map], di), sr);
	atomicAdd(dZ_star_imag_(in[map], di), si);
}

// dL/d tap(o, c, tau, j): summed over every output position: dz += dLo_z * data, dz~ += dLo_z~ * conj(data).
__global__ void gpu_tfconv_backward_taps__(GpuInVar *in, GpuOutVar *out, int N, int B, int Ci, int Co, int T, int F,
		int S, int Bo, int base, int max_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= max_len) {
		return;
	}
	const int n_taps = Co * Ci * T * F;
	const int map = tid / n_taps, k = tid % n_taps;
	const int j = k % F, tau = (k / F) % T, c = (k / (F * T)) % Ci, o = k / (F * T * Ci);
	float gr = 0.f, gi = 0.f, sr = 0.f, si = 0.f;
	for (int t = tau; t < N; ++t) {
		const int ts = t - tau;
		for (int bo = 0; bo < Bo; ++bo) {
			const int b = bo * S + j - F / 2;
			if (b < 0 || b >= B) continue;
			const int oi = (t * Bo + bo) * Co + o, di = (ts * B + b) * Ci + c;
			const float dr = *Z_real_(in[map], di), dm = *Z_imag_(in[map], di);
			const float ar = *dZ_real_(out[map], oi), ai = *dZ_imag_(out[map], oi);
			const float br = *dZ_star_real_(out[map], oi), bi = *dZ_star_imag_(out[map], oi);
			gr += ar * dr - ai * dm;
			gi += ar * dm + ai * dr;
			sr += br * dr + bi * dm;
			si += bi * dr - br * dm;
		}
	}
	atomicAdd(dZ_real_(in[map], base + k), gr);
	atomicAdd(dZ_imag_(in[map], base + k), gi);
	atomicAdd(dZ_star_real_(in[map], base + k), sr);
	atomicAdd(dZ_star_imag_(in[map], base + k), si);
}

void TimeFreqConvGpu::gpu_tfconv_forward() {
	const TimeFreqConv *c = (const TimeFreqConv*) getCpuFun().front();
	const int N = c->nTokens(), B = c->bins(), Ci = c->cIn(), Co = c->cOut(), T = c->tTaps(), F = c->fTaps(), S = c->stride();
	const int Bo = c->outBins(), base = c->tapBase();
	int total_threads = in_.size() * N * Bo * Co;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_tfconv_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, N, B, Ci, Co, T, F, S, Bo, base, total_threads);
	CNET_KSYNC();
}

void TimeFreqConvGpu::gpu_tfconv_backward() {
	const TimeFreqConv *c = (const TimeFreqConv*) getCpuFun().front();
	const int N = c->nTokens(), B = c->bins(), Ci = c->cIn(), Co = c->cOut(), T = c->tTaps(), F = c->fTaps(), S = c->stride();
	const int Bo = c->outBins(), base = c->tapBase();
	int total_threads = in_.size() * N * B * Ci;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_tfconv_backward_data__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, N, B, Ci, Co, T, F, S, Bo, base, total_threads);
	CNET_KSYNC();
	total_threads = in_.size() * c->nTaps();
	no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_tfconv_backward_taps__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, N, B, Ci, Co, T, F, S, Bo, base, total_threads);
	CNET_KSYNC();
}

// ---- interleave K inputs of length L into per-position channels (see Interleave in impl/interleave.h) ----
__global__ void gpu_interleave_forward__(GpuInVar *in, GpuOutVar *out, int L, int K, int max_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= max_len) {
		return;
	}
	const int per_map = L * K;
	const int map = tid / per_map, oi = tid % per_map;
	const int i = oi / K, k = oi % K;
	*Z_real_(out[map], oi) = *Z_real_(in[map], k * L + i);
	*Z_imag_(out[map], oi) = *Z_imag_(in[map], k * L + i);
}

__global__ void gpu_interleave_backward__(GpuInVar *in, GpuOutVar *out, int L, int K, int max_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= max_len) {
		return;
	}
	const int per_map = L * K;
	const int map = tid / per_map, oi = tid % per_map;
	const int i = oi / K, k = oi % K, src = k * L + i;   // one output per input position within a map; atomic because a
	atomicAdd(dZ_real_(in[map], src),      *dZ_real_(out[map], oi));        // layer with several consumers is mapped twice
	atomicAdd(dZ_imag_(in[map], src),      *dZ_imag_(out[map], oi));
	atomicAdd(dZ_star_real_(in[map], src), *dZ_star_real_(out[map], oi));
	atomicAdd(dZ_star_imag_(in[map], src), *dZ_star_imag_(out[map], oi));
}

void InterleaveGpu::gpu_interleave_forward() {
	const Interleave *il = (const Interleave*) getCpuFun().front();
	const int L = il->length(), K = il->k();
	int total_threads = in_.size() * L * K;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_interleave_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, L, K, total_threads);
	CNET_KSYNC();
}

void InterleaveGpu::gpu_interleave_backward() {
	const Interleave *il = (const Interleave*) getCpuFun().front();
	const int L = il->length(), K = il->k();
	int total_threads = in_.size() * L * K;
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;
	gpu_interleave_backward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_out_ptr_, L, K, total_threads);
	CNET_KSYNC();
}
