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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
    __syncthreads();


	*Z_real_(out[current_map], pos_in_map) = *Z_real_(in[current_map], embedding_dim * current_token + pos_in_token);
	*Z_imag_(out[current_map], pos_in_map) = *Z_imag_(in[current_map], embedding_dim * current_token + pos_in_token);
}

void EmbeddingGpu::gpu_embedding_forward() {
	int total_threads = no_out_tokens_ * embedding_dim_ * in_.size();
	int no_blocks = (total_threads + maxNoThreads - 1) / maxNoThreads;

	gpu_embedding_forward__ CUDA2(no_blocks, maxNoThreads) (gpu_in_ptr_, gpu_tokens_, no_out_tokens_, gpu_out_ptr_,
			embedding_dim_, no_out_tokens_ * embedding_dim_, total_threads);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
		*(out[current_map].out_ptr_ + pos_in_map) = 0;
		*(out[current_map].out_ptr_ + pos_in_map + out[current_map].out_length_) = 0;
		return;
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__global__ void gpu_adam_update_input__(GpuInVar in, float l_rate, float beta, int t, int max_len) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_indx >= max_len) {
		return;
	}

//	printf("adam %03d \t lr=%f \t b=%f \t %f \t %f \n",
//	      thread_indx, l_rate, beta, *momentum_real_(in, thread_indx), *dZ_star_real_(in, thread_indx));

	*momentum_real_(in, thread_indx) = *momentum_real_(in, thread_indx) * beta
			+ (1 - beta) * (*dZ_star_real_(in, thread_indx));
	*momentum_imag_(in, thread_indx) = *momentum_imag_(in, thread_indx) * beta
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

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__global__ void gpu_copy_to_clones__(GpuCloneVar *in, int max_input_len, int total_threads) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
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
	int len = max_input_len
		+ (MAX_BLOCK_SIZE - max_input_len % MAX_BLOCK_SIZE) % MAX_BLOCK_SIZE;

	int total_threads = len * no_ancestors;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_copy_to_clones__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (in, len, total_threads);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__global__ void gpu_grad_from_clones__(GpuCloneVar *in, int max_input_len, int total_threads) {
	int thread_indx = blockIdx.x * blockDim.x + threadIdx.x;
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
		atomicAdd(in[current_input].ancestor_ptr_ + start_pos + pos,
				*(in[current_input].clone_array_ptr_[clone_indx] + start_pos + pos));
	}
}

void gpu_grad_from_clones(GpuCloneVar *in, int no_ancestors, int max_input_len) {
	int len = max_input_len
		+ (MAX_BLOCK_SIZE - max_input_len % MAX_BLOCK_SIZE) % MAX_BLOCK_SIZE;

	int total_threads = len * no_ancestors;
	int no_blocks = (total_threads + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

	gpu_grad_from_clones__ CUDA2(no_blocks, MAX_BLOCK_SIZE) (in, len, total_threads);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

