#ifndef GPU_REDUCE_H_
#define GPU_REDUCE_H_

#include "gpumapping.h"
#include "compl.h"
#include "kernels.h"

// Flags.
extern int reduce_block_size;

// Compares argmax_k |z_k|^2 to the label for each batch element; writes 1/0 into
// out_correct[0..no_mappings). Used for cheap in-training accuracy.
void gpu_argmax_correct(GpuInVar *in, int *labels, int *out_correct, int N, int no_mappings);
void gpu_argmax_predict(GpuInVar *in, int *out_pred, int N, int no_mappings);

class LinearGpu;
class FourierGpu;
class L2Gpu;

const int TEST_DATA_PROVIDER = 1;

const int CROSS_ENT_DATA_PROVIDER = 2;

const int LINEAR_DATA_PROVIDER = 3;

const int FFT_DATA_PROVIDER = 4;

const int SOFTMAX_DATA_PROVIDER = 5;

const int NORM_DATA_PROVIDER = 6;

const int T_FFT_DATA_PROVIDER = 7;

inline int getPaddedLength(int block_size, GpuMapping *mp) {
	int padded_segment_len =
			mp->length() % block_size == 0 ?
					mp->length() :
					(mp->length() + block_size - mp->length() % block_size);
	int b_out_length = (padded_segment_len / block_size) * mp->getNoMappings()
			* mp->length();
	return b_out_length;
}

inline int getPaddedOutLength(int block_size, GpuMapping *mp) {
	int padded_segment_len =
			mp->getOutputLength() % block_size == 0 ?
					mp->getOutputLength() :
					(mp->getOutputLength() + block_size - mp->getOutputLength() % block_size);
	int b_out_length = (padded_segment_len / block_size) * mp->getNoMappings()
			* mp->getOutputLength();
	return b_out_length;
}

class LinearGpu : public GpuMapping {

public:
	LinearGpu(int depth) : GpuMapping(depth) {
	}
	virtual ~LinearGpu() {
	}

	void gpu_linear_forward(int block_size);

	void gpu_linear_backward(int label, int block_size);

	virtual void forward() override {
		gpu_linear_forward(reduce_block_size);
	}

	virtual void backward(int label) override {
		gpu_linear_backward(label, reduce_block_size);
	}
};

// Position-wise linear: one shared e_in x e_out weight applied to each of
// n_tokens contiguous e_in slices. Direct kernels (the contraction e_in and the
// output width e_out are small); the weight gradient is accumulated over tokens.
class TokenwiseLinearGpu : public GpuMapping {

public:
	TokenwiseLinearGpu(int depth) : GpuMapping(depth) {
	}
	virtual ~TokenwiseLinearGpu() {
	}

	void gpu_tokenwise_forward();
	void gpu_tokenwise_backward();

	virtual void forward() override {
		gpu_tokenwise_forward();
	}

	virtual void backward(int label) override {
		gpu_tokenwise_backward();
	}
};

class FourierGpu : public GpuMapping {
protected:
	int fft_plan_ = -1;        // cufftHandle (an int); -1 = not created
	cmplx_ *fft_buf_ = NULL;   // interleaved batched scratch (N*M) for the fast path
	int fft_N_ = 0;
	int fft_M_ = 0;

public:
	FourierGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~FourierGpu() {
		free_fft();
	}

	// cuFFT plan + scratch management for the fast (fft_fast) path.
	void ensure_fft_plan(int N, int M);
	void free_fft();

	void gpu_fft_forward(int block_size);

	virtual void forward() override {
		gpu_fft_forward(reduce_block_size);
	}

	void gpu_fft_backward(int label, int block_size);

	virtual void backward(int label) override {
		gpu_fft_backward(label, reduce_block_size);
	}
};

// Inverse (unitary) DFT on GPU. Reuses FourierGpu's cuFFT plan/buffer and the
// shared gather/scatter kernels; only the transform DIRECTION differs:
//   forward  = cuFFT INVERSE (+ 1/sqrt(N)),
//   backward = FourierGpu's backward with the two directions swapped (the
//              adjoint of the inverse map is the forward unitary DFT).
// One batched plan of size N over M = getNoMappings() clones covers the whole
// layer, so per-batch cloning works exactly as for FourierGpu.
class InverseFourierGpu : public FourierGpu {
public:
	InverseFourierGpu(int depth) : FourierGpu(depth) {
	}

	virtual ~InverseFourierGpu() {
	}

	void gpu_ifft_forward(int block_size);
	void gpu_ifft_backward(int label, int block_size);

	virtual void forward() override {
		gpu_ifft_forward(reduce_block_size);
	}

	virtual void backward(int label) override {
		gpu_ifft_backward(label, reduce_block_size);
	}
};

class TrianFourierGpu : public GpuMapping {
	// ---- Bluestein fast path (fft_fast) ----
	// The causal (lower-triangular) DFT  T_p = (1/sqrt N) sum_{q<=p} x_q w^{pq}
	// is the causal half of a Bluestein chirp transform: with pq = (p^2 + q^2 -
	// (p-q)^2)/2 and chirp_k = w^{k^2/2} = e^{i pi k^2 / N},
	//   T_p = chirp_p/sqrt(N) * (a * h)[p],  a_q = x_q chirp_q,  h_k = conj(chirp_k),
	// where (a*h) is a CAUSAL LINEAR convolution (q in 0..p) -- computed via a
	// zero-padded batched FFT of size L >= 2N-1. O(N^2) -> O(N log N). The adjoint
	// (backward) is the same chirp sandwich around a correlation (reversed kernel).
	// Chirp phases use k^2 mod 2N (float cannot hold k^2 exactly for large k), so
	// all chirp/kernel tables are built on the host and the kernel FFTs are done
	// once at setup.
	int bl_plan_ = -1;          // cufftHandle, batched C2C of size L over M clones
	cmplx_ *bl_buf_  = NULL;    // interleaved L*M scratch
	cmplx_ *bl_chirp_ = NULL;   // N : chirp_k = e^{i pi k^2 / N}
	cmplx_ *bl_Hf_   = NULL;    // L : FFT(pad(conj(chirp)))            -- forward conv kernel
	cmplx_ *bl_KrA_  = NULL;    // L : FFT(reverse(conj(chirp)))        -- backward dz   correlation
	cmplx_ *bl_KrB_  = NULL;    // L : FFT(reverse(chirp))              -- backward dz*  correlation
	int bl_N_ = 0, bl_L_ = 0, bl_M_ = 0;

	void ensure_bluestein(int N, int M);
	void free_bluestein();

public:
	TrianFourierGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~TrianFourierGpu() {
		free_bluestein();
	}

	void gpu_T_fft_forward(int block_size);

	virtual void forward() override {
		gpu_T_fft_forward(reduce_block_size);
	}

	void gpu_T_fft_backward(int label, int block_size);

	virtual void backward(int label) override {
		gpu_T_fft_backward(label, reduce_block_size);
	}
};

// Born-rule causal self-attention on GPU (see impl/bornattn.h for the layer and
// its verified Wirtinger backward). Input [Q;K;V] of length 3*N*d per clone ->
// output N*d. Per-clone scratch stores c (N*N complex), A (N*N real) and S (N).
class BornAttentionGpu : public GpuMapping {
	cmplx_ *ba_c_ = NULL;   // c[k,m]   : B*N*N
	float  *ba_A_ = NULL;   // A[k,m]   : B*N*N
	float  *ba_S_ = NULL;   // S[k]     : B*N
	int ba_N_ = 0, ba_d_ = 0, ba_M_ = 0;   // M_ = number of clones (getNoMappings)
	void ensure_scratch(int N, int d, int M);
	void free_scratch();
public:
	BornAttentionGpu(int depth) : GpuMapping(depth) {}
	virtual ~BornAttentionGpu() { free_scratch(); }
	void gpu_born_forward();
	void gpu_born_backward();
	virtual void forward() override { gpu_born_forward(); }
	virtual void backward(int label) override { gpu_born_backward(); }
};

// Per-token complex RMS normalization on the GPU (see impl/tokennorm.h). One
// thread per (clone, token): each token's d features are disjoint across threads,
// so forward writes outputs directly and backward accumulates input gradients
// (atomicAdd, matching the arena's accumulate-across-consumers convention).
class TokenNormGpu : public GpuMapping {
	float *tn_r_ = NULL;   // r_t : B*N  (forward -> backward)
	int tn_N_ = 0, tn_M_ = 0;
	void ensure_scratch(int N, int M);
	void free_scratch();
public:
	TokenNormGpu(int depth) : GpuMapping(depth) {}
	virtual ~TokenNormGpu() { free_scratch(); }
	void gpu_tn_forward();
	void gpu_tn_backward();
	virtual void forward() override { gpu_tn_forward(); }
	virtual void backward(int label) override { gpu_tn_backward(); }
};

class SoftMaxGpu : public GpuMapping {

public:
	SoftMaxGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~SoftMaxGpu() {
	}

	void gpu_soft_max_forward(int block_size);

	virtual void forward() {
		gpu_soft_max_forward(reduce_block_size);
	}

	void gpu_soft_max_backward(int label, int block_size);

	virtual void backward(int label) {
		gpu_soft_max_backward(label, reduce_block_size);
	}
};

class CrossEntropyGpu : public GpuMapping {
	friend class GpuNet;
	friend class CNet;

	int *gpu_labels_ = NULL;

public:

	CrossEntropyGpu(int depth)
		: GpuMapping(depth) {
	}
	virtual ~CrossEntropyGpu() {
	}

	void gpu_cross_ent_forward(int block_size);

	virtual void forward() {
		gpu_cross_ent_forward(reduce_block_size);
	}

	void gpu_cross_ent_backward(int label);

	virtual void backward(int label) {
		gpu_cross_ent_backward(label);
	}

};

// Per-position Born-rule cross entropy on the GPU (autoregressive LM).
// Two target sources:
//   - batch=1 (no clones): pulls the per-position targets from its CPU twin each
//     forward into the self-owned gpu_targets_ (no extra plumbing needed).
//   - batched (B clones): the CNet allocates one B*n_pos targets arena, uploads
//     all B per-position target vectors to it in batchToGpu, and hands the device
//     pointer here via setBatchTargets(); gpu_targets_ stays unused.
// forward() computes every clone-position's ||z_p||^2 and loss over B*n_pos; the
// kernels index in[map_indx] and targets[map_indx*n_pos + p].
class SequenceCrossEntropyGpu : public GpuMapping {
	friend class GpuNet;
	friend class CNet;

	int vocab_ = 0;
	int n_pos_ = 0;
	int *gpu_targets_ = NULL;         // self-owned n_pos targets (batch=1, from CPU twin)
	int *gpu_batch_targets_ = NULL;   // external B*n_pos targets (batched); NOT freed here
	float *gpu_sqnorm_ = NULL;        // B*n_pos ||z_p||^2 (forward -> backward)
	float *gpu_poss_loss_ = NULL;     // B*n_pos -log p (for loss readback)

public:
	SequenceCrossEntropyGpu(int depth) : GpuMapping(depth) {
	}
	virtual ~SequenceCrossEntropyGpu();

	// Called once at allocation in the batched path: the device targets arena
	// (owned by CNet) and the per-position count. Presence of gpu_batch_targets_
	// switches the forward off the CPU-twin pull.
	void setBatchTargets(int *dev, int n_pos) {
		gpu_batch_targets_ = dev;
		n_pos_ = n_pos;
	}

	void gpu_seq_ce_forward();
	void gpu_seq_ce_backward();
	float readLoss();               // mean over B*positions of -log p_{target}

	virtual void forward() {
		gpu_seq_ce_forward();
	}
	virtual void backward(int label) {
		gpu_seq_ce_backward();
	}
};

class L2Gpu : public GpuMapping {
	friend class GpuNet;
	friend class CNet;

public:
	L2Gpu(int depth) : GpuMapping(depth) {
	}

	virtual ~L2Gpu() {
	}

	void gpu_norm_forward(int block_size);

	virtual void forward() {
		gpu_norm_forward(reduce_block_size);
	}

	void gpu_norm_backward();

	virtual void backward(int label) {
		gpu_norm_backward();
	}
};


#endif /* GPU_REDUCE_H_ */
