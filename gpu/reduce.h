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

// Shared batched-cuBLAS GEMM helpers for the complex "position-wise linear"
// pattern Out[N,E_out] = Data[N,E_in] * W[E_out,E_in]^T. Used by TokenwiseLinear
// and by the dense Linear (which is the same layout with N = 1). The caller owns a
// batched-pointer pool that tw_gemm_{forward,backward} (re)allocate and populate ONCE
// per batch size (pointers are static after allocateOnGpu), freed via tw_free_pool.
void tw_free_pool(float **&pool, int &cap);
void tw_gemm_forward(int N, int e_in, int e_out,
		const std::vector<GpuInVar> &in_, const std::vector<GpuOutVar> &out_, float **&pool, int &cap);
void tw_gemm_backward(int N, int e_in, int e_out,
		const std::vector<GpuInVar> &in_, const std::vector<GpuOutVar> &out_, float **&pool, int &cap);

class LinearGpu : public GpuMapping {
	float **d_pool_ = 0;             // forward GEMM pointer pool (built once; CUDA-graph-capturable)
	int tw_cap_ = 0;
	float **d_pool_bwd_ = 0;         // backward GEMM pointer pool (separate so fwd/bwd don't clobber)
	int tw_cap_bwd_ = 0;
public:
	LinearGpu(int depth) : GpuMapping(depth) {
	}
	virtual ~LinearGpu() {
		tw_free_pool(d_pool_, tw_cap_);
		tw_free_pool(d_pool_bwd_, tw_cap_bwd_);
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
	float **d_pool_ = 0;             // forward GEMM pointer pool (built once; CUDA-graph-capturable)
	int tw_cap_ = 0;
	float **d_pool_bwd_ = 0;         // backward GEMM pointer pool (separate so fwd/bwd don't clobber)
	int tw_cap_bwd_ = 0;
public:
	TokenwiseLinearGpu(int depth) : GpuMapping(depth) {
	}
	virtual ~TokenwiseLinearGpu() {
		tw_free_pool(d_pool_, tw_cap_);
		tw_free_pool(d_pool_bwd_, tw_cap_bwd_);
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
	cmplx_ *ba_c_ = NULL;   // c[h][k,m] : B*H*N*N
	float  *ba_A_ = NULL;   // A[h][k,m] : B*H*N*N
	float  *ba_S_ = NULL;   // S[h][k]   : B*H*N
	int ba_N_ = 0, ba_d_ = 0, ba_H_ = 0, ba_M_ = 0, ba_cols_ = 0;   // M_ = number of clones (getNoMappings)
	void ensure_scratch(int N, int d, int H, int M, int cols);   // cols: N (quadratic) or chunk C (linear)
	void free_scratch();
	// Linear (chunkwise density-matrix) mode -- see gpu_born_linear_forward(). Per (clone, head):
	// lifted chunk features cQ/cK [C, d2] (d2 = dh*dh packed Hermitian features), packed value
	// chunk U [2][C][dh], inter-chunk readout num [2][C][dh] / den [C], running state
	// M [2][d2][dh] and Z [d2], and the per-chunk-start snapshots Mhist/Zhist for the backward.
	float *lin_cQ_ = NULL, *lin_cK_ = NULL, *lin_U_ = NULL, *lin_num_ = NULL, *lin_den_ = NULL;
	float *lin_M_ = NULL, *lin_Z_ = NULL, *lin_Mhist_ = NULL, *lin_Zhist_ = NULL, *lin_ones_ = NULL;
	float *lin_Qp_ = NULL, *lin_Kp_ = NULL, *lin_Cr_ = NULL;   // GEMM forward: raw Q/K planes [bh][2][C][dh], scores [bh][2][C][C]
	float *lin_cKw_ = NULL, *lin_cQd_ = NULL;   // decay: keys lifted then weighted by gamma^(C-1-j), queries by gamma^(i+1)
	int *lin_idx_ = NULL;   // feature a -> (m, n) pair table, 2*d2 ints
	// backward scratch (see gpu_born_linear_backward): packed output grads / values per chunk,
	// the C x C intra matrices, per-chunk lifted-space gradients and the suffix state.
	float *bw_Gm_ = NULL, *bw_Gt_ = NULL, *bw_Ut_ = NULL, *bw_Ug_ = NULL, *bw_G4_ = NULL;
	float *bw_Am_ = NULL, *bw_Bm_ = NULL, *bw_W_ = NULL, *bw_Gq_ = NULL, *bw_Gk_ = NULL, *bw_cQs_ = NULL;
	float *bw_dV_ = NULL, *bw_abar_ = NULL, *bw_abarD_ = NULL, *bw_Ms_ = NULL, *bw_Zs_ = NULL;
	void gpu_born_linear_backward();
	int lin_C_ = 0, lin_NC_ = 0, lin_d2_ = 0, lin_B_ = 0, lin_H_ = 0, lin_dh_ = 0;
	void ensure_linear_scratch(int N, int d, int H, int C, int B);
	// Carried state for windowed streaming (linear mode): with carry_ on, a forward starts from
	// carry_M_/carry_Z_ (the state after the previous forward of the same clone) and leaves its final
	// state there; the decay bookkeeping is unchanged because the state is kept at "after the last token".
	float *carry_M_ = NULL, *carry_Z_ = NULL; bool carry_ = false;
	void free_linear_scratch();
	void gpu_born_linear_forward();
public:
	BornAttentionGpu(int depth) : GpuMapping(depth) {}
	virtual ~BornAttentionGpu() { free_scratch(); free_linear_scratch(); }
	void setCarry(bool on) { carry_ = on; }
	void resetCarry();
	void gpu_born_forward();
	void gpu_born_backward();
	virtual void forward() override { gpu_born_forward(); }
	virtual void backward(int label) override { gpu_born_backward(); }
};

// Gated linear attention over complex amplitudes on GPU (see impl/gla.h for the layer and its
// Wirtinger backward). Chunkwise: per (clone, head) a dh x dh complex state S (two planes) carried
// across chunks of C tokens; inside a chunk the C x C amplitudes are formed directly. Complex
// GEMMs are four real strided-batched cuBLAS GEMMs on the [re | im] planes. Backward sweeps the
// chunks in reverse with a suffix state (G, Gb) built from the queries and output gradients.
class GatedLinearAttentionGpu : public GpuMapping {
	// per (clone, head) gate arrays over all tokens: lambda_t = log sigmoid(Re g_t), L_t = chunk-local
	// inclusive cumulative sum; plus the per-chunk totals LC[c]
	float *gl_lam_ = NULL, *gl_L_ = NULL;
	// chunk planes [2][C][dh]: Qs (exp(L) q), Kt (exp(LC-L) k), Qraw, Kraw, V, Oint; state S [2][dh][dh]
	// and its per-chunk snapshots Shist [NC][BH][2][dh][dh]
	float *gl_Qs_ = NULL, *gl_Kt_ = NULL, *gl_Qr_ = NULL, *gl_Kr_ = NULL, *gl_V_ = NULL, *gl_Oint_ = NULL;
	float *gl_S_ = NULL, *gl_Shist_ = NULL;
	// backward: gradient planes, C x C amplitude / weighted-p planes, gate partials, suffix state
	float *bw_Gg_ = NULL, *bw_Gb_ = NULL, *bw_A_ = NULL, *bw_PW_ = NULL, *bw_PbW_ = NULL, *bw_E_ = NULL, *bw_Ecum_ = NULL;
	float *bw_eps_ = NULL, *bw_eta_ = NULL, *bw_Dc_ = NULL;
	float *bw_dQ_ = NULL, *bw_dQb_ = NULL, *bw_dQi_ = NULL, *bw_dQbi_ = NULL;
	float *bw_dK_ = NULL, *bw_dKb_ = NULL, *bw_dKl_ = NULL, *bw_dKlb_ = NULL, *bw_dV_ = NULL, *bw_dVb_ = NULL;
	float *bw_G_ = NULL, *bw_Gbs_ = NULL;
	int gl_N_ = 0, gl_C_ = 0, gl_B_ = 0, gl_H_ = 0, gl_dh_ = 0;
	// Carried state for windowed streaming: when carry_ is on, each forward starts from carry_S_
	// (per clone and head, zero after resetCarry) instead of zero and leaves its final state there,
	// so consecutive N-token forwards of one clone form an unbroken recurrence.
	float *carry_S_ = NULL; bool carry_ = false;
	void ensure_scratch(int N, int d, int H, int C, int B);
	void free_scratch();
public:
	GatedLinearAttentionGpu(int depth) : GpuMapping(depth) {}
	virtual ~GatedLinearAttentionGpu() { free_scratch(); }
	void setCarry(bool on) { carry_ = on; }
	void resetCarry();
	void gpu_gla_forward();
	void gpu_gla_backward();
	virtual void forward() override { gpu_gla_forward(); }
	virtual void backward(int label) override { gpu_gla_backward(); }
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

// Rotary position embedding (RoPE) for complex tokens (see impl/rotary.h). The
// per-(token,feature) unit phasor exp(i theta_dd t) is recomputed in the kernel
// (one thread per element), so no scratch is needed. y=r*x is holomorphic:
// dL/dx = gO*r, dL/dx* = gOb*conj(r).
class RotaryEmbedGpu : public GpuMapping {
public:
	RotaryEmbedGpu(int depth) : GpuMapping(depth) {}
	virtual ~RotaryEmbedGpu() {}
	void gpu_rope_forward();
	void gpu_rope_backward();
	virtual void forward() override { gpu_rope_forward(); }
	virtual void backward(int label) override { gpu_rope_backward(); }
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
	void readLosses(std::vector<float> &out);   // all B*n_pos per-position losses, clone-major
	int nPos() const { return n_pos_; }
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
#ifdef __CUDACC__
		if (l2_scratch_) { cudaFree(l2_scratch_); l2_scratch_ = NULL; }
#endif
	}

	void gpu_norm_forward(int block_size);

	virtual void forward() {
		gpu_norm_forward(reduce_block_size);
	}

	void gpu_norm_backward();

	virtual void backward(int label) {
		gpu_norm_backward();
	}

private:
	// Persistent reduction scratch (allocated once, reused): a per-call cudaMalloc
	// here would both churn and break CUDA-graph capture.
	cmplx_* l2_scratch_ = NULL;
	int     l2_scratch_len_ = 0;
};


#endif /* GPU_REDUCE_H_ */
