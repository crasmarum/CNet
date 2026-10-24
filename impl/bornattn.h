#ifndef IMPL_BORNATTN_H_
#define IMPL_BORNATTN_H_

#include <vector>
#include <cassert>
#include <complex>
#include "cfunc.h"
#include "utils.h"

// Born-rule causal self-attention, fully complex, H heads over a shared [Q;K;V] input.
//
// Input is the concatenation of three complex blocks [Q ; K ; V], each of length
// N*d (token t, dim dd at index t*d + dd). The d dims are split into H heads of
// dh = d/H consecutive dims; head h owns dims [h*dh, (h+1)*dh). Output is O of
// length N*d, head h written into its own dims:
//
//   c_h[k,m] = sum_{j<dh} Q[k,h*dh+j] conj(K[m,h*dh+j])   (keys m <= k; causal)
//   s_h[k,m] = |c_h[k,m]|^2                                (Born-rule measurement score)
//   A_h[k,m] = s_h[k,m] / sum_{m'<=k} s_h[k,m']            (causal normalization, per head)
//   O[k,h*dh+j] = sum_{m<=k} A_h[k,m] V[m,h*dh+j]
//
// H=1 is the original single-head operator (d = dh). The scores ARE measurement
// probabilities of query k against key m, so routing is content-based like
// attention but uses |<Q,K>|^2 instead of a softmaxed dot product. The Wirtinger
// backward (verified to ~1e-9 against finite differences) is below.
class BornAttention: public CFunc {
	int N_, d_, H_;                        // tokens, dims per token, heads (dh = d_/H_)
	bool causal_;                          // true: key m<=k (autoregressive); false: full attention
	float eps_;                            // normalization floor: A=s/(S+eps). Bounds the 1/S
	                                       // gradient blow-up when a query's scores collapse (S->0),
	                                       // the spike that otherwise destabilizes long training.
	std::vector<std::complex<float>> c_;   // c[(h*N+k)*N+m]
	std::vector<float> A_, S_;             // A[(h*N+k)*N+m], S[h*N+k]
	// GPU execution mode. linear_: the exact chunkwise density-matrix form (O(N d^2) time,
	// O(N dh C) scratch instead of O(N^2)); chunk_ = C tokens per chunk (N % C == 0, causal only).
	// The CPU forward/backward above stay quadratic: they are the oracle the linear kernels
	// must reproduce (Proposition 1 of the born_linear paper: identical up to rounding).
	bool linear_ = false;
	int chunk_ = 512;
	// Fixed per-head decay (RetNet schedule) on the Born score: s[k,m] *= gamma_h^(k-m), gamma_h = 1 - 2^-(5+h).
	// Kept exactly in the linear state and in the streaming state (R, M scale by gamma_h per token).
	bool decay_ = false;
	// CPU streaming decode (constant state): N must be 1. Per head h the state is
	//   R[m][n]    = sum_j conj(K_j[m]) K_j[n]             (dh x dh)
	//   Ms[m][n][v]= sum_j conj(K_j[m]) K_j[n] V_j[v]      (dh x dh x dh)
	// so that for the current query s_j = |<Q,K_j>|^2 sums to Q^T R conj(Q) and the numerator to
	// Q^T Ms conj(Q): the inclusive causal sum over every token seen so far, in O(dh^3) per token
	// and O(dh^3) memory, never touching past tokens (born_linear.tex, Corollary 1).
	bool step_ = false;
	std::vector<std::complex<float>> st_R_, st_M_;
public:
	BornAttention(Uid uid, int n_tokens, int dim, bool causal = true, float eps = 0.f, int heads = 1)
		: CFunc(uid, InSize(3 * n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), H_(heads), causal_(causal), eps_(eps),
		  c_((size_t) heads * n_tokens * n_tokens), A_((size_t) heads * n_tokens * n_tokens),
		  S_((size_t) heads * n_tokens) { assert(heads >= 1 && dim % heads == 0); }

	BornAttention(int n_tokens, int dim, bool causal = true, float eps = 0.f, int heads = 1)
		: CFunc(InSize(3 * n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), H_(heads), causal_(causal), eps_(eps),
		  c_((size_t) heads * n_tokens * n_tokens), A_((size_t) heads * n_tokens * n_tokens),
		  S_((size_t) heads * n_tokens) { assert(heads >= 1 && dim % heads == 0); }

	virtual ~BornAttention() {}
	virtual CFunc* clone(Uid uid) {
		BornAttention *b = new BornAttention(uid, N_, d_, causal_, eps_, H_);
		b->setLinear(linear_, chunk_); b->setDecay(decay_);
		return b;
	}
	float eps() const { return eps_; }
	int heads() const { return H_; }
	int headDim() const { return d_ / H_; }
	void setLinear(bool on, int chunk = 512) { linear_ = on; chunk_ = chunk; }
	void setDecay(bool on) { decay_ = on; }
	bool decay() const { return decay_; }
	// Decay schedule gamma_h = 1 - 2^-(base + step*h); default base 5, step 1 (RetNet: 31/32, 63/64, ...). Set once at
	// startup with setDecaySchedule (also copies it to the GPU); not stored in model files, so pass the same flags
	// when evaluating a model trained with a non-default schedule. Half-life of head h ~ 0.69 * 2^(base + step*h) tokens.
	static float& decayBase() { static float b = 5.f; return b; }
	static float& decayStep() { static float st = 1.f; return st; }
	static void setDecaySchedule(float base, float step);
	static float gammaOf(int h) { return 1.f - std::exp2(-(decayBase() + decayStep() * (float) h)); }
	float gamma(int h) const { return decay_ ? gammaOf(h) : 1.f; }
	void setStepMode(bool on) { assert(!on || N_ == 1); step_ = on; resetState(); }
	bool stepMode() const { return step_; }
	void resetState() {
		const int dh = d_ / H_;
		st_R_.assign((size_t) H_ * dh * dh, {0.f, 0.f});
		st_M_.assign((size_t) H_ * dh * dh * dh, {0.f, 0.f});
	}
	bool linear() const { return linear_; }
	int chunk() const { return chunk_; }
	virtual std::string getName() { return "BornAttention_" + std::to_string(uid_); }

	int M() const { return N_ * d_; }
	int nTokens() const { return N_; }
	int dim() const { return d_; }
	bool causal() const { return causal_; }
	// absolute input index of block b (0=Q,1=K,2=V), token t, dim dd.
	inline int qi(int t, int dd) const { return t * d_ + dd; }
	inline int ki(int t, int dd) const { return M() + t * d_ + dd; }
	inline int vi(int t, int dd) const { return 2 * M() + t * d_ + dd; }

	// One token in, one token out, state updated in place (see step_ above).
	void forwardStep() {
		const int d = d_, H = H_, dh = d_ / H_;
		std::vector<std::complex<float>> O(d, {0.f, 0.f});
		for (int h = 0; h < H; ++h) {
			const int d0 = h * dh;
			std::complex<float> *R = &st_R_[(size_t) h * dh * dh], *Ms = &st_M_[(size_t) h * dh * dh * dh];
			if (decay_) {
				const float g = gammaOf(h);
				for (size_t i = 0; i < (size_t) dh * dh; ++i) R[i] *= g;
				for (size_t i = 0; i < (size_t) dh * dh * dh; ++i) Ms[i] *= g;
			}
			for (int m = 0; m < dh; ++m) {
				const std::complex<float> km = std::conj(input().z(ki(0, d0 + m)));
				for (int n = 0; n < dh; ++n) {
					const std::complex<float> kmn = km * input().z(ki(0, d0 + n));
					R[m * dh + n] += kmn;
					std::complex<float> *row = Ms + ((size_t) m * dh + n) * dh;
					for (int v = 0; v < dh; ++v) row[v] += kmn * input().z(vi(0, d0 + v));
				}
			}
			std::complex<float> den = 0.f;
			std::vector<std::complex<float>> num(dh, {0.f, 0.f});
			for (int m = 0; m < dh; ++m) {
				const std::complex<float> qm = input().z(qi(0, d0 + m));
				for (int n = 0; n < dh; ++n) {
					const std::complex<float> w = qm * std::conj(input().z(qi(0, d0 + n)));
					den += w * R[m * dh + n];
					const std::complex<float> *row = Ms + ((size_t) m * dh + n) * dh;
					for (int v = 0; v < dh; ++v) num[v] += w * row[v];
				}
			}
			float denom = den.real() + eps_;
			float inv = denom > 0.f ? 1.f / denom : 0.f;
			for (int v = 0; v < dh; ++v) O[d0 + v] = num[v] * inv;
		}
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < M(); ++p) {
				output(indx).real_[offset(indx) + p] = O[p].real();
				output(indx).imag_[offset(indx) + p] = O[p].imag();
			}
	}

	virtual void forward() {
		assert(no_outputs());
		if (step_) { forwardStep(); return; }
		const int N = N_, d = d_, H = H_, dh = d_ / H_;
		std::vector<std::complex<float>> O(N * d, {0.f, 0.f});
		for (int h = 0; h < H; ++h) {
			const int d0 = h * dh;
			for (int k = 0; k < N; ++k) {
				const int mhi = causal_ ? k : N - 1;
				const size_t cb = ((size_t) h * N + k) * N;
				float Sk = 0.f;
				for (int m = 0; m <= mhi; ++m) {
					std::complex<float> ckm = 0.f;
					for (int j = 0; j < dh; ++j)
						ckm += input().z(qi(k, d0 + j)) * std::conj(input().z(ki(m, d0 + j)));
					c_[cb + m] = ckm;
					float s = std::norm(ckm);           // |ckm|^2
					if (decay_) s *= std::pow(gammaOf(h), (float) std::abs(k - m));   // gamma_h^(k-m)
					A_[cb + m] = s;                     // store s; normalize below
					Sk += s;
				}
				S_[(size_t) h * N + k] = Sk;
				float denom = Sk + eps_;
				float inv = denom > 0.f ? 1.f / denom : 0.f;
				for (int m = 0; m <= mhi; ++m) {
					float a = A_[cb + m] * inv;          // A[k,m]
					A_[cb + m] = a;
					for (int j = 0; j < dh; ++j)
						O[k * d + d0 + j] += a * input().z(vi(m, d0 + j));
				}
			}
		}
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < M(); ++p) {
				output(indx).real_[offset(indx) + p] = O[p].real();
				output(indx).imag_[offset(indx) + p] = O[p].imag();
			}
	}

	virtual void backward() {
		const int N = N_, d = d_, H = H_, dh = d_ / H_;
		// Sum incoming output gradients across all consumers.
		std::vector<std::complex<float>> gO(M(), {0.f, 0.f}), gOb(M(), {0.f, 0.f});
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < M(); ++p) {
				gO[p]  += output(indx).dz(offset(indx) + p);
				gOb[p] += output(indx).dz_star(offset(indx) + p);
			}
		for (int h = 0; h < H; ++h) {
			const int d0 = h * dh;
			for (int k = 0; k < N; ++k) {
				const int mhi = causal_ ? k : N - 1;
				const size_t cb = ((size_t) h * N + k) * N;
				// a[m] = sum_j Re( gO[k,j] V[m,j] + gOb[k,j] conj(V[m,j]) ) over this head's dims
				std::vector<float> a(mhi + 1, 0.f);
				float abar = 0.f;
				for (int m = 0; m <= mhi; ++m) {
					float am = 0.f;
					for (int j = 0; j < dh; ++j) {
						std::complex<float> v = input().z(vi(m, d0 + j));
						am += (gO[k * d + d0 + j] * v + gOb[k * d + d0 + j] * std::conj(v)).real();
					}
					a[m] = am;
					abar += A_[cb + m] * am;
				}
				float denom = S_[(size_t) h * N + k] + eps_;
				float inv = denom > 0.f ? 1.f / denom : 0.f;
				for (int m = 0; m <= mhi; ++m) {
					float b = (a[m] - abar) * inv;            // dL/ds[k,m]-chain (real)
					if (decay_) b *= std::pow(gammaOf(h), (float) std::abs(k - m));   // s = gamma^(k-m) |c|^2
					std::complex<float> ckm = c_[cb + m], ckmc = std::conj(ckm);
					float Akm = A_[cb + m];
					for (int j = 0; j < dh; ++j) {
						const int dd = d0 + j;
						std::complex<float> q = input().z(qi(k, dd));
						std::complex<float> kk = input().z(ki(m, dd));
						// V grads: O[k] += A[k,m] V[m]
						accZ(vi(m, dd),  gO[k * d + dd] * Akm,  gOb[k * d + dd] * Akm);
						// Q grads (query k): dL/dQ += b conj(c) conj(K); dL/dconjQ += b c K
						accZ(qi(k, dd),  b * ckmc * std::conj(kk),  b * ckm * kk);
						// K grads (key m): dL/dK += b c conj(Q); dL/dconjK += b conj(c) Q
						accZ(ki(m, dd),  b * ckm * std::conj(q),  b * ckmc * q);
					}
				}
			}
		}
	}
	virtual void backward(int label) {}

	// Not used (this layer defines gradients directly in backward()).
	virtual std::complex<float> dz(int, int) { return 0; }
	virtual std::complex<float> dz_star(int, int) { return 0; }

private:
	inline void accZ(int p, std::complex<float> dz, std::complex<float> dzstar) {
		input().dz_real_[p]      += dz.real();
		input().dz_imag_[p]      += dz.imag();
		input().dz_star_real_[p] += dzstar.real();
		input().dz_star_imag_[p] += dzstar.imag();
	}
};

#endif /* IMPL_BORNATTN_H_ */
