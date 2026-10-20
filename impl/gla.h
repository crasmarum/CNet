#ifndef IMPL_GLA_H_
#define IMPL_GLA_H_

#include <vector>
#include <cassert>
#include <cmath>
#include <complex>
#include "cfunc.h"
#include "utils.h"

// Gated linear attention over COMPLEX amplitudes (the degree-one counterpart of Born attention).
//
// Input is the concatenation [Q ; K ; V ; G]: Q, K, V are N*d complex (token t, dim dd at t*d + dd),
// split into H heads of dh = d/H consecutive dims; G is N*H complex gate logits, of which the real
// part is used. Per head h, with x_t = Re G[t,h] and the log-gate lambda_t = log sigmoid(x_t) <= 0:
//
//   c[k,m] = s * sum_{j<dh} Q[k,j] conj(K[m,j])         complex amplitude, s = 1/sqrt(dh)
//   w[k,m] = exp(sum_{m<u<=k} lambda_u)                 data-dependent decay (1 on the diagonal)
//   O[k,j] = sum_{m<=k} w[k,m] c[k,m] V[m,j]            complex-weighted value mix, no normalization
//
// which is the recurrence S_t = alpha_t S_{t-1} + conj(K_t) V_t^T, O_t = s S_t^T Q_t with the real
// per-head gate alpha_t = sigmoid(x_t): a dh x dh complex state per head, constant in N. Unlike Born
// attention the weights are amplitudes, not probabilities: contributions interfere by phase (with
// RoPE on Q,K the phase rotates with distance) and nothing sums to one, so the net is expected to
// normalize the output (TokenNorm) as GLA does with its group norm. The CPU forward/backward below
// are the quadratic oracle; the GPU chunkwise form (reduce.cpp) must reproduce them.
//
// Wirtinger backward. O is C-linear in Q and V and anti-linear in K; with gO = dL/dO, gOb = dL/dconj(O):
//   p[k,m]  = sum_j gO[k,j] V[m,j],  pb[k,m] = sum_j gOb[k,j] conj(V[m,j])      (dL/da, dL/dconj a), a = w c
//   V: dL/dV[m]   += sum_k a[k,m] gO[k],             dL/dconjV[m] += sum_k conj(a[k,m]) gOb[k]
//   Q: dL/dQ[k,j] += sum_m p w conj(K[m,j]),         dL/dconjQ    += sum_m pb w K[m,j]
//   K: dL/dK[m,j] += sum_k pb w conj(Q[k,j]),        dL/dconjK    += sum_k p w Q[k,j]
//   gates: e[k,m] = Re(p a + pb conj(a)) (real); dL/dlambda_u = sum_{k>=u} sum_{m<u} e[k,m];
//          dL/dx_u = dL/dlambda_u * sigmoid(-x_u); dL/dG = dL/dconjG = dL/dx / 2 (real part only).
class GatedLinearAttention: public CFunc {
	int N_, d_, H_;
	float scale_;
	int chunk_ = 512;                      // GPU chunk length C (N % C == 0)
	std::vector<std::complex<float>> a_;   // a[(h*N+k)*N+m] = w c  (amplitude incl. gate)
	std::vector<float> lam_;               // lam[h*N+t] = log sigmoid(x_t)
	std::vector<float> Lam_;               // Lam[h*N+t] = inclusive cumulative sum of lam over t
	std::vector<float> x_;                 // x[h*N+t]   = Re G[t,h]
	// CPU streaming decode (constant state): N must be 1. S[h][m][v] = sum_j w conj(K_j[m]) V_j[v].
	bool step_ = false;
	std::vector<std::complex<float>> st_S_;
public:
	GatedLinearAttention(Uid uid, int n_tokens, int dim, int heads = 1, float scale = 0.f)
		: CFunc(uid, InSize(3 * n_tokens * dim + n_tokens * heads), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), H_(heads), scale_(scale > 0.f ? scale : 1.f / std::sqrt((float) (dim / heads))),
		  a_((size_t) heads * n_tokens * n_tokens), lam_((size_t) heads * n_tokens), Lam_((size_t) heads * n_tokens), x_((size_t) heads * n_tokens)
		{ assert(heads >= 1 && dim % heads == 0); }
	GatedLinearAttention(int n_tokens, int dim, int heads = 1, float scale = 0.f)
		: CFunc(InSize(3 * n_tokens * dim + n_tokens * heads), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), H_(heads), scale_(scale > 0.f ? scale : 1.f / std::sqrt((float) (dim / heads))),
		  a_((size_t) heads * n_tokens * n_tokens), lam_((size_t) heads * n_tokens), Lam_((size_t) heads * n_tokens), x_((size_t) heads * n_tokens)
		{ assert(heads >= 1 && dim % heads == 0); }
	virtual ~GatedLinearAttention() {}
	virtual CFunc* clone(Uid uid) {
		auto *c = new GatedLinearAttention(uid, N_, d_, H_, scale_);
		c->step_ = step_; c->chunk_ = chunk_; return c;
	}
	virtual std::string getName() { return "GatedLinearAttention_" + std::to_string(uid_); }
	int nTokens() const { return N_; }
	int dim() const { return d_; }
	int heads() const { return H_; }
	int headDim() const { return d_ / H_; }
	float scale() const { return scale_; }
	void setChunk(int c) { chunk_ = c; }
	int chunk() const { return chunk_; }
	int M() const { return N_ * d_; }
	void setStepMode(bool on) { assert(!on || N_ == 1); step_ = on; resetState(); }
	bool stepMode() const { return step_; }
	void resetState() {
		const int dh = d_ / H_;
		st_S_.assign((size_t) H_ * dh * dh, {0.f, 0.f});
	}
	inline int qi(int t, int dd) const { return t * d_ + dd; }
	inline int ki(int t, int dd) const { return M() + t * d_ + dd; }
	inline int vi(int t, int dd) const { return 2 * M() + t * d_ + dd; }
	inline int gi(int t, int h) const { return 3 * M() + t * H_ + h; }

	static inline float logsigmoid(float x) { return x >= 0.f ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x)); }

	// One token (N=1): S <- alpha S + conj(K) V^T ; O = s S^T Q. The current token is un-gated.
	void forwardStep() {
		const int d = d_, H = H_, dh = d_ / H_;
		std::vector<std::complex<float>> O(d, {0.f, 0.f});
		for (int h = 0; h < H; ++h) {
			const int d0 = h * dh;
			const float alpha = 1.f / (1.f + std::exp(-input().z(gi(0, h)).real()));
			std::complex<float> *S = &st_S_[(size_t) h * dh * dh];
			for (int m = 0; m < dh; ++m) {
				std::complex<float> kc = std::conj(input().z(ki(0, d0 + m)));
				for (int v = 0; v < dh; ++v)
					S[m * dh + v] = alpha * S[m * dh + v] + kc * input().z(vi(0, d0 + v));
			}
			for (int v = 0; v < dh; ++v) {
				std::complex<float> o = 0.f;
				for (int m = 0; m < dh; ++m) o += S[m * dh + v] * input().z(qi(0, d0 + m));
				O[d0 + v] = scale_ * o;
			}
		}
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < d; ++p) {
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
			const float *Lam = &Lam_[(size_t) h * N];
			float acc = 0.f;
			for (int t = 0; t < N; ++t) {
				x_[(size_t) h * N + t] = input().z(gi(t, h)).real();
				lam_[(size_t) h * N + t] = logsigmoid(x_[(size_t) h * N + t]);
				acc += lam_[(size_t) h * N + t]; Lam_[(size_t) h * N + t] = acc;
			}
			for (int k = 0; k < N; ++k) {
				const size_t ab = ((size_t) h * N + k) * N;
				for (int m = 0; m <= k; ++m) {
					std::complex<float> c = 0.f;
					for (int j = 0; j < dh; ++j)
						c += input().z(qi(k, d0 + j)) * std::conj(input().z(ki(m, d0 + j)));
					const float w = std::exp(Lam[k] - Lam[m]);   // product of alpha over m<u<=k
					std::complex<float> a = scale_ * w * c;
					a_[ab + m] = a;
					for (int j = 0; j < dh; ++j) O[k * d + d0 + j] += a * input().z(vi(m, d0 + j));
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
		std::vector<std::complex<float>> gO(M(), {0.f, 0.f}), gOb(M(), {0.f, 0.f});
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < M(); ++p) {
				gO[p]  += output(indx).dz(offset(indx) + p);
				gOb[p] += output(indx).dz_star(offset(indx) + p);
			}
		std::vector<float> e(N);                          // e[m] for the current k: gate-chain terms
		std::vector<double> dLam(N);                      // dL/dLambda_u accumulated via pair sums
		for (int h = 0; h < H; ++h) {
			const int d0 = h * dh;
			std::fill(dLam.begin(), dLam.end(), 0.0);
			for (int k = 0; k < N; ++k) {
				const size_t ab = ((size_t) h * N + k) * N;
				for (int m = 0; m <= k; ++m) {
					const std::complex<float> a = a_[ab + m];
					std::complex<float> p = 0.f, pb = 0.f;
					for (int j = 0; j < dh; ++j) {
						const int dd = d0 + j;
						const std::complex<float> v = input().z(vi(m, dd));
						p  += gO[k * d + dd] * v;
						pb += gOb[k * d + dd] * std::conj(v);
						// V grads: O[k] += a V[m]
						accZ(vi(m, dd), a * gO[k * d + dd], std::conj(a) * gOb[k * d + dd]);
					}
					// a = (s w) c with c = sum_j Q[k,j] conj(K[m,j]); s w real, from the cumulative log-gates
					const float swr = scale_ * std::exp(Lam_[(size_t) h * N + k] - Lam_[(size_t) h * N + m]);
					for (int j = 0; j < dh; ++j) {
						const int dd = d0 + j;
						const std::complex<float> q = input().z(qi(k, dd));
						const std::complex<float> kk = input().z(ki(m, dd));
						accZ(qi(k, dd), p * swr * std::conj(kk), pb * swr * kk);
						accZ(ki(m, dd), pb * swr * std::conj(q), p * swr * q);
					}
					// gate chain: dL/dw-part -> every lambda_u with m < u <= k gets e[k,m]
					e[m] = (p * a + pb * std::conj(a)).real();
				}
				// prefix sums: dLam[u] += sum_{m<u} e[k,m] for u in (0..k]
				double run = 0.0;
				for (int u = 1; u <= k; ++u) { run += e[u - 1]; dLam[u] += run; }
			}
			for (int u = 0; u < N; ++u) {
				const float x = x_[(size_t) h * N + u];
				const float dlam_dx = 1.f / (1.f + std::exp(x));      // sigmoid(-x)
				const float dx = (float) dLam[u] * dlam_dx * 0.5f;     // Wirtinger: d/dG = d/dconjG = (dL/dx)/2
				accZ(gi(u, h), {dx, 0.f}, {dx, 0.f});
			}
		}
	}
	virtual void backward(int label) {}
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

#endif /* IMPL_GLA_H_ */
