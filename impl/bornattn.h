#ifndef IMPL_BORNATTN_H_
#define IMPL_BORNATTN_H_

#include <vector>
#include <complex>
#include "cfunc.h"
#include "utils.h"

// Born-rule causal self-attention (single head), fully complex.
//
// Input is the concatenation of three complex blocks [Q ; K ; V], each of length
// N*d (token t, dim dd at index t*d + dd). Output is O of length N*d:
//
//   c[k,m] = sum_d Q[k,d] conj(K[m,d])          (keys m <= k; causal)
//   s[k,m] = |c[k,m]|^2                          (Born-rule measurement score)
//   A[k,m] = s[k,m] / sum_{m'<=k} s[k,m']        (causal normalization)
//   O[k,d] = sum_{m<=k} A[k,m] V[m,d]
//
// The scores ARE measurement probabilities of query k against key m, so routing
// is content-based like attention but uses |<Q,K>|^2 instead of a softmaxed
// dot product. The Wirtinger backward (verified to ~1e-9 against finite
// differences) is below.
class BornAttention: public CFunc {
	int N_, d_;
	std::vector<std::complex<float>> c_;   // c[k*N+m], m<=k
	std::vector<float> A_, S_;             // A[k*N+m], S[k]
public:
	BornAttention(Uid uid, int n_tokens, int dim)
		: CFunc(uid, InSize(3 * n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), c_(n_tokens * n_tokens), A_(n_tokens * n_tokens), S_(n_tokens) {}

	BornAttention(int n_tokens, int dim)
		: CFunc(InSize(3 * n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), c_(n_tokens * n_tokens), A_(n_tokens * n_tokens), S_(n_tokens) {}

	virtual ~BornAttention() {}
	virtual CFunc* clone(Uid uid) { return new BornAttention(uid, N_, d_); }
	virtual std::string getName() { return "BornAttention_" + std::to_string(uid_); }

	int M() const { return N_ * d_; }
	int nTokens() const { return N_; }
	int dim() const { return d_; }
	// absolute input index of block b (0=Q,1=K,2=V), token t, dim dd.
	inline int qi(int t, int dd) const { return t * d_ + dd; }
	inline int ki(int t, int dd) const { return M() + t * d_ + dd; }
	inline int vi(int t, int dd) const { return 2 * M() + t * d_ + dd; }

	virtual void forward() {
		assert(no_outputs());
		const int N = N_, d = d_;
		std::vector<std::complex<float>> O(N * d, {0.f, 0.f});
		for (int k = 0; k < N; ++k) {
			float Sk = 0.f;
			for (int m = 0; m <= k; ++m) {
				std::complex<float> ckm = 0.f;
				for (int dd = 0; dd < d; ++dd)
					ckm += input().z(qi(k, dd)) * std::conj(input().z(ki(m, dd)));
				c_[k * N + m] = ckm;
				float s = std::norm(ckm);           // |ckm|^2
				A_[k * N + m] = s;                  // store s; normalize below
				Sk += s;
			}
			S_[k] = Sk;
			float inv = Sk > 0.f ? 1.f / Sk : 0.f;
			for (int m = 0; m <= k; ++m) {
				float a = A_[k * N + m] * inv;       // A[k,m]
				A_[k * N + m] = a;
				for (int dd = 0; dd < d; ++dd)
					O[k * d + dd] += a * input().z(vi(m, dd));
			}
		}
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < M(); ++p) {
				output(indx).real_[offset(indx) + p] = O[p].real();
				output(indx).imag_[offset(indx) + p] = O[p].imag();
			}
	}

	virtual void backward() {
		const int N = N_, d = d_;
		// Sum incoming output gradients across all consumers.
		std::vector<std::complex<float>> gO(M(), {0.f, 0.f}), gOb(M(), {0.f, 0.f});
		for (int indx = 0; indx < no_outputs(); ++indx)
			for (int p = 0; p < M(); ++p) {
				gO[p]  += output(indx).dz(offset(indx) + p);
				gOb[p] += output(indx).dz_star(offset(indx) + p);
			}
		for (int k = 0; k < N; ++k) {
			// a[m] = sum_d Re( gO[k,d] V[m,d] + gOb[k,d] conj(V[m,d]) )
			std::vector<float> a(k + 1, 0.f);
			float abar = 0.f;
			for (int m = 0; m <= k; ++m) {
				float am = 0.f;
				for (int dd = 0; dd < d; ++dd) {
					std::complex<float> v = input().z(vi(m, dd));
					am += (gO[k * d + dd] * v + gOb[k * d + dd] * std::conj(v)).real();
				}
				a[m] = am;
				abar += A_[k * N + m] * am;
			}
			float inv = S_[k] > 0.f ? 1.f / S_[k] : 0.f;
			for (int m = 0; m <= k; ++m) {
				float b = (a[m] - abar) * inv;            // dL/ds[k,m]-chain (real)
				std::complex<float> ckm = c_[k * N + m], ckmc = std::conj(ckm);
				float Akm = A_[k * N + m];
				for (int dd = 0; dd < d; ++dd) {
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
