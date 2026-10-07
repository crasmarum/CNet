#ifndef IMPL_TOKENNORM_H_
#define IMPL_TOKENNORM_H_

#include <vector>
#include <complex>
#include <cmath>
#include "cfunc.h"
#include "utils.h"

// Per-token complex RMS normalization (parameter-free), fully complex.
//
// Input is N tokens of d complex features each (token t, dim dd at index
// t*d + dd); output has the same shape. Each token is normalized independently
// by its own root-mean-power so that per-token feature scale is held constant
// through depth -- the complex analogue of (gain-free) RMSNorm/LayerNorm, and
// the per-token counterpart of the global per-block SoftMax L2 norm. A learnable
// affine is intentionally omitted: a following TokenwiseLinear supplies gain and
// bias, which keeps this layer's Wirtinger backward exact and parameter-free.
//
//   s_t   = sum_dd |x[t,dd]|^2                    (per-token power)
//   r_t   = sqrt(s_t / d + eps)                   (real; stored for backward)
//   y[t,dd] = x[t,dd] / r_t
//
// Backward (verified against finite differences). With gO = dL/dy and
// gOb = dL/dy* the output gradients of one token, and
//   T = sum_dd ( gO[dd] x[dd] + gOb[dd] conj(x[dd]) ),
// the exact CR-calculus input gradients are
//   dL/dx[dd]  = gO[dd]/r_t  - conj(x[dd]) * T / (2 d r_t^3)
//   dL/dx*[dd] = gOb[dd]/r_t - x[dd]       * T / (2 d r_t^3)
// (the second term is the shared coupling through r_t across the token's dims).
class TokenNorm: public CFunc {
	int N_, d_;
	float eps_;
	std::vector<float> r_;                 // per-token r_t (forward -> backward)
public:
	TokenNorm(Uid uid, int n_tokens, int dim, float eps = 1e-5f)
		: CFunc(uid, InSize(n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), eps_(eps), r_(n_tokens, 1.f) {}

	TokenNorm(int n_tokens, int dim, float eps = 1e-5f)
		: CFunc(InSize(n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), eps_(eps), r_(n_tokens, 1.f) {}

	virtual ~TokenNorm() {}
	virtual CFunc* clone(Uid uid) { return new TokenNorm(uid, N_, d_, eps_); }
	virtual std::string getName() { return "TokenNorm_" + std::to_string(uid_); }

	int nTokens() const { return N_; }
	int dim() const { return d_; }
	float eps() const { return eps_; }
	int M() const { return N_ * d_; }
	inline int idx(int t, int dd) const { return t * d_ + dd; }

	virtual void forward() {
		assert(no_outputs());
		const int N = N_, d = d_;
		for (int t = 0; t < N; ++t) {
			double s = 0.0;
			for (int dd = 0; dd < d; ++dd) {
				std::complex<float> x = input().z(idx(t, dd));
				s += (double) x.real() * x.real() + (double) x.imag() * x.imag();
			}
			float r = std::sqrt((float) (s / d) + eps_);
			r_[t] = r;
			float inv = 1.f / r;
			for (int dd = 0; dd < d; ++dd) {
				std::complex<float> x = input().z(idx(t, dd));
				std::complex<float> y = x * inv;
				for (int o = 0; o < no_outputs(); ++o) {
					output(o).real_[offset(o) + idx(t, dd)] = y.real();
					output(o).imag_[offset(o) + idx(t, dd)] = y.imag();
				}
			}
		}
	}

	virtual void backward() {
		const int N = N_, d = d_;
		for (int t = 0; t < N; ++t) {
			float r = r_[t];
			float inv = 1.f / r;
			float coup = 1.f / (2.f * (float) d * r * r * r);   // 1/(2 d r^3)
			// Sum output gradients across consumers, and accumulate T over dims.
			std::vector<std::complex<float>> gO(d, {0.f, 0.f}), gOb(d, {0.f, 0.f});
			std::complex<float> T(0.f, 0.f);
			for (int dd = 0; dd < d; ++dd) {
				std::complex<float> go(0.f, 0.f), gob(0.f, 0.f);
				for (int o = 0; o < no_outputs(); ++o) {
					go  += output(o).dz(offset(o) + idx(t, dd));
					gob += output(o).dz_star(offset(o) + idx(t, dd));
				}
				gO[dd] = go; gOb[dd] = gob;
				std::complex<float> x = input().z(idx(t, dd));
				T += go * x + gob * std::conj(x);
			}
			for (int dd = 0; dd < d; ++dd) {
				std::complex<float> x = input().z(idx(t, dd));
				std::complex<float> dz     = gO[dd]  * inv - std::conj(x) * T * coup;
				std::complex<float> dzstar = gOb[dd] * inv - x            * T * coup;
				accZ(idx(t, dd), dz, dzstar);
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

#endif /* IMPL_TOKENNORM_H_ */
