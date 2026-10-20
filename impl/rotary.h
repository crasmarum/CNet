#ifndef IMPL_ROTARY_H_
#define IMPL_ROTARY_H_

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include "cfunc.h"
#include "utils.h"

// Rotary position embedding (RoPE) for complex tokens, parameter-free.
//
// Input is N tokens of d complex features each (token t, dim dd at index t*d+dd).
// Each feature is rotated by a position-dependent unit phasor:
//
//   y[t,dd] = x[t,dd] * exp(i * theta_dd * t),   theta_dd = base^{-dd/d}.
//
// Applied to the queries and keys of BornAttention, the per-feature inner product
// Q[k,dd] conj(K[j,dd]) then carries the RELATIVE phase exp(i*theta_dd*(k-j)), so
// the attention score |<Q_k,K_j>|^2 becomes position-aware (RoPE), which is just a
// complex rotation in this framework. In complex arithmetic a single complex
// dimension is one rotation plane, so there is one frequency per complex feature.
//
// The map y = r*x (r a unit-modulus constant per position/feature) is holomorphic:
// with gO = dL/dy and gOb = dL/dy*, the exact Wirtinger input gradients are
//   dL/dx  = gO  * r,    dL/dx* = gOb * conj(r).
class RotaryEmbed: public CFunc {
	int N_, d_, H_;                          // heads: the frequency ladder restarts every dh = d/H dims
	double base_;
	int pos_ = 0;                            // absolute position of token 0 (streaming decode: N=1 graphs)
	std::vector<std::complex<float>> rot_;   // r[t*d+dd] = exp(i theta_(dd mod dh) t)
	void buildRot() {
		rot_.resize((size_t) N_ * d_);
		const int dh = d_ / H_;
		for (int t = 0; t < N_; ++t)
			for (int dd = 0; dd < d_; ++dd) {
				double theta = std::pow(base_, -(double) (dd % dh) / dh);
				double ang = theta * t;
				rot_[(size_t) t * d_ + dd] = std::complex<float>((float) std::cos(ang), (float) std::sin(ang));
			}
	}
public:
	RotaryEmbed(Uid uid, int n_tokens, int dim, double base = 10000.0, int heads = 1)
		: CFunc(uid, InSize(n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), H_(heads), base_(base) { assert(heads >= 1 && dim % heads == 0); buildRot(); }

	RotaryEmbed(int n_tokens, int dim, double base = 10000.0, int heads = 1)
		: CFunc(InSize(n_tokens * dim), OutSize(n_tokens * dim)),
		  N_(n_tokens), d_(dim), H_(heads), base_(base) { assert(heads >= 1 && dim % heads == 0); buildRot(); }

	virtual ~RotaryEmbed() {}
	virtual CFunc* clone(Uid uid) { return new RotaryEmbed(uid, N_, d_, base_, H_); }
	virtual std::string getName() { return "RotaryEmbed_" + std::to_string(uid_); }

	int nTokens() const { return N_; }
	int dim() const { return d_; }
	double base() const { return base_; }
	int heads() const { return H_; }
	void setPos(int pos) { pos_ = pos; }     // token t of this window sits at absolute position pos + t
	int pos() const { return pos_; }
	std::complex<float> rotAt(int t, int dd) const {
		const int dh = d_ / H_;
		double ang = std::pow(base_, -(double) (dd % dh) / dh) * (double) (pos_ + t);
		return std::complex<float>((float) std::cos(ang), (float) std::sin(ang));
	}
	int M() const { return N_ * d_; }

	virtual void forward() {
		assert(no_outputs());
		for (int p = 0; p < M(); ++p) {
			std::complex<float> y = input().z(p) * (pos_ ? rotAt(p / d_, p % d_) : rot_[p]);
			for (int o = 0; o < no_outputs(); ++o) {
				output(o).real_[offset(o) + p] = y.real();
				output(o).imag_[offset(o) + p] = y.imag();
			}
		}
	}

	virtual void backward() {
		for (int p = 0; p < M(); ++p) {
			std::complex<float> gO(0.f, 0.f), gOb(0.f, 0.f);
			for (int o = 0; o < no_outputs(); ++o) {
				gO  += output(o).dz(offset(o) + p);
				gOb += output(o).dz_star(offset(o) + p);
			}
			std::complex<float> r = rot_[p];
			accZ(p, gO * r, gOb * std::conj(r));
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

#endif /* IMPL_ROTARY_H_ */
