#ifndef IMPL_INTERLEAVE_H_
#define IMPL_INTERLEAVE_H_

#include <cassert>
#include <complex>
#include <string>

#include "cfunc.h"
#include "vars.h"

// Interleave: K inputs of the same length L, concatenated in the input vector (input k at [k L, (k+1) L)),
// become one vector with the K values of each position side by side,
//
//     out(i K + k) = in(k L + i),     0 <= i < L, 0 <= k < K,
//
// so that per-position layers (TimeFreqConv with C_in = K, TokenwiseLinear over L tokens of K features) see the
// K signals of a position as its channels -- e.g. the noisy spectrum and a per-bin projection of the trunk token.
// A permutation: holomorphic, so dz and dz~ are routed back unchanged.
class Interleave: public CFunc {
	friend class ModelSaver;
	int length_, k_;

public:
	Interleave(Uid uid, int length, int k) : CFunc(uid, InSize(length * k), OutSize(length * k)), length_(length), k_(k) {
		assert(length > 0 && k > 0);
	}
	Interleave(int length, int k) : CFunc(InSize(length * k), OutSize(length * k)), length_(length), k_(k) {
		assert(length > 0 && k > 0);
	}
	virtual ~Interleave() {
	}
	virtual CFunc* clone(Uid uid) {
		return new Interleave(uid, length_, k_);
	}
	virtual std::string getName() {
		return "Interleave_" + std::to_string(uid_);
	}
	int length() const { return length_; }
	int k()      const { return k_; }

	virtual void forward() {
		assert(no_outputs());
		const int L = length_, K = k_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int oi = 0; oi < L * K; ++oi) {
				const int i = oi / K, k = oi % K;
				output(indx).real_[offset(indx) + oi] = input().real_[k * L + i];
				output(indx).imag_[offset(indx) + oi] = input().imag_[k * L + i];
			}
		}
	}

	virtual void backward() {
		const int L = length_, K = k_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int oi = 0; oi < L * K; ++oi) {
				const int i = oi / K, k = oi % K, src = k * L + i;
				const auto dz = output(indx).dz(offset(indx) + oi);
				const auto dz_star = output(indx).dz_star(offset(indx) + oi);
				input().dz_real_[src]      += dz.real();
				input().dz_imag_[src]      += dz.imag();
				input().dz_star_real_[src] += dz_star.real();
				input().dz_star_imag_[src] += dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}
	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}
};

#endif /* IMPL_INTERLEAVE_H_ */
