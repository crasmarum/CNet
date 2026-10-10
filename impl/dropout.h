#ifndef IMPL_DROPOUT_H_
#define IMPL_DROPOUT_H_

#include "cfunc.h"
#include "vars.h"

// Complex dropout. Multiplicative per-element mask applied during TRAINING (the
// GPU path, see gpu/kernels.*): each element is kept with prob (1-p) and scaled
// by 1/(1-p) (inverted dropout), so the expected activation is unchanged. The
// mask is a deterministic hash of (seed, step, element) recomputed identically in
// forward and backward, so no mask buffer is stored; the GPU mapping bumps `step`
// each forward for a fresh mask. The CPU path (generate/eval = inference) is the
// identity -- dropout is a no-op at inference.
//
// Motivation: structured multiplicative noise as a symmetry-breaker for the Born
// unigram saddle (where isotropic gradient noise failed). Being part of the
// forward computation, it perturbs the whole backward pass coherently rather than
// averaging out like noise added to the final gradient.
class ComplexDropout : public CFunc {
	float p_;   // drop probability in [0,1)

public:
	ComplexDropout(Uid uid, InSize in_size, float p)
		: CFunc(uid, in_size, OutSize(in_size.value())), p_(p) {}

	ComplexDropout(InSize in_size, float p)
		: CFunc(in_size, OutSize(in_size.value())), p_(p) {}

	virtual ~ComplexDropout() {}

	float dropP() const { return p_; }

	virtual CFunc* clone(Uid uid) {
		return new ComplexDropout(uid, InSize(input().length_), p_);
	}

	virtual std::string getName() {
		return "ComplexDropout_" + std::to_string(uid_);
	}

	// CPU = inference: pass through unchanged (no dropout at generate/eval time).
	virtual void forward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int o = 0; o < out_size_; ++o) {
				output(indx).real_[offset(indx) + o] = input().real_[o];
				output(indx).imag_[offset(indx) + o] = input().imag_[o];
			}
		}
	}

	virtual void backward() {
		for (int oi = 0; oi < no_outputs(); ++oi) {
			#pragma omp parallel for
			for (int i = 0; i < input().length_; ++i) {
				auto dLdz      = output(oi).dz(offset(oi) + i);
				auto dLdz_star = output(oi).dz_star(offset(oi) + i);
				input().dz_real_[i]      += dLdz.real();
				input().dz_imag_[i]      += dLdz.imag();
				input().dz_star_real_[i] += dLdz_star.real();
				input().dz_star_imag_[i] += dLdz_star.imag();
			}
		}
	}

	virtual void backward(int label) {}

	virtual std::complex<float>      dz(int out_indx, int in_index) { return 0; }
	virtual std::complex<float> dz_star(int out_indx, int in_index) { return 0; }
};

#endif /* IMPL_DROPOUT_H_ */
