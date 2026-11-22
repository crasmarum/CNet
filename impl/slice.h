#ifndef IMPL_SLICE_H_
#define IMPL_SLICE_H_

#include <cassert>
#include <complex>
#include <string>
#include "cfunc.h"
#include "vars.h"

// Slice: copy one contiguous range of the input,
//
//     out[i] = in[start + i],   0 <= i < out_size.
//
// It is the identity on the range (holomorphic), so the Wirtinger backward is
//     dL/din[start+i]  += dL/dout[i],     dL/din*[start+i] += dL/dout*[i],
// and every other input position gets no gradient.
//
// Use: split one batch input that carries several per-example signals, e.g. the noisy and
// (negated) clean halves of a speech-enhancement example [x | -s] -> x, -s (the framework has a
// single batch input, CNET_NOTES.md), or read one token (the last one) out of a sequence.
class Slice: public CFunc {
	int start_;

public:
	Slice(Uid uid, InSize in_size, OutSize out_size, int start)
			: CFunc(uid, in_size, out_size), start_(start) {
		assert(start >= 0 && start + out_size.value() <= in_size.value());
	}

	Slice(InSize in_size, OutSize out_size, int start) : CFunc(in_size, out_size), start_(start) {
		assert(start >= 0 && start + out_size.value() <= in_size.value());
	}

	virtual ~Slice() {
	}

	int start() const { return start_; }

	virtual CFunc* clone(Uid uid) {
		return new Slice(uid, InSize(input().length_), OutSize(out_size_), start_);
	}

	virtual std::string getName() {
		return "Slice_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < out_size_; ++i) {
				output(indx).real_[offset(indx) + i] = input().real_[start_ + i];
				output(indx).imag_[offset(indx) + i] = input().imag_[start_ + i];
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < out_size_; ++i) {
				auto dz      = output(indx).dz(offset(indx) + i);
				auto dz_star = output(indx).dz_star(offset(indx) + i);
				input().dz_real_[start_ + i]      += dz.real();
				input().dz_imag_[start_ + i]      += dz.imag();
				input().dz_star_real_[start_ + i] += dz_star.real();
				input().dz_star_imag_[start_ + i] += dz_star.imag();
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

#endif /* IMPL_SLICE_H_ */
