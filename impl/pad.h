#ifndef IMPL_PAD_H_
#define IMPL_PAD_H_

#include <algorithm>
#include <vector>
#include <complex>

#include "cfunc.h"
#include "vars.h"

// Pad: scatter a small set of free kernel taps into a larger zero vector, at a
// fixed index set, producing a spatial-domain padded kernel.  Followed by a
// FourierTrans this yields the frequency-domain Hadamard filter, so that
// Hadamard(FFT(x), FFT(pad(k))) = circular-convolution(x, k) with a kernel of
// only |kernel_index| free taps (the 7x7 local-kernel constraint of
// arXiv:1810.11650).  input().length_ == kernel_index.size() (e.g. 49),
// out_size_ == padded length (e.g. 784).  All non-kernel positions stay 0.
class Pad: public CFunc {
	std::vector<int> kernel_index_;

public:
	Pad(Uid uid, InSize in_size, OutSize out_size, std::vector<int> kernel_index)
			: CFunc(uid, in_size, out_size), kernel_index_(kernel_index) {
		assert((int) kernel_index_.size() == in_size.value());
		assert(out_size.value() >= in_size.value());
	}

	Pad(InSize in_size, OutSize out_size, std::vector<int> kernel_index)
			: CFunc(in_size, out_size), kernel_index_(kernel_index) {
		assert((int) kernel_index_.size() == in_size.value());
		assert(out_size.value() >= in_size.value());
	}

	virtual ~Pad() {
	}

	virtual CFunc* clone(Uid uid) {
		return new Pad(uid, InSize(input().length_), OutSize(out_size_), kernel_index_);
	}

	virtual std::string getName() {
		return "Pad_" + std::to_string(uid_);
	}

	const std::vector<int>& kernel_index() const {
		return kernel_index_;
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			float *re = output(indx).real_ + offset(indx);
			float *im = output(indx).imag_ + offset(indx);
			std::fill(re, re + out_size_, 0.0f);
			std::fill(im, im + out_size_, 0.0f);
			for (int k = 0; k < input().length_; ++k) {
				re[kernel_index_[k]] = input().real_[k];
				im[kernel_index_[k]] = input().imag_[k];
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int k = 0; k < input().length_; ++k) {
				int out_pos = offset(indx) + kernel_index_[k];
				auto dLdz      = output(indx).dz(out_pos);
				auto dLdz_star = output(indx).dz_star(out_pos);
				input().dz_real_[k]      += dLdz.real();
				input().dz_imag_[k]      += dLdz.imag();
				input().dz_star_real_[k] += dLdz_star.real();
				input().dz_star_imag_[k] += dLdz_star.imag();
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

	bool operator==(const Pad& rhs) const {
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()
				|| this->kernel_index_ != rhs.kernel_index_) {
			return false;
		}
		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const Pad& rhs) const {
		return !operator==(rhs);
	}
};

#endif /* IMPL_PAD_H_ */
