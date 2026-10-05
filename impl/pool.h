#ifndef IMPL_POOL_H_
#define IMPL_POOL_H_

#include <complex>

#include "cfunc.h"
#include "vars.h"

// Mean pooling over time: a length-L input is split into out_size contiguous
// segments of width = L/out_size, and each output is the mean of its segment
// (global average pooling when out_size == 1). This builds shift-robust summary
// statistics, e.g. the mean constellation power after a |z|^2 layer.
//
// Mean is a real linear (holomorphic) map w = A z with A_{p,i} = 1/width for i
// in segment p, so the Wirtinger derivatives are simply:
//   dL/dz_i  = dLdz(pool(i))      / width
//   dL/dz~_i = dLdz_star(pool(i)) / width
class MeanPool: public CFunc {
	int width_;   // input samples per output bin

public:
	MeanPool(Uid uid, InSize in_size, OutSize out_size)
			: CFunc(uid, in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		width_ = in_size.value() / out_size.value();
	}

	MeanPool(InSize in_size, OutSize out_size) : CFunc(in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		width_ = in_size.value() / out_size.value();
	}

	virtual ~MeanPool() {
	}

	virtual CFunc* clone(Uid uid) {
		return new MeanPool(uid, InSize(input().length_), OutSize(out_size_));
	}

	virtual std::string getName() {
		return "MeanPool_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int p = 0; p < out_size_; ++p) {
				std::complex<float> sum = 0;
				for (int j = 0; j < width_; ++j) {
					sum += input().z(p * width_ + j);
				}
				sum /= (float) width_;
				output(indx).real_[offset(indx) + p] = sum.real();
				output(indx).imag_[offset(indx) + p] = sum.imag();
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < input().length_; ++i) {
				int p = i / width_;
				auto dz      = output(indx).dz(offset(indx) + p) / (float) width_;
				auto dz_star = output(indx).dz_star(offset(indx) + p) / (float) width_;
				input().dz_real_[i]      += dz.real();
				input().dz_imag_[i]      += dz.imag();
				input().dz_star_real_[i] += dz_star.real();
				input().dz_star_imag_[i] += dz_star.imag();
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

	bool operator==(const MeanPool& rhs) const {
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}
		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const MeanPool& rhs) const {
		return !operator==(rhs);
	}
};

#endif /* IMPL_POOL_H_ */
