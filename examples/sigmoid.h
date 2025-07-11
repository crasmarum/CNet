#ifndef EXAMPLES_SIGMOID_H_
#define EXAMPLES_SIGMOID_H_

#include <complex>

#include "../impl/cfunc.h"
#include "../impl/vars.h"

/**
 * Implementing the complex sigmoid layer, the multivariable
 * complex function CSigmoid : C^n -> C^n
 * C(..., z_i, ...) = (..., 1 / (1 + e ^(-z_i), ...)
 */
class CSigmoid: public CFunc {

public:
	CSigmoid(Uid uid, InSize in_size) :
			CFunc(uid, in_size, OutSize(in_size.value())) {
	}

	CSigmoid(InSize in_size) :
			CFunc(in_size, OutSize(in_size.value())) {
	}

	virtual ~CSigmoid() {
	}

	virtual CFunc* clone(Uid uid) {
		return new CSigmoid(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "CSigmoid_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int i = 0; i < no_outputs(); ++i) {
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				std::complex<float> g = 1.0f / (1.0f + std::exp(-input().z(in_indx)));
				mutable_output(i)->real_[offset(i) + in_indx] = g.real();
				mutable_output(i)->imag_[offset(i) + in_indx] = g.imag();
			}
		}
	}

	virtual std::complex<float> dz(int out_indx, int in_indx) {
		if (out_indx != in_indx) {
			return 0;
		}
		auto g = 1.0f / (1.0f + std::exp(-input().z(in_indx)));
		return g * (1.0f - g);
	}

	virtual std::complex<float> dz_star(int out_indx, int in_indx) {
		return 0;
	}

	bool operator==(const CSigmoid &rhs) const {
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
	bool operator!=(const CSigmoid &rhs) const {
		return !operator==(rhs);
	}
};

#endif /* EXAMPLES_SIGMOID_H_ */
