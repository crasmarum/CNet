#ifndef IMPL_SOFTMAX_H_
#define IMPL_SOFTMAX_H_

#include <cfloat>

#include "cfunc.h"
#include "vars.h"

class SoftMax: public CFunc {
	float square_norm_;
	float norm_2_3;

public:
	SoftMax(Uid uid, InSize in_size) : CFunc(uid, in_size,
			OutSize(in_size.value())), square_norm_(0), norm_2_3(0) {
	}

	SoftMax(InSize in_size) : CFunc(in_size, OutSize(in_size.value())),
		square_norm_(0), norm_2_3(0) {
	}

	virtual ~SoftMax() {
	}

	virtual CFunc* clone(Uid uid) {
		return new SoftMax(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "SoftMax_" + std::to_string(uid_);
	}

	virtual void forward() {
		double sum = 0.0;

		#pragma omp parallel for reduction (+:sum)
		for (int pos = 0; pos < input().length_; ++pos) {
			sum = sum + (std::pow(input().real_[pos], 2) + std::pow(input().imag_[pos], 2));
		}

		square_norm_ = sum;
		sum = sqrt(sum);
		if(sum <= 1.0e-15) {
			sum = 1.0e-15;
		}
		norm_2_3 = pow(sum, 3);
		if(norm_2_3 < 1.0e-15) {
			sum = 1.0e-15;
		}

		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int pos = 0; pos < input().length_; ++pos) {
				output(indx).real_[offset(indx) + pos] = (float) input().real_[pos] / sum;
				output(indx).imag_[offset(indx) + pos] = (float) input().imag_[pos] / sum;
			}
		}
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		if (in_index == out_indx) {
			return 0.5f * (std::complex<float>(2.0f * square_norm_, 0)
					- input().z(in_index) * std::conj(input().z(in_index))) / norm_2_3;
		}
		return -0.5f * input().z(out_indx) * std::conj(input().z(in_index)) / norm_2_3;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
        if (out_indx == in_index) {
    		return -0.5f * input().z(in_index) * input().z(in_index) / norm_2_3;
        }
		return -0.5f * input().z(out_indx) * input().z(in_index) / norm_2_3;
	}

	bool operator==(const SoftMax& rhs) const
	{
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
	bool operator!=(const SoftMax& rhs) const
	{
		return !operator==(rhs);
	}
};



#endif /* IMPL_SOFTMAX_H_ */
